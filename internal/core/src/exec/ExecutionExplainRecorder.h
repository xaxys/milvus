// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <fmt/core.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/Types.h"
#include "pb/internal.pb.h"

namespace milvus::exec {

// Collects what one segment did for one query and groups it by decision, so the
// payload size follows the number of distinct decisions rather than the number of
// filter leaves.
//
// Four counters describe one leaf on one segment:
//   rows_in  rows the leaf was handed (batch sums; smaller than the segment when
//            an upstream conjunction narrowed the input)
//   idx_scan rows it handled through an index path; equals rows_in for an index
//            leaf, zero for a raw leaf. Coverage, not index-internal I/O.
//   raw_scan rows it read from the raw column (raw leaf: rows_in; index leaf that
//            rechecks candidates: the candidates read)
//   rows_out rows where the predicate is true
// Element-level leaves inside MatchExpr report rows_in, idx_scan, raw_scan,
// and rows_out in the flattened struct-element domain rather than the row domain.
class ExecutionExplainRecorder {
 public:
    // Stable per-leaf atomic accumulator stored inside std::deque so pointers
    // handed out at expression compile/bind time remain valid across subsequent
    // slot insertions and allow lock-free hot-path counting and timing.
    struct LeafSlot {
        explicit LeafSlot(int64_t id) noexcept : expr_id(id) {
        }
        LeafSlot(const LeafSlot&) = delete;
        LeafSlot&
        operator=(const LeafSlot&) = delete;

        const int64_t expr_id;
        std::atomic<int64_t> rows_in{0};
        std::atomic<int64_t> raw_scan{0};
        std::atomic<int64_t> rows_out{0};
        std::atomic<int64_t> eval_ns{0};
        std::atomic<int64_t> prefetch_ns{0};
        std::atomic<bool> evaluated{false};
    };

    class ScopedLeafEvalTimer {
     public:
        explicit ScopedLeafEvalTimer(LeafSlot* slot) noexcept
            : slot_(slot),
              start_(slot != nullptr
                         ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point{}) {
        }

        ScopedLeafEvalTimer(ScopedLeafEvalTimer&& other) noexcept
            : slot_(std::exchange(other.slot_, nullptr)), start_(other.start_) {
        }

        ScopedLeafEvalTimer(const ScopedLeafEvalTimer&) = delete;
        ScopedLeafEvalTimer&
        operator=(const ScopedLeafEvalTimer&) = delete;

        ~ScopedLeafEvalTimer() noexcept {
            if (slot_ != nullptr) {
                const auto ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start_)
                        .count();
                if (ns > 0) {
                    slot_->eval_ns.fetch_add(ns, std::memory_order_relaxed);
                }
            }
        }

     private:
        LeafSlot* slot_;
        std::chrono::steady_clock::time_point start_;
    };

    explicit ExecutionExplainRecorder(int64_t rows_total)
        : rows_total_(rows_total) {
    }

    LeafSlot*
    EnsureLeafSlot(int64_t expr_id) {
        if (expr_id == 0) {
            return nullptr;
        }
        std::lock_guard lock(mu_);
        return EnsureLeafSlotLocked(expr_id);
    }

    static void
    AddSlotRows(LeafSlot* slot, int64_t rows_in, int64_t rows_out) noexcept {
        if (slot == nullptr) {
            return;
        }
        if (rows_in > 0) {
            slot->evaluated.store(true, std::memory_order_relaxed);
        }
        if (rows_in != 0) {
            slot->rows_in.fetch_add(rows_in, std::memory_order_relaxed);
        }
        if (rows_out != 0) {
            slot->rows_out.fetch_add(rows_out, std::memory_order_relaxed);
        }
    }

    static void
    AddSlotRawScan(LeafSlot* slot, int64_t rows) noexcept {
        if (slot != nullptr && rows > 0) {
            slot->raw_scan.fetch_add(rows, std::memory_order_relaxed);
        }
    }

    static void
    AddSlotEvalNs(LeafSlot* slot, int64_t ns) noexcept {
        if (slot != nullptr && ns > 0) {
            slot->eval_ns.fetch_add(ns, std::memory_order_relaxed);
        }
    }

    static void
    AddSlotPrefetchNs(LeafSlot* slot, int64_t ns) noexcept {
        if (slot != nullptr && ns > 0) {
            slot->prefetch_ns.fetch_add(ns, std::memory_order_relaxed);
        }
    }

    int32_t
    CurrentEvalOrder() const {
        std::lock_guard lock(mu_);
        return next_eval_order_;
    }

    void
    AdvanceNextEvalOrder(int32_t next) {
        std::lock_guard lock(mu_);
        if (next > next_eval_order_) {
            next_eval_order_ = next;
        }
    }

    void
    SetEvalOrder(int64_t expr_id,
                 int32_t order,
                 FieldId field_id = FieldId(0)) {
        if (expr_id == 0 || order <= 0) {
            return;
        }
        std::lock_guard lock(mu_);
        EnsureLeafSlotLocked(expr_id);
        eval_orders_[expr_id] = order;
        if (field_id.get() != 0) {
            eval_field_ids_[expr_id] = field_id.get();
        }
    }

    void
    MarkEvalEntered(int64_t expr_id) {
        if (expr_id == 0) {
            return;
        }
        std::lock_guard lock(mu_);
        EnsureLeafSlotLocked(expr_id);
        eval_entered_exprs_.insert(expr_id);
    }

    bool
    HasAnyEvalEntered() const {
        std::lock_guard lock(mu_);
        return !eval_entered_exprs_.empty();
    }

    void
    SetRowsTotal(int64_t rows_total) {
        std::lock_guard lock(mu_);
        rows_total_ = rows_total;
    }

    void
    MarkDomainEmpty(int64_t expr_id) {
        if (expr_id == 0) {
            return;
        }
        std::lock_guard lock(mu_);
        if (non_empty_domain_exprs_.count(expr_id) == 0) {
            empty_domain_exprs_.insert(expr_id);
        }
    }

    void
    MarkDomainNonEmpty(int64_t expr_id) {
        if (expr_id == 0) {
            return;
        }
        std::lock_guard lock(mu_);
        non_empty_domain_exprs_.insert(expr_id);
        empty_domain_exprs_.erase(expr_id);
    }

    void
    RegisterOperator(int32_t step, std::string name, std::string domain = "") {
        if (step < 0) {
            return;
        }
        std::lock_guard lock(mu_);
        EnsureOperatorSlotLocked(step);
        auto& op = operators_[step];
        op.step = step;
        op.name = std::move(name);
        if (!domain.empty()) {
            op.domain = std::move(domain);
        }
    }

    void
    SetOperatorDomain(int32_t step, std::string domain) {
        if (step < 0 || domain.empty()) {
            return;
        }
        std::lock_guard lock(mu_);
        if (static_cast<size_t>(step) < operators_.size()) {
            operators_[step].domain = std::move(domain);
        }
    }

    void
    SetOperatorRows(int32_t step, int64_t rows_in, int64_t rows_out) {
        if (step < 0) {
            return;
        }
        std::lock_guard lock(mu_);
        if (static_cast<size_t>(step) < operators_.size()) {
            operators_[step].rows_in = rows_in;
            operators_[step].rows_out = rows_out;
        }
    }

    void
    AddOperatorRowsIn(int32_t step, int64_t delta) {
        if (step < 0 || delta == 0) {
            return;
        }
        std::lock_guard lock(mu_);
        if (static_cast<size_t>(step) < operators_.size()) {
            operators_[step].rows_in += delta;
        }
    }

    void
    AddOperatorRowsOut(int32_t step, int64_t delta) {
        if (step < 0 || delta == 0) {
            return;
        }
        std::lock_guard lock(mu_);
        if (static_cast<size_t>(step) < operators_.size()) {
            operators_[step].rows_out += delta;
        }
    }

    void
    AddOperatorTimeNs(int32_t step, int64_t ns) {
        if (step < 0 || ns <= 0) {
            return;
        }
        std::lock_guard lock(mu_);
        if (static_cast<size_t>(step) < operators_.size()) {
            operators_[step].time_ns += ns;
        }
    }

    // Transfers lazy work performed inside debit_step's GetOutput back to
    // credit_step (e.g., CountingVectorIterator or StrictGroup Phase 2).
    // Allows debit_step's accumulator to go temporarily negative before
    // Driver flushes debit_step's wall duration; Finish() clamps at zero.
    void
    TransferOperatorTimeNs(int32_t credit_step,
                           int32_t debit_step,
                           int64_t ns) {
        if (ns <= 0 || credit_step == debit_step) {
            return;
        }
        std::lock_guard lock(mu_);
        if (credit_step >= 0 &&
            static_cast<size_t>(credit_step) < operators_.size()) {
            operators_[credit_step].time_ns += ns;
        }
        if (debit_step >= 0 &&
            static_cast<size_t>(debit_step) < operators_.size()) {
            operators_[debit_step].time_ns -= ns;
        }
    }

    void
    MarkOperatorCached(int32_t step) {
        if (step < 0) {
            return;
        }
        std::lock_guard lock(mu_);
        if (static_cast<size_t>(step) < operators_.size()) {
            operators_[step].cached = true;
        }
    }

    bool
    IsOperatorCached(int32_t step) const {
        if (step < 0) {
            return false;
        }
        std::lock_guard lock(mu_);
        return static_cast<size_t>(step) < operators_.size() &&
               operators_[step].cached;
    }

    void
    BindExprToOperator(int64_t expr_id, int32_t step) {
        if (expr_id == 0 || step < 0) {
            return;
        }
        std::lock_guard lock(mu_);
        EnsureLeafSlotLocked(expr_id);
        expr_operator_steps_[expr_id] = step;
    }

    void
    SetDefaultFilterStep(int32_t step) {
        if (step < 0) {
            return;
        }
        std::lock_guard lock(mu_);
        if (default_filter_step_ < 0) {
            default_filter_step_ = step;
        }
    }

    // One call per filter leaf, before it runs.
    void
    RecordDecision(FieldId field_id,
                   int64_t expr_id,
                   bool used_index,
                   int64_t index_id,
                   std::string family,
                   bool nested,
                   std::string path,
                   std::string decline_reason) {
        std::lock_guard lock(mu_);
        if (expr_id != 0) {
            EnsureLeafSlotLocked(expr_id);
        }
        auto& group = groups_[Key(field_id,
                                  expr_id,
                                  used_index,
                                  index_id,
                                  family,
                                  nested,
                                  path,
                                  decline_reason)];
        if (group.family.empty()) {
            group.family = std::move(family);
        }
        if (group.path.empty()) {
            group.path = std::move(path);
        }
        if (group.decline_reason.empty()) {
            group.decline_reason = std::move(decline_reason);
        }
        group.field_id = field_id.get();
        group.expr_id = expr_id;
        group.index_id = index_id;
        group.used_index = used_index;
        group.nested = nested;
    }

    // Rows one leaf was handed and rows it matched, per batch. Reported by the
    // node that evaluates the leaf, so it may arrive before or after the leaf
    // recorded its decision; the counters are merged at Finish.
    void
    AddRows(int64_t expr_id, int64_t rows_in, int64_t rows_out) {
        if (expr_id == 0) {
            return;
        }
        AddSlotRows(EnsureLeafSlot(expr_id), rows_in, rows_out);
    }

    // Rows this leaf read from the raw column.
    void
    AddRawScan(int64_t expr_id, int64_t rows) {
        if (expr_id == 0 || rows <= 0) {
            return;
        }
        AddSlotRawScan(EnsureLeafSlot(expr_id), rows);
    }

    void
    AddLeafEvalNs(int64_t expr_id, int64_t ns) {
        if (expr_id == 0 || ns <= 0) {
            return;
        }
        AddSlotEvalNs(EnsureLeafSlot(expr_id), ns);
    }

    void
    AddLeafPrefetchNs(int64_t expr_id, int64_t ns) {
        if (expr_id == 0 || ns <= 0) {
            return;
        }
        AddSlotPrefetchNs(EnsureLeafSlot(expr_id), ns);
    }

    // Per-operand scan counts in the active-row domain for multi-column leaves
    // (CompareExpr / CallExpr), keeping idx_scan and raw_scan aligned with
    // rows_in even under upstream conjunction masks and Growing partial coverage.
    void
    AddOperandScan(FieldId field_id,
                   int64_t expr_id,
                   bool used_index,
                   int64_t index_id,
                   const std::string& family,
                   bool nested,
                   const std::string& path,
                   const std::string& decline_reason,
                   int64_t idx_rows,
                   int64_t raw_rows) {
        if (expr_id == 0 || (idx_rows < 0 && raw_rows < 0)) {
            return;
        }
        std::lock_guard lock(mu_);
        auto it = groups_.find(Key(field_id,
                                   expr_id,
                                   used_index,
                                   index_id,
                                   family,
                                   nested,
                                   path,
                                   decline_reason));
        if (it == groups_.end()) {
            return;
        }
        it->second.has_operand_scan = true;
        if (idx_rows > 0) {
            it->second.operand_idx += idx_rows;
        }
        if (raw_rows > 0) {
            it->second.operand_raw += raw_rows;
        }
    }

    // Hits an index returned in the element domain, when the caller folds them
    // to rows afterwards. They stay separate from rows_out: one row may own
    // several matching elements.
    void
    AddElementHits(FieldId field_id,
                   int64_t index_id,
                   int64_t expr_id,
                   int64_t elements) {
        std::lock_guard lock(mu_);
        for (auto& [key, group] : groups_) {
            if (!group.used_index || group.index_id != index_id) {
                continue;
            }
            // Two leaves on one field can share an index; the expr id keeps
            // their element hits apart. A zero id keeps the old field-wide
            // behavior for callers that predate leaf attribution.
            if (expr_id != 0 ? group.expr_id != expr_id
                             : group.field_id != field_id.get()) {
                continue;
            }
            group.element_hits += elements;
        }
    }

    // Serialized segment node: either operator children carrying filter_group
    // children (Driver execution), or direct filter_group children (standalone
    // expression tests).
    std::string
    Finish() const {
        std::lock_guard lock(mu_);
        proto::internal::ExecutionExplain node;
        node.set_kind("segment");
        node.set_version(1);
        (*node.mutable_counters())["rows_total"] = rows_total_;
        int64_t total_in = 0;
        int64_t total_idx = 0;
        int64_t total_raw = 0;
        int64_t total_out = 0;
        std::vector<proto::internal::ExecutionExplain*> op_nodes;
        if (!operators_.empty()) {
            op_nodes.resize(operators_.size(), nullptr);
            for (size_t i = 0; i < operators_.size(); ++i) {
                const auto& op = operators_[i];
                if (op.name.empty()) {
                    continue;
                }
                auto* op_child = node.add_children();
                op_child->set_kind("operator");
                op_child->set_version(1);
                auto& op_attrs = *op_child->mutable_attributes();
                op_attrs["name"] = op.name;
                op_attrs["step"] = std::to_string(op.step);
                if (!op.domain.empty()) {
                    op_attrs["domain"] = op.domain;
                }
                if (op.cached) {
                    op_attrs["cached"] = "true";
                }
                auto& op_ctrs = *op_child->mutable_counters();
                op_ctrs["rows_in"] = op.rows_in;
                op_ctrs["rows_out"] = op.rows_out;
                op_ctrs["time_us"] = NsToUs(std::max<int64_t>(0, op.time_ns));
                op_nodes[i] = op_child;
            }
        }
        auto resolve_group_parent =
            [&](int64_t expr_id) -> proto::internal::ExecutionExplain* {
            if (op_nodes.empty()) {
                return &node;
            }
            int32_t step = default_filter_step_;
            if (expr_id != 0) {
                if (auto it = expr_operator_steps_.find(expr_id);
                    it != expr_operator_steps_.end()) {
                    step = it->second;
                }
            }
            if (step >= 0 && static_cast<size_t>(step) < op_nodes.size() &&
                op_nodes[step] != nullptr) {
                return op_nodes[step];
            }
            for (auto* candidate : op_nodes) {
                if (candidate != nullptr) {
                    return candidate;
                }
            }
            return &node;
        };
        auto find_slot = [&](int64_t expr_id) -> const LeafSlot* {
            if (expr_id == 0) {
                return nullptr;
            }
            auto it = expr_slots_.find(expr_id);
            return it != expr_slots_.end() ? it->second : nullptr;
        };
        auto is_skipped = [&](int64_t expr_id) {
            if (expr_id == 0 || rows_total_ <= 0 ||
                eval_entered_exprs_.count(expr_id) == 0 ||
                empty_domain_exprs_.count(expr_id) > 0) {
                return false;
            }
            const auto* slot = find_slot(expr_id);
            if (slot != nullptr &&
                (slot->evaluated.load(std::memory_order_relaxed) ||
                 slot->eval_ns.load(std::memory_order_relaxed) > 0)) {
                return false;
            }
            return true;
        };
        struct ExprSegTotals {
            int64_t in{0};
            int64_t idx{0};
            int64_t raw{0};
            int64_t out{0};
        };
        std::set<int64_t> emitted_expr_ids;
        std::map<int64_t, ExprSegTotals> expr_totals;
        for (const auto& [key, group] : groups_) {
            auto* child = resolve_group_parent(group.expr_id)->add_children();
            child->set_kind("filter_group");
            child->set_version(1);
            auto& attrs = *child->mutable_attributes();
            attrs["field_id"] = std::to_string(group.field_id);
            bool skipped = false;
            if (group.expr_id != 0) {
                const bool secondary =
                    !emitted_expr_ids.insert(group.expr_id).second;
                attrs["expr_id"] = std::to_string(group.expr_id);
                if (auto it = eval_orders_.find(group.expr_id);
                    it != eval_orders_.end() && it->second > 0) {
                    attrs["eval_order"] = std::to_string(it->second);
                }
                if (secondary) {
                    attrs["secondary_operand"] = "true";
                }
                skipped = is_skipped(group.expr_id);
                if (skipped) {
                    attrs["skipped"] = "true";
                }
            }
            int64_t rows_in = 0;
            int64_t rows_raw = 0;
            int64_t rows_out = 0;
            int64_t eval_ns = 0;
            int64_t prefetch_ns = 0;
            if (const auto* slot = find_slot(group.expr_id); slot != nullptr) {
                rows_in = slot->rows_in.load(std::memory_order_relaxed);
                rows_raw = slot->raw_scan.load(std::memory_order_relaxed);
                rows_out = slot->rows_out.load(std::memory_order_relaxed);
                eval_ns = slot->eval_ns.load(std::memory_order_relaxed);
                prefetch_ns = slot->prefetch_ns.load(std::memory_order_relaxed);
            }
            const int64_t raw_scan =
                group.has_operand_scan ? group.operand_raw : rows_raw;
            const bool effective_used_index =
                group.used_index && !(group.has_operand_scan && rows_in > 0 &&
                                      group.operand_idx == 0);
            if (effective_used_index) {
                attrs["index_id"] = std::to_string(group.index_id);
            }
            if (!group.family.empty()) {
                attrs["family"] = group.family;
            }
            if (!group.decline_reason.empty()) {
                attrs["decline_reason"] = group.decline_reason;
            } else if (group.used_index && !effective_used_index) {
                attrs["decline_reason"] = "index_unusable";
            }
            if (!group.path.empty()) {
                attrs["path"] = group.path;
            }
            attrs["nested"] = group.nested ? "true" : "false";

            // An index leaf handles its covered input through the index; a
            // partial-coverage value lookup reports the active rows actually
            // served by the value reader.
            const int64_t idx_scan =
                effective_used_index
                    ? (group.has_operand_scan ? group.operand_idx : rows_in)
                    : 0;
            auto& counters = *child->mutable_counters();
            counters["rows_in"] = rows_in;
            counters["idx_scan"] = idx_scan;
            counters["raw_scan"] = raw_scan;
            counters["rows_out"] = rows_out;
            if (group.expr_id != 0) {
                if (!skipped) {
                    counters["eval_us"] = NsToUs(eval_ns);
                }
                if (const int64_t pf_us = NsToUs(prefetch_ns); pf_us > 0) {
                    counters["prefetch_us"] = pf_us;
                }
            }
            if (group.element_hits > 0) {
                counters["element_hits"] = group.element_hits;
            }
            if (group.expr_id != 0) {
                auto& tot = expr_totals[group.expr_id];
                tot.in = std::max(tot.in, rows_in);
                tot.idx = std::max(tot.idx, idx_scan);
                tot.raw = std::max(tot.raw, raw_scan);
                tot.out = std::max(tot.out, rows_out);
            } else {
                total_in += rows_in;
                total_idx += idx_scan;
                total_raw += raw_scan;
                total_out += rows_out;
            }
        }
        for (const auto& [expr_id, tot] : expr_totals) {
            total_in += tot.in;
            total_idx += tot.idx;
            total_raw += tot.raw;
            total_out += tot.out;
        }
        for (const auto& [expr_id, order] : eval_orders_) {
            if (emitted_expr_ids.count(expr_id) > 0 || !is_skipped(expr_id)) {
                continue;
            }
            auto* child = resolve_group_parent(expr_id)->add_children();
            child->set_kind("filter_group");
            child->set_version(1);
            auto& attrs = *child->mutable_attributes();
            if (auto fit = eval_field_ids_.find(expr_id);
                fit != eval_field_ids_.end() && fit->second != 0) {
                attrs["field_id"] = std::to_string(fit->second);
            }
            attrs["expr_id"] = std::to_string(expr_id);
            if (order > 0) {
                attrs["eval_order"] = std::to_string(order);
            }
            attrs["skipped"] = "true";
            auto& counters = *child->mutable_counters();
            counters["rows_in"] = 0;
            counters["idx_scan"] = 0;
            counters["raw_scan"] = 0;
            counters["rows_out"] = 0;
            if (const auto* slot = find_slot(expr_id); slot != nullptr) {
                if (const int64_t pf_us = NsToUs(
                        slot->prefetch_ns.load(std::memory_order_relaxed));
                    pf_us > 0) {
                    counters["prefetch_us"] = pf_us;
                }
            }
        }
        auto& counters = *node.mutable_counters();
        counters["rows_in"] = total_in;
        counters["idx_scan"] = total_idx;
        counters["raw_scan"] = total_raw;
        counters["rows_out"] = total_out;
        return node.SerializeAsString();
    }

 private:
    static int64_t
    NsToUs(int64_t ns) noexcept {
        return ns <= 0 ? 0 : (ns + 500) / 1000;
    }

    struct OperatorSlot {
        int32_t step{0};
        std::string name;
        std::string domain;
        int64_t rows_in{0};
        int64_t rows_out{0};
        int64_t time_ns{0};
        bool cached{false};
    };

    void
    EnsureOperatorSlotLocked(int32_t step) {
        if (static_cast<size_t>(step) >= operators_.size()) {
            operators_.resize(static_cast<size_t>(step) + 1);
        }
    }

    LeafSlot*
    EnsureLeafSlotLocked(int64_t expr_id) const {
        if (expr_id == 0) {
            return nullptr;
        }
        if (auto it = expr_slots_.find(expr_id); it != expr_slots_.end()) {
            return it->second;
        }
        leaf_slots_.emplace_back(expr_id);
        auto* slot = &leaf_slots_.back();
        expr_slots_.emplace(expr_id, slot);
        return slot;
    }

    struct Group {
        int64_t field_id{0};
        int64_t expr_id{0};
        int64_t index_id{0};
        bool used_index{false};
        bool nested{false};
        std::string family;
        std::string path;
        std::string decline_reason;
        int64_t element_hits{0};
        bool has_operand_scan{false};
        int64_t operand_idx{0};
        int64_t operand_raw{0};
    };

    static std::string
    Key(FieldId field_id,
        int64_t expr_id,
        bool used_index,
        int64_t index_id,
        const std::string& family,
        bool nested,
        const std::string& path,
        const std::string& decline_reason) {
        return fmt::format("{}|{}|{}|{}|{}|{}|{}|{}",
                           field_id.get(),
                           expr_id,
                           used_index,
                           index_id,
                           family,
                           nested,
                           path,
                           decline_reason);
    }

    int64_t rows_total_;
    mutable std::mutex mu_;
    mutable std::deque<LeafSlot> leaf_slots_;
    mutable std::unordered_map<int64_t, LeafSlot*> expr_slots_;
    mutable std::map<std::string, Group> groups_;
    mutable int32_t next_eval_order_{1};
    mutable std::map<int64_t, int32_t> eval_orders_;
    mutable std::map<int64_t, int64_t> eval_field_ids_;
    mutable std::set<int64_t> eval_entered_exprs_;
    mutable std::set<int64_t> empty_domain_exprs_;
    mutable std::set<int64_t> non_empty_domain_exprs_;
    mutable std::vector<OperatorSlot> operators_;
    mutable std::map<int64_t, int32_t> expr_operator_steps_;
    mutable int32_t default_filter_step_{-1};
};

}  // namespace milvus::exec
