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

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common/ExplainOperatorKind.h"
#include "common/Types.h"

namespace milvus::exec {

using ExplainOperatorKind = milvus::ExplainOperatorKind;

// Collects what one segment did for one query and serializes a single-segment
// ExplainSegmentGroup. Condition-level row counters live once per expr_id in
// lock-free LeafSlot entries, decoupled from per-operand index/fallback access
// paths recorded in AccessPathGroup.
class ExecutionExplainRecorder {
 public:
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
        std::atomic<int64_t> phase1_idx_scan{0};
        std::atomic<bool> evaluated{false};
    };

    explicit ExecutionExplainRecorder(int64_t rows_total,
                                      bool is_growing = false)
        : rows_total_(rows_total), is_growing_(is_growing) {
    }

    void
    SetGrowing(bool is_growing) {
        std::lock_guard lock(mu_);
        is_growing_ = is_growing;
    }

    bool
    IsGrowing() const {
        std::lock_guard lock(mu_);
        return is_growing_;
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
    AddSlotPhase1IdxScan(LeafSlot* slot, int64_t rows) noexcept {
        if (slot == nullptr) {
            return;
        }
        slot->evaluated.store(true, std::memory_order_relaxed);
        if (rows > 0) {
            slot->phase1_idx_scan.fetch_add(rows, std::memory_order_relaxed);
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

    void
    MarkLeafEvaluated(int64_t expr_id) {
        if (expr_id == 0) {
            return;
        }
        EnsureLeafSlot(expr_id)->evaluated.store(true,
                                                 std::memory_order_relaxed);
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
    RegisterOperator(int32_t step,
                     ExplainOperatorKind kind,
                     std::string domain = "",
                     bool internal_folded = false) {
        if (step < 0 || kind == ExplainOperatorKind::kUnknown) {
            return;
        }
        std::lock_guard lock(mu_);
        EnsureOperatorSlotLocked(step);
        auto& op = operators_[step];
        op.step = step;
        op.kind = kind;
        if (!domain.empty()) {
            op.domain = std::move(domain);
        }
        if (internal_folded) {
            op.internal_folded = true;
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

    int32_t
    NextOperatorStep(int32_t step) const {
        if (step < 0) {
            return -1;
        }
        std::lock_guard lock(mu_);
        for (size_t i = static_cast<size_t>(step) + 1; i < operators_.size();
             ++i) {
            if (operators_[i].kind != ExplainOperatorKind::kUnknown) {
                return static_cast<int32_t>(i);
            }
        }
        return -1;
    }

    int32_t
    active_step() const noexcept {
        return active_step_.load(std::memory_order_relaxed);
    }

    int32_t
    SetActiveStep(int32_t step) noexcept {
        return active_step_.exchange(step, std::memory_order_relaxed);
    }

    class ScopedActiveStep {
     public:
        ScopedActiveStep(ExecutionExplainRecorder* recorder,
                         int32_t step) noexcept
            : recorder_(recorder) {
            if (recorder_ != nullptr) {
                prev_ = recorder_->SetActiveStep(step);
            }
        }
        ~ScopedActiveStep() {
            if (recorder_ != nullptr) {
                recorder_->SetActiveStep(prev_);
            }
        }
        ScopedActiveStep(const ScopedActiveStep&) = delete;
        ScopedActiveStep&
        operator=(const ScopedActiveStep&) = delete;

     private:
        ExecutionExplainRecorder* recorder_;
        int32_t prev_{-1};
    };

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

    void
    AddRows(int64_t expr_id, int64_t rows_in, int64_t rows_out) {
        if (expr_id == 0) {
            return;
        }
        AddSlotRows(EnsureLeafSlot(expr_id), rows_in, rows_out);
    }

    void
    AddRawScan(int64_t expr_id, int64_t rows) {
        if (expr_id == 0 || rows <= 0) {
            return;
        }
        AddSlotRawScan(EnsureLeafSlot(expr_id), rows);
    }

    void
    AddPhase1IdxScan(int64_t expr_id, int64_t rows) {
        if (expr_id == 0) {
            return;
        }
        AddSlotPhase1IdxScan(EnsureLeafSlot(expr_id), rows);
    }

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
            if (expr_id != 0 ? group.expr_id != expr_id
                             : group.field_id != field_id.get()) {
                continue;
            }
            group.element_hits += elements;
        }
    }

    std::string
    Finish() const;

 private:
    struct OperatorSlot {
        int32_t step{0};
        ExplainOperatorKind kind{ExplainOperatorKind::kUnknown};
        std::string domain;
        int64_t rows_in{0};
        int64_t rows_out{0};
        bool cached{false};
        bool internal_folded{false};
    };

    LeafSlot*
    EnsureLeafSlotLocked(int64_t expr_id) {
        auto it = expr_slots_.find(expr_id);
        if (it != expr_slots_.end()) {
            return it->second;
        }
        slots_.emplace_back(expr_id);
        LeafSlot* ptr = &slots_.back();
        expr_slots_.emplace(expr_id, ptr);
        return ptr;
    }

    void
    EnsureOperatorSlotLocked(int32_t step) {
        if (static_cast<size_t>(step) >= operators_.size()) {
            operators_.resize(static_cast<size_t>(step) + 1);
        }
    }

    struct AccessPathKey {
        int64_t field_id{0};
        int64_t expr_id{0};
        bool used_index{false};
        int64_t index_id{0};
        bool nested{false};
        std::string family;
        std::string path;
        std::string decline_reason;

        bool
        operator<(const AccessPathKey& other) const {
            return std::tie(field_id,
                            expr_id,
                            used_index,
                            index_id,
                            nested,
                            family,
                            path,
                            decline_reason) < std::tie(other.field_id,
                                                       other.expr_id,
                                                       other.used_index,
                                                       other.index_id,
                                                       other.nested,
                                                       other.family,
                                                       other.path,
                                                       other.decline_reason);
        }
    };

    struct AccessPathGroup {
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

    static AccessPathKey
    Key(FieldId field_id,
        int64_t expr_id,
        bool used_index,
        int64_t index_id,
        const std::string& family,
        bool nested,
        const std::string& path,
        const std::string& decline_reason) {
        return AccessPathKey{
            .field_id = field_id.get(),
            .expr_id = expr_id,
            .used_index = used_index,
            .index_id = index_id,
            .nested = nested,
            .family = family,
            .path = path,
            .decline_reason = decline_reason,
        };
    }

    int64_t rows_total_;
    bool is_growing_{false};
    mutable std::mutex mu_;
    mutable std::deque<LeafSlot> slots_;
    mutable std::unordered_map<int64_t, LeafSlot*> expr_slots_;
    mutable std::map<AccessPathKey, AccessPathGroup> groups_;
    mutable int32_t next_eval_order_{1};
    mutable std::map<int64_t, int32_t> eval_orders_;
    mutable std::map<int64_t, int64_t> eval_field_ids_;
    mutable std::set<int64_t> eval_entered_exprs_;
    mutable std::set<int64_t> empty_domain_exprs_;
    mutable std::set<int64_t> non_empty_domain_exprs_;
    mutable std::vector<OperatorSlot> operators_;
    mutable std::map<int64_t, int32_t> expr_operator_steps_;
    mutable int32_t default_filter_step_{-1};
    std::atomic<int32_t> active_step_{-1};
};

}  // namespace milvus::exec
