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
#include <map>
#include <mutex>
#include <set>
#include <string>

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
    explicit ExecutionExplainRecorder(int64_t rows_total)
        : rows_total_(rows_total) {
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
        eval_entered_exprs_.insert(expr_id);
    }

    bool
    HasAnyEvalEntered() const {
        std::lock_guard lock(mu_);
        return !eval_entered_exprs_.empty();
    }

    void
    MarkDomainEmpty(int64_t expr_id) {
        if (expr_id == 0) {
            return;
        }
        std::lock_guard lock(mu_);
        empty_domain_exprs_.insert(expr_id);
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
        std::lock_guard lock(mu_);
        if (rows_in > 0) {
            evaluated_exprs_.insert(expr_id);
        }
        auto& rows = pending_rows_[expr_id];
        rows.in += rows_in;
        rows.out += rows_out;
    }

    // Rows this leaf read from the raw column.
    void
    AddRawScan(int64_t expr_id, int64_t rows) {
        if (expr_id == 0 || rows <= 0) {
            return;
        }
        std::lock_guard lock(mu_);
        pending_rows_[expr_id].raw += rows;
    }

    // Per-operand raw scan for multi-column leaves (CompareExpr / CallExpr), so
    // an indexed operand reports raw_scan=0 even when another operand of the
    // same leaf fell back to raw data.
    void
    AddOperandRawScan(FieldId field_id,
                      int64_t expr_id,
                      bool used_index,
                      int64_t index_id,
                      const std::string& family,
                      bool nested,
                      const std::string& path,
                      const std::string& decline_reason,
                      int64_t rows) {
        if (expr_id == 0 || rows < 0) {
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
        it->second.has_operand_raw = true;
        it->second.operand_raw += rows;
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

    // Serialized segment node: one filter_group child per decision group.
    std::string
    Finish() const {
        proto::internal::ExecutionExplain node;
        node.set_kind("segment");
        node.set_version(1);
        (*node.mutable_counters())["rows_total"] = rows_total_;
        int64_t total_in = 0;
        int64_t total_idx = 0;
        int64_t total_raw = 0;
        int64_t total_out = 0;
        std::lock_guard lock(mu_);
        auto is_skipped = [&](int64_t expr_id) {
            return expr_id != 0 && rows_total_ > 0 &&
                   eval_entered_exprs_.count(expr_id) > 0 &&
                   evaluated_exprs_.count(expr_id) == 0 &&
                   empty_domain_exprs_.count(expr_id) == 0;
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
            auto* child = node.add_children();
            child->set_kind("filter_group");
            child->set_version(1);
            auto& attrs = *child->mutable_attributes();
            attrs["field_id"] = std::to_string(group.field_id);
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
                if (is_skipped(group.expr_id)) {
                    attrs["skipped"] = "true";
                }
            }
            if (group.used_index) {
                attrs["index_id"] = std::to_string(group.index_id);
            }
            if (!group.family.empty()) {
                attrs["family"] = group.family;
            }
            if (!group.decline_reason.empty()) {
                attrs["decline_reason"] = group.decline_reason;
            }
            if (!group.path.empty()) {
                attrs["path"] = group.path;
            }
            attrs["nested"] = group.nested ? "true" : "false";

            Rows rows{};
            if (group.expr_id != 0) {
                if (auto it = pending_rows_.find(group.expr_id);
                    it != pending_rows_.end()) {
                    rows = it->second;
                }
            }
            // An index leaf handles its whole input through the index; a raw
            // leaf handles none of it there.
            const int64_t idx_scan = group.used_index ? rows.in : 0;
            const int64_t raw_scan =
                group.has_operand_raw ? group.operand_raw : rows.raw;
            auto& counters = *child->mutable_counters();
            counters["rows_in"] = rows.in;
            counters["idx_scan"] = idx_scan;
            counters["raw_scan"] = raw_scan;
            counters["rows_out"] = rows.out;
            if (group.element_hits > 0) {
                counters["element_hits"] = group.element_hits;
            }
            if (group.expr_id != 0) {
                auto& tot = expr_totals[group.expr_id];
                tot.in = std::max(tot.in, rows.in);
                tot.idx = std::max(tot.idx, idx_scan);
                tot.raw = std::max(tot.raw, raw_scan);
                tot.out = std::max(tot.out, rows.out);
            } else {
                total_in += rows.in;
                total_idx += idx_scan;
                total_raw += raw_scan;
                total_out += rows.out;
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
            auto* child = node.add_children();
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
        }
        auto& counters = *node.mutable_counters();
        counters["rows_in"] = total_in;
        counters["idx_scan"] = total_idx;
        counters["raw_scan"] = total_raw;
        counters["rows_out"] = total_out;
        return node.SerializeAsString();
    }

 private:
    struct Rows {
        int64_t in{0};
        int64_t raw{0};
        int64_t out{0};
    };

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
        bool has_operand_raw{false};
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

    const int64_t rows_total_;
    mutable std::mutex mu_;
    mutable std::map<int64_t, Rows> pending_rows_;
    mutable std::map<std::string, Group> groups_;
    mutable int32_t next_eval_order_{1};
    mutable std::map<int64_t, int32_t> eval_orders_;
    mutable std::map<int64_t, int64_t> eval_field_ids_;
    mutable std::set<int64_t> eval_entered_exprs_;
    mutable std::set<int64_t> evaluated_exprs_;
    mutable std::set<int64_t> empty_domain_exprs_;
};

}  // namespace milvus::exec
