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

#include "exec/ExecutionExplainRecorder.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pb/internal.pb.h"

namespace milvus::exec {

static_assert(static_cast<int>(ExplainOperatorKind::kUnknown) ==
              static_cast<int>(proto::internal::EXPLAIN_OP_UNKNOWN));
static_assert(static_cast<int>(ExplainOperatorKind::kFilter) ==
              static_cast<int>(proto::internal::EXPLAIN_OP_FILTER));
static_assert(static_cast<int>(ExplainOperatorKind::kRandomSample) ==
              static_cast<int>(proto::internal::EXPLAIN_OP_RANDOM_SAMPLE));
static_assert(static_cast<int>(ExplainOperatorKind::kMvcc) ==
              static_cast<int>(proto::internal::EXPLAIN_OP_MVCC));
static_assert(static_cast<int>(ExplainOperatorKind::kElementFilter) ==
              static_cast<int>(proto::internal::EXPLAIN_OP_ELEMENT_FILTER));
static_assert(static_cast<int>(ExplainOperatorKind::kVectorSearch) ==
              static_cast<int>(proto::internal::EXPLAIN_OP_VECTOR_SEARCH));
static_assert(
    static_cast<int>(ExplainOperatorKind::kIterativeElementFilter) ==
    static_cast<int>(proto::internal::EXPLAIN_OP_ITERATIVE_ELEMENT_FILTER));
static_assert(static_cast<int>(ExplainOperatorKind::kIterativeFilter) ==
              static_cast<int>(proto::internal::EXPLAIN_OP_ITERATIVE_FILTER));
static_assert(static_cast<int>(ExplainOperatorKind::kGroupBy) ==
              static_cast<int>(proto::internal::EXPLAIN_OP_GROUP_BY));
static_assert(static_cast<int>(ExplainOperatorKind::kRescore) ==
              static_cast<int>(proto::internal::EXPLAIN_OP_RESCORE));
static_assert(static_cast<int>(ExplainOperatorKind::kProject) ==
              static_cast<int>(proto::internal::EXPLAIN_OP_PROJECT));
static_assert(static_cast<int>(ExplainOperatorKind::kAggregation) ==
              static_cast<int>(proto::internal::EXPLAIN_OP_AGGREGATION));
static_assert(static_cast<int>(ExplainOperatorKind::kOrderBy) ==
              static_cast<int>(proto::internal::EXPLAIN_OP_ORDER_BY));

namespace {

std::pair<std::string, bool>
NormalizeDeclineReason(std::string_view raw, bool is_growing) {
    if (raw == "constant_true" || raw == "constant_false") {
        return {"", true};
    }
    if (raw == "growing") {
        return {"growing", false};
    }
    if (raw == "no_index_on_field") {
        return {is_growing ? "growing" : "no_index", false};
    }
    if (raw == "index_unusable" || raw == "candidate_needs_refine" ||
        raw == "offset_input_with_ngram" || raw == "flat_json_required") {
        return {"unusable", false};
    }
    if (raw == "schema_not_nullable") {
        return {"not_nullable", false};
    }
    if (raw == "cached") {
        return {"cached", false};
    }
    if (raw == "value_type_mismatch") {
        return {"type_mismatch", false};
    }
    if (raw == "json_path_not_covered") {
        return {"path_uncovered", false};
    }
    if (raw == "capability_mismatch") {
        return {"unsupported_op", false};
    }
    if (raw == "literal_not_supported") {
        return {"unsupported_literal", false};
    }
    if (raw == "hinted_index_not_selected") {
        return {"hint_mismatch", false};
    }
    return {"unknown", false};
}

}  // namespace

std::string
ExecutionExplainRecorder::Finish() const {
    std::lock_guard lock(mu_);
    proto::internal::ExecutionExplain explain_pb;
    explain_pb.set_version(1);
    auto* seg_group = explain_pb.mutable_group();
    seg_group->set_segments(1);
    seg_group->set_sealed_segments(is_growing_ ? 0 : 1);
    seg_group->set_growing_segments(is_growing_ ? 1 : 0);
    seg_group->set_rows_total(rows_total_);

    int folded_filter_idx = -1;
    int mvcc_idx = -1;
    for (size_t i = 0; i < operators_.size(); ++i) {
        if (operators_[i].kind == ExplainOperatorKind::kFilter &&
            operators_[i].internal_folded && folded_filter_idx < 0) {
            folded_filter_idx = static_cast<int>(i);
        } else if (operators_[i].kind == ExplainOperatorKind::kMvcc &&
                   mvcc_idx < 0) {
            mvcc_idx = static_cast<int>(i);
        }
    }
    if (folded_filter_idx >= 0 && mvcc_idx >= 0) {
        if (folded_filter_idx >= mvcc_idx) {
            folded_filter_idx = -1;
        } else {
            for (int mid = folded_filter_idx + 1; mid < mvcc_idx; ++mid) {
                if (operators_[mid].kind != ExplainOperatorKind::kUnknown) {
                    folded_filter_idx = -1;
                    break;
                }
            }
        }
    }
    if (folded_filter_idx >= 0 && mvcc_idx >= 0) {
        for (const auto& [_, g] : groups_) {
            int32_t step = default_filter_step_;
            if (g.expr_id != 0) {
                if (auto it = expr_operator_steps_.find(g.expr_id);
                    it != expr_operator_steps_.end()) {
                    step = it->second;
                }
            }
            if (step == folded_filter_idx) {
                folded_filter_idx = -1;
                break;
            }
        }
    }
    const bool fold_filter_into_mvcc =
        (folded_filter_idx >= 0 && mvcc_idx >= 0);

    std::vector<proto::internal::ExplainOperatorStats*> op_nodes;
    if (!operators_.empty()) {
        op_nodes.resize(operators_.size(), nullptr);
        for (size_t i = 0; i < operators_.size(); ++i) {
            const auto& op = operators_[i];
            if (op.kind == ExplainOperatorKind::kUnknown) {
                continue;
            }
            if (fold_filter_into_mvcc &&
                static_cast<int>(i) == folded_filter_idx) {
                continue;
            }
            auto* op_pb = seg_group->add_operators();
            op_pb->set_kind(
                static_cast<proto::internal::ExplainOperatorKind>(op.kind));
            if (!op.domain.empty()) {
                op_pb->set_domain(op.domain);
            }
            op_pb->set_segments(1);
            if (op.cached) {
                op_pb->set_cached_segments(1);
            }
            const int64_t rows_in =
                (fold_filter_into_mvcc && static_cast<int>(i) == mvcc_idx)
                    ? operators_[folded_filter_idx].rows_in
                    : op.rows_in;
            op_pb->set_rows_in(rows_in);
            op_pb->set_rows_out(op.rows_out);
            op_nodes[i] = op_pb;
        }
    }

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
            slot->evaluated.load(std::memory_order_relaxed)) {
            return false;
        }
        return true;
    };

    struct CondKey {
        int64_t expr_id{0};
        int64_t disambig_field_id{0};

        bool
        operator<(const CondKey& other) const {
            const bool a_has = expr_id != 0;
            const bool b_has = other.expr_id != 0;
            if (a_has != b_has) {
                return a_has;
            }
            if (expr_id != other.expr_id) {
                return expr_id < other.expr_id;
            }
            return disambig_field_id < other.disambig_field_id;
        }
    };

    struct UsedKey {
        int64_t index_id{0};
        std::string family;
        std::string path;

        bool
        operator<(const UsedKey& other) const {
            if (index_id != other.index_id) {
                return index_id < other.index_id;
            }
            if (family != other.family) {
                return family < other.family;
            }
            return path < other.path;
        }
    };

    struct UsedAccum {
        int64_t rows_in{0};
        int64_t rows_out{0};
        int64_t idx_scan{0};
        int64_t raw_scan{0};
        int64_t element_hits{0};
    };

    struct ReasonAccum {
        int64_t rows_in{0};
        int64_t rows_out{0};
        int64_t idx_scan{0};
        int64_t raw_scan{0};
    };

    struct CondAccum {
        int64_t expr_id{0};
        int64_t field_id{0};
        bool skipped_all{true};
        bool has_non_skipped{false};
        int64_t rows_in{0};
        int64_t rows_out{0};
        int64_t idx_scan{0};
        int64_t raw_scan{0};
        int64_t element_hits{0};
        std::map<UsedKey, UsedAccum> used;
        std::map<std::string, ReasonAccum> reasons;
    };

    proto::internal::ExplainOperatorStats* standalone_filter = nullptr;
    auto resolve_group_parent =
        [&](int64_t expr_id) -> proto::internal::ExplainOperatorStats* {
        if (!op_nodes.empty()) {
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
        }
        if (standalone_filter == nullptr) {
            standalone_filter = seg_group->add_operators();
            standalone_filter->set_kind(proto::internal::EXPLAIN_OP_FILTER);
            standalone_filter->set_segments(1);
            standalone_filter->set_rows_in(rows_total_);
        }
        return standalone_filter;
    };

    std::map<proto::internal::ExplainOperatorStats*,
             std::map<CondKey, CondAccum>>
        op_conds;
    std::set<int64_t> seen_expr_ids;

    for (const auto& [_, g] : groups_) {
        auto* parent_op = resolve_group_parent(g.expr_id);
        if (g.expr_id != 0) {
            seen_expr_ids.insert(g.expr_id);
        }
        CondKey ck{g.expr_id, g.expr_id == 0 ? g.field_id : 0};
        auto& cond = op_conds[parent_op][ck];
        cond.expr_id = g.expr_id;
        if (cond.field_id == 0 && g.field_id != 0) {
            cond.field_id = g.field_id;
        }

        int64_t rows_in = 0;
        int64_t rows_raw = 0;
        int64_t rows_out = 0;
        int64_t phase1_idx = 0;
        if (const auto* slot = find_slot(g.expr_id); slot != nullptr) {
            rows_in = slot->rows_in.load(std::memory_order_relaxed);
            rows_raw = slot->raw_scan.load(std::memory_order_relaxed);
            rows_out = slot->rows_out.load(std::memory_order_relaxed);
            phase1_idx = slot->phase1_idx_scan.load(std::memory_order_relaxed);
        }
        const bool skipped = is_skipped(g.expr_id);
        if (!skipped) {
            cond.skipped_all = false;
            cond.has_non_skipped = true;
        }

        const int64_t raw_scan = g.has_operand_scan ? g.operand_raw : rows_raw;
        const bool effective_used_index =
            g.used_index &&
            !(g.has_operand_scan && rows_in > 0 && g.operand_idx == 0);
        const int64_t idx_scan =
            effective_used_index
                ? (g.has_operand_scan ? g.operand_idx
                                      : (rows_in > 0 ? rows_in : phase1_idx))
                : 0;

        cond.rows_in = std::max(cond.rows_in, rows_in);
        cond.rows_out = std::max(cond.rows_out, rows_out);
        cond.idx_scan = std::max(cond.idx_scan, idx_scan);
        cond.raw_scan = std::max(cond.raw_scan, raw_scan);
        cond.element_hits += g.element_hits;

        if (!skipped) {
            if (effective_used_index) {
                UsedKey uk{g.index_id, g.family, g.path};
                auto& u = cond.used[uk];
                u.rows_in = std::max(u.rows_in, rows_in);
                u.rows_out = std::max(u.rows_out, rows_out);
                u.idx_scan = std::max(u.idx_scan, idx_scan);
                u.raw_scan = std::max(u.raw_scan, raw_scan);
                u.element_hits += g.element_hits;
            } else {
                std::string_view raw_reason = g.decline_reason;
                if (raw_reason.empty() && g.used_index &&
                    !effective_used_index) {
                    raw_reason = "index_unusable";
                }
                auto [norm_reason, omit] =
                    NormalizeDeclineReason(raw_reason, is_growing_);
                if (!omit) {
                    auto& r = cond.reasons[norm_reason];
                    r.rows_in = std::max(r.rows_in, rows_in);
                    r.rows_out = std::max(r.rows_out, rows_out);
                    r.idx_scan = std::max(r.idx_scan, idx_scan);
                    r.raw_scan = std::max(r.raw_scan, raw_scan);
                }
            }
        }
    }

    for (const auto& [expr_id, _] : eval_orders_) {
        if (seen_expr_ids.count(expr_id) > 0 || !is_skipped(expr_id)) {
            continue;
        }
        auto* parent_op = resolve_group_parent(expr_id);
        CondKey ck{expr_id, 0};
        auto& cond = op_conds[parent_op][ck];
        cond.expr_id = expr_id;
        if (auto fit = eval_field_ids_.find(expr_id);
            fit != eval_field_ids_.end() && fit->second != 0) {
            cond.field_id = fit->second;
        }
        cond.skipped_all = true;
        cond.has_non_skipped = false;
    }

    int64_t standalone_out = 0;
    for (int i = 0; i < seg_group->operators_size(); ++i) {
        auto* op_pb = seg_group->mutable_operators(i);
        auto it = op_conds.find(op_pb);
        if (it != op_conds.end()) {
            for (const auto& [_, cond] : it->second) {
                auto* c_pb = op_pb->add_conditions();
                c_pb->set_expr_id(cond.expr_id);
                c_pb->set_field_id(cond.field_id);
                c_pb->set_segments(1);
                const bool cond_skipped =
                    cond.skipped_all && !cond.has_non_skipped;
                if (cond_skipped) {
                    c_pb->set_skipped_segments(1);
                } else {
                    c_pb->set_rows_in(cond.rows_in);
                    c_pb->set_rows_out(cond.rows_out);
                    c_pb->set_idx_scan(cond.idx_scan);
                    c_pb->set_raw_scan(cond.raw_scan);
                    c_pb->set_element_hits(cond.element_hits);
                    standalone_out = std::max(standalone_out, cond.rows_out);
                }
                if (cond.expr_id != 0) {
                    if (auto oit = eval_orders_.find(cond.expr_id);
                        oit != eval_orders_.end() && oit->second > 0) {
                        auto* eo = c_pb->add_eval_orders();
                        eo->set_eval_order(oit->second);
                        eo->set_segments(1);
                    }
                }
                if (!cond_skipped) {
                    const bool single_access_path =
                        (cond.used.size() + cond.reasons.size()) == 1;
                    for (const auto& [uk, u] : cond.used) {
                        auto* u_pb = c_pb->add_used();
                        u_pb->set_index_id(uk.index_id);
                        if (!uk.family.empty()) {
                            u_pb->set_family(uk.family);
                        }
                        if (!uk.path.empty()) {
                            u_pb->set_path(uk.path);
                        }
                        u_pb->set_segments(1);
                        if (single_access_path) {
                            u_pb->set_rows_in(u.rows_in);
                            u_pb->set_rows_out(u.rows_out);
                        }
                        u_pb->set_idx_scan(u.idx_scan);
                        u_pb->set_raw_scan(u.raw_scan);
                        u_pb->set_element_hits(u.element_hits);
                    }
                    bool has_raw_fallback = false;
                    for (const auto& [reason, r] : cond.reasons) {
                        if (reason != "not_nullable" &&
                            (reason != "cached" || r.raw_scan > 0)) {
                            has_raw_fallback = true;
                        }
                        auto* r_pb = c_pb->add_reasons();
                        r_pb->set_reason(reason);
                        r_pb->set_segments(1);
                        if (single_access_path) {
                            r_pb->set_rows_in(r.rows_in);
                            r_pb->set_rows_out(r.rows_out);
                        }
                        r_pb->set_idx_scan(r.idx_scan);
                        r_pb->set_raw_scan(r.raw_scan);
                    }
                    if (has_raw_fallback) {
                        c_pb->set_raw_scan_segments(1);
                    }
                }
            }
        }
    }
    if (standalone_filter != nullptr) {
        standalone_filter->set_rows_out(standalone_out);
    }

    return explain_pb.SerializeAsString();
}

}  // namespace milvus::exec
