// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "exec/expression/LikeConjunctExpr.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "bitset/bitset.h"
#include "common/EasyAssert.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/UnaryExpr.h"

namespace milvus {

namespace exec {

PhyLikeConjunctExpr::PhyLikeConjunctExpr(
    std::vector<std::shared_ptr<PhyUnaryRangeFilterExpr>> ngram_exprs,
    milvus::OpContext* op_ctx,
    int64_t active_count,
    int64_t batch_size)
    : Expr(DataType::BOOL, {}, "PhyLikeConjunctExpr", op_ctx),
      ngram_exprs_(std::move(ngram_exprs)),
      active_count_(active_count),
      batch_size_(batch_size) {
}

int64_t
PhyLikeConjunctExpr::GetNextBatchSize() {
    return current_pos_ + batch_size_ >= active_count_
               ? active_count_ - current_pos_
               : batch_size_;
}

void
PhyLikeConjunctExpr::Eval(EvalCtx& context, VectorPtr& result) {
    AssertInfo(context.get_offset_input() == nullptr,
               "Offset input is not supported for PhyLikeConjunctExpr");
    AssertInfo(!ngram_exprs_.empty(),
               "PhyLikeConjunctExpr requires at least one expression");

    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        result = nullptr;
        return;
    }

    const auto& bitmap_input = context.get_bitmap_input();
    if (!bitmap_input.empty()) {
        AssertInfo(static_cast<int64_t>(bitmap_input.size()) == real_batch_size,
                   "bitmap_input size {} != real_batch_size {}",
                   bitmap_input.size(),
                   real_batch_size);
        if (bitmap_input.none()) {
            current_pos_ += real_batch_size;
            result = std::make_shared<ColumnVector>(
                TargetBitmap(real_batch_size, false),
                TargetBitmap(real_batch_size, true));
            return;
        }
    }

    auto* explain = context.get_exec_context() == nullptr
                        ? nullptr
                        : context.get_exec_context()->explain();
    const bool track_explain = explain != nullptr;
    const bool track_leaf_rows =
        track_explain &&
        std::any_of(ngram_exprs_.begin(),
                    ngram_exprs_.end(),
                    [](const auto& e) { return e && e->expr_id() != 0; });

    if (cached_phase1_res_ == nullptr) {
        TargetBitmap candidates(active_count_, true);
        if (track_leaf_rows) {
            phase1_cum_.clear();
            phase1_cum_.reserve(ngram_exprs_.size());
        }
        for (auto& expr : ngram_exprs_) {
            const int64_t p1_in =
                track_explain ? static_cast<int64_t>(candidates.count()) : 0;
            expr->EnsureExecPathDetermined();
            expr->EnsureExplainDecisionRecorded();
            auto* slot = expr->explain_slot();
            if (track_explain && expr->expr_id() != 0) {
                if (slot != nullptr) {
                    ExecutionExplainRecorder::AddSlotPhase1IdxScan(slot, p1_in);
                } else {
                    explain->AddPhase1IdxScan(expr->expr_id(), p1_in);
                }
            }
            {
                ExecutionExplainRecorder::ScopedLeafEvalTimer p1_timer(
                    track_explain ? slot : nullptr);
                expr->ExecuteNgramPhase1(candidates);
            }
            if (track_leaf_rows) {
                phase1_cum_.push_back(candidates.clone());
            }
            if (candidates.none()) {
                break;
            }
        }
        cached_phase1_res_ =
            std::make_shared<TargetBitmap>(std::move(candidates));
    }

    // Phase 2: Execute per batch with batch-level bitmap_input
    TargetBitmap batch_candidates;
    batch_candidates.append(*cached_phase1_res_, current_pos_, real_batch_size);
    if (!bitmap_input.empty()) {
        batch_candidates &= bitmap_input;
    }

    auto record_leaf_rows = [&](size_t idx, int64_t in_rows, int64_t out_rows) {
        if (!track_leaf_rows) {
            return;
        }
        auto& expr = ngram_exprs_[idx];
        if (expr->expr_id() == 0) {
            return;
        }
        if (auto* slot = expr->explain_slot(); slot != nullptr) {
            ExecutionExplainRecorder::AddSlotRows(slot, in_rows, out_rows);
        } else {
            explain->AddRows(expr->expr_id(), in_rows, out_rows);
        }
    };

    const int64_t batch_in =
        !track_leaf_rows ? 0
                         : (bitmap_input.empty()
                                ? real_batch_size
                                : static_cast<int64_t>(bitmap_input.count()));
    if (batch_candidates.none()) {
        if (track_leaf_rows && batch_in > 0) {
            int64_t prev_out = batch_in;
            for (size_t i = 0; i < phase1_cum_.size() && prev_out > 0; ++i) {
                TargetBitmap slice;
                slice.append(phase1_cum_[i], current_pos_, real_batch_size);
                if (!bitmap_input.empty()) {
                    slice &= bitmap_input;
                }
                const int64_t out_rows = static_cast<int64_t>(slice.count());
                record_leaf_rows(i, prev_out, out_rows);
                prev_out = out_rows;
            }
        }
    } else {
        int64_t prev_out = batch_in;
        for (size_t i = 0; i < ngram_exprs_.size(); ++i) {
            const int64_t in_rows = (i == 0) ? batch_in : prev_out;
            ngram_exprs_[i]->EnsureExecPathDetermined();
            ngram_exprs_[i]->EnsureExplainDecisionRecorded();
            {
                ExecutionExplainRecorder::ScopedLeafEvalTimer p2_timer(
                    track_explain ? ngram_exprs_[i]->explain_slot() : nullptr);
                ngram_exprs_[i]->ExecuteNgramPhase2(
                    batch_candidates, current_pos_, real_batch_size);
            }
            if (track_leaf_rows) {
                const int64_t out_rows =
                    static_cast<int64_t>(batch_candidates.count());
                record_leaf_rows(i, in_rows, out_rows);
                prev_out = out_rows;
                if (out_rows == 0) {
                    break;
                }
            } else if (batch_candidates.none()) {
                break;
            }
        }
    }

    // For ngram like expression, the valid result is always true as result has all information
    TargetBitmap valid_result(real_batch_size, true);
    current_pos_ += real_batch_size;
    result = std::make_shared<ColumnVector>(std::move(batch_candidates),
                                            std::move(valid_result));
}
}  // namespace exec
}  // namespace milvus
