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

    auto* explain = context.get_exec_context() == nullptr
                        ? nullptr
                        : context.get_exec_context()->explain();
    const bool track_leaf_rows =
        explain != nullptr &&
        std::any_of(ngram_exprs_.begin(),
                    ngram_exprs_.end(),
                    [](const auto& e) { return e && e->expr_id() != 0; });

    // Phase 1: Execute once for entire segment (no bitmap_input needed)
    if (cached_phase1_res_ == nullptr) {
        TargetBitmap candidates(active_count_, true);

        // Execute all ngram index queries, AND merge results. When explain is
        // active on tagged leaves, retain each leaf's Phase 1 bitmap so Phase 2
        // can attribute sequential per-batch (rows_in, rows_out).
        if (track_leaf_rows) {
            phase1_leaf_res_.resize(ngram_exprs_.size());
            for (size_t i = 0; i < ngram_exprs_.size(); ++i) {
                ngram_exprs_[i]->EnsureExplainDecisionRecorded();
                auto leaf_candidates =
                    std::make_shared<TargetBitmap>(active_count_, true);
                ngram_exprs_[i]->ExecuteNgramPhase1(*leaf_candidates);
                phase1_leaf_res_[i] = leaf_candidates;
                candidates &= *leaf_candidates;
                if (candidates.none()) {
                    break;
                }
            }
        } else {
            for (auto& expr : ngram_exprs_) {
                expr->EnsureExplainDecisionRecorded();
                expr->ExecuteNgramPhase1(candidates);
                if (candidates.none()) {
                    break;
                }
            }
        }

        cached_phase1_res_ =
            std::make_shared<TargetBitmap>(std::move(candidates));
    }

    const auto& bitmap_input = context.get_bitmap_input();
    TargetBitmap batch_candidates;

    if (explain != nullptr && !phase1_leaf_res_.empty()) {
        TargetBitmap running;
        if (!bitmap_input.empty()) {
            AssertInfo(
                static_cast<int64_t>(bitmap_input.size()) == real_batch_size,
                "bitmap_input size {} != real_batch_size {}",
                bitmap_input.size(),
                real_batch_size);
            running.append(bitmap_input, 0, real_batch_size);
        } else {
            running = TargetBitmap(real_batch_size, true);
        }

        std::vector<int64_t> c_before(ngram_exprs_.size(), 0);
        std::vector<int64_t> c_after(ngram_exprs_.size(), 0);
        size_t p1_ran = 0;
        for (size_t i = 0; i < ngram_exprs_.size(); ++i) {
            if (running.none() || phase1_leaf_res_[i] == nullptr) {
                break;
            }
            c_before[i] = static_cast<int64_t>(running.count());
            TargetBitmap leaf_slice;
            leaf_slice.append(
                *phase1_leaf_res_[i], current_pos_, real_batch_size);
            running &= leaf_slice;
            c_after[i] = static_cast<int64_t>(running.count());
            p1_ran = i + 1;
        }
        batch_candidates = std::move(running);

        std::vector<int64_t> p2_rejected(ngram_exprs_.size(), 0);
        if (!batch_candidates.none()) {
            for (size_t i = 0; i < ngram_exprs_.size(); ++i) {
                int64_t before = static_cast<int64_t>(batch_candidates.count());
                ngram_exprs_[i]->ExecuteNgramPhase2(
                    batch_candidates, current_pos_, real_batch_size);
                int64_t after = static_cast<int64_t>(batch_candidates.count());
                p2_rejected[i] = before - after;
                if (batch_candidates.none()) {
                    break;
                }
            }
        }

        if (p1_ran > 0) {
            int64_t s = c_before[0];
            for (size_t i = 0; i < p1_ran && s > 0; ++i) {
                int64_t delta = (c_before[i] - c_after[i]) + p2_rejected[i];
                int64_t next_s = std::max<int64_t>(0, s - delta);
                if (ngram_exprs_[i]->expr_id() != 0) {
                    explain->AddRows(ngram_exprs_[i]->expr_id(), s, next_s);
                }
                s = next_s;
            }
        }
    } else {
        // Phase 2: Execute per batch with batch-level bitmap_input
        // Extract current batch from Phase1 result
        batch_candidates.append(
            *cached_phase1_res_, current_pos_, real_batch_size);

        // Apply batch-level pre_filter from previous expressions in conjunction
        if (!bitmap_input.empty()) {
            AssertInfo(
                static_cast<int64_t>(bitmap_input.size()) == real_batch_size,
                "bitmap_input size {} != real_batch_size {}",
                bitmap_input.size(),
                real_batch_size);
            batch_candidates &= bitmap_input;
        }

        // Execute Phase2 (post-filter) on this batch
        if (!batch_candidates.none()) {
            for (auto& expr : ngram_exprs_) {
                expr->ExecuteNgramPhase2(
                    batch_candidates, current_pos_, real_batch_size);
                if (batch_candidates.none()) {
                    break;
                }
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
