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

#include "ConjunctExpr.h"

#include <folly/ScopeGuard.h>
#include <algorithm>
#include <optional>

#include "LikeConjunctExpr.h"
#include "RawExprCacheAdapter.h"
#include "UnaryExpr.h"
#include "common/EasyAssert.h"
#include "common/Tracer.h"
#include "common/ValueOp.h"
#include "exec/QueryContext.h"
#include "exec/expression/Utils.h"
#include "fmt/core.h"
#include "opentelemetry/trace/span.h"

namespace milvus {
namespace exec {

DataType
PhyConjunctFilterExpr::ResolveType(const std::vector<DataType>& inputs) {
    AssertInfo(
        inputs.size() > 0,
        fmt::format(
            "Conjunct expressions expect at least one argument, received: {}",
            inputs.size()));

    for (const auto& type : inputs) {
        AssertInfo(
            type == DataType::BOOL,
            fmt::format("Conjunct expressions expect BOOLEAN, received: {}",
                        type));
    }
    return DataType::BOOL;
}

TargetBitmap
PhyConjunctFilterExpr::BuildActiveBitmap(const ColumnVectorPtr& vec) {
    // Rows that still need the following expressions.
    //
    // For AND: TRUE or NULL rows still need evaluation; only definite FALSE
    //   can stop (NULL AND FALSE = FALSE, so a NULL row can still change).
    //   With a null-rejecting consumer, NULL is already equivalent to FALSE
    //   downstream, so NULL rows are dropped too and only TRUE rows stay
    //   active — restoring the pre-3VL early exit for UNKNOWN-heavy batches.
    // For OR: FALSE or NULL rows still need evaluation; only definite TRUE
    //   can stop. A NULL row can still become TRUE (NULL OR TRUE = TRUE),
    //   so null-rejection does not shrink the active set for OR.
    const size_t size = vec->size();
    TargetBitmapView data(vec->GetRawData(), size);
    TargetBitmapView valid(vec->GetValidRawData(), size);
    if (is_and_) {
        if (null_rejecting_) {
            TargetBitmap active_rows(data);
            active_rows.inplace_and(valid, size);  // data & valid
            return active_rows;
        }
        TargetBitmap active_rows(valid);
        active_rows.inplace_sub(data, size);  // valid & ~data
        active_rows.flip();                   // data | ~valid
        return active_rows;
    }
    TargetBitmap active_rows(data);
    active_rows.inplace_and(valid, size);  // data & valid
    active_rows.flip();                    // ~data | ~valid
    return active_rows;
}

void
PhyConjunctFilterExpr::SkipFollowingExprs(int start) {
    for (int i = start; i < input_order_.size(); ++i) {
        inputs_[input_order_[i]]->MoveCursor();
    }
}

void
PhyConjunctFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
    tracer::AutoSpan span(
        "PhyConjunctFilterExpr::Eval", tracer::GetRootSpan(), true);
    span.SetAttribute("is_and", is_and_);

    auto* explain = context.get_exec_context() == nullptr
                        ? nullptr
                        : context.get_exec_context()->explain();
    TargetBitmap inherited_bitmap;
    // Rows the next child is handed. A scalar-index child may return a
    // batch-wide bitmap even when only a few rows are alive, so its hit count
    // must be masked by this before it is attributed.
    // Snapshotted into a bitset before the child consumes the context bitmap,
    // only when explain recording is active.
    TargetBitmap next_mask;
    bool has_result = false;
    // Rows handed to the next child: the parent's mask for the first one, then
    // the active set the previous child left behind.
    int64_t next_rows_in = -1;
    auto restore_bitmap =
        folly::makeGuard([&context, &inherited_bitmap, &has_result] {
            if (has_result) {
                context.set_bitmap_input(std::move(inherited_bitmap));
            }
        });

    if (input_order_.empty()) {
        input_order_.resize(inputs_.size());
        for (size_t i = 0; i < inputs_.size(); i++) {
            input_order_[i] = i;
        }
    }

    auto has_input_offset = context.get_offset_input() != nullptr;
    if (!like_batch_initialized_ && is_and_ && like_indices_.size() > 1) {
        like_batch_initialized_ = true;
        // Collect LIKE expressions that can use ngram index at runtime.
        // The batch-ngram path evaluates whole batches, so it cannot be
        // used with an offset input; ngram_exprs then stays empty and the
        // else branch below erases the input_order_ slot reserved at
        // compile time. Deciding here — once, in every mode — keeps the
        // invariant that each input_order_ entry has a backing expression
        // in inputs_ before anything indexes inputs_ with it. (An
        // instance's offset mode is fixed for its lifetime, so no later
        // call could have used the batch path anyway.)
        std::vector<std::shared_ptr<PhyUnaryRangeFilterExpr>> ngram_exprs;
        // The fused batch path evaluates eligible LIKE leaves through a shared
        // candidate bitmap. When explain is active, PhyLikeConjunctExpr retains
        // per-leaf Phase 1 candidate bitmaps and tracks Phase 2 filtering per
        // batch to attribute rows accurately without disabling fusion.
        if (!has_input_offset) {
            for (size_t idx : like_indices_) {
                auto input = inputs_[idx];
                if (auto cached =
                        std::dynamic_pointer_cast<RawExprCacheAdapter>(input)) {
                    input = cached->wrapped_expr();
                }
                auto unary_expr =
                    std::dynamic_pointer_cast<PhyUnaryRangeFilterExpr>(input);
                if (unary_expr) {
                    unary_expr->EnsureExecPathDetermined();
                }
                if (unary_expr && unary_expr->CanUseNgramIndex()) {
                    ngram_exprs.push_back(unary_expr);
                    batch_ngram_indices_.insert(idx);
                }
            }
        }

        // Create PhyLikeConjunctExpr and add to inputs_ if we have >= 2 eligible
        if (ngram_exprs.size() >= 2) {
            auto active_count = ngram_exprs[0]->GetActiveCount();
            // The driver may have selected all-at-once execution before this
            // runtime node exists. Inherit its inputs' effective batch size.
            auto batch_size = ngram_exprs[0]->GetNextBatchSize();
            auto like_conjunct = std::make_shared<PhyLikeConjunctExpr>(
                std::move(ngram_exprs), op_ctx_, active_count, batch_size);
            inputs_.push_back(like_conjunct);
            if (explain != nullptr && eval_order_base_ > 0) {
                int32_t next_order = eval_order_base_;
                std::vector<ExprPtr> root{
                    std::shared_ptr<Expr>(this, [](Expr*) {})};
                AssignExplainEvalOrders(root, explain, next_order);
            }
        } else {
            batch_ngram_indices_.clear();
            // Remove the like_conjunct index from input_order_ since we're not
            // creating the batch expression. The index was reserved at compile
            // time but the PhyLikeConjunctExpr is not being created at runtime.
            auto original_size = inputs_.size();
            input_order_.erase(std::remove_if(input_order_.begin(),
                                              input_order_.end(),
                                              [original_size](size_t idx) {
                                                  return idx >= original_size;
                                              }),
                               input_order_.end());
        }
    }

    // Position of the last entry that will actually be evaluated: trailing
    // batch-ngram entries are skipped in the loop and must not force a
    // useless active-bitmap build after the real last input.
    size_t last_eval_pos = input_order_.size();
    for (size_t i = input_order_.size(); i-- > 0;) {
        if (batch_ngram_indices_.count(input_order_[i]) == 0) {
            last_eval_pos = i;
            break;
        }
    }

    for (size_t i = 0; i < input_order_.size(); ++i) {
        size_t idx = input_order_[i];

        // Skip expressions already executed via batch ngram
        if (batch_ngram_indices_.count(idx)) {
            continue;
        }

        const bool first_input = !has_result;
        if (explain != nullptr && !first_input && inputs_[idx] != nullptr &&
            inputs_[idx]->expr_id() != 0) {
            const auto& input = context.get_bitmap_input();
            next_mask.clear();
            if (!input.empty()) {
                next_mask.append(input, 0, input.size());
            }
        }
        VectorPtr input_result;
        inputs_[idx]->Eval(context, input_result);

        ColumnVectorPtr child_bitmap;
        ColumnVectorPtr all_flat_result;
        if (first_input) {
            // The first input needs the parent's mask in context. Take it only
            // after that input succeeds, then restore its storage on every exit.
            inherited_bitmap = context.take_bitmap_input();
            has_result = true;
            result = input_result;
            all_flat_result = GetColumnVector(result);
            child_bitmap = all_flat_result;
        } else {
            auto input_flat_result = GetColumnVector(input_result);
            child_bitmap = input_flat_result;
            all_flat_result = GetColumnVector(result);
            if (is_and_) {
                common::ThreeValuedLogicOp::And(all_flat_result,
                                                input_flat_result);
            } else {
                common::ThreeValuedLogicOp::Or(all_flat_result,
                                               input_flat_result);
            }
        }

        // Attribute this child's input and hits before the accumulated bitmap is
        // mutated further. A child that is itself logical carries no expr_id and
        // counts its own children.
        if (explain != nullptr && child_bitmap != nullptr &&
            inputs_[idx] != nullptr && inputs_[idx]->expr_id() != 0) {
            int64_t rows_in;
            if (first_input) {
                rows_in = inherited_bitmap.empty()
                              ? static_cast<int64_t>(child_bitmap->size())
                              : static_cast<int64_t>(inherited_bitmap.count());
            } else {
                rows_in = next_rows_in >= 0
                              ? next_rows_in
                              : static_cast<int64_t>(child_bitmap->size());
            }
            TargetBitmapView child_view(child_bitmap->GetRawData(),
                                        child_bitmap->size());
            int64_t rows_out;
            if (first_input && !inherited_bitmap.empty()) {
                next_mask.clear();
                next_mask.append(inherited_bitmap, 0, inherited_bitmap.size());
                const auto common =
                    std::min(next_mask.size(), child_view.size());
                rows_out = static_cast<int64_t>(
                    next_mask.inplace_and_with_count(child_view, common));
            } else if (!first_input && !next_mask.empty()) {
                const auto common =
                    std::min(next_mask.size(), child_view.size());
                rows_out = static_cast<int64_t>(
                    next_mask.inplace_and_with_count(child_view, common));
            } else {
                rows_out = static_cast<int64_t>(child_view.count());
            }
            explain->AddRows(inputs_[idx]->expr_id(), rows_in, rows_out);
        }

        // The last evaluated expression needs neither a skip decision nor a
        // bitmap input for a successor.
        if (i == last_eval_pos) {
            break;
        }

        // Build the active-row bitmap once per input: it decides the
        // batch-level early exit, and the same bitmap becomes the row-level
        // input of the next expression.
        auto active_rows = BuildActiveBitmap(all_flat_result);
        // Local results outside the parent's candidates are unspecified and
        // must neither reactivate rows nor prevent a batch-level early exit.
        if (!inherited_bitmap.empty()) {
            active_rows.inplace_and(inherited_bitmap, active_rows.size());
        }
        if (active_rows.none()) {
            SkipFollowingExprs(i + 1);
            return;
        }
        if (explain != nullptr) {
            next_rows_in = static_cast<int64_t>(active_rows.count());
        }

        context.set_bitmap_input(std::move(active_rows));
    }
}

}  //namespace exec
}  // namespace milvus
