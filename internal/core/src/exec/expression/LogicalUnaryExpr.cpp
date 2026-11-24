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

#include "LogicalUnaryExpr.h"

#include <algorithm>

#include "common/EasyAssert.h"
#include "common/Tracer.h"
#include "common/ValueOp.h"
#include "exec/expression/Utils.h"

namespace milvus {
namespace exec {

void
PhyLogicalUnaryExpr::Eval(EvalCtx& context, VectorPtr& result) {
    tracer::AutoSpan span("PhyLogicalUnaryExpr::Eval", tracer::GetRootSpan());

    AssertInfo(inputs_.size() == 1,
               "logical unary expr must has one input, but now {}",
               inputs_.size());

    auto* exec_ctx = context.get_exec_context();
    auto* explain = exec_ctx != nullptr ? exec_ctx->explain() : nullptr;
    const bool explain_leaf = explain != nullptr && inputs_[0] != nullptr &&
                              inputs_[0]->expr_id() != 0;
    const bool has_active = explain_leaf && HasActiveEvalInput(context);
    TargetBitmap mask_copy;
    if (has_active) {
        const auto& incoming_mask = context.get_bitmap_input();
        if (!incoming_mask.empty()) {
            mask_copy.append(incoming_mask, 0, incoming_mask.size());
        }
    }

    ExecutionExplainRecorder::ScopedLeafEvalTimer leaf_timer(
        has_active ? inputs_[0]->explain_slot() : nullptr);
    inputs_[0]->Eval(context, result);
    if (result == nullptr) {
        leaf_timer.Cancel();
        return;
    }
    // The proxy tags the operand of NOT, not the NOT node; without this the
    // condition would carry a decision group but no row counts.
    auto child_bitmap = GetColumnVector(result);
    const bool child_non_empty =
        child_bitmap != nullptr && child_bitmap->size() > 0;
    if (!child_non_empty) {
        leaf_timer.Cancel();
    } else if (has_active && child_bitmap->IsBitmap()) {
        const int64_t rows_in = mask_copy.empty()
                                    ? static_cast<int64_t>(child_bitmap->size())
                                    : static_cast<int64_t>(mask_copy.count());
        const int64_t rows_out =
            CountExplainBitmapMatches(child_bitmap.get(), mask_copy.view());
        ExecutionExplainRecorder::AddSlotRows(
            inputs_[0]->explain_slot(), rows_in, rows_out);
    }
    if (expr_->op_type_ == milvus::expr::LogicalUnaryExpr::OpType::LogicalNot) {
        common::ThreeValuedLogicOp::Not(child_bitmap);
    }
}

}  //namespace exec
}  // namespace milvus
