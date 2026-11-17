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
    TargetBitmap mask_copy;
    if (explain != nullptr && inputs_[0] != nullptr &&
        inputs_[0]->expr_id() != 0) {
        const auto& incoming_mask = context.get_bitmap_input();
        if (!incoming_mask.empty()) {
            mask_copy.append(incoming_mask, 0, incoming_mask.size());
        }
    }

    inputs_[0]->Eval(context, result);
    // The proxy tags the operand of NOT, not the NOT node; without this the
    // condition would carry a decision group but no row counts.
    auto child_bitmap = GetColumnVector(result);
    if (explain != nullptr && child_bitmap != nullptr &&
        inputs_[0] != nullptr && inputs_[0]->expr_id() != 0 &&
        child_bitmap->IsBitmap()) {
        const auto size = child_bitmap->size();
        TargetBitmapView child_view(child_bitmap->GetRawData(), size);
        int64_t rows_in = static_cast<int64_t>(size);
        int64_t rows_out = 0;
        if (!mask_copy.empty()) {
            rows_in = static_cast<int64_t>(mask_copy.count());
            const auto common = std::min(mask_copy.size(), child_view.size());
            rows_out = static_cast<int64_t>(
                mask_copy.inplace_and_with_count(child_view, common));
        } else {
            rows_out = static_cast<int64_t>(child_view.count());
        }
        explain->AddRows(inputs_[0]->expr_id(), rows_in, rows_out);
    }
    if (expr_->op_type_ == milvus::expr::LogicalUnaryExpr::OpType::LogicalNot) {
        common::ThreeValuedLogicOp::Not(GetColumnVector(result));
    }
}

}  //namespace exec
}  // namespace milvus
