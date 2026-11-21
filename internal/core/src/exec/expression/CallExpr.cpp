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

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "common/Tracer.h"
#include "common/Vector.h"
#include "exec/expression/CallExpr.h"
#include "exec/expression/ColumnExpr.h"
#include "exec/expression/EvalCtx.h"
#include "fmt/core.h"
#include "opentelemetry/trace/span.h"

namespace milvus {
namespace exec {

namespace {

std::string
OperandExplainKey(const PhyColumnExpr::ExplainInfo& info) {
    return fmt::format("{}|{}|{}|{}|{}|{}|{}",
                       info.field_id.get(),
                       info.used_index,
                       info.index_id,
                       info.family,
                       info.nested,
                       info.path,
                       info.decline_reason);
}

template <typename Fn>
void
ForEachColumnOperand(const ExprPtr& node, Fn&& fn) {
    if (!node) {
        return;
    }
    if (auto col = std::dynamic_pointer_cast<PhyColumnExpr>(node);
        col != nullptr) {
        fn(*col);
        return;
    }
    if (auto nested_call = std::dynamic_pointer_cast<PhyCallExpr>(node);
        nested_call != nullptr) {
        for (const auto& child : nested_call->GetInputsRef()) {
            ForEachColumnOperand(child, fn);
        }
    }
}

FieldId
FirstLogicalColumnFieldId(const expr::TypedExprPtr& node) {
    if (!node) {
        return FieldId(0);
    }
    if (auto col =
            std::dynamic_pointer_cast<const milvus::expr::ColumnExpr>(node);
        col != nullptr) {
        return col->GetColumn().field_id_;
    }
    if (auto nested_call =
            std::dynamic_pointer_cast<const milvus::expr::CallExpr>(node);
        nested_call != nullptr) {
        for (const auto& child : nested_call->inputs()) {
            auto fid = FirstLogicalColumnFieldId(child);
            if (fid.get() != 0) {
                return fid;
            }
        }
    }
    return FieldId(0);
}

}  // namespace

void
PhyCallExpr::RecordExplainDecision() {
    if (explain_ == nullptr || explain_recorded_ || expr_id() == 0) {
        return;
    }
    explain_recorded_ = true;
    bool recorded_any = false;
    for (const auto& input : this->inputs_) {
        ForEachColumnOperand(input, [&](const PhyColumnExpr& col) {
            auto info = col.ResolveExplainInfo();
            explain_->RecordDecision(info.field_id,
                                     expr_id(),
                                     info.used_index,
                                     info.index_id,
                                     info.family,
                                     info.nested,
                                     info.path,
                                     info.decline_reason);
            explain_->AddOperandScan(info.field_id,
                                     expr_id(),
                                     info.used_index,
                                     info.index_id,
                                     info.family,
                                     info.nested,
                                     info.path,
                                     info.decline_reason,
                                     0,
                                     0);
            operand_explains_.emplace(
                &col,
                CachedOperandExplain{
                    .field_id = info.field_id,
                    .used_index = info.used_index,
                    .index_id = info.index_id,
                    .family = std::move(info.family),
                    .nested = info.nested,
                    .path = std::move(info.path),
                    .decline_reason = std::move(info.decline_reason),
                });
            recorded_any = true;
        });
    }
    if (recorded_any) {
        return;
    }
    FieldId field_id(0);
    for (const auto& param : expr_->inputs()) {
        field_id = FirstLogicalColumnFieldId(param);
        if (field_id.get() != 0) {
            break;
        }
    }
    explain_->RecordDecision(
        field_id,
        expr_id(),
        /*used_index=*/false,
        /*index_id=*/0,
        /*family=*/"",
        /*nested=*/false,
        /*path=*/"",
        milvus::exec::ToString(DeclineReason::kNoIndexOnField));
}

void
PhyCallExpr::Eval(EvalCtx& context, VectorPtr& result) {
    tracer::AutoSpan span("PhyCallExpr::Eval", tracer::GetRootSpan(), true);
    span.SetAttribute("function_name", expr_->fun_name());
    RecordExplainDecision();

    auto offset_input = context.get_offset_input();
    SetHasOffsetInput(offset_input != nullptr);
    AssertInfo(inputs_.size() == expr_->inputs().size(),
               "logical call expr needs {} inputs, but {} inputs are provided",
               expr_->inputs().size(),
               inputs_.size());
    std::vector<VectorPtr> args;
    for (auto& input : this->inputs_) {
        VectorPtr arg_result;
        input->Eval(context, arg_result);
        args.push_back(std::move(arg_result));
    }
    RowVector row_vector(std::move(args));
    this->expr_->function_ptr()(row_vector, result);
    if (explain_ != nullptr && expr_id() != 0 && result != nullptr) {
        const auto& bitmap_input = context.get_bitmap_input();
        const int64_t active_rows =
            !bitmap_input.empty() ? static_cast<int64_t>(bitmap_input.count())
                                  : static_cast<int64_t>(result->size());
        if (active_rows > 0) {
            struct OperandBatchScan {
                PhyColumnExpr::ExplainInfo info;
                int64_t idx{0};
                int64_t raw{0};
            };
            int64_t max_raw_rows = 0;
            std::map<std::string, OperandBatchScan> per_operand_scan;
            for (const auto& input : this->inputs_) {
                ForEachColumnOperand(input, [&](const PhyColumnExpr& col) {
                    const bool used_reader = col.LastBatchUsedValueReader();
                    const int64_t col_idx = used_reader ? active_rows : 0;
                    const int64_t col_raw =
                        !used_reader ? col.LastRawReadRows() : 0;
                    max_raw_rows = std::max(max_raw_rows, col_raw);
                    if (col_idx <= 0 && col_raw <= 0) {
                        return;
                    }
                    PhyColumnExpr::ExplainInfo info;
                    if (auto cached_it = operand_explains_.find(&col);
                        cached_it != operand_explains_.end()) {
                        const auto& c = cached_it->second;
                        info.field_id = c.field_id;
                        info.used_index = c.used_index;
                        info.index_id = c.index_id;
                        info.family = c.family;
                        info.nested = c.nested;
                        info.path = c.path;
                        info.decline_reason = c.decline_reason;
                    } else {
                        info = col.ResolveExplainInfo();
                    }
                    auto key = OperandExplainKey(info);
                    auto [it, inserted] =
                        per_operand_scan.emplace(std::move(key),
                                                 OperandBatchScan{
                                                     .info = std::move(info),
                                                     .idx = col_idx,
                                                     .raw = col_raw,
                                                 });
                    if (!inserted) {
                        it->second.idx = std::max(it->second.idx, col_idx);
                        it->second.raw = std::max(it->second.raw, col_raw);
                    }
                });
            }
            for (const auto& [key, entry] : per_operand_scan) {
                const auto& info = entry.info;
                explain_->AddOperandScan(info.field_id,
                                         expr_id(),
                                         info.used_index,
                                         info.index_id,
                                         info.family,
                                         info.nested,
                                         info.path,
                                         info.decline_reason,
                                         entry.idx,
                                         entry.raw);
            }
            if (max_raw_rows > 0) {
                explain_->AddRawScan(expr_id(), max_raw_rows);
            }
        }
    }
}

}  // namespace exec
}  // namespace milvus
