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

}  // namespace

void
PhyCallExpr::RecordExplainDecisionIfNeeded() {
    if (explain_ == nullptr || explain_recorded_ || expr_id() == 0) {
        return;
    }
    explain_recorded_ = true;
    bool recorded_any = false;
    for (const auto& input : this->inputs_) {
        if (auto col = std::dynamic_pointer_cast<PhyColumnExpr>(input);
            col != nullptr) {
            auto info = col->ResolveExplainInfo();
            explain_->RecordDecision(info.field_id,
                                     expr_id(),
                                     info.used_index,
                                     info.index_id,
                                     info.family,
                                     info.nested,
                                     info.path,
                                     info.decline_reason);
            explain_->AddOperandRawScan(info.field_id,
                                        expr_id(),
                                        info.used_index,
                                        info.index_id,
                                        info.family,
                                        info.nested,
                                        info.path,
                                        info.decline_reason,
                                        0);
            recorded_any = true;
        }
    }
    if (recorded_any) {
        return;
    }
    FieldId field_id(0);
    for (const auto& param : expr_->inputs()) {
        if (auto col =
                std::dynamic_pointer_cast<const milvus::expr::ColumnExpr>(
                    param)) {
            field_id = col->GetColumn().field_id_;
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
    RecordExplainDecisionIfNeeded();

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
        int64_t max_raw_rows = 0;
        std::map<std::string, std::pair<PhyColumnExpr::ExplainInfo, int64_t>>
            per_operand_raw;
        for (const auto& input : this->inputs_) {
            if (auto col = std::dynamic_pointer_cast<PhyColumnExpr>(input);
                col != nullptr) {
                const int64_t col_raw = col->LastRawReadRows();
                max_raw_rows = std::max(max_raw_rows, col_raw);
                if (col_raw > 0) {
                    auto info = col->ResolveExplainInfo();
                    auto key = OperandExplainKey(info);
                    auto [it, inserted] = per_operand_raw.emplace(
                        std::move(key),
                        std::make_pair(std::move(info), col_raw));
                    if (!inserted) {
                        it->second.second =
                            std::max(it->second.second, col_raw);
                    }
                }
            }
        }
        for (const auto& [key, entry] : per_operand_raw) {
            const auto& info = entry.first;
            explain_->AddOperandRawScan(info.field_id,
                                        expr_id(),
                                        info.used_index,
                                        info.index_id,
                                        info.family,
                                        info.nested,
                                        info.path,
                                        info.decline_reason,
                                        entry.second);
        }
        if (max_raw_rows > 0) {
            explain_->AddRawScan(expr_id(), max_raw_rows);
        }
    }
}

}  // namespace exec
}  // namespace milvus
