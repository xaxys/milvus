// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include "query/Plan.h"
#include <utility>
#include "query/generated/ShowExprVisitor.h"
#include "query/ExprImpl.h"

namespace milvus::query {
using Json = nlohmann::json;

#if 1
// THIS CONTAINS EXTRA BODY FOR VISITOR
// WILL BE USED BY GENERATOR
namespace impl {
class ShowExprNodeVisitor : ExprVisitor {
 public:
    using RetType = Json;

 public:
    RetType
    call_child(Expr& expr) {
        assert(!ret_.has_value());
        expr.accept(*this);
        assert(ret_.has_value());
        auto ret = std::move(ret_);
        ret_ = std::nullopt;
        return std::move(ret.value());
    }

    Json
    combine(Json&& extra, UnaryExprBase& expr) {
        auto result = std::move(extra);
        result["child"] = call_child(*expr.child_);
        return result;
    }

    Json
    combine(Json&& extra, BinaryExprBase& expr) {
        auto result = std::move(extra);
        result["left_child"] = call_child(*expr.left_);
        result["right_child"] = call_child(*expr.right_);
        return result;
    }

 private:
    std::optional<RetType> ret_;
};
}  // namespace impl
#endif

void
ShowExprVisitor::visit(UnaryLogicalExpr& expr) {
    Assert(!ret_.has_value());

    // TODO: use magic_enum if available
    Assert(expr.op_type_ == UnaryLogicalOp::LogicalNot);
    auto op_name = "LogicalNot";

    Json extra{
        {"expr_type", "UnaryLogical"},
        {"data_type", datatype_name(expr.data_type_)},
        {"op", op_name},
    };
    ret_ = combine(std::move(extra), expr);
}

void
ShowExprVisitor::visit(BinaryLogicalExpr& expr) {
    Assert(!ret_.has_value());

    // TODO: use magic_enum if available
    auto op_name = [](BinaryLogicalOp op) {
        switch (op) {
            case BinaryLogicalOp::LogicalAnd:
                return "LogicalAnd";
            case BinaryLogicalOp::LogicalOr:
                return "LogicalOr";
            case BinaryLogicalOp::LogicalXor:
                return "LogicalXor";
            default:
                PanicInfo("unsupported op");
        }
    }(expr.op_type_);

    Json extra{
        {"expr_type", "BinaryLogical"},
        {"data_type", datatype_name(expr.data_type_)},
        {"op", op_name},
    };
    ret_ = combine(std::move(extra), expr);
}

template <typename T>
static Json
TermExtract(const TermExpr& expr_raw) {
    auto expr = dynamic_cast<const TermExprImpl<T>*>(&expr_raw);
    Assert(expr);
    return Json{expr->terms_};
}

void
ShowExprVisitor::visit(TermExpr& expr) {
    Assert(!ret_.has_value());
    Assert(!datatype_is_vector(expr.child_->data_type_));
    auto terms = [&] {
        switch (expr.child_->data_type_) {
            case DataType::BOOL:
                return TermExtract<bool>(expr);
            case DataType::INT8:
                return TermExtract<int8_t>(expr);
            case DataType::INT16:
                return TermExtract<int16_t>(expr);
            case DataType::INT32:
                return TermExtract<int32_t>(expr);
            case DataType::INT64:
                return TermExtract<int64_t>(expr);
            case DataType::DOUBLE:
                return TermExtract<double>(expr);
            case DataType::FLOAT:
                return TermExtract<float>(expr);
            default:
                PanicInfo("unsupported type");
        }
    }();

    Json res{
        {"expr_type", "Term"},
        {"data_type", datatype_name(expr.data_type_)},
        {"terms", std::move(terms)},
    };

    ret_ = combine(std::move(res), expr);
}

template <typename T>
static Json
UnaryRangeExtract(const UnaryRangeExpr& expr_raw) {
    using proto::plan::CompareOp;
    using proto::plan::CompareOp_Name;
    auto expr = dynamic_cast<const UnaryRangeExprImpl<T>*>(&expr_raw);
    Assert(expr);
    Json res{
        {"expr_type", "UnaryRange"},
        {"data_type", datatype_name(expr->data_type_)},
        {"op", CompareOp_Name(static_cast<CompareOp>(expr->op_type_))},
        {"value", expr->value_},
    };
    return res;
}

void
ShowExprVisitor::visit(UnaryRangeExpr& expr) {
    Assert(!ret_.has_value());
    Json res;
    switch (expr.child_->data_type_) {
        case DataType::BOOL:
            res = UnaryRangeExtract<bool>(expr);
            break;
        case DataType::INT8:
            res = UnaryRangeExtract<int8_t>(expr);
            break;
        case DataType::INT16:
            res = UnaryRangeExtract<int16_t>(expr);
            break;
        case DataType::INT32:
            res = UnaryRangeExtract<int32_t>(expr);
            break;
        case DataType::INT64:
            res = UnaryRangeExtract<int64_t>(expr);
            break;
        case DataType::DOUBLE:
            res = UnaryRangeExtract<double>(expr);
            break;
        case DataType::FLOAT:
            res = UnaryRangeExtract<float>(expr);
            break;
        default:
            PanicInfo("unsupported type");
    }
    ret_ = combine(std::move(res), expr);
}

template <typename T>
static Json
RangeExtract(const BinaryRangeExpr& expr_raw) {
    auto expr = dynamic_cast<const BinaryRangeExprImpl<T>*>(&expr_raw);
    Assert(expr);
    Json res{
        {"expr_type", "BinaryRange"},
        {"data_type", datatype_name(expr->data_type_)},
        {"lower_inclusive", expr->lower_inclusive_},
        {"upper_inclusive", expr->upper_inclusive_},
        {"lower_value", expr->lower_value_},
        {"upper_value", expr->upper_value_},
    };
    return res;
}

void
ShowExprVisitor::visit(BinaryRangeExpr& expr) {
    Assert(!ret_.has_value());
    Json res;
    switch (expr.child_->data_type_) {
        case DataType::BOOL:
            res = RangeExtract<bool>(expr);
            break;
        case DataType::INT8:
            res = RangeExtract<int8_t>(expr);
            break;
        case DataType::INT16:
            res = RangeExtract<int16_t>(expr);
            break;
        case DataType::INT32:
            res = RangeExtract<int32_t>(expr);
            break;
        case DataType::INT64:
            res = RangeExtract<int64_t>(expr);
            break;
        case DataType::DOUBLE:
            res = RangeExtract<double>(expr);
            break;
        case DataType::FLOAT:
            res = RangeExtract<float>(expr);
            break;
        default:
            PanicInfo("unsupported type");
    }
    ret_ = combine(std::move(res), expr);
}

void
ShowExprVisitor::visit(CompareExpr& expr) {
    using proto::plan::CompareOp;
    using proto::plan::CompareOp_Name;
    Assert(!ret_.has_value());

    Json res{
        {"expr_type", "Compare"},
        {"data_type", datatype_name(expr.data_type_)},
        {"op", CompareOp_Name(static_cast<CompareOp>(expr.op_type_))},
    };
    ret_ = combine(std::move(res), expr);
}

void
ShowExprVisitor::visit(ColumnExpr& expr) {
    Assert(!ret_.has_value());
    Assert(!datatype_is_vector(expr.data_type_));
    Json res{
        {"expr_type", "Column"},
        {"data_type", datatype_name(expr.data_type_)},
        {"field_offset", expr.field_offset_.get()},
    };
    ret_ = res;
}

void
ShowExprVisitor::visit(ArithExpr& expr) {
    using proto::plan::ArithOp;
    using proto::plan::ArithOp_Name;
    Assert(!ret_.has_value());
    Json res{{"expr_type", "Arith"},
             {"data_type", datatype_name(expr.data_type_)},
             {"op", ArithOp_Name(static_cast<ArithOp>(expr.op_type_))}};
    ret_ = combine(std::move(res), expr);
}

template <typename T>
static Json
ValueExtract(const ValueExpr& expr_raw) {
    auto expr = dynamic_cast<const ValueExprImpl<T>*>(&expr_raw);
    Assert(expr);
    Json res{
        {"expr_type", "Value"},
        {"data_type", datatype_name(expr->data_type_)},
        {"value", expr->value_},
    };
    return res;
}

void
ShowExprVisitor::visit(ValueExpr& expr) {
    Assert(!ret_.has_value());
    Json res;
    switch (expr.data_type_) {
        case DataType::BOOL:
            res = ValueExtract<bool>(expr);
            break;
        case DataType::INT8:
            res = ValueExtract<int8_t>(expr);
            break;
        case DataType::INT16:
            res = ValueExtract<int16_t>(expr);
            break;
        case DataType::INT32:
            res = ValueExtract<int32_t>(expr);
            break;
        case DataType::INT64:
            res = ValueExtract<int64_t>(expr);
            break;
        case DataType::DOUBLE:
            res = ValueExtract<double>(expr);
            break;
        case DataType::FLOAT:
            res = ValueExtract<float>(expr);
            break;
        default:
            PanicInfo("unsupported type");
    }
    ret_ = std::move(res);
}

void
ShowExprVisitor::visit(CastExpr& expr) {
    Assert(!ret_.has_value());
    Json res{
        {"expr_type", "Cast"},
        {"data_type", datatype_name(expr.data_type_)},
    };
    ret_ = combine(std::move(res), expr);
}
}  // namespace milvus::query
