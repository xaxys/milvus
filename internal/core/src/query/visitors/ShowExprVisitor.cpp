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
        AssertInfo(!ret_.has_value(), "[ShowExprVisitor]Ret json already has value before visit");

        // TODO: use magic_enum if available
        AssertInfo(expr.op_type_ == UnaryLogicalOp::LogicalNot,  "[ShowExprVisitor]Expr op type isn't LogicNot");
        auto op_name = "LogicalNot";

        Json extra{
                {"expr_type", "UnaryLogical"},
                {"op", op_name},
        };
        ret_ = combine(std::move(extra), expr);
    }

    void
    ShowExprVisitor::visit(BinaryLogicalExpr& expr) {
        AssertInfo(!ret_.has_value(), "[ShowExprVisitor]Ret json already has value before visit");

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
                {"op", op_name},
        };
        ret_ = combine(std::move(extra), expr);
    }

    template <typename T>
    static Json
    GenericValueImplExtract(const GenericValue& gv_raw) {
        using proto::plan::CompareOp;
        using proto::plan::CompareOp_Name;
        auto gv = dynamic_cast<const GenericValueImpl<T>*>(&gv_raw);
        AssertInfo(gv, "[ShowExprVisitor]GenericValue cast to GenericValueImpl failed");
        Json res{
                {"expr_type", "GenericValue"},
                {"data_type", datatype_name(gv->data_type_)},
                {"value", gv->value_},
        };
        return res;
    }

    static Json
    GenericValueExtract(const GenericValue& gv) {
        using proto::plan::CompareOp;
        using proto::plan::CompareOp_Name;

        Json res;
        switch (gv.data_type_) {
            case DataType::BOOL:
                res = GenericValueImplExtract<bool>(gv);
                break;
            case DataType::INT8:
                res = GenericValueImplExtract<int8_t>(gv);
                break;
            case DataType::INT16:
                res = GenericValueImplExtract<int16_t>(gv);
                break;
            case DataType::INT32:
                res = GenericValueImplExtract<int32_t>(gv);
                break;
            case DataType::INT64:
                res = GenericValueImplExtract<int64_t>(gv);
                break;
            case DataType::DOUBLE:
                res = GenericValueImplExtract<double>(gv);
                break;
            case DataType::FLOAT:
                res = GenericValueImplExtract<float>(gv);
                break;
            default:
                PanicInfo("unsupported type");
        }
        return res;
    }

    void
    ShowExprVisitor::visit(TermExpr& expr) {
        AssertInfo(!ret_.has_value(), "[ShowExprVisitor]Ret json already has value before visit");
        Json terms;
        for (const auto& term : expr.terms_) {
            terms.push_back(GenericValueExtract(*term));
        }
        Json res{
                {"expr_type", "Term"},
                {"terms", std::move(terms)},
        };

        ret_ = combine(std::move(res), expr);
    }

    void
    ShowExprVisitor::visit(UnaryRangeExpr& expr) {
        AssertInfo(!ret_.has_value(), "[ShowExprVisitor]Ret json already has value before visit");
        using proto::plan::CompareOp;
        using proto::plan::CompareOp_Name;
        Json res{
                {"expr_type", "UnaryRange"},
                {"op", CompareOp_Name(static_cast<CompareOp>(expr.op_type_))},
                {"value", GenericValueExtract(*expr.value_)},
        };
        ret_ = combine(std::move(res), expr);
    }

    void
    ShowExprVisitor::visit(BinaryRangeExpr& expr) {
        AssertInfo(!ret_.has_value(), "[ShowExprVisitor]Ret json already has value before visit");
        Json res{
                {"expr_type", "BinaryRange"},
                {"lower_inclusive", expr.lower_inclusive_},
                {"upper_inclusive", expr.upper_inclusive_},
                {"lower_value", GenericValueExtract(*expr.lower_value_)},
                {"upper_value", GenericValueExtract(*expr.upper_value_)},
        };
        ret_ = combine(std::move(res), expr);
    }

    void
    ShowExprVisitor::visit(CompareExpr& expr) {
        using proto::plan::CompareOp;
        using proto::plan::CompareOp_Name;
        AssertInfo(!ret_.has_value(), "[ShowExprVisitor]Ret json already has value before visit");

        Json res{
                {"expr_type", "Compare"},
                {"op", CompareOp_Name(static_cast<CompareOp>(expr.op_type_))},
        };
        ret_ = combine(std::move(res), expr);
    }

    void
    ShowExprVisitor::visit(ColumnExpr& expr) {
        AssertInfo(!ret_.has_value(), "[ShowExprVisitor]Ret json already has value before visit");
        AssertInfo(!datatype_is_vector(expr.data_type_), "[ShowExprVisitor]Data type of column isn't vector type");
        Json res{
                {"expr_type", "Column"},
                {"data_type", datatype_name(expr.data_type_)},
                {"field_offset", expr.field_offset_.get()},
        };
        ret_ = res;
    }

    void
    ShowExprVisitor::visit(UnaryArithExpr& expr) {
        using proto::plan::UnaryArithOp;
        using proto::plan::UnaryArithOp_Name;
        AssertInfo(!ret_.has_value(), "[ShowExprVisitor]Ret json already has value before visit");
        Json res{{"expr_type", "UnaryArith"}, {"op", UnaryArithOp_Name(static_cast<UnaryArithOp>(expr.op_type_))}};
        ret_ = combine(std::move(res), expr);
    }

    void
    ShowExprVisitor::visit(BinaryArithExpr& expr) {
        using proto::plan::BinaryArithOp;
        using proto::plan::BinaryArithOp_Name;
        AssertInfo(!ret_.has_value(), "[ShowExprVisitor]Ret json already has value before visit");
        Json res{{"expr_type", "BinaryArith"}, {"op", BinaryArithOp_Name(static_cast<BinaryArithOp>(expr.op_type_))}};
        ret_ = combine(std::move(res), expr);
    }

    void
    ShowExprVisitor::visit(ValueExpr& expr) {
        AssertInfo(!ret_.has_value(), "[ShowExprVisitor]Ret json already has value before visit");
        Json res{
                {"expr_type", "Value"},
                {"value", GenericValueExtract(*expr.value_)},
        };
        ret_ = std::move(res);
    }

    void
    ShowExprVisitor::visit(CastExpr& expr) {
        AssertInfo(!ret_.has_value(), "[ShowExprVisitor]Ret json already has value before visit");
        Json res{
                {"expr_type", "Cast"},
                {"data_type", datatype_name(expr.data_type_)},
        };
        ret_ = combine(std::move(res), expr);
    }
}  // namespace milvus::query
