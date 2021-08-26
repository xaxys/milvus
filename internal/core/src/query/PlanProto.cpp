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

#include "query/PlanProto.h"
#include "ExprImpl.h"
#include <google/protobuf/text_format.h>
#include <query/generated/ExtractInfoPlanNodeVisitor.h>
#include "query/generated/ExtractInfoExprVisitor.h"

namespace milvus::query {
namespace planpb = milvus::proto::plan;

static DataType
getSameType(const DataType& ty1, const DataType& ty2) {
    if (ty1 == DataType::DOUBLE || ty2 == DataType::DOUBLE)
        return DataType::DOUBLE;
    if (ty1 == DataType::FLOAT || ty2 == DataType::FLOAT)
        return DataType::FLOAT;
    if (ty1 == DataType::INT64 || ty2 == DataType::INT64)
        return DataType::INT64;
    if (ty1 == DataType::INT32 || ty2 == DataType::INT32)
        return DataType::INT32;
    if (ty1 == DataType::INT16 || ty2 == DataType::INT16)
        return DataType::INT16;
    if (ty1 == DataType::INT8 || ty2 == DataType::INT8)
        return DataType::INT8;
    if (ty1 == DataType::BOOL || ty2 == DataType::BOOL)
        return DataType::BOOL;
    // for string case
    PanicInfo("can not get same type");
}

template <typename T>
static T
ExtractValue(const planpb::GenericValue& value_proto) {
    if constexpr (std::is_same_v<T, bool>) {
        Assert(value_proto.val_case() == planpb::GenericValue::kBoolVal);
        return static_cast<T>(value_proto.bool_val());
    } else if constexpr (std::is_integral_v<T>) {
        Assert(value_proto.val_case() == planpb::GenericValue::kInt64Val);
        return static_cast<T>(value_proto.int64_val());
    } else if constexpr (std::is_floating_point_v<T>) {
        Assert(value_proto.val_case() == planpb::GenericValue::kFloatVal);
        return static_cast<T>(value_proto.float_val());
    } else {
        static_assert(always_false<T>);
    }
};

template <typename T>
std::unique_ptr<TermExprImpl<T>>
ExtractTermExprImpl(ExprPtr&& child, const planpb::TermExpr& expr_proto) {
    static_assert(std::is_fundamental_v<T>);
    auto result = std::make_unique<TermExprImpl<T>>();
    result->child_ = std::move(child);
    result->data_type_ = DataType::BOOL;

    auto size = expr_proto.values_size();
    for (int i = 0; i < size; ++i) {
        auto term = ExtractValue<T>(expr_proto.values(i));
        result->terms_.emplace_back(term);
    }
    return result;
}

template <typename T>
std::unique_ptr<UnaryRangeExprImpl<T>>
ExtractUnaryRangeExprImpl(ExprPtr&& child, const planpb::UnaryRangeExpr& expr_proto) {
    static_assert(std::is_fundamental_v<T>);
    auto result = std::make_unique<UnaryRangeExprImpl<T>>();
    result->child_ = std::move(child);
    result->data_type_ = DataType::BOOL;
    result->op_type_ = static_cast<CompareOp>(expr_proto.op());
    result->value_ = ExtractValue<T>(expr_proto.value());
    return result;
}

template <typename T>
std::unique_ptr<BinaryRangeExprImpl<T>>
ExtractBinaryRangeExprImpl(ExprPtr&& child, const planpb::BinaryRangeExpr& expr_proto) {
    static_assert(std::is_fundamental_v<T>);
    auto result = std::make_unique<BinaryRangeExprImpl<T>>();
    result->child_ = std::move(child);
    result->data_type_ = DataType::BOOL;
    result->lower_value_ = ExtractValue<T>(expr_proto.lower_value());
    result->upper_value_ = ExtractValue<T>(expr_proto.upper_value());
    return result;
}

std::unique_ptr<VectorPlanNode>
ProtoParser::PlanNodeFromProto(const planpb::PlanNode& plan_node_proto) {
    // TODO: add more buffs
    Assert(plan_node_proto.has_vector_anns());
    auto& anns_proto = plan_node_proto.vector_anns();
    auto expr_opt = [&]() -> std::optional<ExprPtr> {
        if (!anns_proto.has_predicates()) {
            return std::nullopt;
        } else {
            return ParseExpr(anns_proto.predicates());
        }
    }();

    auto& query_info_proto = anns_proto.query_info();

    SearchInfo search_info;
    auto field_id = FieldId(anns_proto.field_id());
    auto field_offset = schema.get_offset(field_id);
    search_info.field_offset_ = field_offset;

    search_info.metric_type_ = GetMetricType(query_info_proto.metric_type());
    search_info.topk_ = query_info_proto.topk();
    search_info.search_params_ = json::parse(query_info_proto.search_params());

    auto plan_node = [&]() -> std::unique_ptr<VectorPlanNode> {
        if (anns_proto.is_binary()) {
            return std::make_unique<BinaryVectorANNS>();
        } else {
            return std::make_unique<FloatVectorANNS>();
        }
    }();
    plan_node->placeholder_tag_ = anns_proto.placeholder_tag();
    plan_node->predicate_ = std::move(expr_opt);
    plan_node->search_info_ = std::move(search_info);
    return plan_node;
}

std::unique_ptr<Plan>
ProtoParser::CreatePlan(const proto::plan::PlanNode& plan_node_proto) {
    auto plan = std::make_unique<Plan>(schema);

    auto plan_node = PlanNodeFromProto(plan_node_proto);
    ExtractedPlanInfo plan_info(schema.size());
    ExtractInfoPlanNodeVisitor extractor(plan_info);
    plan_node->accept(extractor);

    plan->tag2field_["$0"] = plan_node->search_info_.field_offset_;
    plan->plan_node_ = std::move(plan_node);
    plan->extra_info_opt_ = std::move(plan_info);

    for (auto field_id_raw : plan_node_proto.output_field_ids()) {
        auto field_id = FieldId(field_id_raw);
        auto offset = schema.get_offset(field_id);
        plan->target_entries_.push_back(offset);
    }

    return plan;
}

ExprPtr
ProtoParser::ParseUnaryRangeExpr(const proto::plan::UnaryRangeExpr& expr_pb) {
    auto expr = ParseExpr(expr_pb.child());
    auto data_type = expr->data_type_;

    auto result = [&]() -> ExprPtr {
        switch (data_type) {
            case DataType::BOOL: {
                return ExtractUnaryRangeExprImpl<bool>(std::move(expr), expr_pb);
            }
            case DataType::INT8: {
                return ExtractUnaryRangeExprImpl<int8_t>(std::move(expr), expr_pb);
            }
            case DataType::INT16: {
                return ExtractUnaryRangeExprImpl<int16_t>(std::move(expr), expr_pb);
            }
            case DataType::INT32: {
                return ExtractUnaryRangeExprImpl<int32_t>(std::move(expr), expr_pb);
            }
            case DataType::INT64: {
                return ExtractUnaryRangeExprImpl<int64_t>(std::move(expr), expr_pb);
            }
            case DataType::FLOAT: {
                return ExtractUnaryRangeExprImpl<float>(std::move(expr), expr_pb);
            }
            case DataType::DOUBLE: {
                return ExtractUnaryRangeExprImpl<double>(std::move(expr), expr_pb);
            }
            default: {
                PanicInfo("unsupported data type");
            }
        }
    }();
    return result;
}

ExprPtr
ProtoParser::ParseBinaryRangeExpr(const proto::plan::BinaryRangeExpr& expr_pb) {
    auto expr = ParseExpr(expr_pb.child());
    auto data_type = expr->data_type_;

    auto result = [&]() -> ExprPtr {
        switch (data_type) {
            case DataType::BOOL: {
                return ExtractBinaryRangeExprImpl<bool>(std::move(expr), expr_pb);
            }
            case DataType::INT8: {
                return ExtractBinaryRangeExprImpl<int8_t>(std::move(expr), expr_pb);
            }
            case DataType::INT16: {
                return ExtractBinaryRangeExprImpl<int16_t>(std::move(expr), expr_pb);
            }
            case DataType::INT32: {
                return ExtractBinaryRangeExprImpl<int32_t>(std::move(expr), expr_pb);
            }
            case DataType::INT64: {
                return ExtractBinaryRangeExprImpl<int64_t>(std::move(expr), expr_pb);
            }
            case DataType::FLOAT: {
                return ExtractBinaryRangeExprImpl<float>(std::move(expr), expr_pb);
            }
            case DataType::DOUBLE: {
                return ExtractBinaryRangeExprImpl<double>(std::move(expr), expr_pb);
            }
            default: {
                PanicInfo("unsupported data type");
            }
        }
    }();
    return result;
}

ExprPtr
ProtoParser::ParseCompareExpr(const proto::plan::CompareExpr& expr_pb) {
    auto result = std::make_unique<CompareExpr>();
    result->op_type_ = static_cast<CompareOp>(expr_pb.op());
    result->data_type_ = DataType::BOOL;
    result->left_ = ParseExpr(expr_pb.left());
    result->right_ = ParseExpr(expr_pb.right());
    return result;
}

ExprPtr
ProtoParser::ParseTermExpr(const proto::plan::TermExpr& expr_pb) {
    auto expr = this->ParseExpr(expr_pb.child());
    auto data_type = expr->data_type_;

    auto result = [&]() -> ExprPtr {
        switch (data_type) {
            case DataType::BOOL: {
                return ExtractTermExprImpl<bool>(std::move(expr), expr_pb);
            }
            case DataType::INT8: {
                return ExtractTermExprImpl<int8_t>(std::move(expr), expr_pb);
            }
            case DataType::INT16: {
                return ExtractTermExprImpl<int16_t>(std::move(expr), expr_pb);
            }
            case DataType::INT32: {
                return ExtractTermExprImpl<int32_t>(std::move(expr), expr_pb);
            }
            case DataType::INT64: {
                return ExtractTermExprImpl<int64_t>(std::move(expr), expr_pb);
            }
            case DataType::FLOAT: {
                return ExtractTermExprImpl<float>(std::move(expr), expr_pb);
            }
            case DataType::DOUBLE: {
                return ExtractTermExprImpl<double>(std::move(expr), expr_pb);
            }
            default: {
                PanicInfo("unsupported data type");
            }
        }
    }();
    return result;
}

ExprPtr
ProtoParser::ParseUnaryLogicalExpr(const proto::plan::UnaryLogicalExpr& expr_pb) {
    auto result = std::make_unique<UnaryLogicalExpr>();
    result->op_type_ = static_cast<UnaryLogicalOp>(expr_pb.op());
    result->data_type_ = DataType::BOOL;
    Assert(result->op_type_ == UnaryLogicalOp::LogicalNot);
    result->child_ = ParseExpr(expr_pb.child());
    Assert(result->child_->data_type_ == DataType::BOOL);
    return result;
}

ExprPtr
ProtoParser::ParseBinaryLogicalExpr(const proto::plan::BinaryLogicalExpr& expr_pb) {
    auto result = std::make_unique<BinaryLogicalExpr>();
    result->op_type_ = static_cast<BinaryLogicalOp>(expr_pb.op());
    Assert(result->op_type_ != BinaryLogicalOp::InvalidBinaryOp);
    result->data_type_ = DataType::BOOL;
    result->left_ = ParseExpr(expr_pb.left());
    result->right_ = ParseExpr(expr_pb.right());
    Assert(result->left_->data_type_ == DataType::BOOL);
    Assert(result->right_->data_type_ == DataType::BOOL);
    return result;
}

ExprPtr
ProtoParser::ParseColumnExpr(const proto::plan::ColumnExpr& expr_pb) {
    auto result = std::make_unique<ColumnExpr>();
    auto& column_info = expr_pb.column_info();
    result->field_offset_ = schema.get_offset(FieldId(column_info.field_id()));
    result->data_type_ = schema[result->field_offset_].get_data_type();
    Assert(result->data_type_ == static_cast<DataType>(column_info.data_type()));
    return result;
}

ExprPtr
ProtoParser::ParseArithExpr(const proto::plan::ArithExpr& expr_pb) {
    auto result = std::make_unique<ArithExpr>();
    result->op_type_ = static_cast<ArithOp>(expr_pb.op());
    Assert(result->op_type_ != ArithOp::InvalidArithOp);
    result->left_ = this->ParseExpr(expr_pb.left());
    result->right_ = this->ParseExpr(expr_pb.right());
    result->data_type_ = getSameType(result->left_->data_type_, result->right_->data_type_);
    return result;
}

template <typename T, typename U>
static bool
in_limits(const U& value) {
    return value <= std::numeric_limits<T>::max() && value >= std::numeric_limits<T>::lowest();
}

ExprPtr
ProtoParser::ParseValueExpr(const planpb::ValueExpr& expr_pb) {
    using pgv = planpb::GenericValue::ValCase;
    const auto& value_proto = expr_pb.value();
    switch (value_proto.val_case()) {
        case pgv::kBoolVal: {
            auto result = std::make_unique<ValueExprImpl<bool>>();
            result->value_ = static_cast<bool>(value_proto.bool_val());
            result->data_type_ = DataType::BOOL;
            return result;
        }
        case pgv::kInt64Val: {
            auto value = static_cast<int64_t>(value_proto.int64_val());
            if (in_limits<int8_t>(value)) {
                auto result = std::make_unique<ValueExprImpl<int8_t>>();
                result->value_ = value;
                result->data_type_ = DataType::INT8;
                return result;
            }
            if (in_limits<int16_t>(value)) {
                auto result = std::make_unique<ValueExprImpl<int16_t>>();
                result->value_ = value;
                result->data_type_ = DataType::INT16;
                return result;
            }
            if (in_limits<int32_t>(value)) {
                auto result = std::make_unique<ValueExprImpl<int32_t>>();
                result->value_ = value;
                result->data_type_ = DataType::INT32;
                return result;
            }
            {
                auto result = std::make_unique<ValueExprImpl<int64_t>>();
                result->value_ = value;
                result->data_type_ = DataType::INT64;
                return result;
            }
        }
        case pgv::kFloatVal: {
            auto value = static_cast<double>(value_proto.float_val());
            if (in_limits<float>(value)) {
                auto result = std::make_unique<ValueExprImpl<float>>();
                result->value_ = value;
                result->data_type_ = DataType::FLOAT;
                return result;
            }
            {
                auto result = std::make_unique<ValueExprImpl<double>>();
                result->value_ = value;
                result->data_type_ = DataType::DOUBLE;
                return result;
            }
        }
        case proto::plan::GenericValue::VAL_NOT_SET:
            PanicInfo("value not set");
    }
}

ExprPtr
ProtoParser::ParseCastExpr(const planpb::CastExpr& expr_pb) {
    auto result = std::make_unique<CastExpr>();
    result->child_ = ParseExpr(expr_pb.child());
    result->data_type_ = static_cast<DataType>(expr_pb.data_type());
    return result;
}

ExprPtr
ProtoParser::ParseExpr(const proto::plan::Expr& expr_pb) {
    using ppe = proto::plan::Expr;
    switch (expr_pb.expr_case()) {
        case ppe::kTermExpr: {
            return ParseTermExpr(expr_pb.term_expr());
        }
        case ppe::kBinaryLogicalExpr: {
            return ParseBinaryLogicalExpr(expr_pb.binary_logical_expr());
        }
        case ppe::kUnaryLogicalExpr: {
            return ParseUnaryLogicalExpr(expr_pb.unary_logical_expr());
        }
        case ppe::kCompareExpr: {
            return ParseCompareExpr(expr_pb.compare_expr());
        }
        case ppe::kUnaryRangeExpr: {
            return ParseUnaryRangeExpr(expr_pb.unary_range_expr());
        }
        case ppe::kBinaryRangeExpr: {
            return ParseBinaryRangeExpr(expr_pb.binary_range_expr());
        }
        case ppe::kArithExpr: {
            return ParseArithExpr(expr_pb.arith_expr());
        }
        case ppe::kValueExpr: {
            return ParseValueExpr(expr_pb.value_expr());
        }
        case ppe::kColumnExpr: {
            return ParseColumnExpr(expr_pb.column_expr());
        }
        case ppe::kCastExpr: {
            return ParseCastExpr(expr_pb.cast_expr());
        }
        default:
            PanicInfo("unsupported expr proto node");
    }
}
}  // namespace milvus::query
