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

std::unique_ptr<RetrievePlanNode>
ProtoParser::RetrievePlanNodeFromProto(const planpb::PlanNode& plan_node_proto) {
    Assert(plan_node_proto.has_predicates());
    auto& predicate_proto = plan_node_proto.predicates();
    auto expr_opt = [&]() -> ExprPtr { return ParseExpr(predicate_proto); }();

    auto plan_node = [&]() -> std::unique_ptr<RetrievePlanNode> { return std::make_unique<RetrievePlanNode>(); }();
    plan_node->predicate_ = std::move(expr_opt);
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

std::unique_ptr<RetrievePlan>
ProtoParser::CreateRetrievePlan(const proto::plan::PlanNode& plan_node_proto) {
    auto retrieve_plan = std::make_unique<RetrievePlan>(schema);

    auto plan_node = RetrievePlanNodeFromProto(plan_node_proto);
    ExtractedPlanInfo plan_info(schema.size());
    ExtractInfoPlanNodeVisitor extractor(plan_info);
    plan_node->accept(extractor);

    retrieve_plan->plan_node_ = std::move(plan_node);
    for (auto field_id_raw : plan_node_proto.output_field_ids()) {
        auto field_id = FieldId(field_id_raw);
        auto offset = schema.get_offset(field_id);
        retrieve_plan->field_offsets_.push_back(offset);
    }
    return retrieve_plan;
}

ExprPtr
ProtoParser::ParseUnaryRangeExpr(const proto::plan::UnaryRangeExpr& expr_pb) {
    auto result = std::make_unique<UnaryRangeExpr>();
    result->child_ = ParseExpr(expr_pb.child());
    result->op_type_ = static_cast<CompareOp>(expr_pb.op());
    result->value_ = ParseGenericValue(expr_pb.value());
    return result;
}

ExprPtr
ProtoParser::ParseBinaryRangeExpr(const proto::plan::BinaryRangeExpr& expr_pb) {
    auto result = std::make_unique<BinaryRangeExpr>();
    result->child_ = ParseExpr(expr_pb.child());
    result->lower_value_ = ParseGenericValue(expr_pb.lower_value());
    result->upper_value_ = ParseGenericValue(expr_pb.upper_value());
    return result;
}

ExprPtr
ProtoParser::ParseCompareExpr(const proto::plan::CompareExpr& expr_pb) {
    auto result = std::make_unique<CompareExpr>();
    result->op_type_ = static_cast<CompareOp>(expr_pb.op());
    result->left_ = ParseExpr(expr_pb.left());
    result->right_ = ParseExpr(expr_pb.right());
    return result;
}

ExprPtr
ProtoParser::ParseTermExpr(const proto::plan::TermExpr& expr_pb) {
    auto result = std::make_unique<TermExpr>();
    result->child_ = ParseExpr(expr_pb.child());
    for (int i = 0; i < expr_pb.values_size(); i++) {
        auto term = ParseGenericValue(expr_pb.values(i));
        auto data_type = term->data_type_;
        result->values_.emplace_back(std::move(term));
        Assert(data_type == result->values_[0]->data_type_);
    }
    return result;
}

ExprPtr
ProtoParser::ParseUnaryLogicalExpr(const proto::plan::UnaryLogicalExpr& expr_pb) {
    auto result = std::make_unique<UnaryLogicalExpr>();
    result->op_type_ = static_cast<UnaryLogicalOp>(expr_pb.op());
    Assert(result->op_type_ == UnaryLogicalOp::LogicalNot);
    result->child_ = ParseExpr(expr_pb.child());
    return result;
}

ExprPtr
ProtoParser::ParseBinaryLogicalExpr(const proto::plan::BinaryLogicalExpr& expr_pb) {
    auto result = std::make_unique<BinaryLogicalExpr>();
    result->op_type_ = static_cast<BinaryLogicalOp>(expr_pb.op());
    Assert(result->op_type_ != BinaryLogicalOp::InvalidBinaryLogicalOp);
    result->left_ = ParseExpr(expr_pb.left());
    result->right_ = ParseExpr(expr_pb.right());
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
ProtoParser::ParseUnaryArithExpr(const proto::plan::UnaryArithExpr& expr_pb) {
    auto result = std::make_unique<UnaryArithExpr>();
    result->op_type_ = static_cast<UnaryArithOp>(expr_pb.op());
    Assert(result->op_type_ != UnaryArithOp::InvalidUnaryArithOp);
    result->child_ = ParseExpr(expr_pb.child());
    return result;
}

ExprPtr
ProtoParser::ParseBinaryArithExpr(const proto::plan::BinaryArithExpr& expr_pb) {
    auto result = std::make_unique<BinaryArithExpr>();
    result->op_type_ = static_cast<BinaryArithOp>(expr_pb.op());
    Assert(result->op_type_ != BinaryArithOp::InvalidBinaryArithOp);
    result->left_ = ParseExpr(expr_pb.left());
    result->right_ = ParseExpr(expr_pb.right());
    return result;
}

GenericValuePtr
ProtoParser::ParseGenericValue(const proto::plan::GenericValue& gv_pb) {
    using pgv = planpb::GenericValue::ValCase;
    switch (gv_pb.val_case()) {
        case pgv::kBoolVal: {
            auto result = std::make_unique<GenericValueImpl<bool>>();
            result->value_ = static_cast<bool>(gv_pb.bool_val());
            result->data_type_ = DataType::BOOL;
            return result;
        }
        case pgv::kInt64Val: {
            auto result = std::make_unique<GenericValueImpl<int64_t>>();
            result->value_ = static_cast<int64_t>(gv_pb.int64_val());
            ;
            result->data_type_ = DataType::INT64;
            return result;
        }
        case pgv::kFloatVal: {
            auto result = std::make_unique<GenericValueImpl<double>>();
            result->value_ = static_cast<double>(gv_pb.float_val());
            result->data_type_ = DataType::DOUBLE;
            return result;
        }
        case proto::plan::GenericValue::VAL_NOT_SET:
            PanicInfo("value not set");
    }
}

ExprPtr
ProtoParser::ParseValueExpr(const planpb::ValueExpr& expr_pb) {
    auto result = std::make_unique<ValueExpr>();
    result->value_ = ParseGenericValue(expr_pb.value());
    return result;
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
        case ppe::kUnaryArithExpr: {
            return ParseUnaryArithExpr(expr_pb.unary_arith_expr());
        }
        case ppe::kBinaryArithExpr: {
            return ParseBinaryArithExpr(expr_pb.binary_arith_expr());
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
