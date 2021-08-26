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

#include "common/Schema.h"
#include "query/PlanNode.h"
#include "pb/plan.pb.h"
#include "query/Plan.h"
#include <boost/dynamic_bitset.hpp>
#include <memory>

namespace milvus::query {

class ProtoParser {
 public:
    explicit ProtoParser(const Schema& schema) : schema(schema) {
    }

    //    ExprPtr
    //    ExprFromProto(const proto::plan::Expr& expr_proto);

    ExprPtr
    ParseUnaryRangeExpr(const proto::plan::UnaryRangeExpr& expr_pb);

    ExprPtr
    ParseBinaryRangeExpr(const proto::plan::BinaryRangeExpr& expr_pb);

    ExprPtr
    ParseCompareExpr(const proto::plan::CompareExpr& expr_pb);

    ExprPtr
    ParseTermExpr(const proto::plan::TermExpr& expr_pb);

    ExprPtr
    ParseUnaryLogicalExpr(const proto::plan::UnaryLogicalExpr& expr_pb);

    ExprPtr
    ParseBinaryLogicalExpr(const proto::plan::BinaryLogicalExpr& expr_pb);

    ExprPtr
    ParseColumnExpr(const proto::plan::ColumnExpr& expr_pb);

    ExprPtr
    ParseArithExpr(const proto::plan::ArithExpr& expr_pb);

    ExprPtr
    ParseValueExpr(const proto::plan::ValueExpr& expr_pb);

    ExprPtr
    ParseExpr(const proto::plan::Expr& expr_pb);

    std::unique_ptr<VectorPlanNode>
    PlanNodeFromProto(const proto::plan::PlanNode& plan_node_proto);

    std::unique_ptr<Plan>
    CreatePlan(const proto::plan::PlanNode& plan_node_proto);

 private:
    const Schema& schema;
    // boost::dynamic_bitset<> involved_fields;
};

}  // namespace milvus::query
