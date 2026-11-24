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

#include <folly/FBVector.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "NamedType/named_type_impl.hpp"
#include "common/EasyAssert.h"
#include "common/QueryResult.h"
#include "common/Schema.h"
#include "common/Types.h"
#include "common/protobuf_utils.h"
#include "exec/ExecutionExplainRecorder.h"
#include "exec/operator/Utils.h"
#include "expr/ITypeExpr.h"
#include "filemanager/InputStream.h"
#include "gtest/gtest.h"
#include "knowhere/comp/index_param.h"
#include "pb/common.pb.h"
#include "pb/internal.pb.h"
#include "pb/plan.pb.h"
#include "plan/PlanNode.h"
#include "query/Plan.h"
#include "query/PlanNode.h"
#include "rescores/Scorer.h"
#include "segcore/SegmentGrowing.h"
#include "segcore/SegmentSealed.h"
#include "segcore/Types.h"
#include "segcore/storagev1translator/ChunkTranslator.h"
#include "test_utils/DataGen.h"
#include "segcore/test_utils/ConsumerIndexTestUtils.h"
#include "test_utils/storage_test_utils.h"

using namespace milvus;
using namespace milvus::query;
using namespace milvus::segcore;

TEST(Rescorer, Normal) {
    //0. prepare schema
    int dim = 64;
    auto schema = std::make_shared<Schema>();
    auto vec_fid = schema->AddDebugField(
        "fakevec", DataType::VECTOR_FLOAT, dim, knowhere::metric::L2);
    schema->AddDebugField("int8", DataType::INT8);
    schema->AddDebugField("int16", DataType::INT16);
    schema->AddDebugField("int32", DataType::INT32);
    schema->AddDebugField("int64", DataType::INT64);
    auto str_fid = schema->AddDebugField("string", DataType::VARCHAR);
    schema->AddDebugField("bool", DataType::BOOL);
    schema->set_primary_field_id(str_fid);
    size_t N = 50;

    //2. load raw data
    auto raw_data = DataGen(schema, N, 42, 0, 8, 10, false, false);
    auto segment = CreateSealedWithFieldDataLoaded(schema, raw_data);

    //3. load index
    auto vector_data = raw_data.get_col<float>(vec_fid);
    auto load_index_info = milvus::test::consumer::MakeLoadIndexInfo(
        milvus::test::consumer::BuildVectorReader<float>(
            DataType::VECTOR_FLOAT,
            knowhere::IndexEnum::INDEX_HNSW,
            knowhere::metric::L2,
            dim,
            N,
            vector_data.data()),
        DataType::VECTOR_FLOAT,
        vec_fid.get());
    segment->LoadIndex(load_index_info);

    // no result after search
    {
        ScopedSchemaHandle handle(*schema);
        ScopedSchemaHandle::ScorerParams scorer;
        scorer.weight = 4;
        auto plan = handle.ParseSearchWithScorers(schema,
                                                  "int8 >= 100 and int8 < -1",
                                                  "fakevec",
                                                  10,
                                                  "L2",
                                                  R"({"ef": 50})",
                                                  {scorer});
        auto num_queries = 1;
        auto seed = 1024;
        auto ph_group_raw = CreatePlaceholderGroup(num_queries, dim, seed);
        auto ph_group =
            ParsePlaceholderGroup(plan.get(), ph_group_raw.SerializeAsString());
        auto search_result =
            segment->Search(plan.get(), ph_group.get(), MAX_TIMESTAMP);
    }

    // search result not empty but no boost filter
    {
        ScopedSchemaHandle handle(*schema);
        ScopedSchemaHandle::ScorerParams scorer;
        scorer.weight = 4;
        auto plan = handle.ParseSearchWithScorers(schema,
                                                  "int8 >= -1 and int8 < 100",
                                                  "fakevec",
                                                  10,
                                                  "L2",
                                                  R"({"ef": 50})",
                                                  {scorer});
        auto num_queries = 1;
        auto seed = 1024;
        auto ph_group_raw = CreatePlaceholderGroup(num_queries, dim, seed);
        auto ph_group =
            ParsePlaceholderGroup(plan.get(), ph_group_raw.SerializeAsString());
        auto search_result =
            segment->Search(plan.get(), ph_group.get(), MAX_TIMESTAMP);
    }

    // random function with seed
    {
        ScopedSchemaHandle handle(*schema);
        ScopedSchemaHandle::ScorerParams scorer;
        scorer.type = proto::plan::FunctionTypeRandom;
        scorer.weight = 1;
        scorer.params = {{"seed", "123"}};
        auto plan = handle.ParseSearchWithScorers(schema,
                                                  "int8 >= -1 and int8 < 100",
                                                  "fakevec",
                                                  10,
                                                  "L2",
                                                  R"({"ef": 50})",
                                                  {scorer});
        auto num_queries = 1;
        auto seed = 1024;
        auto ph_group_raw = CreatePlaceholderGroup(num_queries, dim, seed);
        auto ph_group =
            ParsePlaceholderGroup(plan.get(), ph_group_raw.SerializeAsString());
        auto search_result =
            segment->Search(plan.get(), ph_group.get(), MAX_TIMESTAMP);
    }

    // random function with field as random seed
    {
        ScopedSchemaHandle handle(*schema);
        ScopedSchemaHandle::ScorerParams scorer;
        scorer.type = proto::plan::FunctionTypeRandom;
        scorer.weight = 1;
        scorer.params = {{"field", "int64"}};
        auto plan = handle.ParseSearchWithScorers(schema,
                                                  "int8 >= -1 and int8 < 100",
                                                  "fakevec",
                                                  10,
                                                  "L2",
                                                  R"({"ef": 50})",
                                                  {scorer});
        auto num_queries = 1;
        auto seed = 1024;
        auto ph_group_raw = CreatePlaceholderGroup(num_queries, dim, seed);
        auto ph_group =
            ParsePlaceholderGroup(plan.get(), ph_group_raw.SerializeAsString());
        auto search_result =
            segment->Search(plan.get(), ph_group.get(), MAX_TIMESTAMP);
    }

    // random function with field and seed
    {
        ScopedSchemaHandle handle(*schema);
        ScopedSchemaHandle::ScorerParams scorer;
        scorer.type = proto::plan::FunctionTypeRandom;
        scorer.weight = 1;
        scorer.params = {{"seed", "123"}, {"field", "int64"}};
        auto plan = handle.ParseSearchWithScorers(schema,
                                                  "int8 >= -1 and int8 < 100",
                                                  "fakevec",
                                                  10,
                                                  "L2",
                                                  R"({"ef": 50})",
                                                  {scorer});
        auto num_queries = 1;
        auto seed = 1024;
        auto ph_group_raw = CreatePlaceholderGroup(num_queries, dim, seed);
        auto ph_group =
            ParsePlaceholderGroup(plan.get(), ph_group_raw.SerializeAsString());
        auto search_result =
            segment->Search(plan.get(), ph_group.get(), MAX_TIMESTAMP);

        auto search_result_same_seed =
            segment->Search(plan.get(), ph_group.get(), MAX_TIMESTAMP);

        // should return same score when use same seed
        for (auto i = 0; i < 10; i++) {
            AssertInfo(search_result->distances_[i] ==
                           search_result_same_seed->distances_[i],
                       "distance not equal %f:%f",
                       search_result->distances_[i],
                       search_result_same_seed->distances_[i]);
        }
    }
}

TEST(Rescorer, ExplainOperatorPipelineAndScorerFilterIsolation) {
    int dim = 16;
    auto schema = std::make_shared<Schema>();
    auto vec_fid = schema->AddDebugField(
        "fakevec", DataType::VECTOR_FLOAT, dim, knowhere::metric::L2);
    schema->AddDebugField("int64", DataType::INT64);
    auto str_fid = schema->AddDebugField("string", DataType::VARCHAR);
    schema->set_primary_field_id(str_fid);
    size_t N = 40;

    auto raw_data = DataGen(schema, N, 42, 0, 8, 10, false, false);
    auto segment = CreateSealedWithFieldDataLoaded(schema, raw_data);

    auto vector_data = raw_data.get_col<float>(vec_fid);
    auto load_index_info = milvus::test::consumer::MakeLoadIndexInfo(
        milvus::test::consumer::BuildVectorReader<float>(
            DataType::VECTOR_FLOAT,
            knowhere::IndexEnum::INDEX_HNSW,
            knowhere::metric::L2,
            dim,
            N,
            vector_data.data()),
        DataType::VECTOR_FLOAT,
        vec_fid.get());
    segment->LoadIndex(load_index_info);

    ScopedSchemaHandle handle(*schema);
    // Build an iterative-filter search plan with a boost filter scorer.
    auto plan_bytes = handle.ParseSearch("int64 >= 0",
                                         "fakevec",
                                         5,
                                         "L2",
                                         R"({"ef": 32})",
                                         /*round_decimal=*/-1,
                                         /*hints=*/"iterative_filter");
    proto::plan::PlanNode plan_proto;
    ParsePlanNodeProto(plan_proto, plan_bytes.data(), plan_bytes.size());
    plan_proto.mutable_plan_options()->set_enable_explain(true);
    plan_proto.mutable_vector_anns()->mutable_predicates()->set_expr_id(88001);

    auto plan = CreateSearchPlanFromPlanNode(schema, plan_proto);
    auto int64_fid = schema->get_field_id(FieldName("int64"));
    proto::plan::GenericValue boost_threshold;
    boost_threshold.set_int64_val(10);
    auto scorer_filter = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(int64_fid, DataType::INT64),
        proto::plan::OpType::GreaterEqual,
        boost_threshold,
        std::vector<proto::plan::GenericValue>());
    scorer_filter->SetExprId(88002);
    proto::plan::ScoreOption score_opt;
    score_opt.set_boost_mode(proto::plan::BoostModeMultiply);
    score_opt.set_function_mode(proto::plan::FunctionModeSum);
    std::vector<std::shared_ptr<rescores::Scorer>> scorers{
        std::make_shared<rescores::WeightScorer>(scorer_filter, 2.0f),
    };
    plan->plan_node_->plannodes_ = std::make_shared<plan::RescoresNode>(
        "rescore",
        scorers,
        score_opt,
        std::vector<plan::PlanNodePtr>{plan->plan_node_->plannodes_});

    auto ph_raw = CreatePlaceholderGroup(1, dim, 1024);
    auto ph = ParsePlaceholderGroup(plan.get(), ph_raw.SerializeAsString());

    auto res = segment->Search(plan.get(), ph.get(), MAX_TIMESTAMP);
    ASSERT_NE(res, nullptr);
    ASSERT_FALSE(res->explain_blob_.empty());

    proto::internal::ExecutionExplain seg_node;
    ASSERT_TRUE(seg_node.ParseFromString(res->explain_blob_));
    const auto& seg_group = seg_node.group();
    EXPECT_EQ(seg_group.segments(), 1);
    EXPECT_EQ(seg_group.sealed_segments(), 1);
    EXPECT_EQ(seg_group.growing_segments(), 0);
    // Expected pipeline: mvcc -> vector_search -> iterative_filter -> rescore
    ASSERT_EQ(seg_group.operators_size(), 4);
    EXPECT_EQ(seg_group.operators(0).kind(), proto::internal::EXPLAIN_OP_MVCC);
    EXPECT_EQ(seg_group.operators(1).kind(),
              proto::internal::EXPLAIN_OP_VECTOR_SEARCH);
    EXPECT_EQ(seg_group.operators(2).kind(),
              proto::internal::EXPLAIN_OP_ITERATIVE_FILTER);
    EXPECT_EQ(seg_group.operators(3).kind(),
              proto::internal::EXPLAIN_OP_RESCORE);

    const auto& vs_op = seg_group.operators(1);
    const auto& if_op = seg_group.operators(2);
    const auto& rs_op = seg_group.operators(3);
    EXPECT_EQ(vs_op.rows_in(), static_cast<int64_t>(N));
    EXPECT_GT(vs_op.rows_out(), 0);
    // CountingVectorIterator transfers exact yielded candidates to iterative_filter.
    EXPECT_EQ(if_op.rows_in(), vs_op.rows_out());
    EXPECT_EQ(if_op.rows_out(), 5);
    EXPECT_EQ(rs_op.rows_in(), 5);
    EXPECT_EQ(rs_op.rows_out(), 5);
    // Only iterative_filter's user predicate appears as a condition; the
    // internal boost-scorer filter expression is completely isolated.
    EXPECT_EQ(if_op.conditions_size(), 1);
    EXPECT_EQ(rs_op.conditions_size(), 0);
    for (int i = 0; i < seg_group.operators_size(); ++i) {
        const auto& op_pb = seg_group.operators(i);
        ASSERT_TRUE(op_pb.has_time());
        EXPECT_EQ(op_pb.time().count(), 1);
        EXPECT_GE(op_pb.time().sum_us(), 0);
        ASSERT_TRUE(op_pb.has_prefetch_wait_time());
        EXPECT_EQ(op_pb.prefetch_wait_time().count(), 1);
        EXPECT_LE(op_pb.prefetch_wait_time().sum_us(), op_pb.time().sum_us());
    }
    const auto& if_cond = if_op.conditions(0);
    ASSERT_TRUE(if_cond.has_eval_time());
    EXPECT_EQ(if_cond.eval_time().count(), 1);
    EXPECT_GE(if_cond.eval_time().sum_us(), 0);

    // Empty-segment Search (active_count = 0 on an empty growing segment) still
    // registers the full operator pipeline skeleton with 0 -> 0 counters.
    auto empty_segment = CreateGrowingSegment(schema, empty_index_meta);
    auto empty_res = empty_segment->Search(plan.get(), ph.get(), MAX_TIMESTAMP);
    ASSERT_NE(empty_res, nullptr);
    proto::internal::ExecutionExplain empty_node;
    ASSERT_TRUE(empty_node.ParseFromString(empty_res->explain_blob_));
    const auto& empty_group = empty_node.group();
    EXPECT_EQ(empty_group.rows_total(), 0);
    EXPECT_EQ(empty_group.growing_segments(), 1);
    EXPECT_EQ(empty_group.sealed_segments(), 0);
    ASSERT_EQ(empty_group.operators_size(), 4);
    EXPECT_EQ(empty_group.operators(0).kind(),
              proto::internal::EXPLAIN_OP_MVCC);
    EXPECT_EQ(empty_group.operators(1).rows_in(), 0);
    EXPECT_EQ(empty_group.operators(1).rows_out(), 0);
    ASSERT_TRUE(empty_group.operators(1).has_time());
    EXPECT_EQ(empty_group.operators(1).time().sum_us(), 0);
}

TEST(Rescorer, CountingVectorIteratorTransfersLazyTimeWithoutUnderflow) {
    class DelayedVectorIterator final : public VectorIterator {
     public:
        explicit DelayedVectorIterator(int count,
                                       std::chrono::milliseconds per_next_delay)
            : total_(count), delay_(per_next_delay) {
        }
        bool
        HasNext() override {
            return pos_ < total_;
        }
        std::optional<std::pair<int64_t, float>>
        Next() override {
            if (pos_ >= total_) {
                return std::nullopt;
            }
            std::this_thread::sleep_for(delay_);
            const int64_t off = pos_++;
            return std::make_pair(off, static_cast<float>(off));
        }

     private:
        int total_;
        int pos_{0};
        std::chrono::milliseconds delay_;
    };

    auto explain = std::make_shared<exec::ExecutionExplainRecorder>(100);
    explain->RegisterOperator(1, ExplainOperatorKind::kVectorSearch);
    explain->RegisterOperator(2, ExplainOperatorKind::kIterativeFilter);
    explain->SetOperatorRows(1, 100, 0);

    const auto op2_start = std::chrono::steady_clock::now();
    {
        auto iter = std::make_shared<exec::CountingVectorIterator>(
            std::make_shared<DelayedVectorIterator>(
                2, std::chrono::milliseconds(3)),
            explain,
            /*out_step=*/1,
            /*in_step=*/2);
        {
            exec::CountingVectorIterator::ScopedBatchTimer batch_timer(
                iter.get());
            while (iter->HasNext()) {
                ASSERT_TRUE(iter->Next().has_value());
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        // ~CountingVectorIterator runs here BEFORE Driver flushes op 2's wall
        // duration, exercising the signed temporary negative accumulator.
    }
    const auto op2_wall_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - op2_start)
            .count();
    explain->AddOperatorTimeNs(2, op2_wall_ns);

    proto::internal::ExecutionExplain seg_node;
    ASSERT_TRUE(seg_node.ParseFromString(explain->Finish()));
    const auto& seg_group = seg_node.group();
    ASSERT_EQ(seg_group.operators_size(), 2);
    const auto& vs_op = seg_group.operators(0);
    const auto& if_op = seg_group.operators(1);
    EXPECT_EQ(vs_op.rows_out(), 2);
    EXPECT_EQ(if_op.rows_in(), 2);
    EXPECT_GE(vs_op.time().sum_us(), 5000);
    EXPECT_GE(if_op.time().sum_us(), 2000);
    // Neither operator double-counts the full ~9ms wall window.
    const int64_t total_wall_us = (op2_wall_ns + 500) / 1000;
    EXPECT_LE(vs_op.time().sum_us() + if_op.time().sum_us(),
              total_wall_us + 200);
}
