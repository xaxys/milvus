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

#include <boost/format.hpp>
#include <boost/optional/optional.hpp>
#include <folly/FBVector.h>
#include <stddef.h>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ExprBatchTestUtils.h"
#include "ConsumerExprTestBase.h"
#include "NamedType/named_type_impl.hpp"
#include "bitset/bitset.h"
#include "common/Consts.h"
#include "common/EasyAssert.h"
#include "common/IndexMeta.h"
#include "common/Schema.h"
#include "common/Types.h"
#include "common/Vector.h"
#include "common/protobuf_utils.h"
#include "exec/expression/EvalCtx.h"
#include "exec/ExecutionExplainRecorder.h"
#include "exec/expression/function/FunctionFactory.h"
#include "expr/ITypeExpr.h"
#include "gtest/gtest.h"
#include "index/Meta.h"
#include "knowhere/comp/index_param.h"
#include "pb/internal.pb.h"
#include "pb/plan.pb.h"
#include "plan/PlanNode.h"
#include "query/ExecPlanNodeVisitor.h"
#include "query/Plan.h"
#include "query/PlanImpl.h"
#include "query/PlanNode.h"
#include "segcore/ChunkedSegmentSealedImpl.h"
#include "segcore/SegcoreConfig.h"
#include "segcore/SegmentGrowing.h"
#include "segcore/SegmentGrowingImpl.h"
#include "segcore/SegmentSealed.h"
#include "segcore/Types.h"
#include "test_utils/DataGen.h"
#include "segcore/test_utils/ConsumerIndexTestUtils.h"
#include "test_utils/GenExprProto.h"
#include "test_utils/cachinglayer_test_utils.h"
#include "test_utils/storage_test_utils.h"

EXPR_TEST_INSTANTIATE();

TEST_P(ExprTest, TestCall) {
    milvus::exec::expression::FunctionFactory& factory =
        milvus::exec::expression::FunctionFactory::Instance();
    factory.Initialize();

    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto varchar_fid = schema->AddDebugField("address", DataType::VARCHAR);
    schema->set_primary_field_id(varchar_fid);

    auto seg = CreateGrowingSegment(schema, empty_index_meta);
    int N = 1000;
    std::vector<std::string> address_col;
    int num_iters = 1;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto raw_data = DataGen(schema, N, iter);
        auto new_address_col = raw_data.get_col<std::string>(varchar_fid);
        address_col.insert(
            address_col.end(), new_address_col.begin(), new_address_col.end());
        seg->PreInsert(N);
        seg->Insert(iter * N,
                    N,
                    raw_data.row_ids_.data(),
                    raw_data.timestamps_.data(),
                    std::make_shared<InsertRecordProto>(*raw_data.raw_));
    }

    auto seg_promote = dynamic_cast<SegmentGrowingImpl*>(seg.get());
    SetSchema(schema);

    std::tuple<std::string, std::function<bool(std::string&)>> test_cases[] = {
        {"empty(address)", [](std::string& v) { return v.empty(); }},
        {"starts_with(address, address)", [](std::string&) { return true; }}};

    for (auto& [expr_str, ref_func] : test_cases) {
        auto plan_str = create_search_plan_from_expr(expr_str);
        auto plan =
            CreateSearchPlanByExpr(schema, plan_str.data(), plan_str.size());
        BitsetType final;
        final = ExecuteQueryExpr(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0],
            seg_promote,
            N * num_iters,
            MAX_TIMESTAMP);
        EXPECT_EQ(final.size(), N * num_iters);

        // specify some offsets and do scalar filtering on these offsets
        milvus::exec::OffsetVector offsets;
        offsets.reserve(N * num_iters / 2);
        for (auto i = 0; i < N * num_iters; ++i) {
            if (i % 2 == 0) {
                offsets.emplace_back(i);
            }
        }
        auto col_vec = milvus::test::gen_filter_res(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0].get(),
            seg_promote,
            N * num_iters,
            MAX_TIMESTAMP,
            &offsets);
        BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
        EXPECT_EQ(view.size(), N * num_iters / 2);

        for (int i = 0; i < N * num_iters; ++i) {
            auto ans = final[i];

            ASSERT_EQ(ans, ref_func(address_col[i]))
                << "@" << i << "!!" << address_col[i];
            if (i % 2 == 0) {
                ASSERT_EQ(view[int(i / 2)], ref_func(address_col[i]))
                    << "@" << i << "!!" << address_col[i];
            }
        }
    }

    // Test incorrect function calls (wrong number of arguments)
    std::string incorrect_test_cases[] = {
        "empty(address, address)",  // empty() takes 1 arg, not 2
        "starts_with(address)"      // starts_with() takes 2 args, not 1
    };
    for (auto& expr_str : incorrect_test_cases) {
        EXPECT_ANY_THROW({
            auto plan_str = create_search_plan_from_expr(expr_str);
            CreateSearchPlanByExpr(schema, plan_str.data(), plan_str.size());
        });
    }
}

TEST_P(ExprTest, TestCompare) {
    // Test cases: expression string and expected comparison function
    std::vector<std::tuple<std::string, std::function<bool(int, int64_t)>>>
        testcases = {
            {"age1 < age2", [](int a, int64_t b) { return a < b; }},
            {"age1 <= age2", [](int a, int64_t b) { return a <= b; }},
            {"age1 > age2", [](int a, int64_t b) { return a > b; }},
            {"age1 >= age2", [](int a, int64_t b) { return a >= b; }},
            {"age1 == age2", [](int a, int64_t b) { return a == b; }},
            {"age1 != age2", [](int a, int64_t b) { return a != b; }},
        };

    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto i32_fid = schema->AddDebugField("age1", DataType::INT32);
    auto i64_fid = schema->AddDebugField("age2", DataType::INT64);
    schema->set_primary_field_id(i64_fid);

    auto seg = CreateGrowingSegment(schema, empty_index_meta);
    int N = 1000;
    std::vector<int> age1_col;
    std::vector<int64_t> age2_col;
    int num_iters = 1;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto raw_data = DataGen(schema, N, iter);
        auto new_age1_col = raw_data.get_col<int>(i32_fid);
        auto new_age2_col = raw_data.get_col<int64_t>(i64_fid);
        age1_col.insert(
            age1_col.end(), new_age1_col.begin(), new_age1_col.end());
        age2_col.insert(
            age2_col.end(), new_age2_col.begin(), new_age2_col.end());
        seg->PreInsert(N);
        seg->Insert(iter * N,
                    N,
                    raw_data.row_ids_.data(),
                    raw_data.timestamps_.data(),
                    std::make_shared<InsertRecordProto>(*raw_data.raw_));
    }

    auto seg_promote = dynamic_cast<SegmentGrowingImpl*>(seg.get());
    query::ExecPlanNodeVisitor visitor(*seg_promote, MAX_TIMESTAMP);
    SetSchema(schema);
    for (auto [expr_str, ref_func] : testcases) {
        auto plan_str = create_search_plan_from_expr(expr_str);
        auto plan =
            CreateSearchPlanByExpr(schema, plan_str.data(), plan_str.size());
        BitsetType final;
        final = ExecuteQueryExpr(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0],
            seg_promote,
            N * num_iters,
            MAX_TIMESTAMP);
        EXPECT_EQ(final.size(), N * num_iters);

        // specify some offsets and do scalar filtering on these offsets
        milvus::exec::OffsetVector offsets;
        offsets.reserve(N * num_iters / 2);
        for (auto i = 0; i < N * num_iters; ++i) {
            if (i % 2 == 0) {
                offsets.emplace_back(i);
            }
        }
        auto col_vec = milvus::test::gen_filter_res(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0].get(),
            seg_promote,
            N * num_iters,
            MAX_TIMESTAMP,
            &offsets);
        BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
        EXPECT_EQ(view.size(), N * num_iters / 2);

        for (int i = 0; i < N * num_iters; ++i) {
            auto ans = final[i];

            auto val1 = age1_col[i];
            auto val2 = age2_col[i];
            auto ref = ref_func(val1, val2);
            ASSERT_EQ(ans, ref) << expr_str << "@" << i << "!!"
                                << boost::format("[%1%, %2%]") % val1 % val2;
            if (i % 2 == 0) {
                ASSERT_EQ(view[int(i / 2)], ref)
                    << expr_str << "@" << i << "!!"
                    << boost::format("[%1%, %2%]") % val1 % val2;
            }
        }
    }
}

TEST_P(ExprTest, TestCompareNullable) {
    // Test cases: expression string and expected comparison function
    // Compares age1 (non-nullable INT32) with nullable (nullable INT64)
    std::vector<
        std::tuple<std::string, std::function<bool(int, int64_t, bool)>>>
        testcases = {
            {"age1 < nullable",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a < b;
             }},
            {"age1 <= nullable",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a <= b;
             }},
            {"age1 > nullable",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a > b;
             }},
            {"age1 >= nullable",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a >= b;
             }},
            {"age1 == nullable",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a == b;
             }},
            {"age1 != nullable",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a != b;
             }},
        };

    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto i32_fid = schema->AddDebugField("age1", DataType::INT32);
    auto i64_fid = schema->AddDebugField("age2", DataType::INT64);
    auto nullable_fid =
        schema->AddDebugField("nullable", DataType::INT64, true);
    schema->set_primary_field_id(i64_fid);

    auto seg = CreateGrowingSegment(schema, empty_index_meta);
    int N = 1000;
    std::vector<int> age1_col;
    std::vector<int64_t> nullable_col;
    FixedVector<bool> valid_data_col;
    int num_iters = 1;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto raw_data = DataGen(schema, N, iter);
        auto new_age1_col = raw_data.get_col<int>(i32_fid);
        auto new_nullable_col = raw_data.get_col<int64_t>(nullable_fid);
        valid_data_col = raw_data.get_col_valid(nullable_fid);
        age1_col.insert(
            age1_col.end(), new_age1_col.begin(), new_age1_col.end());
        nullable_col.insert(nullable_col.end(),
                            new_nullable_col.begin(),
                            new_nullable_col.end());
        seg->PreInsert(N);
        seg->Insert(iter * N,
                    N,
                    raw_data.row_ids_.data(),
                    raw_data.timestamps_.data(),
                    std::make_shared<InsertRecordProto>(*raw_data.raw_));
    }

    auto seg_promote = dynamic_cast<SegmentGrowingImpl*>(seg.get());
    query::ExecPlanNodeVisitor visitor(*seg_promote, MAX_TIMESTAMP);
    SetSchema(schema);
    for (auto [expr_str, ref_func] : testcases) {
        auto plan_str = create_search_plan_from_expr(expr_str);
        auto plan =
            CreateSearchPlanByExpr(schema, plan_str.data(), plan_str.size());
        BitsetType final;
        final = ExecuteQueryExpr(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0],
            seg_promote,
            N * num_iters,
            MAX_TIMESTAMP);
        EXPECT_EQ(final.size(), N * num_iters);

        // specify some offsets and do scalar filtering on these offsets
        milvus::exec::OffsetVector offsets;
        offsets.reserve(N * num_iters / 2);
        for (auto i = 0; i < N * num_iters; ++i) {
            if (i % 2 == 0) {
                offsets.emplace_back(i);
            }
        }
        auto col_vec = milvus::test::gen_filter_res(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0].get(),
            seg_promote,
            N * num_iters,
            MAX_TIMESTAMP,
            &offsets);
        BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
        EXPECT_EQ(view.size(), N * num_iters / 2);

        for (int i = 0; i < N * num_iters; ++i) {
            auto ans = final[i];

            auto val1 = age1_col[i];
            auto val2 = nullable_col[i];
            auto ref = ref_func(val1, val2, valid_data_col[i]);
            ASSERT_EQ(ans, ref) << expr_str << "@" << i << "!!"
                                << boost::format("[%1%, %2%]") % val1 % val2;
            if (i % 2 == 0) {
                ASSERT_EQ(view[int(i / 2)], ref)
                    << expr_str << "@" << i << "!!"
                    << boost::format("[%1%, %2%]") % val1 % val2;
            }
        }
    }
}

TEST_P(ExprTest, TestCompareNullable2) {
    // Test cases: expression string and expected comparison function
    // Compares nullable (nullable INT64) with age1 (non-nullable INT32) - swapped operand order
    std::vector<
        std::tuple<std::string, std::function<bool(int, int64_t, bool)>>>
        testcases = {
            {"nullable < age1",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a < b;
             }},
            {"nullable <= age1",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a <= b;
             }},
            {"nullable > age1",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a > b;
             }},
            {"nullable >= age1",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a >= b;
             }},
            {"nullable == age1",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a == b;
             }},
            {"nullable != age1",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a != b;
             }},
        };

    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto i32_fid = schema->AddDebugField("age1", DataType::INT32);
    auto i64_fid = schema->AddDebugField("age2", DataType::INT64);
    auto nullable_fid =
        schema->AddDebugField("nullable", DataType::INT64, true);
    schema->set_primary_field_id(i64_fid);

    auto seg = CreateGrowingSegment(schema, empty_index_meta);
    int N = 1000;
    std::vector<int> age1_col;
    std::vector<int64_t> nullable_col;
    FixedVector<bool> valid_data_col;
    int num_iters = 1;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto raw_data = DataGen(schema, N, iter);
        auto new_age1_col = raw_data.get_col<int>(i32_fid);
        auto new_nullable_col = raw_data.get_col<int64_t>(nullable_fid);
        valid_data_col = raw_data.get_col_valid(nullable_fid);
        age1_col.insert(
            age1_col.end(), new_age1_col.begin(), new_age1_col.end());
        nullable_col.insert(nullable_col.end(),
                            new_nullable_col.begin(),
                            new_nullable_col.end());
        seg->PreInsert(N);
        seg->Insert(iter * N,
                    N,
                    raw_data.row_ids_.data(),
                    raw_data.timestamps_.data(),
                    std::make_shared<InsertRecordProto>(*raw_data.raw_));
    }

    auto seg_promote = dynamic_cast<SegmentGrowingImpl*>(seg.get());
    query::ExecPlanNodeVisitor visitor(*seg_promote, MAX_TIMESTAMP);
    SetSchema(schema);
    for (auto [expr_str, ref_func] : testcases) {
        auto plan_str = create_search_plan_from_expr(expr_str);
        auto plan =
            CreateSearchPlanByExpr(schema, plan_str.data(), plan_str.size());
        BitsetType final;
        final = ExecuteQueryExpr(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0],
            seg_promote,
            N * num_iters,
            MAX_TIMESTAMP);
        EXPECT_EQ(final.size(), N * num_iters);

        // specify some offsets and do scalar filtering on these offsets
        milvus::exec::OffsetVector offsets;
        offsets.reserve(N * num_iters / 2);
        for (auto i = 0; i < N * num_iters; ++i) {
            if (i % 2 == 0) {
                offsets.emplace_back(i);
            }
        }
        auto col_vec = milvus::test::gen_filter_res(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0].get(),
            seg_promote,
            N * num_iters,
            MAX_TIMESTAMP,
            &offsets);
        BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
        EXPECT_EQ(view.size(), N * num_iters / 2);

        for (int i = 0; i < N * num_iters; ++i) {
            auto ans = final[i];

            auto val2 = age1_col[i];
            auto val1 = nullable_col[i];
            auto ref = ref_func(val1, val2, valid_data_col[i]);
            ASSERT_EQ(ans, ref) << expr_str << "@" << i << "!!"
                                << boost::format("[%1%, %2%]") % val1 % val2;
            if (i % 2 == 0) {
                ASSERT_EQ(view[int(i / 2)], ref)
                    << expr_str << "@" << i << "!!"
                    << boost::format("[%1%, %2%]") % val1 % val2;
            }
        }
    }
}

TEST_P(ExprTest, TestCompareWithScalarIndex) {
    std::vector<std::tuple<std::string, std::function<bool(int, int64_t)>>>
        testcases = {
            {"age32 < age64", [](int a, int64_t b) { return a < b; }},
            {"age32 <= age64", [](int a, int64_t b) { return a <= b; }},
            {"age32 > age64", [](int a, int64_t b) { return a > b; }},
            {"age32 >= age64", [](int a, int64_t b) { return a >= b; }},
            {"age32 == age64", [](int a, int64_t b) { return a == b; }},
            {"age32 != age64", [](int a, int64_t b) { return a != b; }},
        };

    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto i32_fid = schema->AddDebugField("age32", DataType::INT32);
    auto i64_fid = schema->AddDebugField("age64", DataType::INT64);
    schema->set_primary_field_id(i64_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);

    // load index for int32 field
    auto age32_col = raw_data.get_col<int32_t>(i32_fid);
    age32_col[0] = 1000;
    test::expr_index::InstallIndex(
        *seg,
        i32_fid,
        DataType::INT32,
        test::consumer::BuildScalarReader<int32_t>(i32_fid,
                                                   DataType::INT32,
                                                   index::ASCENDING_SORT,
                                                   N,
                                                   age32_col.data()));

    // load index for int64 field
    auto age64_col = raw_data.get_col<int64_t>(i64_fid);
    age64_col[0] = 2000;
    test::expr_index::InstallIndex(
        *seg,
        i64_fid,
        DataType::INT64,
        test::consumer::BuildScalarReader<int64_t>(i64_fid,
                                                   DataType::INT64,
                                                   index::ASCENDING_SORT,
                                                   N,
                                                   age64_col.data()));

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
    SetSchema(schema);
    for (auto [expr_str, ref_func] : testcases) {
        auto plan_str = create_search_plan_from_expr(expr_str);
        auto plan =
            CreateSearchPlanByExpr(schema, plan_str.data(), plan_str.size());
        BitsetType final;
        final = ExecuteQueryExpr(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0],
            seg.get(),
            N,
            MAX_TIMESTAMP);
        EXPECT_EQ(final.size(), N);

        // specify some offsets and do scalar filtering on these offsets
        milvus::exec::OffsetVector offsets;
        offsets.reserve(N / 2);
        for (auto i = 0; i < N; ++i) {
            if (i % 2 == 0) {
                offsets.emplace_back(i);
            }
        }
        auto col_vec = milvus::test::gen_filter_res(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0].get(),
            seg.get(),
            N,
            MAX_TIMESTAMP,
            &offsets);
        BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
        EXPECT_EQ(view.size(), N / 2);

        for (int i = 0; i < N; ++i) {
            auto ans = final[i];
            auto val1 = age32_col[i];
            auto val2 = age64_col[i];
            auto ref = ref_func(val1, val2);
            ASSERT_EQ(ans, ref) << expr_str << "@" << i << "!!"
                                << boost::format("[%1%, %2%]") % val1 % val2;
            if (i % 2 == 0) {
                ASSERT_EQ(view[int(i / 2)], ref)
                    << expr_str << "@" << i << "!!"
                    << boost::format("[%1%, %2%]") % val1 % val2;
            }
        }
    }
}

TEST_P(ExprTest, TestCompareWithScalarIndexNullable) {
    std::vector<
        std::tuple<std::string, std::function<bool(int, int64_t, bool)>>>
        testcases = {
            {"nullable < age64",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a < b;
             }},
            {"nullable <= age64",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a <= b;
             }},
            {"nullable > age64",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a > b;
             }},
            {"nullable >= age64",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a >= b;
             }},
            {"nullable == age64",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a == b;
             }},
            {"nullable != age64",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a != b;
             }},
        };

    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto nullable_fid =
        schema->AddDebugField("nullable", DataType::INT32, true);
    auto i64_fid = schema->AddDebugField("age64", DataType::INT64);
    schema->set_primary_field_id(i64_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);

    // load index for int32 field
    auto nullable_col = raw_data.get_col<int32_t>(nullable_fid);
    nullable_col[0] = 1000;
    auto valid_data_col = raw_data.get_col_valid(nullable_fid);
    test::expr_index::InstallIndex(
        *seg,
        nullable_fid,
        DataType::INT32,
        test::consumer::BuildScalarReader<int32_t>(nullable_fid,
                                                   DataType::INT32,
                                                   index::ASCENDING_SORT,
                                                   N,
                                                   nullable_col.data(),
                                                   valid_data_col.data()));

    // load index for int64 field
    auto age64_col = raw_data.get_col<int64_t>(i64_fid);
    age64_col[0] = 2000;
    test::expr_index::InstallIndex(
        *seg,
        i64_fid,
        DataType::INT64,
        test::consumer::BuildScalarReader<int64_t>(i64_fid,
                                                   DataType::INT64,
                                                   index::ASCENDING_SORT,
                                                   N,
                                                   age64_col.data()));

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
    SetSchema(schema);
    for (auto [expr_str, ref_func] : testcases) {
        auto plan_str = create_search_plan_from_expr(expr_str);
        auto plan =
            CreateSearchPlanByExpr(schema, plan_str.data(), plan_str.size());
        BitsetType final;
        final = ExecuteQueryExpr(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0],
            seg.get(),
            N,
            MAX_TIMESTAMP);
        EXPECT_EQ(final.size(), N);

        // specify some offsets and do scalar filtering on these offsets
        milvus::exec::OffsetVector offsets;
        offsets.reserve(N / 2);
        for (auto i = 0; i < N; ++i) {
            if (i % 2 == 0) {
                offsets.emplace_back(i);
            }
        }
        auto col_vec = milvus::test::gen_filter_res(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0].get(),
            seg.get(),
            N,
            MAX_TIMESTAMP,
            &offsets);
        BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
        EXPECT_EQ(view.size(), N / 2);

        for (int i = 0; i < N; ++i) {
            auto ans = final[i];
            auto val1 = nullable_col[i];
            auto val2 = age64_col[i];
            auto ref = ref_func(val1, val2, valid_data_col[i]);
            ASSERT_EQ(ans, ref) << expr_str << "@" << i << "!!"
                                << boost::format("[%1%, %2%]") % val1 % val2;
            if (i % 2 == 0) {
                ASSERT_EQ(view[int(i / 2)], ref)
                    << expr_str << "@" << i << "!!"
                    << boost::format("[%1%, %2%]") % val1 % val2;
            }
        }
    }
}

TEST_P(ExprTest, TestCompareWithScalarIndexNullable2) {
    std::vector<
        std::tuple<std::string, std::function<bool(int, int64_t, bool)>>>
        testcases = {
            {"age64 < nullable",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a < b;
             }},
            {"age64 <= nullable",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a <= b;
             }},
            {"age64 > nullable",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a > b;
             }},
            {"age64 >= nullable",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a >= b;
             }},
            {"age64 == nullable",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a == b;
             }},
            {"age64 != nullable",
             [](int a, int64_t b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a != b;
             }},
        };

    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto nullable_fid =
        schema->AddDebugField("nullable", DataType::INT32, true);
    auto i64_fid = schema->AddDebugField("age64", DataType::INT64);
    schema->set_primary_field_id(i64_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);

    // load index for int32 field
    auto nullable_col = raw_data.get_col<int32_t>(nullable_fid);
    nullable_col[0] = 1000;
    auto valid_data_col = raw_data.get_col_valid(nullable_fid);
    test::expr_index::InstallIndex(
        *seg,
        nullable_fid,
        DataType::INT32,
        test::consumer::BuildScalarReader<int32_t>(nullable_fid,
                                                   DataType::INT32,
                                                   index::ASCENDING_SORT,
                                                   N,
                                                   nullable_col.data(),
                                                   valid_data_col.data()));

    // load index for int64 field
    auto age64_col = raw_data.get_col<int64_t>(i64_fid);
    age64_col[0] = 2000;
    test::expr_index::InstallIndex(
        *seg,
        i64_fid,
        DataType::INT64,
        test::consumer::BuildScalarReader<int64_t>(i64_fid,
                                                   DataType::INT64,
                                                   index::ASCENDING_SORT,
                                                   N,
                                                   age64_col.data()));

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
    SetSchema(schema);
    for (auto [expr_str, ref_func] : testcases) {
        auto plan_str = create_search_plan_from_expr(expr_str);
        auto plan =
            CreateSearchPlanByExpr(schema, plan_str.data(), plan_str.size());
        BitsetType final;
        final = ExecuteQueryExpr(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0],
            seg.get(),
            N,
            MAX_TIMESTAMP);
        EXPECT_EQ(final.size(), N);

        // specify some offsets and do scalar filtering on these offsets
        milvus::exec::OffsetVector offsets;
        offsets.reserve(N / 2);
        for (auto i = 0; i < N; ++i) {
            if (i % 2 == 0) {
                offsets.emplace_back(i);
            }
        }
        auto col_vec = milvus::test::gen_filter_res(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0].get(),
            seg.get(),
            N,
            MAX_TIMESTAMP,
            &offsets);
        BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
        EXPECT_EQ(view.size(), N / 2);

        for (int i = 0; i < N; ++i) {
            auto ans = final[i];
            auto val2 = nullable_col[i];
            auto val1 = age64_col[i];
            auto ref = ref_func(val1, val2, valid_data_col[i]);
            ASSERT_EQ(ans, ref) << expr_str << "@" << i << "!!"
                                << boost::format("[%1%, %2%]") % val1 % val2;
            if (i % 2 == 0) {
                ASSERT_EQ(view[int(i / 2)], ref)
                    << expr_str << "@" << i << "!!"
                    << boost::format("[%1%, %2%]") % val1 % val2;
            }
        }
    }
}

TEST_P(ExprTest, test_term_pk_with_sorted) {
    auto schema = std::make_shared<Schema>();
    schema->AddField(FieldName("Timestamp"),
                     FieldId(1),
                     DataType::INT64,
                     false,
                     std::nullopt);
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    schema->AddDebugField("string1", DataType::VARCHAR);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    schema->set_primary_field_id(int64_fid);

    auto seg = CreateSealedSegment(
        schema, nullptr, 1, SegcoreConfig::default_config(), true);
    int N = 100000;
    auto raw_data = DataGen(schema, N);
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    std::vector<proto::plan::GenericValue> retrieve_ints;
    for (int i = 0; i < 10; ++i) {
        proto::plan::GenericValue val;
        val.set_int64_val(i);
        retrieve_ints.push_back(val);
    }
    auto expr = std::make_shared<expr::TermFilterExpr>(
        expr::ColumnInfo(int64_fid, DataType::INT64), retrieve_ints);
    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
    BitsetType final;
    auto plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    EXPECT_EQ(final.size(), N);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(final[i], true);
    }
    for (int i = 10; i < N; ++i) {
        EXPECT_EQ(final[i], false);
    }
    retrieve_ints.clear();
    for (int i = 0; i < 10; ++i) {
        proto::plan::GenericValue val;
        val.set_int64_val(i + N);
        retrieve_ints.push_back(val);
    }
    expr = std::make_shared<expr::TermFilterExpr>(
        expr::ColumnInfo(int64_fid, DataType::INT64), retrieve_ints);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    EXPECT_EQ(final.size(), N);

    // specify some offsets and do scalar filtering on these offsets
    milvus::exec::OffsetVector offsets;
    offsets.reserve(N / 2);
    for (auto i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            offsets.emplace_back(i);
        }
    }
    auto col_vec = milvus::test::gen_filter_res(
        plan.get(), seg.get(), N, MAX_TIMESTAMP, &offsets);
    BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
    EXPECT_EQ(view.size(), N / 2);

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(final[i], false);
        if (i % 2 == 0) {
            EXPECT_EQ(view[int(i / 2)], false);
        }
    }
}

TEST_P(ExprTest, TestSealedSegmentGetBatchSize) {
    milvus::test::ExprBatchSizeGuard batch_size_guard(7);
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto bool_fid = schema->AddDebugField("bool", DataType::BOOL);
    auto bool_1_fid = schema->AddDebugField("bool1", DataType::BOOL);
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8);
    auto int8_1_fid = schema->AddDebugField("int81", DataType::INT8);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16);
    auto int16_1_fid = schema->AddDebugField("int161", DataType::INT16);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32);
    auto int32_1_fid = schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    auto int64_1_fid = schema->AddDebugField("int641", DataType::INT64);
    auto float_fid = schema->AddDebugField("float", DataType::FLOAT);
    auto float_1_fid = schema->AddDebugField("float1", DataType::FLOAT);
    auto double_fid = schema->AddDebugField("double", DataType::DOUBLE);
    auto double_1_fid = schema->AddDebugField("double1", DataType::DOUBLE);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    auto str2_fid = schema->AddDebugField("string2", DataType::VARCHAR);
    auto str3_fid = schema->AddDebugField("string3", DataType::VARCHAR);
    auto string_fid = schema->AddDebugField("string", DataType::STRING);
    auto string_1_fid =
        schema->AddDebugField("string1_internal", DataType::STRING);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    size_t N = 1000;
    auto raw_data = DataGen(schema, N);
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
    auto build_expr = [&](enum DataType type) -> expr::TypedExprPtr {
        switch (type) {
            case DataType::BOOL: {
                auto compare_expr = std::make_shared<expr::CompareExpr>(
                    bool_fid,
                    bool_1_fid,
                    DataType::BOOL,
                    DataType::BOOL,
                    proto::plan::OpType::LessThan);
                return compare_expr;
            }
            case DataType::INT8: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(int8_fid,
                                                        int8_1_fid,
                                                        DataType::INT8,
                                                        DataType::INT8,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::INT16: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(int16_fid,
                                                        int16_1_fid,
                                                        DataType::INT16,
                                                        DataType::INT16,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::INT32: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(int32_fid,
                                                        int32_1_fid,
                                                        DataType::INT32,
                                                        DataType::INT32,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::INT64: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(int64_fid,
                                                        int64_1_fid,
                                                        DataType::INT64,
                                                        DataType::INT64,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::FLOAT: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(float_fid,
                                                        float_1_fid,
                                                        DataType::FLOAT,
                                                        DataType::FLOAT,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::DOUBLE: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(double_fid,
                                                        double_1_fid,
                                                        DataType::DOUBLE,
                                                        DataType::DOUBLE,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::VARCHAR: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(str2_fid,
                                                        str3_fid,
                                                        DataType::VARCHAR,
                                                        DataType::VARCHAR,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::STRING: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(string_fid,
                                                        string_1_fid,
                                                        DataType::STRING,
                                                        DataType::STRING,
                                                        OpType::LessThan);
                return compare_expr;
            }
            default:
                return std::make_shared<expr::CompareExpr>(int8_fid,
                                                           int8_1_fid,
                                                           DataType::INT8,
                                                           DataType::INT8,
                                                           OpType::LessThan);
        }
    };
    auto execute = [&](DataType type) {
        auto expr = build_expr(type);
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        return ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    };

    auto check = [&]<typename T>(DataType type, FieldId left, FieldId right) {
        auto result = execute(type);
        auto left_values = raw_data.get_col<T>(left);
        auto right_values = raw_data.get_col<T>(right);
        ASSERT_EQ(result.size(), N);
        for (size_t i = 0; i < N; ++i) {
            ASSERT_EQ(result[i], left_values[i] < right_values[i])
                << "type " << static_cast<int>(type) << " row " << i;
        }
    };

    check.operator()<bool>(DataType::BOOL, bool_fid, bool_1_fid);
    check.operator()<int8_t>(DataType::INT8, int8_fid, int8_1_fid);
    check.operator()<int16_t>(DataType::INT16, int16_fid, int16_1_fid);
    check.operator()<int32_t>(DataType::INT32, int32_fid, int32_1_fid);
    check.operator()<int64_t>(DataType::INT64, int64_fid, int64_1_fid);
    check.operator()<float>(DataType::FLOAT, float_fid, float_1_fid);
    check.operator()<double>(DataType::DOUBLE, double_fid, double_1_fid);
    check.operator()<std::string>(DataType::VARCHAR, str2_fid, str3_fid);
    check.operator()<std::string>(DataType::STRING, string_fid, string_1_fid);

    auto int64_expr = build_expr(DataType::INT64);
    auto int64_plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, int64_expr);
    milvus::exec::OffsetVector offsets{9, 2, 9, 511, 3, 999};
    auto offset_result = milvus::test::gen_filter_res(
        int64_plan.get(), seg.get(), N, MAX_TIMESTAMP, &offsets);
    BitsetTypeView offset_view(offset_result->GetRawData(),
                               offset_result->size());
    auto left_values = raw_data.get_col<int64_t>(int64_fid);
    auto right_values = raw_data.get_col<int64_t>(int64_1_fid);
    ASSERT_EQ(offset_view.size(), offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i) {
        const auto offset = offsets[i];
        ASSERT_EQ(offset_view[i], left_values[offset] < right_values[offset])
            << "offset input position " << i << " row " << offset;
    }

    auto varchar_expr = build_expr(DataType::VARCHAR);
    auto varchar_plan = std::make_shared<plan::FilterBitsNode>(
        DEFAULT_PLANNODE_ID, varchar_expr);
    auto varchar_offset_result = milvus::test::gen_filter_res(
        varchar_plan.get(), seg.get(), N, MAX_TIMESTAMP, &offsets);
    BitsetTypeView varchar_offset_view(varchar_offset_result->GetRawData(),
                                       varchar_offset_result->size());
    auto left_strings = raw_data.get_col<std::string>(str2_fid);
    auto right_strings = raw_data.get_col<std::string>(str3_fid);
    ASSERT_EQ(varchar_offset_view.size(), offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i) {
        const auto offset = offsets[i];
        ASSERT_EQ(varchar_offset_view[i],
                  left_strings[offset] < right_strings[offset])
            << "varchar offset input position " << i << " row " << offset;
    }
}

TEST_P(ExprTest, TestReorder) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("id", DataType::INT64);
    schema->AddDebugField("bool", DataType::BOOL);
    schema->AddDebugField("bool1", DataType::BOOL);
    schema->AddDebugField("int8", DataType::INT8);
    schema->AddDebugField("int81", DataType::INT8);
    schema->AddDebugField("int16", DataType::INT16);
    schema->AddDebugField("int161", DataType::INT16);
    schema->AddDebugField("int32", DataType::INT32);
    schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    auto int64_1_fid = schema->AddDebugField("int641", DataType::INT64);
    schema->AddDebugField("float", DataType::FLOAT);
    schema->AddDebugField("float1", DataType::FLOAT);
    schema->AddDebugField("double", DataType::DOUBLE);
    schema->AddDebugField("double1", DataType::DOUBLE);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    schema->AddDebugField("string2", DataType::VARCHAR);
    auto json_fid = schema->AddDebugField("json", DataType::JSON, false);
    auto str_array_fid =
        schema->AddDebugField("str_array", DataType::ARRAY, DataType::VARCHAR);
    schema->set_primary_field_id(pk);

    auto seg = CreateSealedSegment(schema);
    size_t N = 1000;
    auto raw_data = DataGen(schema, N);
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);

    auto build_expr = [&](int index) -> expr::TypedExprPtr {
        switch (index) {
            case 0: {
                proto::plan::GenericValue val1;
                val1.set_string_val("xxx");
                auto expr1 = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(str1_fid, DataType::VARCHAR),
                    proto::plan::OpType::Equal,
                    val1,
                    std::vector<proto::plan::GenericValue>{});
                proto::plan::GenericValue val2;
                val2.set_int64_val(100);
                auto expr2 = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(int64_fid, DataType::INT64),
                    proto::plan::OpType::LessThan,
                    val2,
                    std::vector<proto::plan::GenericValue>{});
                auto expr3 = std::make_shared<expr::LogicalBinaryExpr>(
                    expr::LogicalBinaryExpr::OpType::And, expr1, expr2);
                return expr3;
            };
            case 1: {
                proto::plan::GenericValue val1;
                val1.set_string_val("xxx");
                auto expr1 = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(json_fid, DataType::JSON, {"int"}),
                    proto::plan::OpType::Equal,
                    val1,
                    std::vector<proto::plan::GenericValue>{});
                proto::plan::GenericValue val2;
                val2.set_int64_val(100);
                auto expr2 = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(int64_fid, DataType::INT64),
                    proto::plan::OpType::LessThan,
                    val2,
                    std::vector<proto::plan::GenericValue>{});
                auto expr3 = std::make_shared<expr::LogicalBinaryExpr>(
                    expr::LogicalBinaryExpr::OpType::And, expr1, expr2);
                return expr3;
            };
            case 2: {
                proto::plan::GenericValue val1;
                val1.set_string_val("12");
                auto expr1 = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(str_array_fid, DataType::ARRAY, {"0"}),
                    proto::plan::OpType::Match,
                    val1,
                    std::vector<proto::plan::GenericValue>{});
                proto::plan::GenericValue val2;
                val2.set_int64_val(100);
                auto expr2 = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(int64_fid, DataType::INT64),
                    proto::plan::OpType::LessThan,
                    val2,
                    std::vector<proto::plan::GenericValue>{});
                auto expr3 = std::make_shared<expr::LogicalBinaryExpr>(
                    expr::LogicalBinaryExpr::OpType::And, expr1, expr2);
                return expr3;
            };
            case 3: {
                auto expr1 =
                    std::make_shared<expr::CompareExpr>(int64_fid,
                                                        int64_1_fid,
                                                        DataType::INT64,
                                                        DataType::INT64,
                                                        OpType::LessThan);
                proto::plan::GenericValue val2;
                val2.set_int64_val(100);
                auto expr2 = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(int64_fid, DataType::INT64),
                    proto::plan::OpType::LessThan,
                    val2,
                    std::vector<proto::plan::GenericValue>{});
                auto expr3 = std::make_shared<expr::LogicalBinaryExpr>(
                    expr::LogicalBinaryExpr::OpType::And, expr1, expr2);
                return expr3;
            };
            default:
                ThrowInfo(ErrorCode::UnexpectedError, "not implement");
        }
    };
    BitsetType final;
    auto expr = build_expr(0);
    auto plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(1);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(2);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(3);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
}

TEST_P(ExprTest, TestCompareExprNullable) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto bool_fid = schema->AddDebugField("bool", DataType::BOOL);
    auto bool_nullable_fid =
        schema->AddDebugField("bool1", DataType::BOOL, true);
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8);
    auto int8_nullable_fid =
        schema->AddDebugField("int81", DataType::INT8, true);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16);
    auto int16_nullable_fid =
        schema->AddDebugField("int161", DataType::INT16, true);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32);
    auto int32_nullable_fid =
        schema->AddDebugField("int321", DataType::INT32, true);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    auto int64_nullable_fid =
        schema->AddDebugField("int641", DataType::INT64, true);
    auto float_fid = schema->AddDebugField("float", DataType::FLOAT);
    auto float_nullable_fid =
        schema->AddDebugField("float1", DataType::FLOAT, true);
    auto double_fid = schema->AddDebugField("double", DataType::DOUBLE);
    auto double_nullable_fid =
        schema->AddDebugField("double1", DataType::DOUBLE, true);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    auto str2_fid = schema->AddDebugField("string2", DataType::VARCHAR);
    auto str_nullable_fid =
        schema->AddDebugField("string3", DataType::VARCHAR, true);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    size_t N = 1000;
    auto raw_data = DataGen(schema, N);
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
    auto build_expr = [&](enum DataType type) -> expr::TypedExprPtr {
        switch (type) {
            case DataType::BOOL: {
                auto compare_expr = std::make_shared<expr::CompareExpr>(
                    bool_fid,
                    bool_nullable_fid,
                    DataType::BOOL,
                    DataType::BOOL,
                    proto::plan::OpType::LessThan);
                return compare_expr;
            }
            case DataType::INT8: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(int8_fid,
                                                        int8_nullable_fid,
                                                        DataType::INT8,
                                                        DataType::INT8,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::INT16: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(int16_fid,
                                                        int16_nullable_fid,
                                                        DataType::INT16,
                                                        DataType::INT16,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::INT32: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(int32_fid,
                                                        int32_nullable_fid,
                                                        DataType::INT32,
                                                        DataType::INT32,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::INT64: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(int64_fid,
                                                        int64_nullable_fid,
                                                        DataType::INT64,
                                                        DataType::INT64,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::FLOAT: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(float_fid,
                                                        float_nullable_fid,
                                                        DataType::FLOAT,
                                                        DataType::FLOAT,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::DOUBLE: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(double_fid,
                                                        double_nullable_fid,
                                                        DataType::DOUBLE,
                                                        DataType::DOUBLE,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::VARCHAR: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(str2_fid,
                                                        str_nullable_fid,
                                                        DataType::VARCHAR,
                                                        DataType::VARCHAR,
                                                        OpType::LessThan);
                return compare_expr;
            }
            default:
                return std::make_shared<expr::CompareExpr>(int8_fid,
                                                           int8_nullable_fid,
                                                           DataType::INT8,
                                                           DataType::INT8,
                                                           OpType::LessThan);
        }
    };
    auto expr = build_expr(DataType::BOOL);
    BitsetType final;
    auto plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(DataType::INT8);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(DataType::INT16);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(DataType::INT32);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(DataType::INT64);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(DataType::FLOAT);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(DataType::DOUBLE);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
}

TEST_P(ExprTest, TestCompareExprNullable2) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto bool_fid = schema->AddDebugField("bool", DataType::BOOL);
    auto bool_nullable_fid =
        schema->AddDebugField("bool1", DataType::BOOL, true);
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8);
    auto int8_nullable_fid =
        schema->AddDebugField("int81", DataType::INT8, true);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16);
    auto int16_nullable_fid =
        schema->AddDebugField("int161", DataType::INT16, true);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32);
    auto int32_nullable_fid =
        schema->AddDebugField("int321", DataType::INT32, true);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    auto int64_nullable_fid =
        schema->AddDebugField("int641", DataType::INT64, true);
    auto float_fid = schema->AddDebugField("float", DataType::FLOAT);
    auto float_nullable_fid =
        schema->AddDebugField("float1", DataType::FLOAT, true);
    auto double_fid = schema->AddDebugField("double", DataType::DOUBLE);
    auto double_nullable_fid =
        schema->AddDebugField("double1", DataType::DOUBLE, true);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    auto str2_fid = schema->AddDebugField("string2", DataType::VARCHAR);
    auto str_nullable_fid =
        schema->AddDebugField("string3", DataType::VARCHAR, true);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    size_t N = 1000;
    auto raw_data = DataGen(schema, N);
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
    auto build_expr = [&](enum DataType type) -> expr::TypedExprPtr {
        switch (type) {
            case DataType::BOOL: {
                auto compare_expr = std::make_shared<expr::CompareExpr>(
                    bool_fid,
                    bool_nullable_fid,
                    DataType::BOOL,
                    DataType::BOOL,
                    proto::plan::OpType::LessThan);
                return compare_expr;
            }
            case DataType::INT8: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(int8_nullable_fid,
                                                        int8_fid,
                                                        DataType::INT8,
                                                        DataType::INT8,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::INT16: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(int16_nullable_fid,
                                                        int16_fid,
                                                        DataType::INT16,
                                                        DataType::INT16,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::INT32: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(int32_nullable_fid,
                                                        int32_fid,
                                                        DataType::INT32,
                                                        DataType::INT32,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::INT64: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(int64_nullable_fid,
                                                        int64_fid,
                                                        DataType::INT64,
                                                        DataType::INT64,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::FLOAT: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(float_nullable_fid,
                                                        float_fid,
                                                        DataType::FLOAT,
                                                        DataType::FLOAT,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::DOUBLE: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(double_nullable_fid,
                                                        double_fid,
                                                        DataType::DOUBLE,
                                                        DataType::DOUBLE,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case DataType::VARCHAR: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(str_nullable_fid,
                                                        str2_fid,
                                                        DataType::VARCHAR,
                                                        DataType::VARCHAR,
                                                        OpType::LessThan);
                return compare_expr;
            }
            default:
                return std::make_shared<expr::CompareExpr>(int8_nullable_fid,
                                                           int8_fid,
                                                           DataType::INT8,
                                                           DataType::INT8,
                                                           OpType::LessThan);
        }
    };
    auto expr = build_expr(DataType::BOOL);
    BitsetType final;
    auto plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(DataType::INT8);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(DataType::INT16);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(DataType::INT32);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(DataType::INT64);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(DataType::FLOAT);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    expr = build_expr(DataType::DOUBLE);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
}

TEST_P(ExprTest, TestCompareNullableLeftWithOffsetInputSealed) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto pk_fid = schema->AddDebugField("age64", DataType::INT64);
    auto nullable_fid =
        schema->AddDebugField("nullable", DataType::INT64, true);
    schema->set_primary_field_id(pk_fid);

    auto seg = CreateSealedSegment(schema);
    const int N = 1000;
    auto raw_data = DataGen(schema, N);
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    auto nullable_col = raw_data.get_col<int64_t>(nullable_fid);
    auto pk_col = raw_data.get_col<int64_t>(pk_fid);
    auto valid_data_col = raw_data.get_col_valid(nullable_fid);

    SetSchema(schema);
    auto plan_str = create_search_plan_from_expr("nullable >= age64");
    auto plan =
        CreateSearchPlanByExpr(schema, plan_str.data(), plan_str.size());

    milvus::exec::OffsetVector offsets;
    offsets.reserve(N);
    for (auto i = 0; i < N; ++i) {
        offsets.emplace_back(i);
    }

    auto col_vec = milvus::test::gen_filter_res(
        plan->plan_node_->plannodes_->sources()[0]->sources()[0].get(),
        seg.get(),
        N,
        MAX_TIMESTAMP,
        &offsets);
    BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
    BitsetTypeView valid_view(col_vec->GetValidRawData(), col_vec->size());
    EXPECT_EQ(view.size(), N);

    for (int i = 0; i < N; ++i) {
        auto ref = valid_data_col[i] && nullable_col[i] >= pk_col[i];
        ASSERT_EQ(view[i], ref) << "@" << i;
        ASSERT_EQ(valid_view[i], valid_data_col[i]) << "@" << i;
    }
}

TEST(CompareExplainTest, ExplainRecordsValueReaderAndRawScanCounts) {
    auto schema = std::make_shared<Schema>();
    auto i32_fid = schema->AddDebugField("age32", DataType::INT32);
    auto i64_fid = schema->AddDebugField("age64", DataType::INT64);
    schema->set_primary_field_id(i64_fid);

    constexpr int N = 64;
    auto raw_data = DataGen(schema, N);
    auto age32_col = raw_data.get_col<int32_t>(i32_fid);
    auto age64_col = raw_data.get_col<int64_t>(i64_fid);
    int64_t expected_matches = 0;
    for (int i = 0; i < N; ++i) {
        age32_col[i] = (i % 4 == 0) ? 10 : 100;
        age64_col[i] = 50;
        if (age32_col[i] < age64_col[i]) {
            ++expected_matches;
        }
    }

    auto cmp_expr = std::make_shared<expr::CompareExpr>(
        i32_fid, i64_fid, DataType::INT32, DataType::INT64, OpType::LessThan);
    cmp_expr->SetExprId(99001);
    auto plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, cmp_expr);

    auto attrs_of = [](const proto::internal::ExecutionExplain& n) {
        std::map<std::string, std::string> out;
        for (const auto& [k, v] : n.attributes()) {
            out[k] = v;
        }
        return out;
    };
    auto counters_of = [](const proto::internal::ExecutionExplain& n) {
        std::map<std::string, int64_t> out;
        for (const auto& [k, v] : n.counters()) {
            out[k] = v;
        }
        return out;
    };

    // 1. Both columns served by STL_SORT value readers: both operand groups
    // recorded, idx_scan=N, raw_scan=0, segment counters count rows once.
    {
        auto seg = CreateSealedSegment(schema);
        test::expr_index::InstallIndex(
            *seg,
            i32_fid,
            DataType::INT32,
            test::consumer::BuildScalarReader<int32_t>(i32_fid,
                                                       DataType::INT32,
                                                       index::ASCENDING_SORT,
                                                       N,
                                                       age32_col.data()));
        test::expr_index::InstallIndex(
            *seg,
            i64_fid,
            DataType::INT64,
            test::consumer::BuildScalarReader<int64_t>(i64_fid,
                                                       DataType::INT64,
                                                       index::ASCENDING_SORT,
                                                       N,
                                                       age64_col.data()));

        auto explain = std::make_shared<exec::ExecutionExplainRecorder>(N);
        auto res = milvus::test::gen_filter_res(
            plan.get(), seg.get(), N, MAX_TIMESTAMP, nullptr, explain);
        ASSERT_NE(res, nullptr);

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 2);
        const auto seg_ctrs = counters_of(node);
        EXPECT_EQ(seg_ctrs.at("rows_in"), N);
        EXPECT_EQ(seg_ctrs.at("idx_scan"), N);
        EXPECT_EQ(seg_ctrs.at("raw_scan"), 0);
        EXPECT_EQ(seg_ctrs.at("rows_out"), expected_matches);

        std::map<std::string, int> by_field;
        int secondary_count = 0;
        for (int i = 0; i < node.children_size(); ++i) {
            const auto attrs = attrs_of(node.children(i));
            const auto ctrs = counters_of(node.children(i));
            by_field[attrs.at("field_id")] = i;
            if (attrs.count("secondary_operand") > 0 &&
                attrs.at("secondary_operand") == "true") {
                ++secondary_count;
            }
            EXPECT_EQ(attrs.at("expr_id"), "99001");
            EXPECT_EQ(attrs.count("index_id"), 1u);
            EXPECT_EQ(attrs.at("family"), index::families::kSort);
            EXPECT_EQ(ctrs.at("rows_in"), N);
            EXPECT_EQ(ctrs.at("idx_scan"), N);
            EXPECT_EQ(ctrs.at("raw_scan"), 0);
            EXPECT_EQ(ctrs.at("rows_out"), expected_matches);
        }
        EXPECT_EQ(by_field.count(std::to_string(i32_fid.get())), 1u);
        EXPECT_EQ(by_field.count(std::to_string(i64_fid.get())), 1u);
        EXPECT_EQ(secondary_count, 1);
    }

    // 2. Both columns raw with no index: both operand groups report
    // decline_reason=no_index_on_field without double-counting segment rows.
    {
        test::expr_index::RawFieldFiles files;
        auto seg = CreateSealedSegment(schema);
        auto i32_field =
            std::make_shared<FieldData<int32_t>>(DataType::INT32, false);
        i32_field->FillFieldData(age32_col.data(), N);
        auto i64_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        i64_field->FillFieldData(age64_col.data(), N);
        seg->LoadFieldData(files.Prepare(i32_fid, {i32_field}));
        seg->LoadFieldData(files.Prepare(i64_fid, {i64_field}));

        auto explain = std::make_shared<exec::ExecutionExplainRecorder>(N);
        auto res = milvus::test::gen_filter_res(
            plan.get(), seg.get(), N, MAX_TIMESTAMP, nullptr, explain);
        ASSERT_NE(res, nullptr);

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 2);
        const auto seg_ctrs = counters_of(node);
        EXPECT_EQ(seg_ctrs.at("rows_in"), N);
        EXPECT_EQ(seg_ctrs.at("idx_scan"), 0);
        EXPECT_EQ(seg_ctrs.at("raw_scan"), N);
        EXPECT_EQ(seg_ctrs.at("rows_out"), expected_matches);

        for (int i = 0; i < node.children_size(); ++i) {
            const auto attrs = attrs_of(node.children(i));
            const auto ctrs = counters_of(node.children(i));
            EXPECT_EQ(attrs.at("expr_id"), "99001");
            EXPECT_EQ(attrs.count("index_id"), 0u);
            EXPECT_EQ(attrs.at("decline_reason"), "no_index_on_field");
            EXPECT_EQ(ctrs.at("rows_in"), N);
            EXPECT_EQ(ctrs.at("idx_scan"), 0);
            EXPECT_EQ(ctrs.at("raw_scan"), N);
            EXPECT_EQ(ctrs.at("rows_out"), expected_matches);
        }
    }

    // 3. Inverted index (no IScalarValueReader) on s1 + no index on s2:
    // s1 reports capability_mismatch and s2 reports no_index_on_field.
    {
        auto str_schema = std::make_shared<Schema>();
        auto s1_fid = str_schema->AddDebugField("s1", DataType::VARCHAR);
        auto s2_fid = str_schema->AddDebugField("s2", DataType::VARCHAR);
        const std::vector<std::string> s1_vals{"a", "z", "b", "y"};
        const std::vector<std::string> s2_vals{"m", "m", "m", "m"};
        auto s1_field = test::expr_index::StringField(s1_vals);
        auto s2_field = test::expr_index::StringField(s2_vals);

        test::expr_index::RawFieldFiles files;
        auto seg = CreateSealedSegment(str_schema);
        seg->LoadFieldData(files.Prepare(s1_fid, {s1_field}));
        seg->LoadFieldData(files.Prepare(s2_fid, {s2_field}));
        test::expr_index::InstallIndex(
            *seg,
            s1_fid,
            DataType::VARCHAR,
            test::expr_index::BuildIndex(s1_fid,
                                         DataType::VARCHAR,
                                         index::INVERTED_INDEX_TYPE,
                                         {s1_field}));

        auto str_cmp = std::make_shared<expr::CompareExpr>(s1_fid,
                                                           s2_fid,
                                                           DataType::VARCHAR,
                                                           DataType::VARCHAR,
                                                           OpType::LessThan);
        str_cmp->SetExprId(99002);
        auto str_plan = std::make_shared<plan::FilterBitsNode>(
            DEFAULT_PLANNODE_ID, str_cmp);

        auto explain_unusable =
            std::make_shared<exec::ExecutionExplainRecorder>(4);
        auto res = milvus::test::gen_filter_res(str_plan.get(),
                                                seg.get(),
                                                4,
                                                MAX_TIMESTAMP,
                                                nullptr,
                                                explain_unusable);
        ASSERT_NE(res, nullptr);

        proto::internal::ExecutionExplain unusable_node;
        ASSERT_TRUE(unusable_node.ParseFromString(explain_unusable->Finish()));
        ASSERT_EQ(unusable_node.children_size(), 2);
        const auto seg_ctrs = counters_of(unusable_node);
        EXPECT_EQ(seg_ctrs.at("rows_in"), 4);
        EXPECT_EQ(seg_ctrs.at("idx_scan"), 0);
        EXPECT_EQ(seg_ctrs.at("raw_scan"), 4);
        EXPECT_EQ(seg_ctrs.at("rows_out"), 2);

        std::map<std::string, std::map<std::string, std::string>> by_field;
        for (int i = 0; i < unusable_node.children_size(); ++i) {
            auto u_attrs = attrs_of(unusable_node.children(i));
            const auto u_ctrs = counters_of(unusable_node.children(i));
            EXPECT_EQ(u_attrs.at("expr_id"), "99002");
            EXPECT_EQ(u_attrs.count("index_id"), 0u);
            EXPECT_EQ(u_ctrs.at("rows_in"), 4);
            EXPECT_EQ(u_ctrs.at("idx_scan"), 0);
            EXPECT_EQ(u_ctrs.at("raw_scan"), 4);
            EXPECT_EQ(u_ctrs.at("rows_out"), 2);
            by_field[u_attrs.at("field_id")] = std::move(u_attrs);
        }
        EXPECT_EQ(
            by_field.at(std::to_string(s1_fid.get())).at("decline_reason"),
            "capability_mismatch");
        EXPECT_EQ(
            by_field.at(std::to_string(s2_fid.get())).at("decline_reason"),
            "no_index_on_field");
    }

    // 4. One STL_SORT value reader (raw_scan=0) + one raw column (raw_scan=N).
    {
        test::expr_index::RawFieldFiles files;
        auto seg = CreateSealedSegment(schema);
        test::expr_index::InstallIndex(
            *seg,
            i32_fid,
            DataType::INT32,
            test::consumer::BuildScalarReader<int32_t>(i32_fid,
                                                       DataType::INT32,
                                                       index::ASCENDING_SORT,
                                                       N,
                                                       age32_col.data()));
        auto i64_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        i64_field->FillFieldData(age64_col.data(), N);
        seg->LoadFieldData(files.Prepare(i64_fid, {i64_field}));

        auto explain = std::make_shared<exec::ExecutionExplainRecorder>(N);
        auto res = milvus::test::gen_filter_res(
            plan.get(), seg.get(), N, MAX_TIMESTAMP, nullptr, explain);
        ASSERT_NE(res, nullptr);

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 2);
        const auto seg_ctrs = counters_of(node);
        EXPECT_EQ(seg_ctrs.at("rows_in"), N);
        EXPECT_EQ(seg_ctrs.at("idx_scan"), N);
        EXPECT_EQ(seg_ctrs.at("raw_scan"), N);
        EXPECT_EQ(seg_ctrs.at("rows_out"), expected_matches);

        for (int i = 0; i < node.children_size(); ++i) {
            const auto attrs = attrs_of(node.children(i));
            const auto ctrs = counters_of(node.children(i));
            if (attrs.at("field_id") == std::to_string(i32_fid.get())) {
                EXPECT_EQ(attrs.count("index_id"), 1u);
                EXPECT_EQ(attrs.at("family"), index::families::kSort);
                EXPECT_EQ(ctrs.at("idx_scan"), N);
                EXPECT_EQ(ctrs.at("raw_scan"), 0);
            } else {
                EXPECT_EQ(attrs.at("field_id"), std::to_string(i64_fid.get()));
                EXPECT_EQ(attrs.count("index_id"), 0u);
                EXPECT_EQ(attrs.at("decline_reason"), "no_index_on_field");
                EXPECT_EQ(ctrs.at("idx_scan"), 0);
                EXPECT_EQ(ctrs.at("raw_scan"), N);
            }
            EXPECT_EQ(ctrs.at("rows_in"), N);
            EXPECT_EQ(ctrs.at("rows_out"), expected_matches);
        }
    }

    // 5. Same-column compare (i32 == i32) emits a single group with raw_scan=N.
    {
        test::expr_index::RawFieldFiles files;
        auto seg = CreateSealedSegment(schema);
        auto i32_field =
            std::make_shared<FieldData<int32_t>>(DataType::INT32, false);
        i32_field->FillFieldData(age32_col.data(), N);
        seg->LoadFieldData(files.Prepare(i32_fid, {i32_field}));

        auto self_cmp = std::make_shared<expr::CompareExpr>(
            i32_fid, i32_fid, DataType::INT32, DataType::INT32, OpType::Equal);
        self_cmp->SetExprId(99003);
        auto self_plan = std::make_shared<plan::FilterBitsNode>(
            DEFAULT_PLANNODE_ID, self_cmp);

        auto explain = std::make_shared<exec::ExecutionExplainRecorder>(N);
        auto res = milvus::test::gen_filter_res(
            self_plan.get(), seg.get(), N, MAX_TIMESTAMP, nullptr, explain);
        ASSERT_NE(res, nullptr);

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto ctrs = counters_of(node.children(0));
        EXPECT_EQ(ctrs.at("rows_in"), N);
        EXPECT_EQ(ctrs.at("raw_scan"), N);
        EXPECT_EQ(ctrs.at("rows_out"), N);
    }

    // 6. CompareExpr under upstream AND mask reports active rows_in == raw_scan,
    // and partial-coverage operand scan preserves effective_used_index=true
    // whenever operand_idx > 0 even if un-indexed tail rows exceed idx_scan.
    {
        test::expr_index::RawFieldFiles files;
        auto seg = CreateSealedSegment(schema);
        auto i32_field =
            std::make_shared<FieldData<int32_t>>(DataType::INT32, false);
        i32_field->FillFieldData(age32_col.data(), N);
        auto i64_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        i64_field->FillFieldData(age64_col.data(), N);
        seg->LoadFieldData(files.Prepare(i32_fid, {i32_field}));
        seg->LoadFieldData(files.Prepare(i64_fid, {i64_field}));

        proto::plan::GenericValue val10;
        val10.set_int64_val(10);
        auto pre_eq = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(i32_fid, DataType::INT32),
            OpType::Equal,
            val10,
            std::vector<proto::plan::GenericValue>());
        pre_eq->SetExprId(99004);
        auto masked_cmp = std::make_shared<expr::CompareExpr>(i32_fid,
                                                              i64_fid,
                                                              DataType::INT32,
                                                              DataType::INT64,
                                                              OpType::LessThan);
        masked_cmp->SetExprId(99005);
        auto and_expr = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, pre_eq, masked_cmp);
        auto and_plan = std::make_shared<plan::FilterBitsNode>(
            DEFAULT_PLANNODE_ID, and_expr);

        auto explain = std::make_shared<exec::ExecutionExplainRecorder>(N);
        auto res = milvus::test::gen_filter_res(
            and_plan.get(), seg.get(), N, MAX_TIMESTAMP, nullptr, explain);
        ASSERT_NE(res, nullptr);

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        int cmp_groups = 0;
        for (int i = 0; i < node.children_size(); ++i) {
            const auto attrs = attrs_of(node.children(i));
            if (attrs.at("expr_id") != "99005") {
                continue;
            }
            ++cmp_groups;
            const auto ctrs = counters_of(node.children(i));
            EXPECT_EQ(ctrs.at("rows_in"), expected_matches);
            EXPECT_EQ(ctrs.at("raw_scan"), expected_matches);
            EXPECT_EQ(ctrs.at("idx_scan"), 0);
            EXPECT_EQ(ctrs.at("rows_out"), expected_matches);
        }
        EXPECT_EQ(cmp_groups, 2);

        exec::ExecutionExplainRecorder partial_rec(200);
        partial_rec.RecordDecision(
            i32_fid, 99006, true, 7001, "sort", false, "", "");
        partial_rec.AddRows(99006, 60, 15);
        partial_rec.AddOperandScan(
            i32_fid, 99006, true, 7001, "sort", false, "", "", 10, 50);
        partial_rec.AddRawScan(99006, 50);
        proto::internal::ExecutionExplain partial_node;
        ASSERT_TRUE(partial_node.ParseFromString(partial_rec.Finish()));
        ASSERT_EQ(partial_node.children_size(), 1);
        const auto p_attrs = attrs_of(partial_node.children(0));
        const auto p_ctrs = counters_of(partial_node.children(0));
        EXPECT_EQ(p_attrs.at("index_id"), "7001");
        EXPECT_EQ(p_attrs.count("decline_reason"), 0u);
        EXPECT_EQ(p_ctrs.at("rows_in"), 60);
        EXPECT_EQ(p_ctrs.at("idx_scan"), 10);
        EXPECT_EQ(p_ctrs.at("raw_scan"), 50);
    }
}
