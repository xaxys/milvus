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

#include <gtest/gtest.h>
#include "query/deprecated/ParserDeprecated.h"
#include <google/protobuf/text_format.h>
#include "query/PlanProto.h"
#include "query/Expr.h"
#include "query/PlanNode.h"
#include "query/generated/ExprVisitor.h"
#include "query/generated/PlanNodeVisitor.h"
#include "test_utils/DataGen.h"
#include "query/generated/ShowPlanNodeVisitor.h"
#include "query/generated/ExecExprVisitor.h"
#include "query/Plan.h"
#include "utils/tools.h"
#include <regex>
#include <boost/format.hpp>
#include "pb/plan.pb.h"
#include "segcore/SegmentGrowingImpl.h"
using namespace milvus;

TEST(Expr, Naive) {
    SUCCEED();
    using namespace milvus::wtf;
    std::string dsl_string = R"(
{
    "bool": {
        "must": [
            {
                "term": {
                    "A": [
                        1,
                        2,
                        5
                    ]
                }
            },
            {
                "range": {
                    "B": {
                        "GT": 1,
                        "LT": 100
                    }
                }
            },
            {
                "vector": {
                    "Vec": {
                        "metric_type": "L2",
                        "params": {
                            "nprobe": 10
                        },
                        "query": "$0",
                        "topk": 10
                    }
                }
            }
        ]
    }
})";
}

TEST(Expr, Range) {
    SUCCEED();
    using namespace milvus;
    using namespace milvus::query;
    using namespace milvus::segcore;
    std::string dsl_string = R"(
{
    "bool": {
        "must": [
            {
                "range": {
                    "age": {
                        "GT": 1,
                        "LT": 100
                    }
                }
            },
            {
                "vector": {
                    "fakevec": {
                        "metric_type": "L2",
                        "params": {
                            "nprobe": 10
                        },
                        "query": "$0",
                        "topk": 10
                    }
                }
            }
        ]
    }
})";
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", DataType::VECTOR_FLOAT, 16, MetricType::METRIC_L2);
    schema->AddDebugField("age", DataType::INT32);
    auto plan = CreatePlan(*schema, dsl_string);
    ShowPlanNodeVisitor shower;
    Assert(plan->tag2field_.at("$0") == schema->get_offset(FieldName("fakevec")));
    auto out = shower.call_child(*plan->plan_node_);
    std::cout << out.dump(4);
}

TEST(Expr, RangeBinary) {
    SUCCEED();
    using namespace milvus;
    using namespace milvus::query;
    using namespace milvus::segcore;
    std::string dsl_string = R"(
{
    "bool": {
        "must": [
            {
                "range": {
                    "age": {
                        "GT": 1,
                        "LT": 100
                    }
                }
            },
            {
                "vector": {
                    "fakevec": {
                        "metric_type": "Jaccard",
                        "params": {
                            "nprobe": 10
                        },
                        "query": "$0",
                        "topk": 10
                    }
                }
            }
        ]
    }
})";
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", DataType::VECTOR_BINARY, 512, MetricType::METRIC_Jaccard);
    schema->AddDebugField("age", DataType::INT32);
    auto plan = CreatePlan(*schema, dsl_string);
    ShowPlanNodeVisitor shower;
    Assert(plan->tag2field_.at("$0") == schema->get_offset(FieldName("fakevec")));
    auto out = shower.call_child(*plan->plan_node_);
    std::cout << out.dump(4);
}

TEST(Expr, InvalidRange) {
    SUCCEED();
    using namespace milvus;
    using namespace milvus::query;
    using namespace milvus::segcore;
    std::string dsl_string = R"(
{
    "bool": {
        "must": [
            {
                "range": {
                    "age": {
                        "GT": 1,
                        "LT": "100"
                    }
                }
            },
            {
                "vector": {
                    "fakevec": {
                        "metric_type": "L2",
                        "params": {
                            "nprobe": 10
                        },
                        "query": "$0",
                        "topk": 10
                    }
                }
            }
        ]
    }
})";
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", DataType::VECTOR_FLOAT, 16, MetricType::METRIC_L2);
    schema->AddDebugField("age", DataType::INT32);
    ASSERT_ANY_THROW(CreatePlan(*schema, dsl_string));
}

TEST(Expr, InvalidDSL) {
    SUCCEED();
    using namespace milvus;
    using namespace milvus::query;
    using namespace milvus::segcore;
    std::string dsl_string = R"(
{
    "float": {
        "must": [
            {
                "range": {
                    "age": {
                        "GT": 1,
                        "LT": 100
                    }
                }
            },
            {
                "vector": {
                    "fakevec": {
                        "metric_type": "L2",
                        "params": {
                            "nprobe": 10
                        },
                        "query": "$0",
                        "topk": 10
                    }
                }
            }
        ]
    }
})";

    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", DataType::VECTOR_FLOAT, 16, MetricType::METRIC_L2);
    schema->AddDebugField("age", DataType::INT32);
    ASSERT_ANY_THROW(CreatePlan(*schema, dsl_string));
}

TEST(Expr, ShowExecutor) {
    using namespace milvus::query;
    using namespace milvus::segcore;
    auto node = std::make_unique<FloatVectorANNS>();
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", DataType::VECTOR_FLOAT, 16, MetricType::METRIC_L2);
    int64_t num_queries = 100L;
    auto raw_data = DataGen(schema, num_queries);
    auto& info = node->search_info_;

    info.metric_type_ = MetricType::METRIC_L2;
    info.topk_ = 20;
    info.field_offset_ = FieldOffset(0);
    node->predicate_ = std::nullopt;
    ShowPlanNodeVisitor show_visitor;
    PlanNodePtr base(node.release());
    auto res = show_visitor.call_child(*base);
    auto dup = res;
    dup["data"] = "...collased...";
    std::cout << dup.dump(4);
}

TEST(Expr, TestRange) {
    using namespace milvus::query;
    using namespace milvus::segcore;
    std::vector<std::tuple<std::string, std::function<bool(int)>>> testcases = {
        {R"("GT": 2000, "LT": 3000)", [](int v) { return 2000 < v && v < 3000; }},
        {R"("GE": 2000, "LT": 3000)", [](int v) { return 2000 <= v && v < 3000; }},
        {R"("GT": 2000, "LE": 3000)", [](int v) { return 2000 < v && v <= 3000; }},
        {R"("GE": 2000, "LE": 3000)", [](int v) { return 2000 <= v && v <= 3000; }},
        {R"("GE": 2000)", [](int v) { return v >= 2000; }},
        {R"("GT": 2000)", [](int v) { return v > 2000; }},
        {R"("LE": 2000)", [](int v) { return v <= 2000; }},
        {R"("LT": 2000)", [](int v) { return v < 2000; }},
        {R"("EQ": 2000)", [](int v) { return v == 2000; }},
        {R"("NE": 2000)", [](int v) { return v != 2000; }},
    };

    std::string dsl_string_tmp = R"(
{
    "bool": {
        "must": [
            {
                "range": {
                    "age": {
                        @@@@
                    }
                }
            },
            {
                "vector": {
                    "fakevec": {
                        "metric_type": "L2",
                        "params": {
                            "nprobe": 10
                        },
                        "query": "$0",
                        "topk": 10
                    }
                }
            }
        ]
    }
})";
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", DataType::VECTOR_FLOAT, 16, MetricType::METRIC_L2);
    schema->AddDebugField("age", DataType::INT32);

    auto seg = CreateGrowingSegment(schema);
    int N = 1000;
    std::vector<int> age_col;
    int num_iters = 100;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto raw_data = DataGen(schema, N, iter);
        auto new_age_col = raw_data.get_col<int>(1);
        age_col.insert(age_col.end(), new_age_col.begin(), new_age_col.end());
        seg->PreInsert(N);
        seg->Insert(iter * N, N, raw_data.row_ids_.data(), raw_data.timestamps_.data(), raw_data.raw_);
    }

    auto seg_promote = dynamic_cast<SegmentGrowingImpl*>(seg.get());
    ExecExprVisitor visitor(*seg_promote, seg_promote->get_row_count(), MAX_TIMESTAMP);
    for (auto [clause, ref_func] : testcases) {
        auto loc = dsl_string_tmp.find("@@@@");
        auto dsl_string = dsl_string_tmp;
        dsl_string.replace(loc, 4, clause);
        auto plan = CreatePlan(*schema, dsl_string);
        auto final = visitor.call_child(*plan->plan_node_->predicate_.value());
        EXPECT_EQ(final.length(), N * num_iters);

        Assert(final.is_array());
        Assert(final.type()->Equals(arrow::BooleanType()));
        const auto& array = final.array_as<arrow::BooleanArray>();
        for (int i = 0; i < N * num_iters; ++i) {
            Assert(array->IsValid(i));
            auto ans = array->GetView(i);
            auto val = age_col[i];
            auto ref = ref_func(val);
            ASSERT_EQ(ans, ref) << clause << "@" << i << "!!" << val;
        }
    }
}

TEST(Expr, TestRangeBool) {
    using namespace milvus::query;
    using namespace milvus::segcore;
    std::vector<std::tuple<std::string, std::string, std::function<bool(bool)>>> testcases = {
        {"Equal", "true", [](bool f) { return f == true; }},
        {"Equal", "false", [](bool f) { return f == false; }},
        {"NotEqual", "true", [](bool f) { return f != true; }},
        {"NotEqual", "false", [](bool f) { return f != false; }},
    };

    // f op bool_constant
    auto string_tpl = R"(
vector_anns: <
  field_id: 20000
  predicates: <
    unary_range_expr: <
      child: <
        column_expr: <
          column_info: <
            field_id: 20001
            data_type: Bool
          >
        >
      >
      value: <
        bool_val: %2%
      >
      op: %1%
    >
  >
  query_info: <
    topk: 10
    metric_type: "L2"
    search_params: "{\"nprobe\": 10}"
  >
  placeholder_tag: "$0"
>
)";

    auto schema = std::make_shared<Schema>();
    schema->AddField(FieldName("fakevec"), FieldId(20000), DataType::VECTOR_FLOAT, 16, MetricType::METRIC_L2);
    schema->AddField(FieldName("BoolField"), FieldId(20001), DataType::BOOL);

    auto seg = CreateGrowingSegment(schema);
    int N = 10000;
    std::vector<uint8_t> bool_col;
    int num_iters = 100;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto raw_data = DataGen(schema, N, iter);
        auto new_bool_col = raw_data.get_col<uint8_t>(1);
        bool_col.insert(bool_col.end(), new_bool_col.begin(), new_bool_col.end());
        seg->PreInsert(N);
        seg->Insert(iter * N, N, raw_data.row_ids_.data(), raw_data.timestamps_.data(), raw_data.raw_);
    }

    auto seg_promote = dynamic_cast<SegmentGrowingImpl*>(seg.get());
    ExecExprVisitor visitor(*seg_promote, seg_promote->get_row_count(), MAX_TIMESTAMP);
    for (auto [clause, bool_constant, ref_func] : testcases) {
        auto proto_text = boost::str(boost::format(string_tpl) % clause % bool_constant);
        proto::plan::PlanNode node_proto;
        google::protobuf::TextFormat::ParseFromString(proto_text, &node_proto);
        // std::cout << node_proto.DebugString();
        auto plan = ProtoParser(*schema).CreatePlan(node_proto);
        // std::cout << ShowPlanNodeVisitor().call_child(*plan->plan_node_) << std::endl;
        auto final = visitor.call_child(*plan->plan_node_->predicate_.value());
        EXPECT_EQ(final.length(), N * num_iters);

        Assert(final.is_array());
        Assert(final.type()->Equals(arrow::BooleanType()));
        const auto& array = final.array_as<arrow::BooleanArray>();
        for (int i = 0; i < N * num_iters; ++i) {
            auto ans = array->GetView(i);

            auto val = bool_col[i];
            auto ref = ref_func(val);
            ASSERT_EQ(ans, ref) << clause << "@" << i << "!!" << boost::format("%1%") % val;
        }
    }
}

TEST(Expr, TestTerm) {
    using namespace milvus::query;
    using namespace milvus::segcore;
    auto vec_2k_3k = [] {
        std::string buf = "[";
        for (int i = 2000; i < 3000 - 1; ++i) {
            buf += std::to_string(i) + ", ";
        }
        buf += std::to_string(2999) + "]";
        return buf;
    }();

    std::vector<std::tuple<std::string, std::function<bool(int)>>> testcases = {
        {R"([2000, 3000])", [](int v) { return v == 2000 || v == 3000; }},
        {R"([2000])", [](int v) { return v == 2000; }},
        {R"([3000])", [](int v) { return v == 3000; }},
        {R"([])", [](int v) { return false; }},
        {vec_2k_3k, [](int v) { return 2000 <= v && v < 3000; }},
    };

    std::string dsl_string_tmp = R"(
{
    "bool": {
        "must": [
            {
                "term": {
                    "age": {
                        "values": @@@@
                    }
                }
            },
            {
                "vector": {
                    "fakevec": {
                        "metric_type": "L2",
                        "params": {
                            "nprobe": 10
                        },
                        "query": "$0",
                        "topk": 10
                    }
                }
            }
        ]
    }
})";
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", DataType::VECTOR_FLOAT, 16, MetricType::METRIC_L2);
    schema->AddDebugField("age", DataType::INT32);

    auto seg = CreateGrowingSegment(schema);
    int N = 1000;
    std::vector<int> age_col;
    int num_iters = 100;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto raw_data = DataGen(schema, N, iter);
        auto new_age_col = raw_data.get_col<int>(1);
        age_col.insert(age_col.end(), new_age_col.begin(), new_age_col.end());
        seg->PreInsert(N);
        seg->Insert(iter * N, N, raw_data.row_ids_.data(), raw_data.timestamps_.data(), raw_data.raw_);
    }

    auto seg_promote = dynamic_cast<SegmentGrowingImpl*>(seg.get());
    ExecExprVisitor visitor(*seg_promote, seg_promote->get_row_count(), MAX_TIMESTAMP);
    for (auto [clause, ref_func] : testcases) {
        auto loc = dsl_string_tmp.find("@@@@");
        auto dsl_string = dsl_string_tmp;
        dsl_string.replace(loc, 4, clause);
        auto plan = CreatePlan(*schema, dsl_string);
        auto final = visitor.call_child(*plan->plan_node_->predicate_.value());
        if (final.is_scalar()) {
            // for [] case
            for (int i = 0; i < N * num_iters; ++i) {
                auto val = age_col[i];
                auto ref = ref_func(val);
                ASSERT_EQ(false, ref) << clause << "@" << i << "!!" << val;
            }
            continue;
        }
        EXPECT_EQ(final.length(), N * num_iters);
        Assert(final.is_array());
        Assert(final.type()->Equals(arrow::BooleanType()));
        const auto& array = final.array_as<arrow::BooleanArray>();
        for (int i = 0; i < N * num_iters; ++i) {
            auto ans = array->GetView(i);
            auto val = age_col[i];
            auto ref = ref_func(val);
            ASSERT_EQ(ans, ref) << clause << "@" << i << "!!" << val;
        }
    }
}

TEST(Expr, TestSimpleDsl) {
    using namespace milvus::query;
    using namespace milvus::segcore;

    auto vec_dsl = Json::parse(R"(
            {
                "vector": {
                    "fakevec": {
                        "metric_type": "L2",
                        "params": {
                            "nprobe": 10
                        },
                        "query": "$0",
                        "topk": 10
                    }
                }
            }
)");

    int N = 32;
    auto get_item = [&](int base, int bit = 1) {
        std::vector<int> terms;
        // note: random gen range is [0, 2N)
        for (int i = 0; i < N * 2; ++i) {
            if (((i >> base) & 0x1) == bit) {
                terms.push_back(i);
            }
        }
        Json s;
        s["term"]["age"]["values"] = terms;
        return s;
    };
    // std::cout << get_item(0).dump(-2);
    // std::cout << vec_dsl.dump(-2);
    std::vector<std::tuple<Json, std::function<bool(int)>>> testcases;
    {
        Json dsl;
        dsl["must"] = Json::array({vec_dsl, get_item(0), get_item(1), get_item(2, 0), get_item(3)});
        testcases.emplace_back(dsl, [](int x) { return (x & 0b1111) == 0b1011; });
    }

    {
        Json dsl;
        Json sub_dsl;
        sub_dsl["must"] = Json::array({get_item(0), get_item(1), get_item(2, 0), get_item(3)});
        dsl["must"] = Json::array({sub_dsl, vec_dsl});
        testcases.emplace_back(dsl, [](int x) { return (x & 0b1111) == 0b1011; });
    }

    {
        Json dsl;
        Json sub_dsl;
        sub_dsl["should"] = Json::array({get_item(0), get_item(1), get_item(2, 0), get_item(3)});
        dsl["must"] = Json::array({sub_dsl, vec_dsl});
        testcases.emplace_back(dsl, [](int x) { return !!((x & 0b1111) ^ 0b0100); });
    }

    {
        Json dsl;
        Json sub_dsl;
        sub_dsl["must_not"] = Json::array({get_item(0), get_item(1), get_item(2, 0), get_item(3)});
        dsl["must"] = Json::array({sub_dsl, vec_dsl});
        testcases.emplace_back(dsl, [](int x) { return (x & 0b1111) != 0b1011; });
    }

    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", DataType::VECTOR_FLOAT, 16, MetricType::METRIC_L2);
    schema->AddDebugField("age", DataType::INT32);

    auto seg = CreateGrowingSegment(schema);
    std::vector<int> age_col;
    int num_iters = 100;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto raw_data = DataGen(schema, N, iter);
        auto new_age_col = raw_data.get_col<int>(1);
        age_col.insert(age_col.end(), new_age_col.begin(), new_age_col.end());
        seg->PreInsert(N);
        seg->Insert(iter * N, N, raw_data.row_ids_.data(), raw_data.timestamps_.data(), raw_data.raw_);
    }

    auto seg_promote = dynamic_cast<SegmentGrowingImpl*>(seg.get());
    ExecExprVisitor visitor(*seg_promote, seg_promote->get_row_count(), MAX_TIMESTAMP);
    for (auto [clause, ref_func] : testcases) {
        Json dsl;
        dsl["bool"] = clause;
        // std::cout << dsl.dump(2);
        auto plan = CreatePlan(*schema, dsl.dump());
        auto final = visitor.call_child(*plan->plan_node_->predicate_.value());
        EXPECT_EQ(final.length(), N * num_iters);

        Assert(final.is_array());
        Assert(final.type()->Equals(arrow::BooleanType()));
        const auto& array = final.array_as<arrow::BooleanArray>();
        for (int i = 0; i < N * num_iters; ++i) {
            auto ans = array->GetView(i);
            auto val = age_col[i];
            auto ref = ref_func(val);
            ASSERT_EQ(ans, ref) << clause << "@" << i << "!!" << val;
        }
    }
}

TEST(Expr, TestCompare) {
    using namespace milvus::query;
    using namespace milvus::segcore;
    std::vector<std::tuple<std::string, std::function<bool(int, int64_t)>>> testcases = {
        {R"("LT")", [](int a, int64_t b) { return a < b; }},  {R"("LE")", [](int a, int64_t b) { return a <= b; }},
        {R"("GT")", [](int a, int64_t b) { return a > b; }},  {R"("GE")", [](int a, int64_t b) { return a >= b; }},
        {R"("EQ")", [](int a, int64_t b) { return a == b; }}, {R"("NE")", [](int a, int64_t b) { return a != b; }},
    };

    std::string dsl_string_tpl = R"(
{
    "bool": {
        "must": [
            {
                "compare": {
                    %1%: [
                        "age1",
                        "age2"
                    ]
                }
            },
            {
                "vector": {
                    "fakevec": {
                        "metric_type": "L2",
                        "params": {
                            "nprobe": 10
                        },
                        "query": "$0",
                        "topk": 10
                    }
                }
            }
        ]
    }
})";
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", DataType::VECTOR_FLOAT, 16, MetricType::METRIC_L2);
    schema->AddDebugField("age1", DataType::INT32);
    schema->AddDebugField("age2", DataType::INT64);

    auto seg = CreateGrowingSegment(schema);
    int N = 1000;
    std::vector<int> age1_col;
    std::vector<int64_t> age2_col;
    int num_iters = 100;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto raw_data = DataGen(schema, N, iter);
        auto new_age1_col = raw_data.get_col<int>(1);
        auto new_age2_col = raw_data.get_col<int64_t>(2);
        age1_col.insert(age1_col.end(), new_age1_col.begin(), new_age1_col.end());
        age2_col.insert(age2_col.end(), new_age2_col.begin(), new_age2_col.end());
        seg->PreInsert(N);
        seg->Insert(iter * N, N, raw_data.row_ids_.data(), raw_data.timestamps_.data(), raw_data.raw_);
    }

    auto seg_promote = dynamic_cast<SegmentGrowingImpl*>(seg.get());
    ExecExprVisitor visitor(*seg_promote, seg_promote->get_row_count(), MAX_TIMESTAMP);
    for (auto [clause, ref_func] : testcases) {
        auto dsl_string = boost::str(boost::format(dsl_string_tpl) % clause);
        auto plan = CreatePlan(*schema, dsl_string);
        // std::cout << ShowPlanNodeVisitor().call_child(*plan->plan_node_) << std::endl;
        auto final = visitor.call_child(*plan->plan_node_->predicate_.value());
        EXPECT_EQ(final.length(), N * num_iters);

        Assert(final.is_array());
        Assert(final.type()->Equals(arrow::BooleanType()));
        const auto& array = final.array_as<arrow::BooleanArray>();
        for (int i = 0; i < N * num_iters; ++i) {
            auto ans = array->GetView(i);

            auto val1 = age1_col[i];
            auto val2 = age2_col[i];
            auto ref = ref_func(val1, val2);
            ASSERT_EQ(ans, ref) << clause << "@" << i << "!!" << boost::format("[%1%, %2%]") % val1 % val2;
        }
    }
}

TEST(Expr, TestCompareBool) {
    using namespace milvus::query;
    using namespace milvus::segcore;
    std::vector<std::tuple<std::string, std::function<bool(bool, bool)>>> testcases = {
        {"Equal", [](bool a, bool b) { return a == b; }},
        {"NotEqual", [](bool a, bool b) { return a != b; }},
    };

    // f op bool_constant
    auto string_tpl = R"(
vector_anns: <
  field_id: 20000
  predicates: <
    compare_expr: <
      left: <
        column_expr: <
          column_info: <
            field_id: 20001
            data_type: Bool
          >
        >
      >
      right: <
        column_expr: <
          column_info: <
            field_id: 20002
            data_type: Bool
          >
        >
      >
      op: %1%
    >
  >
  query_info: <
    topk: 10
    metric_type: "L2"
    search_params: "{\"nprobe\": 10}"
  >
  placeholder_tag: "$0"
>
)";

    auto schema = std::make_shared<Schema>();
    schema->AddField(FieldName("fakevec"), FieldId(20000), DataType::VECTOR_FLOAT, 16, MetricType::METRIC_L2);
    schema->AddField(FieldName("BoolField1"), FieldId(20001), DataType::BOOL);
    schema->AddField(FieldName("BoolField2"), FieldId(20002), DataType::BOOL);

    auto seg = CreateGrowingSegment(schema);
    int N = 10000;
    std::vector<uint8_t> bool1_col;
    std::vector<uint8_t> bool2_col;
    int num_iters = 100;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto raw_data = DataGen(schema, N, iter);
        auto new_bool1_col = raw_data.get_col<uint8_t>(1);
        auto new_bool2_col = raw_data.get_col<uint8_t>(2);
        bool1_col.insert(bool1_col.end(), new_bool1_col.begin(), new_bool1_col.end());
        bool2_col.insert(bool2_col.end(), new_bool2_col.begin(), new_bool2_col.end());
        seg->PreInsert(N);
        seg->Insert(iter * N, N, raw_data.row_ids_.data(), raw_data.timestamps_.data(), raw_data.raw_);
    }

    auto seg_promote = dynamic_cast<SegmentGrowingImpl*>(seg.get());
    ExecExprVisitor visitor(*seg_promote, seg_promote->get_row_count(), MAX_TIMESTAMP);
    for (auto [clause, ref_func] : testcases) {
        auto proto_text = boost::str(boost::format(string_tpl) % clause);
        proto::plan::PlanNode node_proto;
        google::protobuf::TextFormat::ParseFromString(proto_text, &node_proto);
        // std::cout << node_proto.DebugString();
        auto plan = ProtoParser(*schema).CreatePlan(node_proto);
        // std::cout << ShowPlanNodeVisitor().call_child(*plan->plan_node_) << std::endl;
        auto final = visitor.call_child(*plan->plan_node_->predicate_.value());
        EXPECT_EQ(final.length(), N * num_iters);

        Assert(final.is_array());
        Assert(final.type()->Equals(arrow::BooleanType()));
        const auto& array = final.array_as<arrow::BooleanArray>();
        for (int i = 0; i < N * num_iters; ++i) {
            auto ans = array->GetView(i);

            auto val1 = bool1_col[i];
            auto val2 = bool2_col[i];
            auto ref = ref_func(val1, val2);
            ASSERT_EQ(ans, ref) << clause << "@" << i << "!!" << boost::format("[%1%, %2%]") % val1 % val2;
        }
    }
}

TEST(Expr, TestBinaryArith) {
    using namespace milvus::query;
    using namespace milvus::segcore;
    std::vector<std::tuple<std::string, std::function<bool(int64_t, int16_t)>>> testcases = {
        {"Add", [](int64_t a, int16_t b) { return (a + b) >= b * b; }},
        {"Subtract", [](int64_t a, int16_t b) { return (a - b) >= b * b; }},
        {"Multiply", [](int64_t a, int16_t b) { return (a * b) >= b * b; }},
        {"Divide", [](int64_t a, int16_t b) { return (a / b) >= b * b; }},
        {"BitAnd", [](int64_t a, int16_t b) { return (a & b) >= b * b; }},
        {"BitOr", [](int64_t a, int16_t b) { return (a | b) >= b * b; }},
        {"BitXor", [](int64_t a, int16_t b) { return (a ^ b) >= b * b; }},
    };

    // (age1 op age2) >= ((double)(age2) ** 2)
    auto string_tpl = R"(
vector_anns: <
  field_id: 20000
  predicates: <
    compare_expr: <
      left: <
        binary_arith_expr: <
          left: <
            column_expr: <
              column_info: <
                field_id: 20001
                data_type: Int64
              >
            >
          >
          right: <
            column_expr: <
              column_info: <
                field_id: 20002
                data_type: Int16
              >
            >
          >
          op: %1%
        >
      >
      right: <
        binary_arith_expr: <
          left: <
            cast_expr: <
              child: <
                column_expr: <
                  column_info: <
                    field_id: 20002
                    data_type: Int16
                  >
                >
              >
              data_type: Double
            >
          >
          right: <
            value_expr: <
              value: <
                int64_val: 2
              >
            >
          >
          op: Power
        >
      >
      op: GreaterEqual
    >
  >
  query_info: <
    topk: 10
    metric_type: "L2"
    search_params: "{\"nprobe\": 10}"
  >
  placeholder_tag: "$0"
>
)";

    auto schema = std::make_shared<Schema>();
    schema->AddField(FieldName("fakevec"), FieldId(20000), DataType::VECTOR_FLOAT, 16, MetricType::METRIC_L2);
    schema->AddField(FieldName("age1"), FieldId(20001), DataType::INT64);
    schema->AddField(FieldName("age2"), FieldId(20002), DataType::INT16);

    auto seg = CreateGrowingSegment(schema);
    int N = 10000;
    std::vector<int64_t> age1_col;
    std::vector<int16_t> age2_col;
    int num_iters = 100;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto raw_data = DataGen(schema, N, iter);
        auto new_age1_col = raw_data.get_col<int64_t>(1);
        auto new_age2_col = raw_data.get_col<int16_t>(2);
        age1_col.insert(age1_col.end(), new_age1_col.begin(), new_age1_col.end());
        age2_col.insert(age2_col.end(), new_age2_col.begin(), new_age2_col.end());
        seg->PreInsert(N);
        seg->Insert(iter * N, N, raw_data.row_ids_.data(), raw_data.timestamps_.data(), raw_data.raw_);
    }

    auto seg_promote = dynamic_cast<SegmentGrowingImpl*>(seg.get());
    ExecExprVisitor visitor(*seg_promote, seg_promote->get_row_count(), MAX_TIMESTAMP);
    for (auto [clause, ref_func] : testcases) {
        auto proto_text = boost::str(boost::format(string_tpl) % clause);
        proto::plan::PlanNode node_proto;
        google::protobuf::TextFormat::ParseFromString(proto_text, &node_proto);
        // std::cout << node_proto.DebugString();
        auto plan = ProtoParser(*schema).CreatePlan(node_proto);
        // std::cout << ShowPlanNodeVisitor().call_child(*plan->plan_node_) << std::endl;
        auto final = visitor.call_child(*plan->plan_node_->predicate_.value());
        EXPECT_EQ(final.length(), N * num_iters);

        Assert(final.is_array());
        Assert(final.type()->Equals(arrow::BooleanType()));
        const auto& array = final.array_as<arrow::BooleanArray>();
        for (int i = 0; i < N * num_iters; ++i) {
            auto ans = array->GetView(i);

            auto val1 = age1_col[i];
            auto val2 = age2_col[i];
            auto ref = ref_func(val1, val2);
            ASSERT_EQ(ans, ref) << clause << "@" << i << "!!" << boost::format("[%1%, %2%]") % val1 % val2;
        }
    }
}

TEST(Expr, TestUnaryArith) {
    using namespace milvus::query;
    using namespace milvus::segcore;
    std::vector<std::tuple<std::string, std::function<bool(int64_t)>>> testcases = {
        {"Minus", [](int64_t a) { return -a > 1234; }},
        {"BitNot", [](int64_t a) { return ~a > 1234; }},
    };

    // op a > 1234
    auto string_tpl = R"(
vector_anns: <
  field_id: 20000
  predicates: <
    compare_expr: <
      left: <
        unary_arith_expr: <
          child: <
            column_expr: <
              column_info: <
                field_id: 20001
                data_type: Int64
              >
            >
          >
          op: %1%
        >
      >
      right: <
        value_expr: <
          value: <
            int64_val: 1234
          >
        >
      >
      op: GreaterThan
    >
  >
  query_info: <
    topk: 10
    metric_type: "L2"
    search_params: "{\"nprobe\": 10}"
  >
  placeholder_tag: "$0"
>
)";

    auto schema = std::make_shared<Schema>();
    schema->AddField(FieldName("fakevec"), FieldId(20000), DataType::VECTOR_FLOAT, 16, MetricType::METRIC_L2);
    schema->AddField(FieldName("age1"), FieldId(20001), DataType::INT64);

    auto seg = CreateGrowingSegment(schema);
    int N = 10000;
    std::vector<int64_t> age1_col;
    int num_iters = 100;
    for (int iter = 0; iter < num_iters; ++iter) {
        auto raw_data = DataGen(schema, N, iter);
        auto new_age1_col = raw_data.get_col<int64_t>(1);
        age1_col.insert(age1_col.end(), new_age1_col.begin(), new_age1_col.end());
        seg->PreInsert(N);
        seg->Insert(iter * N, N, raw_data.row_ids_.data(), raw_data.timestamps_.data(), raw_data.raw_);
    }

    auto seg_promote = dynamic_cast<SegmentGrowingImpl*>(seg.get());
    ExecExprVisitor visitor(*seg_promote, seg_promote->get_row_count(), MAX_TIMESTAMP);
    for (auto [clause, ref_func] : testcases) {
        auto proto_text = boost::str(boost::format(string_tpl) % clause);
        proto::plan::PlanNode node_proto;
        google::protobuf::TextFormat::ParseFromString(proto_text, &node_proto);
        // std::cout << node_proto.DebugString();
        auto plan = ProtoParser(*schema).CreatePlan(node_proto);
        // std::cout << ShowPlanNodeVisitor().call_child(*plan->plan_node_) << std::endl;
        auto final = visitor.call_child(*plan->plan_node_->predicate_.value());
        EXPECT_EQ(final.length(), N * num_iters);

        Assert(final.is_array());
        Assert(final.type()->Equals(arrow::BooleanType()));
        const auto& array = final.array_as<arrow::BooleanArray>();
        for (int i = 0; i < N * num_iters; ++i) {
            auto ans = array->GetView(i);

            auto val = age1_col[i];
            auto ref = ref_func(val);
            ASSERT_EQ(ans, ref) << clause << "@" << i << "!!" << boost::format("%1%") % val;
        }
    }
}
