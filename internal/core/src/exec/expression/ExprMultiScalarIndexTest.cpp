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

// One field, several scalar indexes, end to end with real index objects: a
// sealed segment carries the raw column plus two real indexes of different
// types, both are loaded through LoadIndex(), and every filter is compared
// against the same filter evaluated over the raw column only.
//
// What this pins:
//   * a second scalar index on a field that already has one loads instead of
//     tripping the old AssertInfo(!has_index),
//   * both stay visible as separate capability entries of the field
//     (IndexCapability(field).entries().size() == 2), keyed by their own
//     indexID,
//   * the leaf selects an entry from that metadata alone (exec::DetermineExecPath
//     over index::ReaderCaps) and pins exactly one: Equal with INVERTED+NGRAM,
//     ranges and IN with STL_SORT+INVERTED, and the rows it returns are the rows
//     the raw-data evaluation returns.

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ExprBatchTestUtils.h"
#include "ExprIndexIntegrationTestUtils.h"
#include "common/FieldData.h"
#include "common/Geometry.h"
#include "common/Types.h"
#include "exec/expression/ExprCache.h"
#include "exec/expression/function/FunctionFactory.h"
#include "folly/ScopeGuard.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "gtest/gtest.h"
#include "index/Meta.h"
#include "index/contracts/query/IIndexReaderBase.h"
#include "index/contracts/query/INgramReader.h"
#include "pb/plan.pb.h"
#include "query/ExecPlanNodeVisitor.h"
#include "query/PlanNode.h"
#include "segcore/SegcoreConfig.h"
#include "segcore/SegmentSealed.h"
#include "segcore/Utils.h"
#include "segcore/indexing/FieldIndexCapability.h"
#include "storage/Util.h"
#include "test_utils/DataGen.h"
#include "test_utils/GenExprProto.h"
#include "test_utils/storage_test_utils.h"

using namespace milvus;
using namespace milvus::query;
using namespace milvus::segcore;

namespace {

using milvus::test::expr_index::BuildIndex;
using milvus::test::expr_index::InstallIndex;
using milvus::test::expr_index::OpenedIndex;
using milvus::test::expr_index::RawFieldFiles;

// Row i of the column holds value i, so a filter result reads as "which rows
// matched". The duplicate in each column keeps a row-identity-only
// implementation from agreeing with the raw scan by accident.
const std::vector<std::string> kVarCharValues = {
    /* 0 */ "prefix_alpha",
    /* 1 */ "other_alpha",
    /* 2 */ "prefix_beta",
    /* 3 */ "other_beta",
    /* 4 */ "prefix_gamma",
    /* 5 */ "other_gamma",
    /* 6 */ "prefix_beta",
    /* 7 */ "other_delta",
    /* 8 */ "prefix_epsilon",
    /* 9 */ "other_epsilon",
};

const std::vector<int64_t> kInt64Values = {
    10, 20, 30, 40, 40, 50, 60, 70, 80, 90};

// The rows the raw column must match. Checked explicitly as well, so a broken
// raw evaluation cannot make both sides of a comparison agree on a wrong answer.
void
ExpectHitRows(const ColumnVectorPtr& result,
              const std::vector<size_t>& expected_rows,
              const std::string& what) {
    ASSERT_NE(result, nullptr) << what;
    TargetBitmapView values(result->GetRawData(), result->size());
    for (size_t i = 0; i < result->size(); ++i) {
        const bool expected =
            std::find(expected_rows.begin(), expected_rows.end(), i) !=
            expected_rows.end();
        EXPECT_EQ(values[i], expected) << what << ": row " << i;
    }
}

// Row-by-row equality of a result over the indexed segment with the same result
// over the raw-data segment: matched row ids, hit count and validity bits.
void
ExpectSameFilterResult(const ColumnVectorPtr& raw,
                       const ColumnVectorPtr& indexed,
                       const std::string& what) {
    ASSERT_NE(raw, nullptr) << what;
    ASSERT_NE(indexed, nullptr) << what;
    ASSERT_EQ(raw->size(), indexed->size()) << what;
    TargetBitmapView raw_values(raw->GetRawData(), raw->size());
    TargetBitmapView raw_valid(raw->GetValidRawData(), raw->size());
    TargetBitmapView index_values(indexed->GetRawData(), indexed->size());
    TargetBitmapView index_valid(indexed->GetValidRawData(), indexed->size());
    int64_t raw_hits = 0;
    int64_t index_hits = 0;
    for (size_t i = 0; i < raw->size(); ++i) {
        EXPECT_EQ(index_valid[i], raw_valid[i]) << what << ": row " << i;
        EXPECT_EQ(index_values[i], raw_values[i]) << what << ": row " << i;
        raw_hits += raw_valid[i] && raw_values[i];
        index_hits += index_valid[i] && index_values[i];
    }
    EXPECT_EQ(index_hits, raw_hits) << what;
}

ColumnVectorPtr
EvaluateFilter(
    const expr::TypedExprPtr& filter_expr,
    const segcore::SegmentInternalInterface* segment,
    int64_t row_count,
    std::shared_ptr<milvus::exec::ExecutionExplainRecorder> explain = nullptr,
    const milvus::query::PlanOptions& plan_options =
        milvus::query::PlanOptions()) {
    auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                       filter_expr);
    return milvus::test::gen_filter_res(plan.get(),
                                        segment,
                                        row_count,
                                        MAX_TIMESTAMP,
                                        nullptr,
                                        explain,
                                        plan_options);
}

// Installs one already-opened reader under an explicit indexID, so a test can
// decide which index is the field's smallest and which one is loaded second.
void
InstallIndexWithId(segcore::SegmentSealed& segment,
                   FieldId field_id,
                   DataType field_type,
                   int64_t index_id,
                   OpenedIndex opened,
                   std::map<std::string, std::string> extra_params = {}) {
    segcore::LoadIndexInfo info{};
    info.field_id = field_id.get();
    info.field_type = field_type;
    info.index_id = index_id;
    info.index_engine_version = 3;
    info.index_family = opened.family;
    info.index_value_type = opened.reader->ValueType();
    info.index_caps = opened.caps;
    info.num_rows = opened.reader->Count();
    for (const auto& [key, value] : opened.params.items()) {
        info.index_params.emplace(
            key, value.is_string() ? value.get<std::string>() : value.dump());
    }
    for (auto& [key, value] : extra_params) {
        info.index_params.emplace(std::move(key), std::move(value));
    }
    info.index = std::move(opened.index);
    // The raw column stays loaded in these fixtures, exactly as it does for a
    // field whose index reports no raw data.
    info.load_resource_request = LoadResourceRequest{};
    segment.LoadIndex(info);
}

proto::plan::GenericValue
Int64Literal(int64_t value) {
    proto::plan::GenericValue val;
    val.set_int64_val(value);
    return val;
}

}  // namespace

// Case 1: INVERTED + NGRAM on one VARCHAR field. Equal must return the raw-data
// rows (it is served by INVERTED: an ngram index answers pattern operators and
// nothing else), LIKE 'prefix%' must return the raw-data rows (it is served by
// the NGRAM channel -- PrefixMatch + the literal "prefix" is exactly what the
// planner makes of LIKE 'prefix%').
TEST(MultiScalarIndexExprTest,
     VarCharInvertedAndNgramOnOneFieldBothServeQueries) {
    constexpr int64_t kInvertedIndexId = 3101;
    constexpr int64_t kNgramIndexId = 3102;

    auto schema = std::make_shared<Schema>();
    auto varchar_fid =
        schema->AddDebugField("varchar_field", DataType::VARCHAR);
    auto field_data = milvus::test::expr_index::StringField(kVarCharValues);

    // Oracle: the same column with no index at all.
    RawFieldFiles raw_files;
    auto raw_segment = CreateSealedSegment(schema);
    raw_segment->LoadFieldData(raw_files.Prepare(varchar_fid, {field_data}));

    // Feature under test: one field, two real scalar indexes, both loaded.
    RawFieldFiles indexed_files;
    auto indexed_segment = CreateSealedSegment(schema);
    indexed_segment->LoadFieldData(
        indexed_files.Prepare(varchar_fid, {field_data}));
    InstallIndexWithId(*indexed_segment,
                       varchar_fid,
                       DataType::VARCHAR,
                       kInvertedIndexId,
                       BuildIndex(varchar_fid,
                                  DataType::VARCHAR,
                                  milvus::index::INVERTED_INDEX_TYPE,
                                  {field_data}));
    InstallIndexWithId(*indexed_segment,
                       varchar_fid,
                       DataType::VARCHAR,
                       kNgramIndexId,
                       BuildIndex(varchar_fid,
                                  DataType::VARCHAR,
                                  milvus::index::NGRAM_INDEX_TYPE,
                                  {field_data},
                                  {{milvus::index::MIN_GRAM, 2},
                                   {milvus::index::MAX_GRAM, 4}}));

    // Silently dropping either load has to fail right here: both entries have to
    // be visible as field capability metadata, keyed by their own indexID.
    const auto capability =
        indexed_segment->ResolveIndexCapability(nullptr, varchar_fid);
    ASSERT_EQ(capability.entries().size(), 2u)
        << "INVERTED and NGRAM must both stay loaded on one field";
    EXPECT_EQ(
        capability.entries()[0].key,
        (milvus::segcore::IndexKey{
            varchar_fid,
            milvus::segcore::IndexIdentity::PreBuiltIndex(kInvertedIndexId)}));
    EXPECT_TRUE(capability.entries()[0].caps.predicate);
    EXPECT_FALSE(capability.entries()[0].caps.ngram_candidates);
    EXPECT_EQ(
        capability.entries()[1].key,
        (milvus::segcore::IndexKey{
            varchar_fid,
            milvus::segcore::IndexIdentity::PreBuiltIndex(kNgramIndexId)}));
    EXPECT_FALSE(capability.entries()[1].caps.predicate)
        << "an ngram entry is a candidate family: it never serves an exact op";
    EXPECT_TRUE(capability.entries()[1].caps.ngram_candidates);
    EXPECT_FALSE(capability.entries()[1].caps.exact);

    proto::plan::GenericValue equal_value;
    equal_value.set_string_val("prefix_beta");
    auto equal_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::Equal,
        equal_value,
        std::vector<proto::plan::GenericValue>());
    equal_expr->SetExprId(42001);

    auto raw_equal =
        EvaluateFilter(equal_expr, raw_segment.get(), kVarCharValues.size());
    ExpectHitRows(raw_equal, {2, 6}, "Equal \"prefix_beta\"");
    ExpectSameFilterResult(
        raw_equal,
        EvaluateFilter(
            equal_expr, indexed_segment.get(), kVarCharValues.size()),
        "Equal \"prefix_beta\" over INVERTED+NGRAM");
    // The leaf committed to the scalar-index path rather than the raw column.
    EXPECT_TRUE(milvus::test::CanExprExecuteAllAtOnce(
        equal_expr, indexed_segment.get(), kVarCharValues.size()));

    proto::plan::GenericValue prefix_value;
    prefix_value.set_string_val("prefix");
    auto like_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::PrefixMatch,  // LIKE 'prefix%'
        prefix_value,
        std::vector<proto::plan::GenericValue>());

    auto raw_like =
        EvaluateFilter(like_expr, raw_segment.get(), kVarCharValues.size());
    ExpectHitRows(raw_like, {0, 2, 4, 6, 8}, "LIKE 'prefix%'");
    ExpectSameFilterResult(
        raw_like,
        EvaluateFilter(like_expr, indexed_segment.get(), kVarCharValues.size()),
        "LIKE 'prefix%' over INVERTED+NGRAM");
}

// Case 1b: I4 regression. A field with only an NGRAM index and an Equal leaf: an
// NGRAM index serves pattern leaves only, so the leaf must fall back to the raw
// column rather than borrow it -- a term lookup over n-gram tokens silently
// returns wrong rows.
TEST(MultiScalarIndexExprTest, FamilyPriorityBeatsIndexIdForEquality) {
    // The design doc's priority table prefers BITMAP over INVERTED for an equality
    // predicate. BITMAP is given the LARGER indexID here, so the lowest-indexID
    // tie-break cannot be what produces the choice: only the family rank can.
    auto schema = std::make_shared<Schema>();
    auto varchar_fid =
        schema->AddDebugField("varchar_field", DataType::VARCHAR);
    auto field_data = milvus::test::expr_index::StringField(kVarCharValues);

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(varchar_fid, {field_data}));
    InstallIndexWithId(*segment,
                       varchar_fid,
                       DataType::VARCHAR,
                       30,
                       BuildIndex(varchar_fid,
                                  DataType::VARCHAR,
                                  milvus::index::INVERTED_INDEX_TYPE,
                                  {field_data}));
    InstallIndexWithId(*segment,
                       varchar_fid,
                       DataType::VARCHAR,
                       31,
                       BuildIndex(varchar_fid,
                                  DataType::VARCHAR,
                                  milvus::index::BITMAP_INDEX_TYPE,
                                  {field_data}));

    const auto capability =
        segment->ResolveIndexCapability(nullptr, varchar_fid);
    ASSERT_EQ(capability.entries().size(), 2u);

    exec::ExprIndexRequirement eq{
        .field_id = varchar_fid,
        .reader = exec::RequiredReader::Predicate,
        .value_type = DataType::VARCHAR,
    };
    const auto decision = exec::DetermineExecPath(eq, capability);
    ASSERT_TRUE(decision.key.has_value());
    const auto* chosen = capability.Find(*decision.key);
    ASSERT_NE(chosen, nullptr);
    EXPECT_EQ(chosen->family, "bitmap")
        << "equality must prefer BITMAP over INVERTED even when BITMAP has the "
           "larger indexID";
}

TEST(MultiScalarIndexExprTest, RangePrefersSortOverBitmap) {
    // The design doc's table asks for SORT first on ordered comparisons and BITMAP
    // first on equality. Both arrive as RequiredReader::Predicate; only the leaf's
    // is_range flag separates them. SORT gets the LARGER indexID so the lowest-id
    // tie-break cannot be what produces the range choice.
    auto schema = std::make_shared<Schema>();
    auto int_fid = schema->AddDebugField("int64_field", DataType::INT64);
    auto field_data =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    field_data->FillFieldData(kInt64Values.data(), kInt64Values.size());

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(int_fid, {field_data}));
    InstallIndexWithId(*segment,
                       int_fid,
                       DataType::INT64,
                       40,
                       BuildIndex(int_fid,
                                  DataType::INT64,
                                  milvus::index::BITMAP_INDEX_TYPE,
                                  {field_data}));
    InstallIndexWithId(*segment,
                       int_fid,
                       DataType::INT64,
                       41,
                       BuildIndex(int_fid,
                                  DataType::INT64,
                                  milvus::index::ASCENDING_SORT,
                                  {field_data}));

    const auto capability = segment->ResolveIndexCapability(nullptr, int_fid);
    ASSERT_EQ(capability.entries().size(), 2u);

    auto family_for = [&](bool is_range) {
        exec::ExprIndexRequirement req{
            .field_id = int_fid,
            .reader = exec::RequiredReader::Predicate,
            .value_type = DataType::INT64,
            .is_range = is_range,
        };
        const auto decision = exec::DetermineExecPath(req, capability);
        EXPECT_TRUE(decision.key.has_value());
        const auto* chosen = capability.Find(*decision.key);
        EXPECT_NE(chosen, nullptr);
        return chosen == nullptr ? std::string{} : chosen->family;
    };

    EXPECT_EQ(family_for(true), "sort")
        << "an ordered comparison must prefer SORT over BITMAP";
    EXPECT_EQ(family_for(false), "bitmap")
        << "equality must still prefer BITMAP over SORT";
}

TEST(MultiScalarIndexExprTest, EqualNeverUsesAnNgramOnlyField) {
    constexpr int64_t kNgramIndexId = 3111;

    auto schema = std::make_shared<Schema>();
    auto varchar_fid =
        schema->AddDebugField("varchar_field", DataType::VARCHAR);
    auto field_data = milvus::test::expr_index::StringField(kVarCharValues);

    RawFieldFiles raw_files;
    auto raw_segment = CreateSealedSegment(schema);
    raw_segment->LoadFieldData(raw_files.Prepare(varchar_fid, {field_data}));

    RawFieldFiles indexed_files;
    auto indexed_segment = CreateSealedSegment(schema);
    indexed_segment->LoadFieldData(
        indexed_files.Prepare(varchar_fid, {field_data}));
    InstallIndexWithId(*indexed_segment,
                       varchar_fid,
                       DataType::VARCHAR,
                       kNgramIndexId,
                       BuildIndex(varchar_fid,
                                  DataType::VARCHAR,
                                  milvus::index::NGRAM_INDEX_TYPE,
                                  {field_data},
                                  {{milvus::index::MIN_GRAM, 2},
                                   {milvus::index::MAX_GRAM, 4}}));

    const auto capability =
        indexed_segment->ResolveIndexCapability(nullptr, varchar_fid);
    ASSERT_EQ(capability.entries().size(), 1u);
    EXPECT_TRUE(capability.entries()[0].caps.ngram_candidates);

    proto::plan::GenericValue equal_value;
    equal_value.set_string_val("prefix_beta");
    auto equal_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::Equal,
        equal_value,
        std::vector<proto::plan::GenericValue>());

    auto raw_equal =
        EvaluateFilter(equal_expr, raw_segment.get(), kVarCharValues.size());
    ExpectSameFilterResult(
        raw_equal,
        EvaluateFilter(
            equal_expr, indexed_segment.get(), kVarCharValues.size()),
        "Equal \"prefix_beta\" over an NGRAM-only field");
    EXPECT_FALSE(milvus::test::CanExprExecuteAllAtOnce(
        equal_expr, indexed_segment.get(), kVarCharValues.size()))
        << "NGRAM cannot serve Equal => fall back to the raw column";
}

// Case 2: STL_SORT + INVERTED on one INT64 field. Open-ended ranges and `in`
// must return the raw-data rows. Capability metadata is ordered by index
// identity, so both indexID orders are run: whichever of the two the leaf lands
// on has to be able to serve the operator (neither may be handed a lookup shape
// it cannot answer instead of falling back to the raw column).
TEST(MultiScalarIndexExprTest,
     Int64StlSortAndInvertedOnOneFieldBothServeQueries) {
    auto schema = std::make_shared<Schema>();
    auto int64_fid = schema->AddDebugField("int64_field", DataType::INT64);
    auto field_data =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    field_data->FillFieldData(kInt64Values.data(), kInt64Values.size());

    // Oracle: the same column with no index at all.
    RawFieldFiles raw_files;
    auto raw_segment = CreateSealedSegment(schema);
    raw_segment->LoadFieldData(raw_files.Prepare(int64_fid, {field_data}));

    auto greater_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(int64_fid, DataType::INT64),
        proto::plan::OpType::GreaterThan,
        Int64Literal(45),
        std::vector<proto::plan::GenericValue>());
    auto less_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(int64_fid, DataType::INT64),
        proto::plan::OpType::LessThan,
        Int64Literal(35),
        std::vector<proto::plan::GenericValue>());
    auto in_expr = std::make_shared<expr::TermFilterExpr>(
        expr::ColumnInfo(int64_fid, DataType::INT64),
        std::vector<proto::plan::GenericValue>{
            Int64Literal(30), Int64Literal(90), Int64Literal(15)},
        false);

    auto raw_greater =
        EvaluateFilter(greater_expr, raw_segment.get(), kInt64Values.size());
    ExpectHitRows(raw_greater, {5, 6, 7, 8, 9}, "> 45");
    auto raw_less =
        EvaluateFilter(less_expr, raw_segment.get(), kInt64Values.size());
    ExpectHitRows(raw_less, {0, 1, 2}, "< 35");
    auto raw_in =
        EvaluateFilter(in_expr, raw_segment.get(), kInt64Values.size());
    ExpectHitRows(raw_in, {2, 9}, "in [30, 90, 15]");

    struct IndexIdOrder {
        int64_t stl_sort_id;
        int64_t inverted_id;
    };
    const std::vector<IndexIdOrder> orderings = {
        {3301, 3302},  // STL_SORT is the first capability entry
        {3312, 3311},  // INVERTED is the first capability entry
    };

    for (const auto& ordering : orderings) {
        RawFieldFiles indexed_files;
        auto indexed_segment = CreateSealedSegment(schema);
        indexed_segment->LoadFieldData(
            indexed_files.Prepare(int64_fid, {field_data}));
        InstallIndexWithId(*indexed_segment,
                           int64_fid,
                           DataType::INT64,
                           ordering.stl_sort_id,
                           BuildIndex(int64_fid,
                                      DataType::INT64,
                                      milvus::index::ASCENDING_SORT,
                                      {field_data}));
        InstallIndexWithId(*indexed_segment,
                           int64_fid,
                           DataType::INT64,
                           ordering.inverted_id,
                           BuildIndex(int64_fid,
                                      DataType::INT64,
                                      milvus::index::INVERTED_INDEX_TYPE,
                                      {field_data}));

        // Silently dropping either load has to fail right here.
        const auto capability =
            indexed_segment->ResolveIndexCapability(nullptr, int64_fid);
        ASSERT_EQ(capability.entries().size(), 2u)
            << "STL_SORT and INVERTED must both stay loaded";
        EXPECT_EQ(capability.entries()[0].key.identity.ToString(),
                  std::to_string(
                      std::min(ordering.stl_sort_id, ordering.inverted_id)));
        EXPECT_EQ(capability.entries()[1].key.identity.ToString(),
                  std::to_string(
                      std::max(ordering.stl_sort_id, ordering.inverted_id)));
        // Both are exact predicate readers, so either may serve any of the three
        // operators below.
        for (const auto& entry : capability.entries()) {
            EXPECT_TRUE(entry.caps.predicate);
            EXPECT_TRUE(entry.caps.exact);
        }

        ExpectSameFilterResult(
            raw_greater,
            EvaluateFilter(
                greater_expr, indexed_segment.get(), kInt64Values.size()),
            "> 45 over STL_SORT+INVERTED");
        ExpectSameFilterResult(
            raw_less,
            EvaluateFilter(
                less_expr, indexed_segment.get(), kInt64Values.size()),
            "< 35 over STL_SORT+INVERTED");
        ExpectSameFilterResult(
            raw_in,
            EvaluateFilter(in_expr, indexed_segment.get(), kInt64Values.size()),
            "in [30, 90, 15] over STL_SORT+INVERTED");

        // None of the three may have fallen back to the raw column.
        EXPECT_TRUE(milvus::test::CanExprExecuteAllAtOnce(
            greater_expr, indexed_segment.get(), kInt64Values.size()));
        EXPECT_TRUE(milvus::test::CanExprExecuteAllAtOnce(
            less_expr, indexed_segment.get(), kInt64Values.size()));
        EXPECT_TRUE(milvus::test::CanExprExecuteAllAtOnce(
            in_expr, indexed_segment.get(), kInt64Values.size()));
    }
}

// Case 3: two casts on one JSON path are two different entries (the identity is
// the indexID, not the path), so both load, and the typed selection keeps its
// rule: an exact path plus a compatible cast, never an NGRAM sibling.
TEST(MultiScalarIndexExprTest,
     JsonSamePathTwoCastTypesBothLoadAndBothServeQueries) {
    constexpr int64_t kDoubleCastIndexId = 3401;
    constexpr int64_t kVarcharCastIndexId = 3402;

    auto schema = std::make_shared<Schema>();
    auto json_fid = schema->AddDebugField("json_field", DataType::JSON);
    const std::vector<std::string> json_rows = {
        R"({"s": "alpha", "n": 1})",
        R"({"s": "beta", "n": 2})",
        R"({"s": "alpha", "n": 3})",
        R"({"s": "gamma", "n": 4})",
    };
    auto json_field = milvus::test::expr_index::JsonField(json_rows);

    RawFieldFiles indexed_files;
    auto indexed_segment = CreateSealedSegment(schema);
    indexed_segment->LoadFieldData(
        indexed_files.Prepare(json_fid, {json_field}));
    InstallIndexWithId(
        *indexed_segment,
        json_fid,
        DataType::JSON,
        kDoubleCastIndexId,
        BuildIndex(json_fid,
                   DataType::JSON,
                   milvus::index::INVERTED_INDEX_TYPE,
                   {json_field},
                   {{JSON_PATH, "/n"}, {JSON_CAST_TYPE, "DOUBLE"}}),
        {{JSON_PATH, "/n"}, {JSON_CAST_TYPE, "DOUBLE"}});
    InstallIndexWithId(
        *indexed_segment,
        json_fid,
        DataType::JSON,
        kVarcharCastIndexId,
        BuildIndex(json_fid,
                   DataType::JSON,
                   milvus::index::INVERTED_INDEX_TYPE,
                   {json_field},
                   {{JSON_PATH, "/s"}, {JSON_CAST_TYPE, "VARCHAR"}}),
        {{JSON_PATH, "/s"}, {JSON_CAST_TYPE, "VARCHAR"}});

    const auto capability =
        indexed_segment->ResolveIndexCapability(nullptr, json_fid);
    ASSERT_EQ(capability.entries().size(), 2u)
        << "two casts on one field are two identities (different indexIDs)";
    EXPECT_EQ(capability.entries()[0].json_path, "/n");
    EXPECT_EQ(capability.entries()[1].json_path, "/s");

    proto::plan::GenericValue equal_value;
    equal_value.set_string_val("alpha");
    auto equal_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(
            json_fid, DataType::JSON, std::vector<std::string>{"s"}),
        proto::plan::OpType::Equal,
        equal_value);
    auto plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, equal_expr);
    auto result = milvus::test::gen_filter_res(
        plan.get(), indexed_segment.get(), json_rows.size(), MAX_TIMESTAMP);
    ExpectHitRows(result, {0, 2}, "JSON /s == \"alpha\"");
    // The VARCHAR cast at path /s served it; the DOUBLE cast at /n cannot.
    EXPECT_TRUE(milvus::test::CanExprExecuteAllAtOnce(
        equal_expr, indexed_segment.get(), json_rows.size()));
}

// Case 4: a flat JSON index over the whole document and a typed index on one
// path both cover the same query. The typed index on the exact path wins for its
// own path; a path only the flat one covers still executes on an index. Neither
// selection may fall back to the raw column, and neither may borrow an NGRAM
// entry, which is why this case also loads one on the same field.
TEST(MultiScalarIndexExprTest, JsonTypedPathBeatsFlatRouterOnTheSamePath) {
    constexpr int64_t kFlatIndexId = 3501;
    constexpr int64_t kTypedIndexId = 3502;
    constexpr int64_t kNgramIndexId = 3503;

    auto schema = std::make_shared<Schema>();
    auto json_fid = schema->AddDebugField("json_field", DataType::JSON);
    const std::vector<std::string> json_rows = {
        R"({"a": {"b": 1, "c": 3}})",
        R"({"a": {"b": 2, "c": 4}})",
        R"({"a": {"c": 3}})",
        R"({"a": {"b": 1, "c": 5}})",
    };
    auto json_field = milvus::test::expr_index::JsonField(json_rows);

    RawFieldFiles raw_files;
    auto raw_segment = CreateSealedSegment(schema);
    raw_segment->LoadFieldData(raw_files.Prepare(json_fid, {json_field}));

    RawFieldFiles indexed_files;
    auto indexed_segment = CreateSealedSegment(schema);
    indexed_segment->LoadFieldData(
        indexed_files.Prepare(json_fid, {json_field}));
    // The flat index routes every path of the document.
    InstallIndexWithId(*indexed_segment,
                       json_fid,
                       DataType::JSON,
                       kFlatIndexId,
                       BuildIndex(json_fid,
                                  DataType::JSON,
                                  milvus::index::INVERTED_INDEX_TYPE,
                                  {json_field},
                                  {{JSON_PATH, ""}, {JSON_CAST_TYPE, "JSON"}}),
                       {{JSON_PATH, ""}, {JSON_CAST_TYPE, "JSON"}});
    // The typed index answers exactly one leaf of it.
    InstallIndexWithId(
        *indexed_segment,
        json_fid,
        DataType::JSON,
        kTypedIndexId,
        BuildIndex(json_fid,
                   DataType::JSON,
                   milvus::index::INVERTED_INDEX_TYPE,
                   {json_field},
                   {{JSON_PATH, "/a/b"}, {JSON_CAST_TYPE, "DOUBLE"}}),
        {{JSON_PATH, "/a/b"}, {JSON_CAST_TYPE, "DOUBLE"}});
    // Candidate-only sibling: it must never be selected for an exact operator,
    // even though its indexID is the largest of the three.
    InstallIndexWithId(*indexed_segment,
                       json_fid,
                       DataType::JSON,
                       kNgramIndexId,
                       BuildIndex(json_fid,
                                  DataType::JSON,
                                  milvus::index::NGRAM_INDEX_TYPE,
                                  {json_field},
                                  {{JSON_PATH, "/a/b"},
                                   {JSON_CAST_TYPE, "VARCHAR"},
                                   {milvus::index::MIN_GRAM, 2},
                                   {milvus::index::MAX_GRAM, 4}}),
                       {{JSON_PATH, "/a/b"}, {JSON_CAST_TYPE, "VARCHAR"}});

    const auto capability =
        indexed_segment->ResolveIndexCapability(nullptr, json_fid);
    ASSERT_EQ(capability.entries().size(), 3u);
    // Capability metadata is ordered by index identity, not by load order.
    EXPECT_EQ(capability.entries()[0].key.identity.ToString(),
              std::to_string(kFlatIndexId));
    EXPECT_EQ(capability.entries()[1].key.identity.ToString(),
              std::to_string(kTypedIndexId));
    EXPECT_EQ(capability.entries()[2].key.identity.ToString(),
              std::to_string(kNgramIndexId));
    EXPECT_TRUE(capability.entries()[0].caps.json_paths);
    EXPECT_EQ(capability.entries()[0].json_path, "");
    EXPECT_EQ(capability.entries()[1].json_path, "/a/b");
    EXPECT_TRUE(capability.entries()[2].caps.ngram_candidates);
    EXPECT_FALSE(capability.entries()[2].caps.predicate);

    auto make_equal = [&](std::vector<std::string> path, int64_t value) {
        return std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(json_fid, DataType::JSON, std::move(path)),
            proto::plan::OpType::Equal,
            Int64Literal(value));
    };

    // /a/b is covered by the typed index and by the flat router; the exact path
    // wins, and the NGRAM entry is never a candidate for equality.
    auto typed_expr = make_equal({"a", "b"}, 1);
    auto typed_plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, typed_expr);
    auto typed_result = milvus::test::gen_filter_res(typed_plan.get(),
                                                     indexed_segment.get(),
                                                     json_rows.size(),
                                                     MAX_TIMESTAMP);
    ExpectHitRows(typed_result, {0, 3}, "JSON /a/b == 1");
    EXPECT_TRUE(milvus::test::CanExprExecuteAllAtOnce(
        typed_expr, indexed_segment.get(), json_rows.size()));

    // /a/c is only covered by the flat router, so that is the one that serves it.
    auto flat_expr = make_equal({"a", "c"}, 3);
    auto flat_plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, flat_expr);
    auto flat_result = milvus::test::gen_filter_res(flat_plan.get(),
                                                    indexed_segment.get(),
                                                    json_rows.size(),
                                                    MAX_TIMESTAMP);
    ExpectHitRows(flat_result, {0, 2}, "JSON /a/c == 3");
    EXPECT_TRUE(milvus::test::CanExprExecuteAllAtOnce(
        flat_expr, indexed_segment.get(), json_rows.size()));

    // The same predicates over the raw column must agree: selection changes the
    // path, never the answer.
    auto raw_typed_expr = make_equal({"a", "b"}, 1);
    auto raw_typed_plan = std::make_shared<plan::FilterBitsNode>(
        DEFAULT_PLANNODE_ID, raw_typed_expr);
    ExpectSameFilterResult(milvus::test::gen_filter_res(raw_typed_plan.get(),
                                                        raw_segment.get(),
                                                        json_rows.size(),
                                                        MAX_TIMESTAMP),
                           typed_result,
                           "JSON /a/b == 1 against the raw column");
}

// Selection rejects an index that cannot serve this literal BEFORE pinning, so a
// pattern query with a literal shorter than the index's min_gram never pins at all.
TEST(MultiScalarIndexExprTest, NgramTooShortLiteralRejectedBeforePin) {
    constexpr FieldId kField(100);

    milvus::segcore::IndexCapabilityEntry entry{milvus::segcore::IndexKey{
        kField, milvus::segcore::IndexIdentity::PreBuiltIndex(5001)}};
    entry.family = "ngram";
    entry.value_type = DataType::VARCHAR;
    entry.caps.ngram_candidates = true;
    entry.caps.exact = false;
    entry.params.ngram.min_gram = 3;
    entry.params.ngram.max_gram = 5;

    milvus::exec::ExprIndexRequirement req;
    req.field_id = kField;
    req.reader = milvus::exec::RequiredReader::Ngram;
    req.has_pattern = true;
    req.pattern = "%ab%";
    req.pattern_op = milvus::index::PatternOp::Match;

    const milvus::segcore::FieldIndexCapability one_field(kField, {entry});
    const auto short_literal = milvus::exec::DetermineExecPath(req, one_field);
    EXPECT_EQ(short_literal.path, milvus::exec::ExprExecPath::RawData)
        << "min_gram 3 cannot answer a 2-character literal; selection must say "
           "so";
    ASSERT_NE(short_literal.fallback_reason,
              milvus::exec::DeclineReason::kNone);
    EXPECT_STREQ(milvus::exec::ToString(short_literal.fallback_reason),
                 "literal_not_supported");

    req.pattern = "%abcd%";
    const auto long_literal = milvus::exec::DetermineExecPath(req, one_field);
    EXPECT_EQ(long_literal.path, milvus::exec::ExprExecPath::ScalarIndex);
    ASSERT_TRUE(long_literal.key.has_value());
    EXPECT_EQ(long_literal.key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(5001));
}

// A query hint outranks the rule-based score: the named index is used even when
// another one scores better, and the hint's own order decides among several.
TEST(MultiScalarIndexExprTest, HintOutranksScore) {
    constexpr FieldId kField(100);
    const auto make_entry = [](int64_t id, const char* family) {
        milvus::segcore::IndexCapabilityEntry entry{milvus::segcore::IndexKey{
            kField, milvus::segcore::IndexIdentity::PreBuiltIndex(id)}};
        entry.family = family;
        entry.value_type = DataType::INT64;
        entry.caps.predicate = true;
        entry.caps.exact = true;
        return entry;
    };
    // BITMAP scores higher than INVERTED on equality, so without a hint it wins.
    milvus::segcore::FieldIndexCapability caps(
        kField, {make_entry(6001, "inverted"), make_entry(6002, "bitmap")});

    milvus::exec::ExprIndexRequirement req;
    req.field_id = kField;
    req.reader = milvus::exec::RequiredReader::Predicate;
    req.value_type = DataType::INT64;
    EXPECT_EQ(milvus::exec::DetermineExecPath(req, caps).key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(6002));

    // The hint names the fewer-scoring index: it must win.
    const std::vector<std::pair<int64_t, int64_t>> hint{{kField.get(), 6001}};
    req.index_hints = &hint;
    EXPECT_EQ(milvus::exec::DetermineExecPath(req, caps).key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(6001));

    // Hints of another field leave this field's selection untouched.
    const std::vector<std::pair<int64_t, int64_t>> other_field{{999, 6001}};
    req.index_hints = &other_field;
    EXPECT_EQ(milvus::exec::DetermineExecPath(req, caps).key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(6002));
}

// strict: a hint that names this field's indexes but cannot be used fails loudly
// instead of silently selecting another index.
TEST(MultiScalarIndexExprTest, StrictHintFailsWhenUnusable) {
    constexpr FieldId kField(100);
    milvus::segcore::IndexCapabilityEntry entry{milvus::segcore::IndexKey{
        kField, milvus::segcore::IndexIdentity::PreBuiltIndex(6101)}};
    entry.family = "inverted";
    entry.value_type =
        DataType::VARCHAR;  // does not match the requirement below
    entry.caps.predicate = true;
    entry.caps.exact = true;
    milvus::segcore::FieldIndexCapability caps(kField, {entry});

    milvus::exec::ExprIndexRequirement req;
    req.field_id = kField;
    req.reader = milvus::exec::RequiredReader::Predicate;
    req.value_type = DataType::INT64;
    const std::vector<std::pair<int64_t, int64_t>> hint{{kField.get(), 6101}};
    req.index_hints = &hint;

    // Without strict, the field simply falls back to raw.
    EXPECT_EQ(milvus::exec::DetermineExecPath(req, caps).path,
              milvus::exec::ExprExecPath::RawData);

    req.index_hint_strict = true;
    EXPECT_THROW(milvus::exec::DetermineExecPath(req, caps),
                 milvus::SegcoreError);
}

// The hint's own order decides between two eligible indexes: family priority and
// score only order the candidates nobody asked for.
TEST(MultiScalarIndexExprTest, HintOrderOutranksFamilyPriority) {
    constexpr FieldId kField(100);
    auto make_entry = [&](int64_t index_id, const char* family) {
        milvus::segcore::IndexCapabilityEntry entry{milvus::segcore::IndexKey{
            kField, milvus::segcore::IndexIdentity::PreBuiltIndex(index_id)}};
        entry.family = family;
        entry.value_type = DataType::INT64;
        entry.caps.predicate = true;
        entry.caps.exact = true;
        return entry;
    };
    // BITMAP (family rank 3 for equality) is listed first and would win on the
    // rule-based score; the hint asks for INVERTED first.
    milvus::segcore::FieldIndexCapability caps(
        kField, {make_entry(6101, "bitmap"), make_entry(6102, "inverted")});

    milvus::exec::ExprIndexRequirement req;
    req.field_id = kField;
    req.reader = milvus::exec::RequiredReader::Predicate;
    req.value_type = DataType::INT64;
    const std::vector<std::pair<int64_t, int64_t>> hint{{kField.get(), 6102},
                                                        {kField.get(), 6101}};
    req.index_hints = &hint;

    auto decision = milvus::exec::DetermineExecPath(req, caps);
    ASSERT_TRUE(decision.key.has_value());
    EXPECT_EQ(decision.key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(6102))
        << "the hinted order must win over the family priority";
}

// A strict hint is not violated by a field that has no index entry at all: a
// growing segment (or one whose index is not loaded yet) still answers from raw
// data, and failing the whole query there would make strict hints unusable on any
// collection that takes writes.
TEST(MultiScalarIndexExprTest, StrictHintAllowsRawWhenFieldHasNoEntry) {
    constexpr FieldId kField(100);
    milvus::segcore::FieldIndexCapability caps(kField, {});

    milvus::exec::ExprIndexRequirement req;
    req.field_id = kField;
    req.reader = milvus::exec::RequiredReader::Predicate;
    req.value_type = DataType::INT64;
    const std::vector<std::pair<int64_t, int64_t>> hint{{kField.get(), 6101}};
    req.index_hints = &hint;
    req.index_hint_strict = true;

    auto decision = milvus::exec::DetermineExecPath(req, caps);
    EXPECT_EQ(decision.path, milvus::exec::ExprExecPath::RawData);
}

// A hint naming an index on j/a must not govern the j/b leaf: the same JSON field
// feeds every path predicate, and only the hinted path was asked for.
TEST(MultiScalarIndexExprTest, StrictHintIsScopedToItsJsonPath) {
    constexpr FieldId kField(100);
    const std::string path_a = "/a";
    const std::string path_b = "/b";
    auto make_entry = [&](int64_t index_id, const std::string& path) {
        milvus::segcore::IndexCapabilityEntry entry{milvus::segcore::IndexKey{
            kField, milvus::segcore::IndexIdentity::PreBuiltIndex(index_id)}};
        entry.family = "inverted";
        entry.json_path = path;
        entry.json_cast_type = JsonCastType::FromString("DOUBLE");
        entry.value_type = DataType::DOUBLE;
        entry.caps.predicate = true;
        entry.caps.exact = true;
        return entry;
    };
    milvus::segcore::FieldIndexCapability caps(
        kField, {make_entry(6101, path_a), make_entry(6102, path_b)});

    const std::vector<std::pair<int64_t, int64_t>> hint{{kField.get(), 6101}};

    milvus::exec::ExprIndexRequirement req_a;
    req_a.field_id = kField;
    req_a.reader = milvus::exec::RequiredReader::Predicate;
    req_a.value_type = DataType::DOUBLE;
    req_a.is_json_field = true;
    req_a.json_path = path_a;
    req_a.index_hints = &hint;
    req_a.index_hint_strict = true;
    EXPECT_EQ(milvus::exec::DetermineExecPath(req_a, caps).path,
              milvus::exec::ExprExecPath::ScalarIndex);

    milvus::exec::ExprIndexRequirement req_b = req_a;
    req_b.json_path = path_b;
    // The j/a hint does not govern the j/b leaf: no error, and the selection is
    // free to use the index that actually covers j/b.
    auto decision_b = milvus::exec::DetermineExecPath(req_b, caps);
    EXPECT_EQ(decision_b.path, milvus::exec::ExprExecPath::ScalarIndex);
    ASSERT_TRUE(decision_b.key.has_value());
    EXPECT_EQ(decision_b.key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(6102));
}

// Copies a protobuf string map into a std::map so assertions read it with
// ordinary lookups.
std::map<std::string, int64_t>
CountersOf(const proto::internal::ExecutionExplain& node) {
    std::map<std::string, int64_t> out;
    for (const auto& [key, value] : node.counters()) {
        out[key] = value;
    }
    return out;
}

std::map<std::string, std::string>
AttributesOf(const proto::internal::ExecutionExplain& node) {
    std::map<std::string, std::string> out;
    for (const auto& [key, value] : node.attributes()) {
        out[key] = value;
    }
    return out;
}

// The explain payload answers "what did this filter do, per leaf": which index
// served it, or why the raw column did. One field, one filter, two segments --
// with and without an index -- so the two decision groups can be compared.
TEST(MultiScalarIndexExprTest, ExplainRecordsIndexAndRawDecisions) {
    constexpr int64_t kIndexId = 4201;

    auto schema = std::make_shared<Schema>();
    auto varchar_fid =
        schema->AddDebugField("varchar_field", DataType::VARCHAR);
    auto field_data = milvus::test::expr_index::StringField(kVarCharValues);

    proto::plan::GenericValue equal_value;
    equal_value.set_string_val("prefix_beta");
    auto equal_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::Equal,
        equal_value,
        std::vector<proto::plan::GenericValue>());
    equal_expr->SetExprId(43001);

    // No index: the leaf falls back to the raw column and says why.
    {
        RawFieldFiles files;
        auto segment = CreateSealedSegment(schema);
        segment->LoadFieldData(files.Prepare(varchar_fid, {field_data}));

        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            kVarCharValues.size());
        auto result = EvaluateFilter(
            equal_expr, segment.get(), kVarCharValues.size(), explain);
        ExpectHitRows(result, {2, 6}, "Equal \"prefix_beta\" without index");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        EXPECT_EQ(node.kind(), "segment");
        const auto counters = CountersOf(node);
        EXPECT_EQ(counters.at("rows_total"),
                  static_cast<int64_t>(kVarCharValues.size()));
        EXPECT_EQ(counters.at("idx_scan"), 0)
            << "a raw leaf handles nothing through an index";
        EXPECT_EQ(counters.at("rows_out"), 2);
        ASSERT_EQ(node.children_size(), 1);
        const auto& group = node.children(0);
        EXPECT_EQ(group.kind(), "filter_group");
        const auto attributes = AttributesOf(group);
        EXPECT_EQ(attributes.at("field_id"), std::to_string(varchar_fid.get()));
        EXPECT_EQ(attributes.at("decline_reason"), "no_index_on_field");
        const auto group_counters = CountersOf(group);
        EXPECT_EQ(group_counters.at("rows_in"),
                  static_cast<int64_t>(kVarCharValues.size()));
        EXPECT_EQ(group_counters.at("raw_scan"),
                  static_cast<int64_t>(kVarCharValues.size()))
            << "the raw leaf read every row once";
        EXPECT_EQ(group_counters.at("idx_scan"), 0);
        EXPECT_EQ(group_counters.at("rows_out"), 2);
    }

    // INVERTED serves the leaf: the group names the index and counts its rows.
    {
        RawFieldFiles files;
        auto segment = CreateSealedSegment(schema);
        segment->LoadFieldData(files.Prepare(varchar_fid, {field_data}));
        InstallIndexWithId(*segment,
                           varchar_fid,
                           DataType::VARCHAR,
                           kIndexId,
                           BuildIndex(varchar_fid,
                                      DataType::VARCHAR,
                                      milvus::index::INVERTED_INDEX_TYPE,
                                      {field_data}));

        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            kVarCharValues.size());
        auto result = EvaluateFilter(
            equal_expr, segment.get(), kVarCharValues.size(), explain);
        ExpectHitRows(result, {2, 6}, "Equal \"prefix_beta\" over INVERTED");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        EXPECT_EQ(CountersOf(node).at("rows_out"), 2);
        EXPECT_EQ(CountersOf(node).at("idx_scan"),
                  static_cast<int64_t>(kVarCharValues.size()));
        ASSERT_EQ(node.children_size(), 1);
        const auto& group = node.children(0);
        const auto attributes = AttributesOf(group);
        EXPECT_EQ(attributes.at("index_id"), std::to_string(kIndexId));
        EXPECT_EQ(attributes.find("decline_reason"), attributes.end())
            << "an index leaf declined nothing";
        const auto group_counters = CountersOf(group);
        EXPECT_EQ(group_counters.at("rows_in"),
                  static_cast<int64_t>(kVarCharValues.size()));
        EXPECT_EQ(group_counters.at("idx_scan"),
                  static_cast<int64_t>(kVarCharValues.size()))
            << "the index handled every input row";
        EXPECT_EQ(group_counters.at("raw_scan"), 0)
            << "an exact index leaf reads no raw rows";
        EXPECT_EQ(group_counters.at("rows_out"), 2);
    }
}

// A pattern leaf tries the NGRAM pass first. With two hinted NGRAM indexes on
// one field, the first one may reject the literal (min_gram too long) while the
// second can serve it; the hint's order must not be read as "only the first
// hinted index counts", or the leaf drops a usable index and a strict hint
// fails the query.
TEST(MultiScalarIndexExprTest, HintedSecondNgramServesPattern) {
    constexpr int64_t kLongGramIndexId = 4301;
    constexpr int64_t kShortGramIndexId = 4302;

    auto schema = std::make_shared<Schema>();
    auto varchar_fid =
        schema->AddDebugField("varchar_field", DataType::VARCHAR);
    const std::vector<std::string> values = {
        "abc", "abd", "xab", "zzz", "abcd", "abc"};
    auto field_data = milvus::test::expr_index::StringField(values);

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(varchar_fid, {field_data}));
    // min_gram 3 cannot answer a two-character literal; min_gram 2 can.
    InstallIndexWithId(*segment,
                       varchar_fid,
                       DataType::VARCHAR,
                       kLongGramIndexId,
                       BuildIndex(varchar_fid,
                                  DataType::VARCHAR,
                                  milvus::index::NGRAM_INDEX_TYPE,
                                  {field_data},
                                  {{milvus::index::MIN_GRAM, 3},
                                   {milvus::index::MAX_GRAM, 4}}));
    InstallIndexWithId(*segment,
                       varchar_fid,
                       DataType::VARCHAR,
                       kShortGramIndexId,
                       BuildIndex(varchar_fid,
                                  DataType::VARCHAR,
                                  milvus::index::NGRAM_INDEX_TYPE,
                                  {field_data},
                                  {{milvus::index::MIN_GRAM, 2},
                                   {milvus::index::MAX_GRAM, 3}}));

    proto::plan::GenericValue prefix_value;
    prefix_value.set_string_val("ab");
    auto like_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::PrefixMatch,
        prefix_value,
        std::vector<proto::plan::GenericValue>());

    milvus::query::PlanOptions options;
    options.index_hints = {{varchar_fid.get(), kLongGramIndexId},
                           {varchar_fid.get(), kShortGramIndexId}};
    options.index_hint_strict = true;

    auto explain =
        std::make_shared<milvus::exec::ExecutionExplainRecorder>(values.size());
    ColumnVectorPtr result;
    ASSERT_NO_THROW(
        result = EvaluateFilter(
            like_expr, segment.get(), values.size(), explain, options))
        << "the second hinted ngram index can serve the literal";

    ExpectHitRows(
        result, {0, 1, 4, 5}, "LIKE 'ab%' over two hinted ngram indexes");

    proto::internal::ExecutionExplain node;
    ASSERT_TRUE(node.ParseFromString(explain->Finish()));
    ASSERT_EQ(node.children_size(), 1);
    const auto attributes = AttributesOf(node.children(0));
    ASSERT_TRUE(attributes.count("index_id") == 1)
        << "the leaf served the pattern from an index";
    EXPECT_EQ(attributes.at("index_id"), std::to_string(kShortGramIndexId))
        << "the index that can handle a two-character literal must serve it";
}

// A conjunction evaluates its right leaf only on the rows the left leaf left
// alive. The right leaf's group must report the rows it really processed, not
// one batch per batch of the whole filter.
TEST(MultiScalarIndexExprTest, ExplainSkipsShortCircuitedLeaf) {
    auto schema = std::make_shared<Schema>();
    auto left_fid = schema->AddDebugField("left_varchar", DataType::VARCHAR);
    auto right_fid = schema->AddDebugField("right_varchar", DataType::VARCHAR);
    auto field_data = milvus::test::expr_index::StringField(kVarCharValues);

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(left_fid, {field_data}));
    segment->LoadFieldData(files.Prepare(right_fid, {field_data}));

    // The left leaf matches nothing, so the right leaf has no row to evaluate.
    proto::plan::GenericValue never_value;
    never_value.set_string_val("no-such-prefix");
    auto left = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(left_fid, DataType::VARCHAR),
        proto::plan::OpType::PrefixMatch,
        never_value,
        std::vector<proto::plan::GenericValue>());
    left->SetExprId(45001);
    proto::plan::GenericValue prefix_value;
    prefix_value.set_string_val("prefix");
    auto right = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(right_fid, DataType::VARCHAR),
        proto::plan::OpType::PrefixMatch,
        prefix_value,
        std::vector<proto::plan::GenericValue>());
    right->SetExprId(45002);
    auto both = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, left, right);

    auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
        kVarCharValues.size());
    auto result =
        EvaluateFilter(both, segment.get(), kVarCharValues.size(), explain);
    ExpectHitRows(result, {}, "AND that matches nothing");

    proto::internal::ExecutionExplain node;
    ASSERT_TRUE(node.ParseFromString(explain->Finish()));
    ASSERT_EQ(node.children_size(), 2);
    bool saw_left = false;
    bool saw_right = false;
    for (int i = 0; i < node.children_size(); ++i) {
        const auto attributes = AttributesOf(node.children(i));
        const auto counters = CountersOf(node.children(i));
        if (attributes.at("field_id") == std::to_string(left_fid.get())) {
            saw_left = true;
            EXPECT_EQ(attributes.count("skipped"), 0u);
            EXPECT_EQ(counters.at("rows_in"),
                      static_cast<int64_t>(kVarCharValues.size()))
                << "the left leaf evaluated every row";
            EXPECT_EQ(counters.at("raw_scan"),
                      static_cast<int64_t>(kVarCharValues.size()));
            EXPECT_EQ(counters.at("rows_out"), 0);
        } else {
            EXPECT_EQ(attributes.at("field_id"),
                      std::to_string(right_fid.get()));
            saw_right = true;
            EXPECT_EQ(attributes.at("skipped"), "true");
            EXPECT_EQ(counters.at("rows_in"), 0)
                << "the right leaf was handed no row to evaluate";
            EXPECT_EQ(counters.at("raw_scan"), 0);
            EXPECT_EQ(counters.at("rows_out"), 0);
        }
    }
    EXPECT_TRUE(saw_left && saw_right);
}

// A hint is scoped to the JSON path it names. With an NGRAM index on j/a and
// another on j/b, hinting the j/a index (even in strict mode) must not steer the
// j/b leaf away from its own index: the two-pass NGRAM/PatternMatch logic has to
// apply the same path check the selector does.
TEST(MultiScalarIndexExprTest, HintOnOneJsonPathKeepsOtherPathNgram) {
    constexpr int64_t kPathAIndexId = 4401;
    constexpr int64_t kPathBIndexId = 4402;

    auto schema = std::make_shared<Schema>();
    auto json_fid = schema->AddDebugField("json_field", DataType::JSON);
    std::vector<std::string> rows = {
        R"({"a": "zzzabc", "b": "abcde"})",
        R"({"a": "abc", "b": "zzz"})",
        R"({"a": "q", "b": "xabcx"})",
        R"({"a": "abcabc", "b": "abc"})",
    };
    auto field_data = milvus::test::expr_index::JsonField(rows);

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(json_fid, {field_data}));
    InstallIndexWithId(*segment,
                       json_fid,
                       DataType::JSON,
                       kPathAIndexId,
                       BuildIndex(json_fid,
                                  DataType::JSON,
                                  milvus::index::NGRAM_INDEX_TYPE,
                                  {field_data},
                                  {{JSON_PATH, "/a"},
                                   {JSON_CAST_TYPE, "VARCHAR"},
                                   {milvus::index::MIN_GRAM, 2},
                                   {milvus::index::MAX_GRAM, 4}}),
                       {{JSON_PATH, "/a"}, {JSON_CAST_TYPE, "VARCHAR"}});
    InstallIndexWithId(*segment,
                       json_fid,
                       DataType::JSON,
                       kPathBIndexId,
                       BuildIndex(json_fid,
                                  DataType::JSON,
                                  milvus::index::NGRAM_INDEX_TYPE,
                                  {field_data},
                                  {{JSON_PATH, "/b"},
                                   {JSON_CAST_TYPE, "VARCHAR"},
                                   {milvus::index::MIN_GRAM, 2},
                                   {milvus::index::MAX_GRAM, 4}}),
                       {{JSON_PATH, "/b"}, {JSON_CAST_TYPE, "VARCHAR"}});

    proto::plan::GenericValue inner_value;
    inner_value.set_string_val("abc");
    auto expr_b = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(json_fid, DataType::JSON, {"b"}),
        proto::plan::OpType::InnerMatch,
        inner_value,
        std::vector<proto::plan::GenericValue>());

    // The hint names the j/a index, in strict mode, and must neither fail the
    // query nor stop the j/b index from serving its own path.
    milvus::query::PlanOptions options;
    options.index_hints = {{json_fid.get(), kPathAIndexId}};
    options.index_hint_strict = true;

    auto explain =
        std::make_shared<milvus::exec::ExecutionExplainRecorder>(rows.size());
    auto result =
        EvaluateFilter(expr_b, segment.get(), rows.size(), explain, options);
    ExpectHitRows(result, {0, 2, 3}, "JSON j[\"b\"] like \"%abc%\"");
    EXPECT_TRUE(milvus::test::CanExprExecuteAllAtOnce(
        expr_b, segment.get(), rows.size()))
        << "the j/b ngram index must still serve its own path";

    proto::internal::ExecutionExplain node;
    ASSERT_TRUE(node.ParseFromString(explain->Finish()));
    ASSERT_EQ(node.children_size(), 1);
    const auto attributes = AttributesOf(node.children(0));
    const auto selected = attributes.find("index_id");
    ASSERT_NE(selected, attributes.end())
        << "the j/b ngram index must serve its own path";
    EXPECT_EQ(selected->second, std::to_string(kPathBIndexId));
}

// Two leaves on the same field can make the same decision. The plan assigns
// them different expr_ids, and the explain must keep them apart: the proxy folds
// decisions by expr_id to rebuild each condition, so merging them here would
// lose which leaf scanned what.
TEST(MultiScalarIndexExprTest, ExplainKeepsLeavesWithSameDecisionApart) {
    auto schema = std::make_shared<Schema>();
    auto int_fid = schema->AddDebugField("int64_field", DataType::INT64);

    std::vector<int64_t> values = {10, 20, 30, 40, 40, 50, 60, 70, 80, 90};
    auto int_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    int_field->FillFieldData(values.data(), values.size());

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(int_fid, {int_field}));
    InstallIndexWithId(*segment,
                       int_fid,
                       DataType::INT64,
                       7101,
                       BuildIndex(int_fid,
                                  DataType::INT64,
                                  milvus::index::ASCENDING_SORT,
                                  {int_field}));

    auto lower = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(int_fid, DataType::INT64),
        proto::plan::OpType::GreaterThan,
        Int64Literal(20),
        std::vector<proto::plan::GenericValue>());
    lower->SetExprId(71001);
    auto upper = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(int_fid, DataType::INT64),
        proto::plan::OpType::LessThan,
        Int64Literal(60),
        std::vector<proto::plan::GenericValue>());
    upper->SetExprId(71002);
    auto both = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, lower, upper);

    auto explain =
        std::make_shared<milvus::exec::ExecutionExplainRecorder>(values.size());
    auto result = EvaluateFilter(both, segment.get(), values.size(), explain);
    ExpectHitRows(result, {2, 3, 4, 5}, "int64 in (20, 60)");

    proto::internal::ExecutionExplain node;
    ASSERT_TRUE(node.ParseFromString(explain->Finish()));
    ASSERT_EQ(node.children_size(), 2)
        << "each expr_id keeps its own filter group";
    EXPECT_EQ(AttributesOf(node.children(0)).at("expr_id"), "71001");
    EXPECT_EQ(AttributesOf(node.children(1)).at("expr_id"), "71002");
    for (int i = 0; i < node.children_size(); ++i) {
        const auto attributes = AttributesOf(node.children(i));
        EXPECT_EQ(attributes.at("field_id"), std::to_string(int_fid.get()));
        EXPECT_EQ(attributes.at("family"), "sort");
    }
}

// Non-exact NGRAM LIKE leaves still fuse into PhyLikeConjunctExpr when explain
// is active, while keeping per-leaf (rows_in, idx_scan, raw_scan, rows_out)
// and sequential eval_order accurate.
TEST(MultiScalarIndexExprTest, ExplainFusedNonExactNgramLikes) {
    constexpr int64_t kNgramIndexId = 7201;

    auto schema = std::make_shared<Schema>();
    auto varchar_fid =
        schema->AddDebugField("varchar_field", DataType::VARCHAR);
    const std::vector<std::string> values = {
        /* 0 */ "11abcd22wxyz33",
        /* 1 */ "11abczbcd22wxyz33",
        /* 2 */ "11abcd22wxyaxyz33",
        /* 3 */ "11abcd22none33",
        /* 4 */ "11none22wxyz33",
        /* 5 */ "none",
    };
    auto field_data = milvus::test::expr_index::StringField(values);

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(varchar_fid, {field_data}));
    InstallIndexWithId(*segment,
                       varchar_fid,
                       DataType::VARCHAR,
                       kNgramIndexId,
                       BuildIndex(varchar_fid,
                                  DataType::VARCHAR,
                                  milvus::index::NGRAM_INDEX_TYPE,
                                  {field_data},
                                  {{milvus::index::MIN_GRAM, 2},
                                   {milvus::index::MAX_GRAM, 3}}));

    proto::plan::GenericValue v1;
    v1.set_string_val("abcd");
    auto like1 = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::InnerMatch,
        v1,
        std::vector<proto::plan::GenericValue>());
    like1->SetExprId(72001);

    proto::plan::GenericValue v2;
    v2.set_string_val("wxyz");
    auto like2 = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::InnerMatch,
        v2,
        std::vector<proto::plan::GenericValue>());
    like2->SetExprId(72002);

    auto both = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, like1, like2);

    auto explain =
        std::make_shared<milvus::exec::ExecutionExplainRecorder>(values.size());
    auto result = EvaluateFilter(both, segment.get(), values.size(), explain);
    ExpectHitRows(result, {0}, "fused non-exact LIKE AND");

    proto::internal::ExecutionExplain node;
    ASSERT_TRUE(node.ParseFromString(explain->Finish()));
    ASSERT_EQ(node.children_size(), 2);

    std::map<std::string, const proto::internal::ExecutionExplain*> by_id;
    for (int i = 0; i < node.children_size(); ++i) {
        const auto& child = node.children(i);
        by_id[AttributesOf(child).at("expr_id")] = &child;
    }
    ASSERT_EQ(by_id.count("72001"), 1);
    ASSERT_EQ(by_id.count("72002"), 1);

    const auto attrs1 = AttributesOf(*by_id.at("72001"));
    const auto ctrs1 = CountersOf(*by_id.at("72001"));
    EXPECT_EQ(attrs1.at("family"), "ngram");
    EXPECT_EQ(attrs1.at("index_id"), std::to_string(kNgramIndexId));
    EXPECT_EQ(attrs1.at("eval_order"), "1");
    EXPECT_EQ(ctrs1.at("rows_in"), 6);
    EXPECT_EQ(ctrs1.at("idx_scan"), 6);
    EXPECT_EQ(ctrs1.at("raw_scan"), 3);
    EXPECT_LE(ctrs1.at("raw_scan"), ctrs1.at("rows_in"));
    EXPECT_EQ(ctrs1.at("rows_out"), 3);

    const auto attrs2 = AttributesOf(*by_id.at("72002"));
    const auto ctrs2 = CountersOf(*by_id.at("72002"));
    EXPECT_EQ(attrs2.at("family"), "ngram");
    EXPECT_EQ(attrs2.at("index_id"), std::to_string(kNgramIndexId));
    EXPECT_EQ(attrs2.at("eval_order"), "2");
    EXPECT_EQ(ctrs2.at("rows_in"), 3);
    EXPECT_EQ(ctrs2.at("idx_scan"), 3);
    EXPECT_EQ(ctrs2.at("raw_scan"), 2);
    EXPECT_LE(ctrs2.at("raw_scan"), ctrs2.at("rows_in"));
    EXPECT_EQ(ctrs2.at("rows_out"), 1);
}

// GIS coarse/refine split-fusion preserves every original GIS leaf's expr_id
// and attributes both R-Tree and raw paths (including refinement false
// positives) without emitting spurious internal nodes.
TEST(MultiScalarIndexExprTest, ExplainGISSplitFusionIndexedAndRaw) {
    struct LocalGISSplitFusionGuard {
        bool saved;
        LocalGISSplitFusionGuard()
            : saved(SegcoreConfig::default_config()
                        .get_enable_gis_split_fusion()) {
            SegcoreConfig::default_config().set_enable_gis_split_fusion(true);
        }
        ~LocalGISSplitFusionGuard() {
            SegcoreConfig::default_config().set_enable_gis_split_fusion(saved);
        }
    } split_guard;

    auto schema = std::make_shared<Schema>();
    auto int_fid = schema->AddDebugField("int64_field", DataType::INT64);
    auto geo_fid = schema->AddDebugField("geo", DataType::GEOMETRY);

    const std::vector<int64_t> int_vals = {10, 20, 30, 40, 50, 60};
    auto int_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    int_field->FillFieldData(int_vals.data(), int_vals.size());

    auto make_wkb = [](const char* wkt) {
        return Geometry(GetThreadLocalGEOSContext(), wkt).to_wkb_string();
    };
    const std::vector<std::string> wkbs = {
        /* 0: inside box & inside diagonal triangle */ make_wkb("POINT(8 2)"),
        /* 1: inside coarse MBR of triangle, outside triangle */
        make_wkb("POINT(2 8)"),
        /* 2: inside first box only */ make_wkb("POINT(-3 -3)"),
        /* 3: inside both, rejected by int64 < 40 */ make_wkb("POINT(7 1)"),
        /* 4: outside */ make_wkb("POINT(50 50)"),
        /* 5: outside */ make_wkb("POINT(-50 -50)"),
    };
    auto geo_field =
        storage::CreateFieldData(DataType::GEOMETRY, DataType::NONE, false);
    geo_field->FillFieldData(wkbs.data(), wkbs.size());

    auto int_lt = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(int_fid, DataType::INT64),
        proto::plan::OpType::LessThan,
        Int64Literal(40),
        std::vector<proto::plan::GenericValue>());
    int_lt->SetExprId(73001);

    auto gis_1 = std::make_shared<expr::GISFunctionFilterExpr>(
        expr::ColumnInfo(geo_fid, DataType::GEOMETRY),
        proto::plan::GISFunctionFilterExpr_GISOp_Intersects,
        "POLYGON((0 0, 10 0, 10 10, 0 0))");
    gis_1->SetExprId(73002);

    auto gis_2 = std::make_shared<expr::GISFunctionFilterExpr>(
        expr::ColumnInfo(geo_fid, DataType::GEOMETRY),
        proto::plan::GISFunctionFilterExpr_GISOp_Within,
        "POLYGON((-10 -10, 20 -10, 20 20, -10 20, -10 -10))");
    gis_2->SetExprId(73003);

    auto inner = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, int_lt, gis_1);
    auto all = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, inner, gis_2);

    // Case A: raw geometry column (no R-Tree).
    {
        RawFieldFiles files;
        auto segment = CreateSealedSegment(schema);
        segment->LoadFieldData(files.Prepare(int_fid, {int_field}));
        segment->LoadFieldData(files.Prepare(geo_fid, {geo_field}));

        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            wkbs.size());
        auto result = EvaluateFilter(all, segment.get(), wkbs.size(), explain);
        ExpectHitRows(result, {0}, "GIS split-fusion without R-Tree");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 3);

        std::map<std::string, const proto::internal::ExecutionExplain*> by_id;
        for (int i = 0; i < node.children_size(); ++i) {
            const auto& child = node.children(i);
            by_id[AttributesOf(child).at("expr_id")] = &child;
        }
        ASSERT_EQ(by_id.count("73001"), 1);
        ASSERT_EQ(by_id.count("73002"), 1);
        ASSERT_EQ(by_id.count("73003"), 1);

        EXPECT_EQ(AttributesOf(*by_id.at("73001")).at("eval_order"), "1");
        EXPECT_EQ(CountersOf(*by_id.at("73001")).at("rows_out"), 3);

        const auto g1_attrs = AttributesOf(*by_id.at("73002"));
        const auto g1_ctrs = CountersOf(*by_id.at("73002"));
        EXPECT_EQ(g1_attrs.at("decline_reason"), "no_index_on_field");
        EXPECT_EQ(g1_attrs.at("eval_order"), "2");
        EXPECT_EQ(g1_ctrs.at("rows_in"), 3);
        EXPECT_EQ(g1_ctrs.at("raw_scan"), 3);
        EXPECT_EQ(g1_ctrs.at("rows_out"), 1);

        const auto g2_attrs = AttributesOf(*by_id.at("73003"));
        const auto g2_ctrs = CountersOf(*by_id.at("73003"));
        EXPECT_EQ(g2_attrs.at("decline_reason"), "no_index_on_field");
        EXPECT_EQ(g2_attrs.at("eval_order"), "3");
        EXPECT_EQ(g2_ctrs.at("rows_in"), 1);
        EXPECT_EQ(g2_ctrs.at("raw_scan"), 1);
        EXPECT_EQ(g2_ctrs.at("rows_out"), 1);
    }

    // Case B: with R-Tree index installed.
    {
        constexpr int64_t kRTreeIndexId = 7310;
        RawFieldFiles files;
        auto segment = CreateSealedSegment(schema);
        segment->LoadFieldData(files.Prepare(int_fid, {int_field}));
        segment->LoadFieldData(files.Prepare(geo_fid, {geo_field}));
        InstallIndexWithId(*segment,
                           geo_fid,
                           DataType::GEOMETRY,
                           kRTreeIndexId,
                           BuildIndex(geo_fid,
                                      DataType::GEOMETRY,
                                      milvus::index::RTREE_INDEX_TYPE,
                                      {geo_field}));

        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            wkbs.size());
        auto result = EvaluateFilter(all, segment.get(), wkbs.size(), explain);
        ExpectHitRows(result, {0}, "GIS split-fusion with R-Tree");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 3);

        std::map<std::string, const proto::internal::ExecutionExplain*> by_id;
        for (int i = 0; i < node.children_size(); ++i) {
            const auto& child = node.children(i);
            by_id[AttributesOf(child).at("expr_id")] = &child;
        }
        ASSERT_EQ(by_id.count("73001"), 1);
        ASSERT_EQ(by_id.count("73002"), 1);
        ASSERT_EQ(by_id.count("73003"), 1);

        EXPECT_EQ(AttributesOf(*by_id.at("73001")).at("eval_order"), "1");

        const auto g1_attrs = AttributesOf(*by_id.at("73002"));
        const auto g1_ctrs = CountersOf(*by_id.at("73002"));
        EXPECT_EQ(g1_attrs.at("index_id"), std::to_string(kRTreeIndexId));
        EXPECT_EQ(g1_attrs.at("family"), "rtree");
        EXPECT_EQ(g1_attrs.at("eval_order"), "2");
        EXPECT_EQ(g1_ctrs.at("rows_in"), 3);
        EXPECT_EQ(g1_ctrs.at("idx_scan"), 3);
        // Candidates after coarse for {0,1,2} intersecting triangle MBR: rows 0 & 1.
        EXPECT_EQ(g1_ctrs.at("raw_scan"), 2);
        EXPECT_LE(g1_ctrs.at("raw_scan"), g1_ctrs.at("rows_in"));
        EXPECT_EQ(g1_ctrs.at("rows_out"), 1);

        const auto g2_attrs = AttributesOf(*by_id.at("73003"));
        const auto g2_ctrs = CountersOf(*by_id.at("73003"));
        EXPECT_EQ(g2_attrs.at("index_id"), std::to_string(kRTreeIndexId));
        EXPECT_EQ(g2_attrs.at("family"), "rtree");
        EXPECT_EQ(g2_attrs.at("eval_order"), "3");
        EXPECT_EQ(g2_ctrs.at("rows_in"), 2);
        EXPECT_EQ(g2_ctrs.at("idx_scan"), 2);
        // Only row 0 survived g1 exact refinement when g2 ran.
        EXPECT_EQ(g2_ctrs.at("raw_scan"), 1);
        EXPECT_LE(g2_ctrs.at("raw_scan"), g2_ctrs.at("rows_in"));
        EXPECT_EQ(g2_ctrs.at("rows_out"), 1);
    }

    // Case C1: POINT(2 8) falls inside both MBRs, outside gis_1 triangle.
    // Refine short-circuits at gis_1, and neither predicate may keep coarse
    // false positives in rows_out.
    {
        constexpr int64_t kRTreeIndexId = 7311;
        const std::vector<std::string> c1_wkbs = {
            make_wkb("POINT(2 8)"),
            make_wkb("POINT(50 50)"),
        };
        auto c1_geo =
            storage::CreateFieldData(DataType::GEOMETRY, DataType::NONE, false);
        c1_geo->FillFieldData(c1_wkbs.data(), c1_wkbs.size());

        RawFieldFiles files;
        auto segment = CreateSealedSegment(schema);
        segment->LoadFieldData(files.Prepare(geo_fid, {c1_geo}));
        InstallIndexWithId(*segment,
                           geo_fid,
                           DataType::GEOMETRY,
                           kRTreeIndexId,
                           BuildIndex(geo_fid,
                                      DataType::GEOMETRY,
                                      milvus::index::RTREE_INDEX_TYPE,
                                      {c1_geo}));

        auto c1_expr = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, gis_1, gis_2);
        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            c1_wkbs.size());
        auto result =
            EvaluateFilter(c1_expr, segment.get(), c1_wkbs.size(), explain);
        ExpectHitRows(result, {}, "GIS C1: zero exact hits inside both MBRs");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 2);
        for (int i = 0; i < node.children_size(); ++i) {
            EXPECT_EQ(CountersOf(node.children(i)).at("rows_out"), 0);
        }
    }

    // Case C2: gis_1 triangle MBR covers both {POINT(8 2), POINT(2 8)},
    // whereas a tight gis_2 box covers only POINT(8 2). POINT(2 8) is pruned by
    // gis_2's coarse MBR before refinement, so gis_1 must still report exact
    // rows_out=1 rather than coarse=2.
    {
        constexpr int64_t kRTreeIndexId = 7312;
        const std::vector<std::string> c2_wkbs = {
            make_wkb("POINT(8 2)"),
            make_wkb("POINT(2 8)"),
        };
        auto c2_geo =
            storage::CreateFieldData(DataType::GEOMETRY, DataType::NONE, false);
        c2_geo->FillFieldData(c2_wkbs.data(), c2_wkbs.size());

        RawFieldFiles files;
        auto segment = CreateSealedSegment(schema);
        segment->LoadFieldData(files.Prepare(geo_fid, {c2_geo}));
        InstallIndexWithId(*segment,
                           geo_fid,
                           DataType::GEOMETRY,
                           kRTreeIndexId,
                           BuildIndex(geo_fid,
                                      DataType::GEOMETRY,
                                      milvus::index::RTREE_INDEX_TYPE,
                                      {c2_geo}));

        auto gis_tight = std::make_shared<expr::GISFunctionFilterExpr>(
            expr::ColumnInfo(geo_fid, DataType::GEOMETRY),
            proto::plan::GISFunctionFilterExpr_GISOp_Within,
            "POLYGON((5 0, 10 0, 10 5, 5 5, 5 0))");
        gis_tight->SetExprId(73004);
        auto c2_expr = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, gis_1, gis_tight);

        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            c2_wkbs.size());
        auto result =
            EvaluateFilter(c2_expr, segment.get(), c2_wkbs.size(), explain);
        ExpectHitRows(result, {0}, "GIS C2: sibling coarse prunes false MBR");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 2);
        std::map<std::string, const proto::internal::ExecutionExplain*> by_id;
        for (int i = 0; i < node.children_size(); ++i) {
            const auto& child = node.children(i);
            by_id[AttributesOf(child).at("expr_id")] = &child;
        }
        const auto g1_ctrs = CountersOf(*by_id.at("73002"));
        const auto gt_ctrs = CountersOf(*by_id.at("73004"));
        EXPECT_EQ(g1_ctrs.at("rows_in"), 2);
        EXPECT_EQ(g1_ctrs.at("raw_scan"), 1);
        EXPECT_EQ(g1_ctrs.at("rows_out"), 1);
        EXPECT_EQ(gt_ctrs.at("rows_in"), 2);
        EXPECT_EQ(gt_ctrs.at("raw_scan"), 1);
        EXPECT_EQ(gt_ctrs.at("rows_out"), 1);
    }
}

// ReorderConjunctExpr puts numeric predicates ahead of varchar predicates; the
// explain attributes must report the actual execution order via eval_order.
TEST(MultiScalarIndexExprTest, ExplainEvalOrderReflectsOptimizerReorder) {
    auto schema = std::make_shared<Schema>();
    auto varchar_fid =
        schema->AddDebugField("varchar_field", DataType::VARCHAR);
    auto int_fid = schema->AddDebugField("int64_field", DataType::INT64);

    auto str_field = milvus::test::expr_index::StringField(kVarCharValues);
    auto int_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    int_field->FillFieldData(kInt64Values.data(), kInt64Values.size());

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(varchar_fid, {str_field}));
    segment->LoadFieldData(files.Prepare(int_fid, {int_field}));

    proto::plan::GenericValue beta_val;
    beta_val.set_string_val("prefix_beta");
    auto str_eq = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::Equal,
        beta_val,
        std::vector<proto::plan::GenericValue>());
    str_eq->SetExprId(74001);  // AST first

    auto int_gt = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(int_fid, DataType::INT64),
        proto::plan::OpType::GreaterThan,
        Int64Literal(20),
        std::vector<proto::plan::GenericValue>());
    int_gt->SetExprId(74002);  // AST second, executed first

    auto both = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, str_eq, int_gt);

    auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
        kVarCharValues.size());
    auto result =
        EvaluateFilter(both, segment.get(), kVarCharValues.size(), explain);
    ExpectHitRows(result, {2, 6}, "varchar == 'prefix_beta' AND int64 > 20");

    proto::internal::ExecutionExplain node;
    ASSERT_TRUE(node.ParseFromString(explain->Finish()));
    ASSERT_EQ(node.children_size(), 2);

    std::map<std::string, std::string> orders;
    for (int i = 0; i < node.children_size(); ++i) {
        const auto attrs = AttributesOf(node.children(i));
        orders[attrs.at("expr_id")] = attrs.at("eval_order");
    }
    EXPECT_EQ(orders.at("74002"), "1");
    EXPECT_EQ(orders.at("74001"), "2");
}

// PhyCallExpr (e.g. empty(varchar)) records a raw decision and counts so it
// survives ExecutionExplainRecorder::Finish() both standalone and in an AND.
TEST(MultiScalarIndexExprTest, ExplainCallExprSurvivesSerialization) {
    auto schema = std::make_shared<Schema>();
    auto varchar_fid =
        schema->AddDebugField("varchar_field", DataType::VARCHAR);
    auto int_fid = schema->AddDebugField("int64_field", DataType::INT64);

    const std::vector<std::string> str_vals = {
        "", "a", "", "b", "c", "d", "e", "f", "g", "h"};
    auto str_field = milvus::test::expr_index::StringField(str_vals);
    auto int_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    int_field->FillFieldData(kInt64Values.data(), kInt64Values.size());

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(varchar_fid, {str_field}));
    segment->LoadFieldData(files.Prepare(int_fid, {int_field}));

    auto& factory = milvus::exec::expression::FunctionFactory::Instance();
    factory.Initialize();
    auto empty_fn = factory.GetFilterFunction(
        milvus::exec::expression::FilterFunctionRegisterKey{
            "empty", {DataType::VARCHAR}});
    ASSERT_NE(empty_fn, nullptr);

    auto col_expr = std::make_shared<expr::ColumnExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR));
    auto call_expr = std::make_shared<expr::CallExpr>(
        "empty", std::vector<expr::TypedExprPtr>{col_expr}, empty_fn);
    call_expr->SetExprId(75001);

    // Standalone CallExpr.
    {
        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            str_vals.size());
        auto result =
            EvaluateFilter(call_expr, segment.get(), str_vals.size(), explain);
        ExpectHitRows(result, {0, 2}, "standalone empty(varchar)");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto attrs = AttributesOf(node.children(0));
        const auto ctrs = CountersOf(node.children(0));
        EXPECT_EQ(attrs.at("expr_id"), "75001");
        EXPECT_EQ(attrs.at("field_id"), std::to_string(varchar_fid.get()));
        EXPECT_EQ(attrs.at("decline_reason"), "no_index_on_field");
        EXPECT_EQ(ctrs.at("rows_in"), 10);
        EXPECT_EQ(ctrs.at("raw_scan"), 10);
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("rows_out"), 2);
    }

    // Combined under AND: int64 < 30 matches rows {0, 1}.
    {
        auto int_lt = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(int_fid, DataType::INT64),
            proto::plan::OpType::LessThan,
            Int64Literal(30),
            std::vector<proto::plan::GenericValue>());
        int_lt->SetExprId(75002);
        auto both = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, call_expr, int_lt);

        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            str_vals.size());
        auto result =
            EvaluateFilter(both, segment.get(), str_vals.size(), explain);
        ExpectHitRows(result, {0}, "int64 < 30 AND empty(varchar)");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 2);
        std::map<std::string, const proto::internal::ExecutionExplain*> by_id;
        for (int i = 0; i < node.children_size(); ++i) {
            const auto& child = node.children(i);
            by_id[AttributesOf(child).at("expr_id")] = &child;
        }
        ASSERT_EQ(by_id.count("75001"), 1);
        const auto call_ctrs = CountersOf(*by_id.at("75001"));
        EXPECT_EQ(call_ctrs.at("rows_in"), 2);
        EXPECT_EQ(call_ctrs.at("raw_scan"), 10);
        EXPECT_EQ(call_ctrs.at("rows_out"), 1);
    }

    // Indexed CallExpr via IScalarValueReader (MARISA): reports index decision
    // with idx_scan=rows_in and raw_scan=0.
    {
        constexpr int64_t kMarisaIndexId = 75010;
        RawFieldFiles indexed_files;
        auto indexed_segment = CreateSealedSegment(schema);
        indexed_segment->LoadFieldData(
            indexed_files.Prepare(varchar_fid, {str_field}));
        InstallIndexWithId(*indexed_segment,
                           varchar_fid,
                           DataType::VARCHAR,
                           kMarisaIndexId,
                           BuildIndex(varchar_fid,
                                      DataType::VARCHAR,
                                      milvus::index::MARISA_TRIE,
                                      {str_field}));

        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            str_vals.size());
        auto result = EvaluateFilter(
            call_expr, indexed_segment.get(), str_vals.size(), explain);
        ExpectHitRows(result, {0, 2}, "indexed empty(varchar) via MARISA");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto attrs = AttributesOf(node.children(0));
        const auto ctrs = CountersOf(node.children(0));
        EXPECT_EQ(attrs.at("expr_id"), "75001");
        EXPECT_EQ(attrs.at("field_id"), std::to_string(varchar_fid.get()));
        EXPECT_EQ(attrs.at("index_id"), std::to_string(kMarisaIndexId));
        EXPECT_EQ(attrs.at("family"), "marisa");
        EXPECT_EQ(attrs.find("decline_reason"), attrs.end());
        EXPECT_EQ(ctrs.at("rows_in"), 10);
        EXPECT_EQ(ctrs.at("idx_scan"), 10);
        EXPECT_EQ(ctrs.at("raw_scan"), 0);
        EXPECT_EQ(ctrs.at("rows_out"), 2);
    }
}

// Offset-input raw evaluation via Take / chunk fallback must credit raw_scan,
// and must not falsely credit PkIndex or bypassed ScalarIndex with idx_scan.
TEST(MultiScalarIndexExprTest, ExplainOffsetInputTakeRecordsRawScan) {
    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("pk_field", DataType::INT64);
    schema->set_primary_field_id(pk_fid);
    auto int_fid = schema->AddDebugField("int64_field", DataType::INT64);
    auto pk_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    pk_field->FillFieldData(kInt64Values.data(), kInt64Values.size());
    auto int_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    int_field->FillFieldData(kInt64Values.data(), kInt64Values.size());

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(pk_fid, {pk_field}));
    segment->LoadFieldData(files.Prepare(int_fid, {int_field}));

    auto int_gt = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(int_fid, DataType::INT64),
        proto::plan::OpType::GreaterThan,
        Int64Literal(20),
        std::vector<proto::plan::GenericValue>());
    int_gt->SetExprId(76001);

    // Offsets {0(10), 2(30), 4(40), 6(60)}: 3 hits.
    FixedVector<int32_t> offsets = {0, 2, 4, 6};
    {
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, int_gt);
        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            kInt64Values.size());
        auto result = milvus::test::gen_filter_res(plan.get(),
                                                   segment.get(),
                                                   kInt64Values.size(),
                                                   MAX_TIMESTAMP,
                                                   &offsets,
                                                   explain);
        ExpectHitRows(result, {1, 2, 3}, "offset input unindexed int64 > 20");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto attrs = AttributesOf(node.children(0));
        const auto ctrs = CountersOf(node.children(0));
        EXPECT_EQ(attrs.at("expr_id"), "76001");
        EXPECT_EQ(attrs.at("decline_reason"), "no_index_on_field");
        EXPECT_EQ(ctrs.at("rows_in"), 4);
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), 4);
        EXPECT_EQ(ctrs.at("rows_out"), 3);
    }

    // PK range under offset input bypasses PkIndex and scans raw data: idx_scan
    // must be 0, never double-counted with raw_scan.
    {
        auto pk_gt = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(pk_fid, DataType::INT64),
            proto::plan::OpType::GreaterThan,
            Int64Literal(20),
            std::vector<proto::plan::GenericValue>());
        pk_gt->SetExprId(76002);
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, pk_gt);
        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            kInt64Values.size());
        auto result = milvus::test::gen_filter_res(plan.get(),
                                                   segment.get(),
                                                   kInt64Values.size(),
                                                   MAX_TIMESTAMP,
                                                   &offsets,
                                                   explain);
        ExpectHitRows(result, {1, 2, 3}, "offset input pk > 20");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto attrs = AttributesOf(node.children(0));
        const auto ctrs = CountersOf(node.children(0));
        EXPECT_EQ(attrs.at("expr_id"), "76002");
        EXPECT_EQ(attrs.find("family"), attrs.end());
        EXPECT_EQ(attrs.at("decline_reason"), "index_unusable");
        EXPECT_EQ(ctrs.at("rows_in"), 4);
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), 4);
        EXPECT_EQ(ctrs.at("rows_out"), 3);
    }

    // ScalarIndex (INVERTED) with raw data still present also bypasses the
    // index under offset input.
    {
        InstallIndexWithId(*segment,
                           int_fid,
                           DataType::INT64,
                           76010,
                           BuildIndex(int_fid,
                                      DataType::INT64,
                                      milvus::index::INVERTED_INDEX_TYPE,
                                      {int_field}));
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, int_gt);
        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            kInt64Values.size());
        auto result = milvus::test::gen_filter_res(plan.get(),
                                                   segment.get(),
                                                   kInt64Values.size(),
                                                   MAX_TIMESTAMP,
                                                   &offsets,
                                                   explain);
        ExpectHitRows(result, {1, 2, 3}, "offset input inverted int64 > 20");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto attrs = AttributesOf(node.children(0));
        const auto ctrs = CountersOf(node.children(0));
        EXPECT_EQ(attrs.at("expr_id"), "76001");
        EXPECT_EQ(attrs.find("index_id"), attrs.end());
        EXPECT_EQ(attrs.at("decline_reason"), "index_unusable");
        EXPECT_EQ(ctrs.at("rows_in"), 4);
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), 4);
        EXPECT_EQ(ctrs.at("rows_out"), 3);
    }
}

// IS [NOT] NULL on a non-nullable field short-circuits from schema metadata in
// PreCheckNullable, so neither idx_scan nor raw_scan is credited.
TEST(MultiScalarIndexExprTest, ExplainNonNullableIsNotNullSkipsIndexCredit) {
    auto schema = std::make_shared<Schema>();
    auto varchar_fid =
        schema->AddDebugField("varchar_field", DataType::VARCHAR);
    auto str_field = milvus::test::expr_index::StringField(kVarCharValues);

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(varchar_fid, {str_field}));
    InstallIndexWithId(*segment,
                       varchar_fid,
                       DataType::VARCHAR,
                       78010,
                       BuildIndex(varchar_fid,
                                  DataType::VARCHAR,
                                  milvus::index::INVERTED_INDEX_TYPE,
                                  {str_field}));

    auto not_null = std::make_shared<expr::NullExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::NullExpr_NullOp_IsNotNull);
    not_null->SetExprId(78001);

    auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
        kVarCharValues.size());
    auto result =
        EvaluateFilter(not_null, segment.get(), kVarCharValues.size(), explain);
    ExpectHitRows(result,
                  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                  "non-nullable varchar IS NOT NULL");

    proto::internal::ExecutionExplain node;
    ASSERT_TRUE(node.ParseFromString(explain->Finish()));
    ASSERT_EQ(node.children_size(), 1);
    const auto attrs = AttributesOf(node.children(0));
    const auto ctrs = CountersOf(node.children(0));
    EXPECT_EQ(attrs.at("expr_id"), "78001");
    EXPECT_EQ(attrs.find("index_id"), attrs.end());
    EXPECT_EQ(attrs.at("decline_reason"), "schema_not_nullable");
    EXPECT_EQ(ctrs.at("rows_in"), 10);
    EXPECT_EQ(ctrs.at("idx_scan"), 0);
    EXPECT_EQ(ctrs.at("raw_scan"), 0);
    EXPECT_EQ(ctrs.at("rows_out"), 10);
}

// A single leaf inside MatchExpr is evaluated directly without a parent
// conjunction; PhyMatchFilterExpr must attribute its element-domain counts.
TEST(MultiScalarIndexExprTest, ExplainMatchAnySingleLeafCreditsElementCounts) {
    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("id", DataType::INT64);
    schema->set_primary_field_id(pk_fid);
    auto sub_int_fid = schema->AddDebugArrayField(
        "struct_array[sub_int]", DataType::INT32, true);

    constexpr int64_t row_count = 3;
    auto insert_data = std::make_unique<InsertRecordProto>();
    std::vector<int64_t> ids = {0, 1, 2};
    auto id_array =
        CreateDataArrayFrom(ids.data(), nullptr, row_count, (*schema)[pk_fid]);
    insert_data->mutable_fields_data()->AddAllocated(id_array.release());

    auto* sub_field = insert_data->add_fields_data();
    sub_field->set_field_id(sub_int_fid.get());
    sub_field->set_field_name("struct_array[sub_int]");
    sub_field->set_type(proto::schema::DataType::Array);
    auto* array_data = sub_field->mutable_scalars()->mutable_array_data();
    array_data->set_element_type(proto::schema::DataType::Int32);
    for (int64_t r = 0; r < row_count; ++r) {
        sub_field->mutable_scalars()->add_valid_data(true);
    }
    auto append_row = [array_data](std::initializer_list<int32_t> values) {
        auto* row = array_data->add_data();
        for (auto v : values) {
            row->mutable_int_data()->add_data(v);
        }
    };
    append_row({1, 2});
    append_row({9001});
    append_row({5});
    insert_data->set_num_rows(row_count);

    GeneratedData generated;
    generated.schema_ = schema;
    generated.raw_ = insert_data.release();
    generated.row_ids_ = {0, 1, 2};
    generated.timestamps_ = {0, 1, 2};
    auto segment = CreateSealedWithFieldDataLoaded(schema, generated);

    expr::ColumnInfo sub_col(
        sub_int_fid, DataType::INT32, DataType::INT32, {}, true);
    sub_col.element_level_ = true;
    auto sub_ge = std::make_shared<expr::UnaryRangeFilterExpr>(
        sub_col,
        proto::plan::OpType::GreaterEqual,
        Int64Literal(9000),
        std::vector<proto::plan::GenericValue>());
    sub_ge->SetExprId(77001);

    auto match_expr = std::make_shared<expr::MatchExpr>(
        "struct_array", expr::MatchType::MatchAny, 0, sub_ge);

    auto explain =
        std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
    auto result = EvaluateFilter(match_expr, segment.get(), row_count, explain);
    ExpectHitRows(result, {1}, "match_any(struct_array, $[sub_int] >= 9000)");

    proto::internal::ExecutionExplain node;
    ASSERT_TRUE(node.ParseFromString(explain->Finish()));
    ASSERT_EQ(node.children_size(), 1);
    const auto attrs = AttributesOf(node.children(0));
    const auto ctrs = CountersOf(node.children(0));
    EXPECT_EQ(attrs.at("expr_id"), "77001");
    EXPECT_EQ(attrs.at("field_id"), std::to_string(sub_int_fid.get()));
    EXPECT_EQ(ctrs.at("rows_in"), 4);
    EXPECT_EQ(ctrs.at("idx_scan"), 0);
    EXPECT_EQ(ctrs.at("raw_scan"), 4);
    EXPECT_EQ(ctrs.at("rows_out"), 1);
}

TEST(MultiScalarIndexExprTest, ExplainNotOperandRespectsCandidateMask) {
    auto schema = std::make_shared<Schema>();
    auto a_fid = schema->AddDebugField("a", DataType::INT64);
    auto b_fid = schema->AddDebugField("b", DataType::INT64);

    // Row 0 has a==1, b==4; rows 1..9 have a==2, b==2.
    std::vector<int64_t> a_vals(10, 2);
    a_vals[0] = 1;
    std::vector<int64_t> b_vals(10, 2);
    b_vals[0] = 4;

    auto a_field = std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    a_field->FillFieldData(a_vals.data(), a_vals.size());
    auto b_field = std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    b_field->FillFieldData(b_vals.data(), b_vals.size());

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(a_fid, {a_field}));
    segment->LoadFieldData(files.Prepare(b_fid, {b_field}));
    InstallIndexWithId(
        *segment,
        b_fid,
        DataType::INT64,
        78001,
        BuildIndex(
            b_fid, DataType::INT64, milvus::index::ASCENDING_SORT, {b_field}));

    auto a_eq_1 = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(a_fid, DataType::INT64),
        proto::plan::OpType::Equal,
        Int64Literal(1),
        std::vector<proto::plan::GenericValue>());
    a_eq_1->SetExprId(78101);

    auto b_in_2_3 = std::make_shared<expr::TermFilterExpr>(
        expr::ColumnInfo(b_fid, DataType::INT64),
        std::vector<proto::plan::GenericValue>{Int64Literal(2),
                                               Int64Literal(3)});
    b_in_2_3->SetExprId(78102);

    auto not_b = std::make_shared<expr::LogicalUnaryExpr>(
        expr::LogicalUnaryExpr::OpType::LogicalNot, b_in_2_3);
    auto conj = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, a_eq_1, not_b);

    auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(10);
    auto result = EvaluateFilter(conj, segment.get(), 10, explain);
    ExpectHitRows(result, {0}, "a == 1 AND NOT (b IN [2, 3])");

    proto::internal::ExecutionExplain node;
    ASSERT_TRUE(node.ParseFromString(explain->Finish()));
    ASSERT_EQ(node.children_size(), 2);

    bool found_b = false;
    for (int i = 0; i < node.children_size(); ++i) {
        const auto attrs = AttributesOf(node.children(i));
        const auto ctrs = CountersOf(node.children(i));
        if (attrs.at("expr_id") == "78102") {
            found_b = true;
            EXPECT_EQ(ctrs.at("rows_in"), 1);
            EXPECT_EQ(ctrs.at("idx_scan"), 1);
            EXPECT_EQ(ctrs.at("raw_scan"), 0);
            EXPECT_EQ(ctrs.at("rows_out"), 0);
        }
    }
    EXPECT_TRUE(found_b);
}

TEST(MultiScalarIndexExprTest, ExplainNullableIsNullRecordsRawScan) {
    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("id", DataType::INT64);
    schema->set_primary_field_id(pk_fid);
    auto opt_fid = schema->AddDebugField("opt", DataType::INT64, true);

    constexpr int64_t row_count = 6;
    std::vector<int64_t> ids = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> vals = {10, 20, 30, 40, 50, 60};
    bool valids[row_count] = {true, false, true, false, true, true};

    auto insert_data = std::make_unique<InsertRecordProto>();
    auto id_arr =
        CreateDataArrayFrom(ids.data(), nullptr, row_count, (*schema)[pk_fid]);
    insert_data->mutable_fields_data()->AddAllocated(id_arr.release());
    auto opt_arr =
        CreateDataArrayFrom(vals.data(), valids, row_count, (*schema)[opt_fid]);
    insert_data->mutable_fields_data()->AddAllocated(opt_arr.release());
    insert_data->set_num_rows(row_count);

    GeneratedData generated;
    generated.schema_ = schema;
    generated.raw_ = insert_data.release();
    generated.row_ids_ = {0, 1, 2, 3, 4, 5};
    generated.timestamps_ = {0, 1, 2, 3, 4, 5};
    auto segment = CreateSealedWithFieldDataLoaded(schema, generated);

    auto is_null = std::make_shared<expr::NullExpr>(
        expr::ColumnInfo(opt_fid, DataType::INT64, {}, true),
        proto::plan::NullExpr_NullOp_IsNull);
    is_null->SetExprId(78201);

    {
        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        auto res = EvaluateFilter(is_null, segment.get(), row_count, explain);
        ExpectHitRows(res, {1, 3}, "nullable opt IS NULL");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto attrs = AttributesOf(node.children(0));
        const auto ctrs = CountersOf(node.children(0));
        EXPECT_EQ(attrs.at("expr_id"), "78201");
        EXPECT_EQ(attrs.at("decline_reason"), "no_index_on_field");
        EXPECT_EQ(ctrs.at("rows_in"), row_count);
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), row_count);
        EXPECT_EQ(ctrs.at("rows_out"), 2);
    }

    // Offset-input validity check must also record raw_scan.
    {
        FixedVector<int32_t> offsets = {0, 1, 3, 5};
        auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                           is_null);
        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        auto res = milvus::test::gen_filter_res(plan.get(),
                                                segment.get(),
                                                row_count,
                                                MAX_TIMESTAMP,
                                                &offsets,
                                                explain);
        ExpectHitRows(res, {1, 2}, "nullable opt IS NULL by offsets");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto ctrs = CountersOf(node.children(0));
        EXPECT_EQ(ctrs.at("rows_in"), 4);
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), 4);
        EXPECT_EQ(ctrs.at("rows_out"), 2);
    }
}

TEST(MultiScalarIndexExprTest,
     ExplainEntityTTLCollectionHidesInternalPredicates) {
    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("id", DataType::INT64);
    schema->set_primary_field_id(pk_fid);
    auto ttl_fid = schema->AddDebugField("ttl_ts", DataType::TIMESTAMPTZ, true);
    schema->set_ttl_field_id(ttl_fid);
    auto val_fid = schema->AddDebugField("val", DataType::INT64);

    constexpr int64_t row_count = 4;
    // query_ts = (1000ULL << 18) => physical time 1,000,000 us.
    constexpr uint64_t query_ts = 1000ULL << 18;
    std::vector<int64_t> ids = {0, 1, 2, 3};
    std::vector<int64_t> ttls = {500000, 1500000, 2000000, 500000};
    bool ttl_valids[row_count] = {true, true, true, false};
    std::vector<int64_t> vals = {1, 1, 2, 1};

    auto insert_data = std::make_unique<InsertRecordProto>();
    insert_data->mutable_fields_data()->AddAllocated(
        CreateDataArrayFrom(ids.data(), nullptr, row_count, (*schema)[pk_fid])
            .release());
    insert_data->mutable_fields_data()->AddAllocated(
        CreateDataArrayFrom(
            ttls.data(), ttl_valids, row_count, (*schema)[ttl_fid])
            .release());
    insert_data->mutable_fields_data()->AddAllocated(
        CreateDataArrayFrom(vals.data(), nullptr, row_count, (*schema)[val_fid])
            .release());
    insert_data->set_num_rows(row_count);

    GeneratedData generated;
    generated.schema_ = schema;
    generated.raw_ = insert_data.release();
    generated.row_ids_ = {0, 1, 2, 3};
    generated.timestamps_ = {0, 1, 2, 3};
    auto segment = CreateSealedWithFieldDataLoaded(schema, generated);

    auto eval_on_sealed =
        [&](const expr::TypedExprPtr& filter_expr,
            std::shared_ptr<milvus::exec::ExecutionExplainRecorder> explain) {
            std::vector<milvus::expr::TypedExprPtr> filters{filter_expr};
            auto query_context = std::make_shared<milvus::exec::QueryContext>(
                DEAFULT_QUERY_ID, segment.get(), row_count, query_ts);
            query_context->set_read_schema(schema);
            query_context->set_read_snapshot(segment->CaptureReadSnapshot());
            query_context->SetExplain(explain);
            milvus::exec::ExecContext exec_context(query_context.get());
            exec_context.SetExplain(explain);
            milvus::exec::ExprSet exprs(
                filters, &exec_context, /*null_rejecting=*/true);
            std::vector<VectorPtr> results;
            milvus::exec::EvalCtx eval_ctx(&exec_context);
            exprs.Eval(0, 1, true, eval_ctx, results);
            return milvus::test::GetColumnVectorForTest(results[0]);
        };

    // 1. Unfiltered search on a TTL collection uses an untagged AlwaysTrueExpr:
    // TTL still filters expired row 0, and explain emits zero phantom groups.
    {
        auto always_true = std::make_shared<expr::AlwaysTrueExpr>();
        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        auto res = eval_on_sealed(always_true, explain);
        ExpectHitRows(res, {1, 2, 3}, "unfiltered TTL collection");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        EXPECT_EQ(node.children_size(), 0);
    }

    // 2. Filtered search (val == 1) emits only the user leaf (78301), never
    // phantom field_<ttl_fid> groups.
    {
        auto val_eq_1 = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(val_fid, DataType::INT64),
            proto::plan::OpType::Equal,
            Int64Literal(1),
            std::vector<proto::plan::GenericValue>());
        val_eq_1->SetExprId(78301);

        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        auto res = eval_on_sealed(val_eq_1, explain);
        ExpectHitRows(res, {1, 3}, "val == 1 on TTL collection");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto attrs = AttributesOf(node.children(0));
        EXPECT_EQ(attrs.at("expr_id"), "78301");
        EXPECT_EQ(attrs.at("field_id"), std::to_string(val_fid.get()));
    }

    // 3. Top-level AND conjunction (id >= 0 AND val == 1) appends the TTL
    // filter into the existing PhyConjunctFilterExpr without emitting TTL
    // leaves or shifting eval_order gaps.
    {
        auto id_ge_0 = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(pk_fid, DataType::INT64),
            proto::plan::OpType::GreaterEqual,
            Int64Literal(0),
            std::vector<proto::plan::GenericValue>());
        id_ge_0->SetExprId(78302);
        auto val_eq_1 = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(val_fid, DataType::INT64),
            proto::plan::OpType::Equal,
            Int64Literal(1),
            std::vector<proto::plan::GenericValue>());
        val_eq_1->SetExprId(78303);
        auto conj = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, id_ge_0, val_eq_1);

        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        auto res = eval_on_sealed(conj, explain);
        ExpectHitRows(res, {1, 3}, "id >= 0 AND val == 1 on TTL collection");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 2);
        std::map<std::string, std::map<std::string, std::string>> by_id;
        for (const auto& child : node.children()) {
            auto attrs = AttributesOf(child);
            by_id[attrs.at("expr_id")] = std::move(attrs);
        }
        ASSERT_EQ(by_id.count("78302"), 1u);
        ASSERT_EQ(by_id.count("78303"), 1u);
        EXPECT_EQ(by_id.at("78302").at("eval_order"), "1");
        EXPECT_EQ(by_id.at("78303").at("eval_order"), "2");
    }
}

TEST(MultiScalarIndexExprTest,
     ExplainTwoColumnCallAndShortCircuitedCompareLeaves) {
    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("id", DataType::INT64);
    schema->set_primary_field_id(pk_fid);
    auto name_fid = schema->AddDebugField("name", DataType::VARCHAR);
    auto pfx_fid = schema->AddDebugField("pfx", DataType::VARCHAR);

    const std::vector<int64_t> ids = {0, 1, 2, 3};
    const std::vector<std::string> names = {
        "alpha_one", "beta_two", "alpha_three", "gamma"};
    const std::vector<std::string> pfxs = {"alpha", "alpha", "beta", "gamma"};
    const int64_t row_count = static_cast<int64_t>(ids.size());

    auto id_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    id_field->FillFieldData(ids.data(), row_count);
    auto name_field = milvus::test::expr_index::StringField(names);
    auto pfx_field = milvus::test::expr_index::StringField(pfxs);

    constexpr int64_t kNameMarisaIndexId = 78401;
    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(pk_fid, {id_field}));
    segment->LoadFieldData(files.Prepare(name_fid, {name_field}));
    segment->LoadFieldData(files.Prepare(pfx_fid, {pfx_field}));
    InstallIndexWithId(*segment,
                       name_fid,
                       DataType::VARCHAR,
                       kNameMarisaIndexId,
                       BuildIndex(name_fid,
                                  DataType::VARCHAR,
                                  milvus::index::MARISA_TRIE,
                                  {name_field}));

    // 1. Two-column CallExpr starts_with(name, pfx): name uses MARISA_TRIE
    // (raw_scan=0) and pfx falls back to raw scan (raw_scan=4).
    {
        auto& factory = milvus::exec::expression::FunctionFactory::Instance();
        factory.Initialize();
        auto starts_with_fn = factory.GetFilterFunction(
            milvus::exec::expression::FilterFunctionRegisterKey{
                "starts_with", {DataType::VARCHAR, DataType::VARCHAR}});
        ASSERT_NE(starts_with_fn, nullptr);

        auto name_col = std::make_shared<expr::ColumnExpr>(
            expr::ColumnInfo(name_fid, DataType::VARCHAR));
        auto pfx_col = std::make_shared<expr::ColumnExpr>(
            expr::ColumnInfo(pfx_fid, DataType::VARCHAR));
        auto call_expr = std::make_shared<expr::CallExpr>(
            "starts_with",
            std::vector<expr::TypedExprPtr>{name_col, pfx_col},
            starts_with_fn);
        call_expr->SetExprId(78410);

        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        auto res = EvaluateFilter(call_expr, segment.get(), row_count, explain);
        ExpectHitRows(res, {0, 3}, "starts_with(name, pfx)");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 2);
        const auto seg_ctrs = CountersOf(node);
        EXPECT_EQ(seg_ctrs.at("rows_in"), row_count);
        EXPECT_EQ(seg_ctrs.at("idx_scan"), row_count);
        EXPECT_EQ(seg_ctrs.at("raw_scan"), row_count);
        EXPECT_EQ(seg_ctrs.at("rows_out"), 2);

        for (int i = 0; i < node.children_size(); ++i) {
            const auto attrs = AttributesOf(node.children(i));
            const auto ctrs = CountersOf(node.children(i));
            EXPECT_EQ(attrs.at("expr_id"), "78410");
            if (attrs.at("field_id") == std::to_string(name_fid.get())) {
                EXPECT_EQ(attrs.at("index_id"),
                          std::to_string(kNameMarisaIndexId));
                EXPECT_EQ(attrs.at("family"), "marisa");
                EXPECT_EQ(ctrs.at("idx_scan"), row_count);
                EXPECT_EQ(ctrs.at("raw_scan"), 0);
            } else {
                EXPECT_EQ(attrs.at("field_id"), std::to_string(pfx_fid.get()));
                EXPECT_EQ(attrs.at("decline_reason"), "no_index_on_field");
                EXPECT_EQ(ctrs.at("idx_scan"), 0);
                EXPECT_EQ(ctrs.at("raw_scan"), row_count);
            }
            EXPECT_EQ(ctrs.at("rows_in"), row_count);
            EXPECT_EQ(ctrs.at("rows_out"), 2);
        }
    }

    // 2. Short-circuited CompareExpr (pfx == "nope" AND name < pfx) never
    // enters CompareExpr::Eval, yet Finish() emits its skipped filter_group
    // with eval_order preserved.
    {
        proto::plan::GenericValue nope;
        nope.set_string_val("nope");
        auto never = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(pfx_fid, DataType::VARCHAR),
            proto::plan::OpType::Equal,
            nope,
            std::vector<proto::plan::GenericValue>());
        never->SetExprId(78421);
        auto cmp = std::make_shared<expr::CompareExpr>(name_fid,
                                                       pfx_fid,
                                                       DataType::VARCHAR,
                                                       DataType::VARCHAR,
                                                       OpType::LessThan);
        cmp->SetExprId(78422);
        auto conj = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, never, cmp);

        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        auto res = EvaluateFilter(conj, segment.get(), row_count, explain);
        ExpectHitRows(res, {}, "short-circuited compare");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 2);
        std::map<std::string, const proto::internal::ExecutionExplain*> by_id;
        for (int i = 0; i < node.children_size(); ++i) {
            const auto& child = node.children(i);
            by_id[AttributesOf(child).at("expr_id")] = &child;
        }
        ASSERT_EQ(by_id.count("78421"), 1u);
        ASSERT_EQ(by_id.count("78422"), 1u);
        const auto never_attrs = AttributesOf(*by_id.at("78421"));
        EXPECT_EQ(never_attrs.count("skipped"), 0u);

        const auto cmp_attrs = AttributesOf(*by_id.at("78422"));
        const auto cmp_ctrs = CountersOf(*by_id.at("78422"));
        EXPECT_EQ(cmp_attrs.at("eval_order"), "2");
        EXPECT_EQ(cmp_attrs.at("skipped"), "true");
        EXPECT_EQ(cmp_ctrs.at("rows_in"), 0);
        EXPECT_EQ(cmp_ctrs.at("idx_scan"), 0);
        EXPECT_EQ(cmp_ctrs.at("raw_scan"), 0);
        EXPECT_EQ(cmp_ctrs.at("rows_out"), 0);
    }

    // 3. Recorder with rows_total=0 or without Eval entered does NOT mark
    // un-evaluated leaves as skipped.
    {
        auto empty_explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(0);
        empty_explain->SetEvalOrder(78430, 1);
        empty_explain->MarkEvalEntered(78430);
        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(empty_explain->Finish()));
        EXPECT_EQ(node.children_size(), 0);
    }
}

TEST(MultiScalarIndexExprTest, ExplainWholeFilterCacheHitReportsZeroIdxScan) {
    using milvus::exec::ExprResCacheManager;
    auto& mgr = ExprResCacheManager::Instance();
    const bool prev_enabled = ExprResCacheManager::IsEnabled();
    ExprResCacheManager::SetEnabled(true);
    mgr.Clear();
    mgr.SetCapacityBytes(1ULL << 20);
    auto restore = folly::makeGuard([&] {
        mgr.Clear();
        ExprResCacheManager::SetEnabled(prev_enabled);
    });

    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("id", DataType::INT64);
    schema->set_primary_field_id(pk_fid);
    auto val_fid = schema->AddDebugField("val", DataType::INT64);

    const std::vector<int64_t> vals = {10, 20, 30, 40, 50, 60};
    const int64_t row_count = static_cast<int64_t>(vals.size());
    auto pk_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    pk_field->FillFieldData(vals.data(), row_count);
    auto val_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    val_field->FillFieldData(vals.data(), row_count);

    constexpr int64_t kSortIndexId = 78501;
    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(pk_fid, {pk_field}));
    segment->LoadFieldData(files.Prepare(val_fid, {val_field}));
    InstallIndexWithId(*segment,
                       val_fid,
                       DataType::INT64,
                       kSortIndexId,
                       BuildIndex(val_fid,
                                  DataType::INT64,
                                  milvus::index::ASCENDING_SORT,
                                  {val_field}));

    auto leaf = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(val_fid, DataType::INT64),
        proto::plan::OpType::GreaterThan,
        Int64Literal(30),
        std::vector<proto::plan::GenericValue>());
    leaf->SetExprId(78510);
    auto filter_plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, leaf);

    auto run_once =
        [&](const std::shared_ptr<milvus::exec::ExecutionExplainRecorder>&
                explain) {
            auto fragment = plan::PlanFragment(filter_plan);
            auto query_context = std::make_shared<milvus::exec::QueryContext>(
                DEAFULT_QUERY_ID, segment.get(), row_count, MAX_TIMESTAMP);
            query_context->set_read_schema(schema);
            query_context->set_read_snapshot(segment->CaptureReadSnapshot());
            query_context->set_enable_expr_cache(true);
            query_context->set_enable_sub_expr_cache_write(false);
            query_context->SetExplain(explain);
            auto row = ExecPlanNodeVisitor::ExecuteTask(
                fragment, std::move(query_context));
            ASSERT_NE(row, nullptr);
        };

    // Cold run evaluates the sort index and populates ExprResCacheManager.
    {
        auto cold_explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        run_once(cold_explain);
        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(cold_explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto& op = node.children(0);
        EXPECT_EQ(op.kind(), "operator");
        const auto op_attrs = AttributesOf(op);
        const auto op_ctrs = CountersOf(op);
        EXPECT_EQ(op_attrs.at("name"), "filter");
        EXPECT_EQ(op_attrs.at("step"), "0");
        EXPECT_EQ(op_attrs.count("cached"), 0u);
        EXPECT_EQ(op_ctrs.at("rows_in"), row_count);
        EXPECT_EQ(op_ctrs.at("rows_out"), 3);
        ASSERT_EQ(op.children_size(), 1);
        const auto ctrs = CountersOf(op.children(0));
        EXPECT_EQ(ctrs.at("rows_in"), row_count);
        EXPECT_EQ(ctrs.at("idx_scan"), row_count);
        EXPECT_EQ(ctrs.at("rows_out"), 3);
    }

    // Warm run hits the whole-filter cache without scanning the index or
    // marking the leaf as skipped, while marking the filter operator cached.
    {
        auto warm_explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        run_once(warm_explain);
        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(warm_explain->Finish()));
        const auto seg_ctrs = CountersOf(node);
        EXPECT_EQ(seg_ctrs.at("idx_scan"), 0);
        EXPECT_EQ(seg_ctrs.at("raw_scan"), 0);
        ASSERT_EQ(node.children_size(), 1);
        const auto& op = node.children(0);
        EXPECT_EQ(op.kind(), "operator");
        const auto op_attrs = AttributesOf(op);
        const auto op_ctrs = CountersOf(op);
        EXPECT_EQ(op_attrs.at("name"), "filter");
        EXPECT_EQ(op_attrs.at("cached"), "true");
        EXPECT_EQ(op_ctrs.at("rows_in"), row_count);
        EXPECT_EQ(op_ctrs.at("rows_out"), 3);
        EXPECT_EQ(op.children_size(), 0);
    }
}

// Background PrefetchAsync() and capability probes (CanExecuteAllAtOnce() /
// CanUseNestedIndex()) run before Eval() binds has_offset_input_. When Eval()
// subsequently runs with an offset input and bypasses the index in favor of raw
// data, explain must record the actual raw fallback (idx_scan=0, raw_scan=N,
// decline_reason=index_unusable) rather than a premature index decision.
// Conversely, BinaryRange on a projected JSON index still serves offset inputs
// through the index and must keep its index attribution.
TEST(MultiScalarIndexExprTest,
     ExplainPrefetchAsyncBeforeOffsetInputRecordsActualPath) {
    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("pk_field", DataType::INT64);
    schema->set_primary_field_id(pk_fid);
    auto int_fid = schema->AddDebugField("int64_field", DataType::INT64);
    auto json_fid = schema->AddDebugField("json_field", DataType::JSON);

    auto pk_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    pk_field->FillFieldData(kInt64Values.data(), kInt64Values.size());
    auto int_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    int_field->FillFieldData(kInt64Values.data(), kInt64Values.size());

    std::vector<std::string> json_rows;
    json_rows.reserve(kInt64Values.size());
    for (int64_t v : kInt64Values) {
        json_rows.push_back(fmt::format(R"({{"score": {}}})", v));
    }
    auto json_field = milvus::test::expr_index::JsonField(json_rows);

    constexpr int64_t kInvertedId = 78601;
    constexpr int64_t kSortId = 78602;
    constexpr int64_t kJsonIndexId = 78603;

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(pk_fid, {pk_field}));
    segment->LoadFieldData(files.Prepare(int_fid, {int_field}));
    segment->LoadFieldData(files.Prepare(json_fid, {json_field}));
    InstallIndexWithId(*segment,
                       int_fid,
                       DataType::INT64,
                       kInvertedId,
                       BuildIndex(int_fid,
                                  DataType::INT64,
                                  milvus::index::INVERTED_INDEX_TYPE,
                                  {int_field}));
    InstallIndexWithId(*segment,
                       int_fid,
                       DataType::INT64,
                       kSortId,
                       BuildIndex(int_fid,
                                  DataType::INT64,
                                  milvus::index::ASCENDING_SORT,
                                  {int_field}));
    InstallIndexWithId(
        *segment,
        json_fid,
        DataType::JSON,
        kJsonIndexId,
        BuildIndex(json_fid,
                   DataType::JSON,
                   milvus::index::INVERTED_INDEX_TYPE,
                   {json_field},
                   {{JSON_PATH, "/score"}, {JSON_CAST_TYPE, "DOUBLE"}}),
        {{JSON_PATH, "/score"}, {JSON_CAST_TYPE, "DOUBLE"}});

    auto pool = std::make_shared<folly::CPUThreadPoolExecutor>(1);
    FixedVector<int32_t> offsets = {0, 2, 4, 6};

    auto eval_after_prefetch =
        [&](const expr::TypedExprPtr& filter_expr,
            const std::shared_ptr<milvus::exec::ExecutionExplainRecorder>&
                explain) {
            std::vector<milvus::expr::TypedExprPtr> filters{filter_expr};
            auto query_context = std::make_shared<milvus::exec::QueryContext>(
                DEAFULT_QUERY_ID,
                segment.get(),
                kInt64Values.size(),
                MAX_TIMESTAMP);
            query_context->set_read_schema(schema);
            query_context->set_read_snapshot(segment->CaptureReadSnapshot());
            query_context->SetExplain(explain);
            milvus::exec::ExecContext exec_context(query_context.get());
            exec_context.SetExplain(explain);
            milvus::exec::ExprSet exprs(filters, &exec_context);
            // Trigger both capability probing and background prefetch while
            // has_offset_input_ is still false, mirroring Driver::PrefetchAsync
            // and ElementFilterBitsNode::EvaluateElementExpression.
            (void)exprs.CanExecuteAllAtOnce();
            for (const auto& compiled : exprs.exprs()) {
                (void)compiled->CanUseNestedIndex();
            }
            exprs.PrefetchAsync(pool);
            exprs.WaitPrefetch();

            std::vector<VectorPtr> results;
            milvus::exec::EvalCtx eval_ctx(&exec_context, &offsets);
            exprs.Eval(0, 1, true, eval_ctx, results);
            return milvus::test::GetColumnVectorForTest(results[0]);
        };

    // 1. UnaryRangeFilterExpr (int64 == 40): INVERTED selected in prefetch,
    // bypassed under offset_input -> raw fallback.
    {
        auto eq_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(int_fid, DataType::INT64),
            proto::plan::OpType::Equal,
            Int64Literal(40),
            std::vector<proto::plan::GenericValue>());
        eq_expr->SetExprId(78611);

        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            kInt64Values.size());
        auto res = eval_after_prefetch(eq_expr, explain);
        ExpectHitRows(res, {2}, "prefetch + offset UnaryRange == 40");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto attrs = AttributesOf(node.children(0));
        const auto ctrs = CountersOf(node.children(0));
        EXPECT_EQ(attrs.find("index_id"), attrs.end());
        EXPECT_EQ(attrs.at("decline_reason"), "index_unusable");
        EXPECT_EQ(ctrs.at("rows_in"), 4);
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), 4);
        EXPECT_EQ(ctrs.at("rows_out"), 1);
    }

    // 2. BinaryRangeFilterExpr (20 < int64 < 60): STL_SORT selected in
    // prefetch, bypassed under offset_input -> raw fallback.
    {
        auto range_expr = std::make_shared<expr::BinaryRangeFilterExpr>(
            expr::ColumnInfo(int_fid, DataType::INT64),
            Int64Literal(20),
            Int64Literal(60),
            /*lower_inclusive=*/false,
            /*upper_inclusive=*/false);
        range_expr->SetExprId(78612);

        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            kInt64Values.size());
        auto res = eval_after_prefetch(range_expr, explain);
        ExpectHitRows(res, {1, 2}, "prefetch + offset BinaryRange (20, 60)");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto attrs = AttributesOf(node.children(0));
        const auto ctrs = CountersOf(node.children(0));
        EXPECT_EQ(attrs.find("index_id"), attrs.end());
        EXPECT_EQ(attrs.at("decline_reason"), "index_unusable");
        EXPECT_EQ(ctrs.at("rows_in"), 4);
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), 4);
        EXPECT_EQ(ctrs.at("rows_out"), 2);
    }

    // 3. TermFilterExpr (int64 IN [30, 60]): INVERTED selected in prefetch,
    // bypassed under offset_input -> raw fallback.
    {
        auto term_expr = std::make_shared<expr::TermFilterExpr>(
            expr::ColumnInfo(int_fid, DataType::INT64),
            std::vector<proto::plan::GenericValue>{Int64Literal(30),
                                                   Int64Literal(60)});
        term_expr->SetExprId(78613);

        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            kInt64Values.size());
        auto res = eval_after_prefetch(term_expr, explain);
        ExpectHitRows(res, {1, 3}, "prefetch + offset Term IN [30, 60]");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto attrs = AttributesOf(node.children(0));
        const auto ctrs = CountersOf(node.children(0));
        EXPECT_EQ(attrs.find("index_id"), attrs.end());
        EXPECT_EQ(attrs.at("decline_reason"), "index_unusable");
        EXPECT_EQ(ctrs.at("rows_in"), 4);
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), 4);
        EXPECT_EQ(ctrs.at("rows_out"), 2);
    }

    // 4. Projected JSON BinaryRangeFilterExpr (20 < json["score"] < 60):
    // Eval() serves offset_input from the projected JSON index and gathers the
    // requested offsets without scanning raw JSON.
    {
        auto json_range = std::make_shared<expr::BinaryRangeFilterExpr>(
            expr::ColumnInfo(json_fid, DataType::JSON, {"score"}),
            Int64Literal(20),
            Int64Literal(60),
            /*lower_inclusive=*/false,
            /*upper_inclusive=*/false);
        json_range->SetExprId(78614);

        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            kInt64Values.size());
        auto res = eval_after_prefetch(json_range, explain);
        ExpectHitRows(res, {1, 2}, "prefetch + offset JSON BinaryRange");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto attrs = AttributesOf(node.children(0));
        const auto ctrs = CountersOf(node.children(0));
        EXPECT_EQ(attrs.at("index_id"), std::to_string(kJsonIndexId));
        EXPECT_EQ(attrs.at("family"), "inverted");
        EXPECT_EQ(ctrs.at("rows_in"), 4);
        EXPECT_EQ(ctrs.at("idx_scan"), 4);
        EXPECT_EQ(ctrs.at("raw_scan"), 0);
        EXPECT_EQ(ctrs.at("rows_out"), 2);
    }
}

TEST(MultiScalarIndexExprTest,
     ExplainRetrieveCandidateBudgetAndInternalFilterSuppression) {
    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("id", DataType::INT64);
    schema->set_primary_field_id(pk_fid);
    auto val_fid = schema->AddDebugField("val", DataType::INT64);
    auto ttl_fid =
        schema->AddDebugField("ttl", DataType::INT64, /*nullable=*/true);
    schema->set_ttl_field_id(ttl_fid);
    auto ns_fid = schema->AddDebugField("ns", DataType::VARCHAR);
    schema->set_namespace_field_id(ns_fid);

    const int64_t row_count = 4;
    const std::vector<int64_t> ids = {1, 2, 3, 4};
    const std::vector<int64_t> vals = {10, 20, 30, 40};
    const std::vector<int64_t> ttl_vals = {
        9999999999999LL, 9999999999999LL, 9999999999999LL, 9999999999999LL};
    const std::vector<std::string> ns_vals = {"ns_a", "ns_a", "ns_a", "ns_b"};

    auto insert_data = std::make_unique<InsertRecordProto>();
    insert_data->mutable_fields_data()->AddAllocated(
        CreateDataArrayFrom(ids.data(), nullptr, row_count, (*schema)[pk_fid])
            .release());
    insert_data->mutable_fields_data()->AddAllocated(
        CreateDataArrayFrom(vals.data(), nullptr, row_count, (*schema)[val_fid])
            .release());
    bool ttl_valid_bool[4] = {true, true, true, true};
    insert_data->mutable_fields_data()->AddAllocated(
        CreateDataArrayFrom(
            ttl_vals.data(), ttl_valid_bool, row_count, (*schema)[ttl_fid])
            .release());
    auto* ns_fd = insert_data->add_fields_data();
    ns_fd->set_field_id(ns_fid.get());
    ns_fd->set_field_name("ns");
    ns_fd->set_type(proto::schema::DataType::VarChar);
    for (const auto& s : ns_vals) {
        ns_fd->mutable_scalars()->mutable_string_data()->add_data(s);
    }
    insert_data->set_num_rows(row_count);

    GeneratedData generated;
    generated.schema_ = schema;
    generated.raw_ = insert_data.release();
    generated.row_ids_ = {0, 1, 2, 3};
    generated.timestamps_ = {1, 2, 3, 4};
    auto segment = CreateSealedWithFieldDataLoaded(schema, generated);

    // User leaf (expr_id != 0) + untagged Namespace leaf (expr_id == 0) +
    // runtime-injected Entity TTL leaf (expr_id == 0). Only the user leaf must
    // appear under the filter operator.
    auto user_leaf = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(val_fid, DataType::INT64),
        proto::plan::OpType::GreaterEqual,
        Int64Literal(20),
        std::vector<proto::plan::GenericValue>());
    user_leaf->SetExprId(91001);

    proto::plan::GenericValue ns_val;
    ns_val.set_string_val("ns_a");
    auto ns_leaf = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(ns_fid, DataType::VARCHAR),
        proto::plan::OpType::Equal,
        ns_val,
        std::vector<proto::plan::GenericValue>());
    // Untagged internal namespace predicate (expr_id == 0).
    auto merged = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, user_leaf, ns_leaf);

    auto filter_node =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, merged);
    auto mvcc_node = std::make_shared<plan::MvccNode>(
        "2", std::vector<plan::PlanNodePtr>{filter_node});

    auto retrieve_plan = std::make_unique<query::RetrievePlan>(schema);
    retrieve_plan->plan_node_ = std::make_unique<query::RetrievePlanNode>();
    retrieve_plan->plan_node_->plannodes_ = mvcc_node;
    retrieve_plan->plan_node_->limit_ = Unlimited;
    retrieve_plan->plan_node_->plan_options_.enable_explain = true;
    retrieve_plan->mode_ = query::RetrieveMode::Candidates;
    retrieve_plan->max_candidate_rows_ = 0;

    // Candidate offsets for 2 matching rows fit comfortably in 64 bytes, while
    // the serialized explain payload is larger than 64 bytes. Attaching explain
    // after the candidate byte-budget check must let the call succeed AND emit
    // the explain payload.
    auto res = segment->Retrieve(nullptr,
                                 retrieve_plan.get(),
                                 MAX_TIMESTAMP,
                                 /*limit_size=*/64,
                                 /*ignore_non_pk=*/true,
                                 folly::CancellationToken(),
                                 /*consistency_level=*/0,
                                 /*collection_ttl=*/0,
                                 /*entity_ttl_physical_time_us=*/1000);
    ASSERT_NE(res, nullptr);
    EXPECT_EQ(res->offset_size(), 2);
    ASSERT_FALSE(res->explain().empty());

    proto::internal::ExecutionExplain seg_explain;
    ASSERT_TRUE(seg_explain.ParseFromString(res->explain()));
    ASSERT_EQ(seg_explain.children_size(), 2);

    const auto& filter_op = seg_explain.children(0);
    EXPECT_EQ(filter_op.kind(), "operator");
    EXPECT_EQ(AttributesOf(filter_op).at("name"), "filter");
    EXPECT_EQ(CountersOf(filter_op).at("rows_in"), row_count);
    EXPECT_EQ(CountersOf(filter_op).at("rows_out"), 2);
    // Only the user leaf (91001) is emitted; internal TTL and Namespace leaves
    // are suppressed.
    ASSERT_EQ(filter_op.children_size(), 1);
    EXPECT_EQ(AttributesOf(filter_op.children(0)).at("expr_id"), "91001");

    const auto& mvcc_op = seg_explain.children(1);
    EXPECT_EQ(mvcc_op.kind(), "operator");
    EXPECT_EQ(AttributesOf(mvcc_op).at("name"), "mvcc");
    EXPECT_EQ(CountersOf(mvcc_op).at("rows_in"), 2);
    EXPECT_EQ(CountersOf(mvcc_op).at("rows_out"), 2);
}

TEST(MultiScalarIndexExprTest, ExplainArrayRawScanAndAlwaysTrueDeferred) {
    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("id", DataType::INT64);
    schema->set_primary_field_id(pk_fid);
    auto arr_fid =
        schema->AddDebugField("tags", DataType::ARRAY, DataType::INT64);

    const int64_t row_count = 4;
    const std::vector<int64_t> ids = {0, 1, 2, 3};
    auto pk_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    pk_field->FillFieldData(ids.data(), row_count);

    std::vector<proto::schema::ScalarField> arrays(row_count);
    arrays[0].mutable_long_data()->add_data(10);
    arrays[0].mutable_long_data()->add_data(20);
    arrays[1].mutable_long_data()->add_data(30);
    arrays[2].mutable_long_data()->add_data(10);
    arrays[2].mutable_long_data()->add_data(20);
    arrays[3].mutable_long_data()->add_data(10);

    auto arr_field =
        std::make_shared<FieldData<milvus::Array>>(DataType::ARRAY, false);
    std::vector<milvus::Array> arr_vals;
    for (const auto& sa : arrays) {
        arr_vals.emplace_back(sa);
    }
    arr_field->FillFieldData(arr_vals.data(), row_count);

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(pk_fid, {pk_field}));
    segment->LoadFieldData(files.Prepare(arr_fid, {arr_field}));

    // 1. array_length(tags) == 2 via ExecArrayLengthFromChunks records raw_scan.
    {
        auto len_expr = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            expr::ColumnInfo(arr_fid, DataType::ARRAY, DataType::INT64),
            proto::plan::OpType::Equal,
            proto::plan::ArithOpType::ArrayLength,
            Int64Literal(2),
            Int64Literal(0));
        len_expr->SetExprId(92001);

        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        auto res = EvaluateFilter(len_expr, segment.get(), row_count, explain);
        ExpectHitRows(res, {0, 2}, "array_length == 2");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        const auto ctrs = CountersOf(node.children(0));
        EXPECT_EQ(ctrs.at("rows_in"), row_count);
        EXPECT_EQ(ctrs.at("raw_scan"), row_count);
        EXPECT_EQ(ctrs.at("rows_out"), 2);
    }

    // 2. AlwaysTrueExpr compiled without running Eval() (e.g. whole-filter
    // cache hit) must NOT record a premature decision.
    {
        auto always_true = std::make_shared<expr::AlwaysTrueExpr>();
        always_true->SetExprId(92002);
        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        auto qc = std::make_shared<milvus::exec::QueryContext>(
            DEAFULT_QUERY_ID, segment.get(), row_count, MAX_TIMESTAMP);
        qc->set_read_schema(schema);
        qc->SetExplain(explain);
        milvus::exec::ExecContext ec(qc.get());
        std::vector<expr::TypedExprPtr> filters{always_true};
        milvus::exec::ExprSet compiled(filters, &ec);

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        EXPECT_EQ(node.children_size(), 0);
    }
}

TEST(MultiScalarIndexExprTest, ExplainFineGrainedDeclineReasonsAndConstants) {
    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk_fid);
    auto v_fid = schema->AddDebugField("v", DataType::VARCHAR);
    auto i8_fid = schema->AddDebugField("i8", DataType::INT8);
    auto json_fid = schema->AddDebugField("j", DataType::JSON);

    const int64_t n = static_cast<int64_t>(kVarCharValues.size());
    std::vector<int64_t> pks(n);
    std::vector<int8_t> i8s(n);
    std::vector<std::string> jsons;
    for (int64_t i = 0; i < n; ++i) {
        pks[i] = i;
        i8s[i] = static_cast<int8_t>(i);
        jsons.push_back(fmt::format(R"({{"a": "s{}", "b": {}}})", i, i));
    }

    auto pk_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    pk_field->FillFieldData(pks.data(), n);
    auto v_field = milvus::test::expr_index::StringField(kVarCharValues);
    auto i8_field = std::make_shared<FieldData<int8_t>>(DataType::INT8, false);
    i8_field->FillFieldData(i8s.data(), n);
    auto json_field = milvus::test::expr_index::JsonField(jsons);

    RawFieldFiles files;
    auto seg = CreateSealedSegment(schema);
    seg->LoadFieldData(files.Prepare(pk_fid, {pk_field}));
    seg->LoadFieldData(files.Prepare(v_fid, {v_field}));
    seg->LoadFieldData(files.Prepare(i8_fid, {i8_field}));
    seg->LoadFieldData(files.Prepare(json_fid, {json_field}));

    constexpr int64_t kInvertedId = 93001;
    constexpr int64_t kNgramId = 93002;
    constexpr int64_t kI8SortId = 93003;
    constexpr int64_t kJsonBId = 93004;
    constexpr int64_t kJsonAId = 93005;

    InstallIndexWithId(*seg,
                       v_fid,
                       DataType::VARCHAR,
                       kInvertedId,
                       BuildIndex(v_fid,
                                  DataType::VARCHAR,
                                  milvus::index::INVERTED_INDEX_TYPE,
                                  {v_field}));
    InstallIndexWithId(*seg,
                       v_fid,
                       DataType::VARCHAR,
                       kNgramId,
                       BuildIndex(v_fid,
                                  DataType::VARCHAR,
                                  milvus::index::NGRAM_INDEX_TYPE,
                                  {v_field},
                                  {{milvus::index::MIN_GRAM, 3},
                                   {milvus::index::MAX_GRAM, 4}}));
    InstallIndexWithId(
        *seg,
        i8_fid,
        DataType::INT8,
        kI8SortId,
        BuildIndex(
            i8_fid, DataType::INT8, milvus::index::ASCENDING_SORT, {i8_field}));
    InstallIndexWithId(
        *seg,
        json_fid,
        DataType::JSON,
        kJsonBId,
        BuildIndex(json_fid,
                   DataType::JSON,
                   milvus::index::INVERTED_INDEX_TYPE,
                   {json_field},
                   {{JSON_PATH, "/b"}, {JSON_CAST_TYPE, "DOUBLE"}}),
        {{JSON_PATH, "/b"}, {JSON_CAST_TYPE, "DOUBLE"}});
    InstallIndexWithId(
        *seg,
        json_fid,
        DataType::JSON,
        kJsonAId,
        BuildIndex(json_fid,
                   DataType::JSON,
                   milvus::index::INVERTED_INDEX_TYPE,
                   {json_field},
                   {{JSON_PATH, "/a"}, {JSON_CAST_TYPE, "VARCHAR"}}),
        {{JSON_PATH, "/a"}, {JSON_CAST_TYPE, "VARCHAR"}});

    auto eval_reason = [&](const expr::TypedExprPtr& filter_expr,
                           const milvus::query::PlanOptions& opts = {}) {
        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(n);
        (void)EvaluateFilter(filter_expr, seg.get(), n, explain, opts);
        proto::internal::ExecutionExplain node;
        EXPECT_TRUE(node.ParseFromString(explain->Finish()));
        EXPECT_EQ(node.children_size(), 1);
        return std::make_pair(AttributesOf(node.children(0)),
                              CountersOf(node.children(0)));
    };

    // 1. NGRAM + INVERTED with 2-char literal (< min_gram=3) -> literal_not_supported.
    {
        proto::plan::GenericValue pat;
        pat.set_string_val("ab");
        auto short_like = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(v_fid, DataType::VARCHAR),
            proto::plan::OpType::InnerMatch,
            pat,
            std::vector<proto::plan::GenericValue>());
        short_like->SetExprId(93101);
        auto [attrs, ctrs] = eval_reason(short_like);
        EXPECT_EQ(attrs.at("decline_reason"), "literal_not_supported");
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), n);
    }

    // 2. NGRAM + INVERTED with valid 4-char literal, soft-hinting INVERTED -> hinted_index_not_selected.
    {
        proto::plan::GenericValue pat;
        pat.set_string_val("beta");
        auto hinted_like = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(v_fid, DataType::VARCHAR),
            proto::plan::OpType::InnerMatch,
            pat,
            std::vector<proto::plan::GenericValue>());
        hinted_like->SetExprId(93102);
        milvus::query::PlanOptions opts;
        opts.index_hints = {{v_fid.get(), kInvertedId}};
        opts.index_hint_strict = false;
        auto [attrs, ctrs] = eval_reason(hinted_like, opts);
        EXPECT_EQ(attrs.at("decline_reason"), "hinted_index_not_selected");
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), n);
    }

    // 2b. ASCENDING_SORT on INT8 queried with unsupported_op (Equal on JSON/ARRAY or STL_SORT on InnerMatch):
    // test INVERTED-only VARCHAR on InnerMatch -> capability_mismatch.
    {
        RawFieldFiles inv_files;
        auto inv_seg = CreateSealedSegment(schema);
        inv_seg->LoadFieldData(inv_files.Prepare(pk_fid, {pk_field}));
        inv_seg->LoadFieldData(inv_files.Prepare(v_fid, {v_field}));
        InstallIndexWithId(*inv_seg,
                           v_fid,
                           DataType::VARCHAR,
                           kInvertedId,
                           BuildIndex(v_fid,
                                      DataType::VARCHAR,
                                      milvus::index::INVERTED_INDEX_TYPE,
                                      {v_field}));
        proto::plan::GenericValue pat;
        pat.set_string_val("beta");
        auto inner_like = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(v_fid, DataType::VARCHAR),
            proto::plan::OpType::InnerMatch,
            pat,
            std::vector<proto::plan::GenericValue>());
        inner_like->SetExprId(93107);
        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(n);
        (void)EvaluateFilter(inner_like, inv_seg.get(), n, explain);
        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(node.children_size(), 1);
        EXPECT_EQ(AttributesOf(node.children(0)).at("decline_reason"),
                  "capability_mismatch");
    }

    // 3. JSON /c (uncovered path) -> json_path_not_covered.
    {
        auto uncovered = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(json_fid, DataType::JSON, {"c"}),
            proto::plan::OpType::Equal,
            Int64Literal(1),
            std::vector<proto::plan::GenericValue>());
        uncovered->SetExprId(93103);
        auto [attrs, _] = eval_reason(uncovered);
        EXPECT_EQ(attrs.at("decline_reason"), "json_path_not_covered");
    }

    // 4. JSON /a (VARCHAR index) queried with INT64 literal -> value_type_mismatch
    // (must NOT be shadowed by /b's json_path_not_covered).
    {
        auto mismatch = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(json_fid, DataType::JSON, {"a"}),
            proto::plan::OpType::Equal,
            Int64Literal(1),
            std::vector<proto::plan::GenericValue>());
        mismatch->SetExprId(93104);
        auto [attrs, _] = eval_reason(mismatch);
        EXPECT_EQ(attrs.at("decline_reason"), "value_type_mismatch");
    }

    // 5. Indexed INT8 overflow (i8 > 200) -> constant_false with 0 idx_scan & 0 raw_scan.
    {
        auto overflow_gt = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(i8_fid, DataType::INT8),
            proto::plan::OpType::GreaterThan,
            Int64Literal(200),
            std::vector<proto::plan::GenericValue>());
        overflow_gt->SetExprId(93105);
        auto [attrs, ctrs] = eval_reason(overflow_gt);
        EXPECT_EQ(attrs.count("index_id"), 0u);
        EXPECT_EQ(attrs.at("decline_reason"), "constant_false");
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), 0);
        EXPECT_EQ(ctrs.at("rows_out"), 0);
    }

    // 6. Indexed JSON json_contains_any(j['a'], []) -> constant_false with 0 idx_scan & 0 raw_scan.
    {
        auto empty_contains = std::make_shared<expr::JsonContainsExpr>(
            expr::ColumnInfo(json_fid, DataType::JSON, {"a"}),
            proto::plan::JSONContainsExpr_JSONOp_ContainsAny,
            /*same_type=*/true,
            std::vector<proto::plan::GenericValue>{});
        empty_contains->SetExprId(93106);
        auto [attrs, ctrs] = eval_reason(empty_contains);
        EXPECT_EQ(attrs.count("index_id"), 0u);
        EXPECT_EQ(attrs.at("decline_reason"), "constant_false");
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), 0);
        EXPECT_EQ(ctrs.at("rows_out"), 0);
    }
}

TEST(MultiScalarIndexExprTest,
     ExplainLeafCacheHitReportsCachedWithZeroRawScan) {
    using milvus::exec::ExprResCacheManager;
    auto& mgr = ExprResCacheManager::Instance();
    const bool prev_enabled = ExprResCacheManager::IsEnabled();
    ExprResCacheManager::SetEnabled(true);
    mgr.Clear();
    mgr.SetCapacityBytes(1ULL << 20);
    auto restore = folly::makeGuard([&] {
        mgr.Clear();
        ExprResCacheManager::SetEnabled(prev_enabled);
    });

    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("id", DataType::INT64);
    schema->set_primary_field_id(pk_fid);
    auto raw_fid = schema->AddDebugField("raw_str", DataType::VARCHAR);

    const int64_t n = static_cast<int64_t>(kVarCharValues.size());
    std::vector<int64_t> pks(n);
    for (int64_t i = 0; i < n; ++i) {
        pks[i] = i;
    }
    auto pk_field =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    pk_field->FillFieldData(pks.data(), n);
    auto str_field = milvus::test::expr_index::StringField(kVarCharValues);

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(pk_fid, {pk_field}));
    segment->LoadFieldData(files.Prepare(raw_fid, {str_field}));

    proto::plan::GenericValue target;
    target.set_string_val("prefix_beta");
    auto eq = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(raw_fid, DataType::VARCHAR),
        proto::plan::OpType::Equal,
        target,
        std::vector<proto::plan::GenericValue>());
    eq->SetExprId(93201);

    auto eval_with_sub_cache = [&]() {
        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(n);
        auto qc = std::make_shared<milvus::exec::QueryContext>(
            DEAFULT_QUERY_ID, segment.get(), n, MAX_TIMESTAMP);
        qc->set_read_schema(schema);
        qc->set_read_snapshot(segment->CaptureReadSnapshot());
        qc->set_enable_sub_expr_cache_write(true);
        qc->SetExplain(explain);
        milvus::exec::ExecContext ec(qc.get());
        ec.SetExplain(explain);
        std::vector<expr::TypedExprPtr> filters{eq};
        milvus::exec::ExprSet exprs(filters, &ec, /*null_rejecting=*/true);
        std::vector<VectorPtr> results;
        milvus::exec::EvalCtx eval_ctx(&ec);
        exprs.Eval(0, 1, true, eval_ctx, results);
        return explain;
    };

    // Cold run scans raw_str and populates RawExprCacheAdapter.
    auto cold = eval_with_sub_cache();
    proto::internal::ExecutionExplain cold_node;
    ASSERT_TRUE(cold_node.ParseFromString(cold->Finish()));
    ASSERT_EQ(cold_node.children_size(), 1);
    EXPECT_EQ(AttributesOf(cold_node.children(0)).at("decline_reason"),
              "no_index_on_field");
    EXPECT_EQ(CountersOf(cold_node.children(0)).at("raw_scan"), n);

    // Warm run hits RawExprCacheAdapter: decline_reason="cached", raw_scan=0.
    auto warm = eval_with_sub_cache();
    proto::internal::ExecutionExplain warm_node;
    ASSERT_TRUE(warm_node.ParseFromString(warm->Finish()));
    ASSERT_EQ(warm_node.children_size(), 1);
    const auto warm_attrs = AttributesOf(warm_node.children(0));
    const auto warm_ctrs = CountersOf(warm_node.children(0));
    EXPECT_EQ(warm_attrs.at("decline_reason"), "cached");
    EXPECT_EQ(warm_ctrs.at("idx_scan"), 0);
    EXPECT_EQ(warm_ctrs.at("raw_scan"), 0);
    EXPECT_EQ(warm_ctrs.at("rows_in"), n);
    EXPECT_EQ(warm_ctrs.at("rows_out"), 2);
}
