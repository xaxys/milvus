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
#include "common/Types.h"
#include "gtest/gtest.h"
#include "index/Meta.h"
#include "index/contracts/query/IIndexReaderBase.h"
#include "index/contracts/query/INgramReader.h"
#include "pb/plan.pb.h"
#include "query/PlanNode.h"
#include "segcore/SegmentSealed.h"
#include "segcore/indexing/FieldIndexCapability.h"
#include "test_utils/GenExprProto.h"

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
EvaluateFilter(const expr::TypedExprPtr& filter_expr,
               const segcore::SegmentInternalInterface* segment,
               int64_t row_count,
               milvus::exec::ExecutionExplainRecorder* explain = nullptr,
               const milvus::query::PlanOptions& plan_options =
                   milvus::query::PlanOptions()) {
    auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                       filter_expr);
    return milvus::test::gen_filter_res(
        plan.get(), segment, row_count, MAX_TIMESTAMP, nullptr, explain, plan_options);
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
    const auto capability = indexed_segment->ResolveIndexCapability(nullptr, varchar_fid);
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
    auto varchar_fid = schema->AddDebugField("varchar_field", DataType::VARCHAR);
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

    const auto capability = segment->ResolveIndexCapability(nullptr, varchar_fid);
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

    const auto capability = indexed_segment->ResolveIndexCapability(nullptr, varchar_fid);
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
        const auto capability = indexed_segment->ResolveIndexCapability(nullptr, int64_fid);
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

    const auto capability = indexed_segment->ResolveIndexCapability(nullptr, json_fid);
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

    const auto capability = indexed_segment->ResolveIndexCapability(nullptr, json_fid);
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

    milvus::segcore::IndexCapabilityEntry entry{
        milvus::segcore::IndexKey{
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
        << "min_gram 3 cannot answer a 2-character literal; selection must say so";
    ASSERT_NE(short_literal.fallback_reason, milvus::exec::DeclineReason::kNone);
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
    entry.value_type = DataType::VARCHAR;  // does not match the requirement below
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
    EXPECT_THROW(milvus::exec::DetermineExecPath(req, caps), milvus::SegcoreError);
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
    const std::vector<std::pair<int64_t, int64_t>> hint{
        {kField.get(), 6102}, {kField.get(), 6101}};
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

    // No index: the leaf falls back to the raw column and says why.
    {
        RawFieldFiles files;
        auto segment = CreateSealedSegment(schema);
        segment->LoadFieldData(files.Prepare(varchar_fid, {field_data}));

        milvus::exec::ExecutionExplainRecorder explain(kVarCharValues.size());
        auto result = EvaluateFilter(
            equal_expr, segment.get(), kVarCharValues.size(), &explain);
        ExpectHitRows(result, {2, 6}, "Equal \"prefix_beta\" without index");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain.Finish()));
        EXPECT_EQ(node.kind(), "segment");
        const auto counters = CountersOf(node);
        EXPECT_EQ(counters.at("rows_total"),
                  static_cast<int64_t>(kVarCharValues.size()));
        EXPECT_EQ(counters.at("rows_matched"), 0)
            << "nothing was matched by an index";
        ASSERT_EQ(node.children_size(), 1);
        const auto& group = node.children(0);
        EXPECT_EQ(group.kind(), "filter_group");
        const auto attributes = AttributesOf(group);
        EXPECT_EQ(attributes.at("field_id"),
                  std::to_string(varchar_fid.get()));
        EXPECT_EQ(attributes.at("decline_reason"), "no_index_on_field");
        const auto group_counters = CountersOf(group);
        EXPECT_EQ(group_counters.at("leaves"), 1);
        EXPECT_EQ(group_counters.at("rows_matched"), 0)
            << "a raw leaf has no index hits to report";
        EXPECT_EQ(group_counters.at("rows_scanned"),
                  static_cast<int64_t>(kVarCharValues.size()))
            << "the raw leaf evaluated every row once";
    }

    // INVERTED serves the leaf: the group names the index and counts its rows.
    {
        RawFieldFiles files;
        auto segment = CreateSealedSegment(schema);
        segment->LoadFieldData(files.Prepare(varchar_fid, {field_data}));
        InstallIndexWithId(
            *segment,
            varchar_fid,
            DataType::VARCHAR,
            kIndexId,
            BuildIndex(varchar_fid,
                       DataType::VARCHAR,
                       milvus::index::INVERTED_INDEX_TYPE,
                       {field_data}));

        milvus::exec::ExecutionExplainRecorder explain(kVarCharValues.size());
        auto result = EvaluateFilter(
            equal_expr, segment.get(), kVarCharValues.size(), &explain);
        ExpectHitRows(result, {2, 6}, "Equal \"prefix_beta\" over INVERTED");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain.Finish()));
        EXPECT_EQ(CountersOf(node).at("rows_matched"), 2);
        ASSERT_EQ(node.children_size(), 1);
        const auto& group = node.children(0);
        const auto attributes = AttributesOf(group);
        EXPECT_EQ(attributes.at("index_id"), std::to_string(kIndexId));
        EXPECT_EQ(attributes.find("decline_reason"), attributes.end())
            << "an index leaf declined nothing";
        const auto group_counters = CountersOf(group);
        EXPECT_EQ(group_counters.at("rows_matched"), 2);
        EXPECT_EQ(group_counters.at("leaves"), 1);
        EXPECT_EQ(group_counters.at("rows_scanned"), 0)
            << "an index leaf scans no rows";
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
    auto varchar_fid = schema->AddDebugField("varchar_field", DataType::VARCHAR);
    const std::vector<std::string> values = {
        "abc", "abd", "xab", "zzz", "abcd", "abc"};
    auto field_data = milvus::test::expr_index::StringField(values);

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(varchar_fid, {field_data}));
    // min_gram 3 cannot answer a two-character literal; min_gram 2 can.
    InstallIndexWithId(
        *segment,
        varchar_fid,
        DataType::VARCHAR,
        kLongGramIndexId,
        BuildIndex(varchar_fid,
                   DataType::VARCHAR,
                   milvus::index::NGRAM_INDEX_TYPE,
                   {field_data},
                   {{milvus::index::MIN_GRAM, 3}, {milvus::index::MAX_GRAM, 4}}));
    InstallIndexWithId(
        *segment,
        varchar_fid,
        DataType::VARCHAR,
        kShortGramIndexId,
        BuildIndex(varchar_fid,
                   DataType::VARCHAR,
                   milvus::index::NGRAM_INDEX_TYPE,
                   {field_data},
                   {{milvus::index::MIN_GRAM, 2}, {milvus::index::MAX_GRAM, 3}}));

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

    milvus::exec::ExecutionExplainRecorder explain(values.size());
    ColumnVectorPtr result;
    ASSERT_NO_THROW(result = EvaluateFilter(
                        like_expr, segment.get(), values.size(), &explain, options))
        << "the second hinted ngram index can serve the literal";

    ExpectHitRows(result, {0, 1, 4, 5}, "LIKE 'ab%' over two hinted ngram indexes");

    proto::internal::ExecutionExplain node;
    ASSERT_TRUE(node.ParseFromString(explain.Finish()));
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
    proto::plan::GenericValue prefix_value;
    prefix_value.set_string_val("prefix");
    auto right = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(right_fid, DataType::VARCHAR),
        proto::plan::OpType::PrefixMatch,
        prefix_value,
        std::vector<proto::plan::GenericValue>());
    auto both = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, left, right);

    milvus::exec::ExecutionExplainRecorder explain(kVarCharValues.size());
    auto result =
        EvaluateFilter(both, segment.get(), kVarCharValues.size(), &explain);
    ExpectHitRows(result, {}, "AND that matches nothing");

    proto::internal::ExecutionExplain node;
    ASSERT_TRUE(node.ParseFromString(explain.Finish()));
    ASSERT_EQ(node.children_size(), 2);
    bool saw_left = false;
    bool saw_right = false;
    for (int i = 0; i < node.children_size(); ++i) {
        const auto attributes = AttributesOf(node.children(i));
        const auto counters = CountersOf(node.children(i));
        if (attributes.at("field_id") == std::to_string(left_fid.get())) {
            saw_left = true;
            EXPECT_EQ(counters.at("rows_scanned"),
                      static_cast<int64_t>(kVarCharValues.size()))
                << "the left leaf evaluated every row";
        } else {
            EXPECT_EQ(attributes.at("field_id"),
                      std::to_string(right_fid.get()));
            saw_right = true;
            EXPECT_EQ(counters.at("rows_scanned"), 0)
                << "the right leaf was handed no row to evaluate";
        }
    }
    EXPECT_TRUE(saw_left && saw_right);
}
