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
               int64_t row_count) {
    auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                       filter_expr);
    return milvus::test::gen_filter_res(
        plan.get(), segment, row_count, MAX_TIMESTAMP);
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
    std::unique_ptr<
        milvus::cachinglayer::Translator<milvus::index::IIndexReaderBase>>
        translator =
            std::make_unique<milvus::test::expr_index::ReaderTranslator>(
                std::move(opened.reader), nullptr);
    info.cache_index =
        milvus::cachinglayer::Manager::GetInstance().CreateCacheSlot(
            std::move(translator));
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
    const auto capability = indexed_segment->IndexCapability(varchar_fid);
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

    const auto capability = segment->IndexCapability(varchar_fid);
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

    const auto capability = indexed_segment->IndexCapability(varchar_fid);
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
        const auto capability = indexed_segment->IndexCapability(int64_fid);
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

    const auto capability = indexed_segment->IndexCapability(json_fid);
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

    const auto capability = indexed_segment->IndexCapability(json_fid);
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
