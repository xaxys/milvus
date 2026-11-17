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

    proto::plan::GenericValue inner_value;
    inner_value.set_string_val("beta");
    auto inner_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::InnerMatch,  // LIKE '%beta%'
        inner_value,
        std::vector<proto::plan::GenericValue>());
    auto raw_inner =
        EvaluateFilter(inner_expr, raw_segment.get(), kVarCharValues.size());
    ExpectSameFilterResult(
        raw_inner,
        EvaluateFilter(
            inner_expr, indexed_segment.get(), kVarCharValues.size()),
        "LIKE '%beta%' over INVERTED+NGRAM");
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

    milvus::exec::ExprIndexRequirement req;
    req.field_id = kField;
    req.reader = milvus::exec::RequiredReader::PatternMatch;
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

TEST(MultiScalarIndexExprTest,
     PatternMatchInvertedDoesNotStarveSiblingFmIndexOrBitmap) {
    constexpr FieldId kField(100);
    auto make_pattern_entry = [](int64_t id, const char* family) {
        milvus::segcore::IndexCapabilityEntry entry{milvus::segcore::IndexKey{
            kField, milvus::segcore::IndexIdentity::PreBuiltIndex(id)}};
        entry.family = family;
        entry.value_type = DataType::VARCHAR;
        entry.caps.pattern_match = true;
        entry.caps.exact = true;
        return entry;
    };
    auto make_ngram_entry = [](int64_t id) {
        milvus::segcore::IndexCapabilityEntry entry{milvus::segcore::IndexKey{
            kField, milvus::segcore::IndexIdentity::PreBuiltIndex(id)}};
        entry.family = "ngram";
        entry.value_type = DataType::VARCHAR;
        entry.caps.ngram_candidates = true;
        entry.caps.exact = false;
        entry.params.ngram.min_gram = 3;
        return entry;
    };

    // INVERTED is created first (lowest index_id), followed by BITMAP and FMINDEX.
    milvus::segcore::FieldIndexCapability caps(
        kField,
        {make_pattern_entry(7001, "inverted"),
         make_pattern_entry(7002, "bitmap"),
         make_pattern_entry(7003, "fmindex")});

    for (const auto op : {milvus::index::PatternOp::InnerMatch,
                          milvus::index::PatternOp::PostfixMatch,
                          milvus::index::PatternOp::RegexMatch}) {
        milvus::exec::ExprIndexRequirement req;
        req.field_id = kField;
        req.reader = milvus::exec::RequiredReader::PatternMatch;
        req.value_type = DataType::VARCHAR;
        req.has_pattern = true;
        req.pattern_op = op;
        req.pattern = "abc";
        req.accepts_candidates = true;

        const auto decision = milvus::exec::DetermineExecPath(req, caps);
        EXPECT_EQ(decision.path, milvus::exec::ExprExecPath::ScalarIndex);
        ASSERT_TRUE(decision.key.has_value());
        EXPECT_EQ(decision.key->identity,
                  milvus::segcore::IndexIdentity::PreBuiltIndex(7003));
    }

    // With NGRAM also present, non-prefix pattern ops prefer NGRAM over
    // exact dictionary/FM families, while PrefixMatch still prefers MARISA.
    milvus::segcore::FieldIndexCapability with_ngram_caps(
        kField,
        {make_pattern_entry(7001, "inverted"),
         make_pattern_entry(7002, "bitmap"),
         make_pattern_entry(7003, "fmindex"),
         make_ngram_entry(7004)});
    {
        milvus::exec::ExprIndexRequirement req;
        req.field_id = kField;
        req.reader = milvus::exec::RequiredReader::PatternMatch;
        req.value_type = DataType::VARCHAR;
        req.has_pattern = true;
        req.pattern_op = milvus::index::PatternOp::InnerMatch;
        req.pattern = "abc";
        req.accepts_candidates = true;
        const auto decision =
            milvus::exec::DetermineExecPath(req, with_ngram_caps);
        ASSERT_TRUE(decision.key.has_value());
        EXPECT_EQ(decision.key->identity,
                  milvus::segcore::IndexIdentity::PreBuiltIndex(7004));
    }

    // PrefixMatch prefers marisa / sort > inverted > fmindex > bitmap > ngram.
    milvus::segcore::FieldIndexCapability prefix_caps(
        kField,
        {make_ngram_entry(7010),
         make_pattern_entry(7011, "fmindex"),
         make_pattern_entry(7012, "inverted"),
         make_pattern_entry(7013, "marisa")});
    milvus::exec::ExprIndexRequirement prefix_req;
    prefix_req.field_id = kField;
    prefix_req.reader = milvus::exec::RequiredReader::PatternMatch;
    prefix_req.value_type = DataType::VARCHAR;
    prefix_req.has_pattern = true;
    prefix_req.pattern_op = milvus::index::PatternOp::PrefixMatch;
    prefix_req.pattern = "abc";
    const auto prefix_decision =
        milvus::exec::DetermineExecPath(prefix_req, prefix_caps);
    ASSERT_TRUE(prefix_decision.key.has_value());
    EXPECT_EQ(prefix_decision.key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(7013));
}

TEST(MultiScalarIndexExprTest,
     WholeArrayEqualityExcludesBitmapAndSelectsSiblingInverted) {
    constexpr FieldId kField(100);
    auto make_entry = [](int64_t id, const char* family) {
        milvus::segcore::IndexCapabilityEntry entry{milvus::segcore::IndexKey{
            kField, milvus::segcore::IndexIdentity::PreBuiltIndex(id)}};
        entry.family = family;
        entry.value_type = DataType::INT64;
        entry.caps.predicate = true;
        entry.caps.exact = true;
        return entry;
    };

    milvus::segcore::FieldIndexCapability caps(
        kField, {make_entry(7101, "bitmap"), make_entry(7102, "inverted")});

    milvus::exec::ExprIndexRequirement elem_req;
    elem_req.field_id = kField;
    elem_req.reader = milvus::exec::RequiredReader::Predicate;
    elem_req.value_type = DataType::INT64;
    EXPECT_EQ(milvus::exec::DetermineExecPath(elem_req, caps).key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(7101));

    milvus::exec::ExprIndexRequirement whole_array_req = elem_req;
    whole_array_req.is_whole_array = true;
    const auto decision =
        milvus::exec::DetermineExecPath(whole_array_req, caps);
    EXPECT_EQ(decision.path, milvus::exec::ExprExecPath::ScalarIndex);
    ASSERT_TRUE(decision.key.has_value());
    EXPECT_EQ(decision.key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(7102));
}

TEST(MultiScalarIndexExprTest,
     JsonCastFunctionOutranksSiblingWithoutCastFunction) {
    constexpr FieldId kField(100);
    milvus::segcore::IndexCapabilityEntry plain{milvus::segcore::IndexKey{
        kField, milvus::segcore::IndexIdentity::PreBuiltIndex(7201)}};
    plain.family = "bitmap";
    plain.json_path = "/n";
    plain.json_cast_type = JsonCastType::FromString("DOUBLE");
    plain.value_type = DataType::DOUBLE;
    plain.caps.predicate = true;
    plain.caps.exact = true;

    milvus::segcore::IndexCapabilityEntry with_cast_fn{
        milvus::segcore::IndexKey{
            kField, milvus::segcore::IndexIdentity::PreBuiltIndex(7202)}};
    with_cast_fn.family = "inverted";
    with_cast_fn.json_path = "/n";
    with_cast_fn.json_cast_type = JsonCastType::FromString("DOUBLE");
    with_cast_fn.json_cast_function = "STRING_TO_DOUBLE";
    with_cast_fn.value_type = DataType::DOUBLE;
    with_cast_fn.caps.predicate = true;
    with_cast_fn.caps.exact = true;

    milvus::segcore::FieldIndexCapability caps(kField, {plain, with_cast_fn});
    milvus::exec::ExprIndexRequirement req;
    req.field_id = kField;
    req.reader = milvus::exec::RequiredReader::Predicate;
    req.value_type = DataType::DOUBLE;
    req.is_json_field = true;
    req.json_path = "/n";

    const auto decision = milvus::exec::DetermineExecPath(req, caps);
    ASSERT_TRUE(decision.key.has_value());
    EXPECT_EQ(decision.key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(7202));
}

TEST(MultiScalarIndexExprTest,
     NullReaderSkipsNestedEntryToSelectRowDomainSibling) {
    constexpr FieldId kField(100);
    milvus::segcore::IndexCapabilityEntry nested_entry{
        milvus::segcore::IndexKey{
            kField, milvus::segcore::IndexIdentity::PreBuiltIndex(7251)}};
    nested_entry.family = "inverted";
    nested_entry.value_type = DataType::INT64;
    nested_entry.caps.predicate = true;
    nested_entry.caps.nested = true;
    nested_entry.caps.exact = false;

    milvus::segcore::IndexCapabilityEntry row_entry{milvus::segcore::IndexKey{
        kField, milvus::segcore::IndexIdentity::PreBuiltIndex(7252)}};
    row_entry.family = "inverted";
    row_entry.value_type = DataType::INT64;
    row_entry.caps.predicate = true;
    row_entry.caps.nested = false;
    row_entry.caps.exact = true;

    milvus::segcore::FieldIndexCapability caps(kField,
                                               {nested_entry, row_entry});
    milvus::exec::ExprIndexRequirement req;
    req.field_id = kField;
    req.reader = milvus::exec::RequiredReader::Null;
    req.value_type = DataType::INT64;

    const auto decision = milvus::exec::DetermineExecPath(req, caps);
    ASSERT_TRUE(decision.key.has_value());
    EXPECT_EQ(decision.key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(7252));
}

TEST(MultiScalarIndexExprTest, NullableArrayNullExprSelectsScalarIndex) {
    auto schema = std::make_shared<Schema>();
    auto arr_fid = schema->AddDebugArrayField(
        "arr_int", DataType::INT64, /*nullable=*/true);

    std::vector<Array> values;
    for (const auto& row :
         std::vector<std::vector<int64_t>>{{1, 2}, {}, {3}, {}, {4, 5}}) {
        proto::schema::ScalarField data;
        auto* out = data.mutable_long_data();
        for (int64_t v : row) {
            out->add_data(v);
        }
        values.emplace_back(data);
    }
    // Rows 0, 2, 4 are non-null (0b10101 = 0x15); rows 1, 3 are NULL.
    const uint8_t validity = 0x15;
    auto field_data = std::make_shared<FieldData<Array>>(DataType::ARRAY, true);
    field_data->FillFieldData(values.data(), &validity, values.size(), 0);

    RawFieldFiles raw_files;
    auto raw_segment = CreateSealedSegment(schema);
    raw_segment->LoadFieldData(raw_files.Prepare(arr_fid, {field_data}));

    // Indexed segment WITHOUT loading raw field data: proves PhyNullExpr uses
    // the scalar index (element_type=INT64) rather than falling back to raw scan!
    auto indexed_segment = CreateSealedSegment(schema);
    auto opened = BuildIndex(arr_fid,
                             DataType::ARRAY,
                             milvus::index::INVERTED_INDEX_TYPE,
                             {field_data},
                             Config::object(),
                             DataType::INT64);
    InstallIndex(*indexed_segment,
                 arr_fid,
                 DataType::ARRAY,
                 std::move(opened),
                 DataType::INT64);

    expr::ColumnInfo col_info(arr_fid,
                              DataType::ARRAY,
                              DataType::INT64,
                              std::vector<std::string>{},
                              /*nullable=*/true);
    auto is_null_expr = std::make_shared<expr::NullExpr>(
        col_info, proto::plan::NullExpr_NullOp_IsNull);
    auto is_not_null_expr = std::make_shared<expr::NullExpr>(
        col_info, proto::plan::NullExpr_NullOp_IsNotNull);

    EXPECT_TRUE(milvus::test::CanExprExecuteAllAtOnce(
        is_null_expr, indexed_segment.get(), values.size()));
    EXPECT_TRUE(milvus::test::CanExprExecuteAllAtOnce(
        is_not_null_expr, indexed_segment.get(), values.size()));

    auto raw_is_null =
        EvaluateFilter(is_null_expr, raw_segment.get(), values.size());
    auto idx_is_null =
        EvaluateFilter(is_null_expr, indexed_segment.get(), values.size());
    ExpectHitRows(idx_is_null, {1, 3}, "ARRAY<INT64> IS NULL via index");
    ExpectSameFilterResult(raw_is_null, idx_is_null, "ARRAY<INT64> IS NULL");

    auto raw_is_not_null =
        EvaluateFilter(is_not_null_expr, raw_segment.get(), values.size());
    auto idx_is_not_null =
        EvaluateFilter(is_not_null_expr, indexed_segment.get(), values.size());
    ExpectHitRows(
        idx_is_not_null, {0, 2, 4}, "ARRAY<INT64> IS NOT NULL via index");
    ExpectSameFilterResult(
        raw_is_not_null, idx_is_not_null, "ARRAY<INT64> IS NOT NULL");
}

TEST(MultiScalarIndexExprTest, MultiNgramSelectsUsableMinGram) {
    constexpr FieldId kField(100);
    auto make_ngram = [](int64_t id, size_t min_gram, size_t /*max_gram*/) {
        milvus::segcore::IndexCapabilityEntry entry{milvus::segcore::IndexKey{
            kField, milvus::segcore::IndexIdentity::PreBuiltIndex(id)}};
        entry.family = "ngram";
        entry.value_type = DataType::VARCHAR;
        entry.caps.ngram_candidates = true;
        entry.caps.exact = false;
        entry.params.ngram.min_gram = min_gram;
        return entry;
    };

    milvus::segcore::FieldIndexCapability caps(
        kField, {make_ngram(7301, 4, 6), make_ngram(7302, 2, 4)});

    milvus::exec::ExprIndexRequirement req;
    req.field_id = kField;
    req.reader = milvus::exec::RequiredReader::PatternMatch;
    req.has_pattern = true;
    req.pattern = "abc";
    req.pattern_op = milvus::index::PatternOp::InnerMatch;

    const auto decision = milvus::exec::DetermineExecPath(req, caps);
    EXPECT_EQ(decision.path, milvus::exec::ExprExecPath::ScalarIndex);
    ASSERT_TRUE(decision.key.has_value());
    EXPECT_EQ(decision.key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(7302));
}

TEST(MultiScalarIndexExprTest,
     ExactTypedJsonPathOutranksSamePathJsonFlatIndex) {
    constexpr FieldId kJsonField(102);

    milvus::segcore::IndexCapabilityEntry flat_entry{milvus::segcore::IndexKey{
        kJsonField, milvus::segcore::IndexIdentity::PreBuiltIndex(7401)}};
    flat_entry.json_path = "/a";
    flat_entry.json_cast_type = JsonCastType::FromString("JSON");
    flat_entry.family = "json_flat";
    flat_entry.value_type = DataType::JSON;
    flat_entry.caps = {.predicate = true, .json_paths = true, .exact = true};

    milvus::segcore::IndexCapabilityEntry sort_entry{milvus::segcore::IndexKey{
        kJsonField, milvus::segcore::IndexIdentity::PreBuiltIndex(7402)}};
    sort_entry.json_path = "/a";
    sort_entry.json_cast_type = JsonCastType::FromString("VARCHAR");
    sort_entry.family = "sort";
    sort_entry.value_type = DataType::VARCHAR;
    sort_entry.caps = {.predicate = true,
                       .value_lookup = true,
                       .cheap_value_lookup = true,
                       .exact = true};

    milvus::segcore::FieldIndexCapability caps(kJsonField,
                                               {flat_entry, sort_entry});

    milvus::exec::ExprIndexRequirement req;
    req.field_id = kJsonField;
    req.reader = milvus::exec::RequiredReader::Predicate;
    req.value_type = DataType::VARCHAR;
    req.is_json_field = true;
    req.json_path = "/a";

    const auto decision = milvus::exec::DetermineExecPath(req, caps);
    EXPECT_EQ(decision.path, milvus::exec::ExprExecPath::ScalarIndex);
    ASSERT_TRUE(decision.key.has_value());
    EXPECT_EQ(decision.key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(7402));
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
    milvus::segcore::FieldIndexCapability caps(
        kField, {make_entry(6001, "inverted"), make_entry(6002, "bitmap")});

    milvus::exec::ExprIndexRequirement req;
    req.field_id = kField;
    req.reader = milvus::exec::RequiredReader::Predicate;
    req.value_type = DataType::INT64;
    EXPECT_EQ(milvus::exec::DetermineExecPath(req, caps).key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(6002));

    const std::vector<milvus::IndexHintOption> hint{
        {.field_id = kField.get(), .index_id = 6001}};
    req.index_hints = &hint;
    EXPECT_EQ(milvus::exec::DetermineExecPath(req, caps).key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(6001));

    const std::vector<milvus::IndexHintOption> other_field{
        {.field_id = 999, .index_id = 6001}};
    req.index_hints = &other_field;
    EXPECT_EQ(milvus::exec::DetermineExecPath(req, caps).key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(6002));
}

TEST(MultiScalarIndexExprTest,
     StrictHintRejectsLoadedIncompatibleAndAllowsAbsentIndex) {
    constexpr FieldId kField(100);
    milvus::segcore::IndexCapabilityEntry ngram_entry{milvus::segcore::IndexKey{
        kField, milvus::segcore::IndexIdentity::PreBuiltIndex(6101)}};
    ngram_entry.family = "ngram";
    ngram_entry.value_type = DataType::VARCHAR;
    ngram_entry.caps.ngram_candidates = true;
    ngram_entry.caps.exact = false;
    milvus::segcore::FieldIndexCapability caps(kField, {ngram_entry});

    milvus::exec::ExprIndexRequirement req;
    req.field_id = kField;
    req.reader = milvus::exec::RequiredReader::Predicate;
    req.value_type = DataType::INT64;
    const std::vector<milvus::IndexHintOption> hint{
        {.field_id = kField.get(), .index_id = 6101}};
    req.index_hints = &hint;

    EXPECT_EQ(milvus::exec::DetermineExecPath(req, caps).path,
              milvus::exec::ExprExecPath::RawData);

    req.index_hint_strict = true;
    EXPECT_THROW(milvus::exec::DetermineExecPath(req, caps),
                 milvus::SegcoreError);

    // When a segment does NOT carry the hinted index (6199) at all (for example
    // a newly flushed, low-cardinality, or <1024-row segment), strict mode lets
    // the segment proceed with normal selection instead of failing the query.
    milvus::segcore::IndexCapabilityEntry old_entry{milvus::segcore::IndexKey{
        kField, milvus::segcore::IndexIdentity::PreBuiltIndex(6101)}};
    old_entry.family = "inverted";
    old_entry.value_type = DataType::INT64;
    old_entry.caps.predicate = true;
    old_entry.caps.exact = true;
    milvus::segcore::FieldIndexCapability old_caps(kField, {old_entry});
    const std::vector<milvus::IndexHintOption> unloaded_hint{
        {.field_id = kField.get(), .index_id = 6199}};
    req.index_hints = &unloaded_hint;
    auto decision = milvus::exec::DetermineExecPath(req, old_caps);
    EXPECT_EQ(decision.path, milvus::exec::ExprExecPath::ScalarIndex);
    ASSERT_TRUE(decision.key.has_value());
    EXPECT_EQ(decision.key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(6101));
}

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
    milvus::segcore::FieldIndexCapability caps(
        kField, {make_entry(6101, "bitmap"), make_entry(6102, "inverted")});

    milvus::exec::ExprIndexRequirement req;
    req.field_id = kField;
    req.reader = milvus::exec::RequiredReader::Predicate;
    req.value_type = DataType::INT64;
    const std::vector<milvus::IndexHintOption> hint{
        {.field_id = kField.get(), .index_id = 6102},
        {.field_id = kField.get(), .index_id = 6101},
    };
    req.index_hints = &hint;

    auto decision = milvus::exec::DetermineExecPath(req, caps);
    ASSERT_TRUE(decision.key.has_value());
    EXPECT_EQ(decision.key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(6102));
}

TEST(MultiScalarIndexExprTest, StrictHintAllowsRawWhenFieldHasNoEntry) {
    constexpr FieldId kField(100);
    milvus::segcore::FieldIndexCapability caps(kField, {});

    milvus::exec::ExprIndexRequirement req;
    req.field_id = kField;
    req.reader = milvus::exec::RequiredReader::Predicate;
    req.value_type = DataType::INT64;
    const std::vector<milvus::IndexHintOption> hint{
        {.field_id = kField.get(), .index_id = 6101}};
    req.index_hints = &hint;
    req.index_hint_strict = true;

    auto decision = milvus::exec::DetermineExecPath(req, caps);
    EXPECT_EQ(decision.path, milvus::exec::ExprExecPath::RawData);
}

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

    const std::vector<milvus::IndexHintOption> hint{
        {.field_id = kField.get(), .index_id = 6101}};

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
    auto decision_b = milvus::exec::DetermineExecPath(req_b, caps);
    EXPECT_EQ(decision_b.path, milvus::exec::ExprExecPath::ScalarIndex);
    ASSERT_TRUE(decision_b.key.has_value());
    EXPECT_EQ(decision_b.key->identity,
              milvus::segcore::IndexIdentity::PreBuiltIndex(6102));

    // Conversely, hinting an unloaded index on /b lets the segment fall back to
    // raw data on /b without failing strict mode even when /a is loaded; only a
    // loaded incompatible index on /b fails strict mode.
    const std::vector<milvus::IndexHintOption> unloaded_b_hint{
        {.field_id = kField.get(), .index_id = 6199}};
    milvus::segcore::FieldIndexCapability only_a_caps(
        kField, {make_entry(6101, path_a)});
    req_b.index_hints = &unloaded_b_hint;
    EXPECT_EQ(milvus::exec::DetermineExecPath(req_b, only_a_caps).path,
              milvus::exec::ExprExecPath::RawData);
}

TEST(MultiScalarIndexExprTest,
     StrictHintExemptsConstantFoldAndHonorsValueLookupAndNonInvalidErrors) {
    constexpr int64_t kNgramIndexId = 7451;
    constexpr int64_t kSortIndexId = 7452;

    auto schema = std::make_shared<Schema>();
    auto varchar_fid =
        schema->AddDebugField("varchar_field", DataType::VARCHAR);
    auto left_int_fid = schema->AddDebugField("left_int", DataType::INT64);
    auto right_int_fid = schema->AddDebugField("right_int", DataType::INT64);

    auto varchar_data = milvus::test::expr_index::StringField(kVarCharValues);
    auto int_data =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    int_data->FillFieldData(kInt64Values.data(), kInt64Values.size());

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(varchar_fid, {varchar_data}));
    segment->LoadFieldData(files.Prepare(left_int_fid, {int_data}));
    segment->LoadFieldData(files.Prepare(right_int_fid, {int_data}));

    InstallIndexWithId(*segment,
                       varchar_fid,
                       DataType::VARCHAR,
                       kNgramIndexId,
                       BuildIndex(varchar_fid,
                                  DataType::VARCHAR,
                                  milvus::index::NGRAM_INDEX_TYPE,
                                  {varchar_data},
                                  {{milvus::index::MIN_GRAM, 3},
                                   {milvus::index::MAX_GRAM, 4}}));
    InstallIndexWithId(*segment,
                       left_int_fid,
                       DataType::INT64,
                       kSortIndexId,
                       BuildIndex(left_int_fid,
                                  DataType::INT64,
                                  milvus::index::ASCENDING_SORT,
                                  {int_data}));

    // 1. Non-nullable IS NOT NULL is constant-folded from schema and must not
    //    fail strict mode even when the loaded NGRAM index cannot serve Null
    //    or when strict raw-fallback check runs.
    {
        auto is_not_null = std::make_shared<expr::NullExpr>(
            expr::ColumnInfo(
                varchar_fid, DataType::VARCHAR, {}, /*nullable=*/false),
            proto::plan::NullExpr_NullOp_IsNotNull);
        query::PlanOptions opts;
        opts.index_hints = {
            {.field_id = varchar_fid.get(), .index_id = kNgramIndexId}};
        opts.index_hint_strict = true;

        std::vector<expr::TypedExprPtr> filters{is_not_null};
        auto qc = std::make_shared<exec::QueryContext>(DEAFULT_QUERY_ID,
                                                       segment.get(),
                                                       kVarCharValues.size(),
                                                       MAX_TIMESTAMP,
                                                       0,
                                                       0,
                                                       opts);
        auto ec = std::make_unique<exec::ExecContext>(qc.get());
        auto exprs = std::make_unique<exec::ExprSet>(filters, ec.get());
        std::vector<VectorPtr> results;
        exec::EvalCtx eval_ctx(ec.get());
        ASSERT_NO_THROW(exprs->Eval(0, 1, true, eval_ctx, results));
        auto col = milvus::test::GetColumnVectorForTest(results[0]);
        ASSERT_NE(col, nullptr);
        EXPECT_EQ(col->size(), kVarCharValues.size());
    }

    // 2. PhyCompareFilterExpr forwards PlanOptions into ValueLookupSource:
    //    strict hint on an index without value_lookup (NGRAM on varchar_fid)
    //    throws InvalidParameter, while strict hint on STL_SORT succeeds.
    {
        auto str_cmp =
            std::make_shared<expr::CompareExpr>(varchar_fid,
                                                varchar_fid,
                                                DataType::VARCHAR,
                                                DataType::VARCHAR,
                                                proto::plan::OpType::Equal);
        query::PlanOptions bad_opts;
        bad_opts.index_hints = {
            {.field_id = varchar_fid.get(), .index_id = kNgramIndexId}};
        bad_opts.index_hint_strict = true;

        std::vector<expr::TypedExprPtr> bad_filters{str_cmp};
        auto bad_qc =
            std::make_shared<exec::QueryContext>(DEAFULT_QUERY_ID,
                                                 segment.get(),
                                                 kVarCharValues.size(),
                                                 MAX_TIMESTAMP,
                                                 0,
                                                 0,
                                                 bad_opts);
        auto bad_ec = std::make_unique<exec::ExecContext>(bad_qc.get());
        // ValueLookupSource backs column-value reads rather than filter
        // predicate routing, so a hinted filter index without value_lookup
        // falls back to raw column values instead of failing strict mode.
        auto fallback_exprs =
            std::make_unique<exec::ExprSet>(bad_filters, bad_ec.get());
        std::vector<VectorPtr> fallback_results;
        exec::EvalCtx fallback_eval_ctx(bad_ec.get());
        ASSERT_NO_THROW(fallback_exprs->Eval(
            0, 1, true, fallback_eval_ctx, fallback_results));

        auto int_cmp =
            std::make_shared<expr::CompareExpr>(left_int_fid,
                                                right_int_fid,
                                                DataType::INT64,
                                                DataType::INT64,
                                                proto::plan::OpType::Equal);
        query::PlanOptions ok_opts;
        ok_opts.index_hints = {
            {.field_id = left_int_fid.get(), .index_id = kSortIndexId}};
        ok_opts.index_hint_strict = true;

        std::vector<expr::TypedExprPtr> ok_filters{int_cmp};
        auto ok_qc = std::make_shared<exec::QueryContext>(DEAFULT_QUERY_ID,
                                                          segment.get(),
                                                          kInt64Values.size(),
                                                          MAX_TIMESTAMP,
                                                          0,
                                                          0,
                                                          ok_opts);
        auto ok_ec = std::make_unique<exec::ExecContext>(ok_qc.get());
        auto exprs = std::make_unique<exec::ExprSet>(ok_filters, ok_ec.get());
        std::vector<VectorPtr> results;
        exec::EvalCtx eval_ctx(ok_ec.get());
        ASSERT_NO_THROW(exprs->Eval(0, 1, true, eval_ctx, results));
    }

    // 3. Offset-input evaluation that bypasses a hinted ScalarIndex / Ngram
    //    when raw column chunks are present must fail in strict mode.
    {
        auto eq_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(left_int_fid, DataType::INT64),
            proto::plan::OpType::Equal,
            Int64Literal(kInt64Values[0]));
        query::PlanOptions strict_opts;
        strict_opts.index_hints = {
            {.field_id = left_int_fid.get(), .index_id = kSortIndexId}};
        strict_opts.index_hint_strict = true;

        std::vector<expr::TypedExprPtr> filters{eq_expr};
        auto qc = std::make_shared<exec::QueryContext>(DEAFULT_QUERY_ID,
                                                       segment.get(),
                                                       kInt64Values.size(),
                                                       MAX_TIMESTAMP,
                                                       0,
                                                       0,
                                                       strict_opts);
        auto ec = std::make_unique<exec::ExecContext>(qc.get());
        auto exprs = std::make_unique<exec::ExprSet>(filters, ec.get());
        // Trigger path determination before offset_input is bound (as async
        // prefetch does), then evaluate with OffsetVector input.
        EXPECT_TRUE(exprs->expr(0)->CanExecuteAllAtOnce());
        exec::OffsetVector offsets{0, 2};
        exec::EvalCtx eval_ctx(ec.get(), &offsets);
        std::vector<VectorPtr> results;
        EXPECT_THROW(exprs->Eval(0, 1, true, eval_ctx, results), SegcoreError);
    }
}

TEST(MultiScalarIndexExprTest, PatternMatchHintOrderAndStrictNgramReason) {
    constexpr int64_t kTrieIndexId = 7401;
    constexpr int64_t kNgramIndexId = 7402;

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
                       kTrieIndexId,
                       BuildIndex(varchar_fid,
                                  DataType::VARCHAR,
                                  milvus::index::MARISA_TRIE,
                                  {field_data}));
    InstallIndexWithId(*segment,
                       varchar_fid,
                       DataType::VARCHAR,
                       kNgramIndexId,
                       BuildIndex(varchar_fid,
                                  DataType::VARCHAR,
                                  milvus::index::NGRAM_INDEX_TYPE,
                                  {field_data},
                                  {{milvus::index::MIN_GRAM, 3},
                                   {milvus::index::MAX_GRAM, 4}}));

    auto eval_with_options = [&](const expr::TypedExprPtr& expr,
                                 const query::PlanOptions& opts) {
        std::vector<expr::TypedExprPtr> filters{expr};
        auto qc = std::make_shared<exec::QueryContext>(DEAFULT_QUERY_ID,
                                                       segment.get(),
                                                       kVarCharValues.size(),
                                                       MAX_TIMESTAMP,
                                                       0,
                                                       0,
                                                       opts);
        auto ec = std::make_unique<exec::ExecContext>(qc.get());
        auto exprs = std::make_unique<exec::ExprSet>(filters, ec.get());
        std::vector<VectorPtr> results;
        exec::EvalCtx eval_ctx(ec.get());
        exprs->Eval(0, 1, true, eval_ctx, results);
        return milvus::test::GetColumnVectorForTest(results[0]);
    };

    proto::plan::GenericValue alpha_pat;
    alpha_pat.set_string_val("alpha");
    auto inner_alpha = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::InnerMatch,
        alpha_pat,
        std::vector<proto::plan::GenericValue>{});

    // 1. Hint [Trie, Ngram] must pick Marisa Trie (index 0 in hint order)
    //    rather than letting Ngram preempt it.
    {
        query::PlanOptions opts;
        opts.feature_recorder = std::make_shared<FeatureRecorder>();
        opts.index_hints = {
            {.field_id = varchar_fid.get(), .index_id = kTrieIndexId},
            {.field_id = varchar_fid.get(), .index_id = kNgramIndexId},
        };
        auto res = eval_with_options(inner_alpha, opts);
        ExpectHitRows(res, {0, 1}, "InnerMatch alpha via hinted Trie");
        const uint64_t bits = opts.feature_recorder->Bits();
        const auto has_bit = [bits](FeatureBit b) {
            return (bits & (uint64_t{1} << static_cast<uint32_t>(b))) != 0;
        };
        EXPECT_TRUE(has_bit(FeatureBit::ScalarIndexTrie));
        EXPECT_FALSE(has_bit(FeatureBit::ScalarIndexNgram));
    }

    // 2. Reversing the hint [Ngram, Trie] must pick Ngram.
    {
        query::PlanOptions opts;
        opts.feature_recorder = std::make_shared<FeatureRecorder>();
        opts.index_hints = {
            {.field_id = varchar_fid.get(), .index_id = kNgramIndexId},
            {.field_id = varchar_fid.get(), .index_id = kTrieIndexId},
        };
        auto res = eval_with_options(inner_alpha, opts);
        ExpectHitRows(res, {0, 1}, "InnerMatch alpha via hinted Ngram");
        const uint64_t bits = opts.feature_recorder->Bits();
        const auto has_bit = [bits](FeatureBit b) {
            return (bits & (uint64_t{1} << static_cast<uint32_t>(b))) != 0;
        };
        EXPECT_TRUE(has_bit(FeatureBit::ScalarIndexNgram));
        EXPECT_FALSE(has_bit(FeatureBit::ScalarIndexTrie));
    }

    // 3. Single NGRAM strict hint on a too-short literal ("al", min_gram=3)
    //    must preserve "literal_not_supported" instead of "capability_mismatch".
    {
        proto::plan::GenericValue short_pat;
        short_pat.set_string_val("al");
        auto inner_short = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
            proto::plan::OpType::InnerMatch,
            short_pat,
            std::vector<proto::plan::GenericValue>{});

        query::PlanOptions opts;
        opts.index_hints = {
            {.field_id = varchar_fid.get(), .index_id = kNgramIndexId}};
        opts.index_hint_strict = true;
        try {
            eval_with_options(inner_short, opts);
            FAIL() << "expected SegcoreError for strict too-short ngram hint";
        } catch (const SegcoreError& err) {
            EXPECT_EQ(err.get_error_code(), ErrorCode::InvalidParameter);
            EXPECT_NE(std::string(err.what()).find("literal_not_supported"),
                      std::string::npos)
                << err.what();
        }
    }
}

TEST(MultiScalarIndexExprTest, StrictHintParentJsonFlatRejectsIntegerSubpath) {
    constexpr FieldId kJsonField(102);

    milvus::segcore::IndexCapabilityEntry flat_entry{milvus::segcore::IndexKey{
        kJsonField, milvus::segcore::IndexIdentity::PreBuiltIndex(7501)}};
    flat_entry.json_path = "/a";
    flat_entry.json_cast_type = JsonCastType::FromString("JSON");
    flat_entry.family = "json_flat";
    flat_entry.value_type = DataType::JSON;
    flat_entry.caps = {.predicate = true, .json_paths = true, .exact = true};

    milvus::segcore::FieldIndexCapability caps(kJsonField, {flat_entry});
    std::vector<milvus::IndexHintOption> hints{
        {.field_id = kJsonField.get(), .index_id = 7501}};

    milvus::exec::ExprIndexRequirement req;
    req.field_id = kJsonField;
    req.reader = milvus::exec::RequiredReader::Predicate;
    req.value_type = DataType::VARCHAR;
    req.is_json_field = true;
    req.json_path = "/a/0/x";
    req.index_hints = &hints;
    req.index_hint_strict = true;

    try {
        (void)milvus::exec::DetermineExecPath(req, caps);
        FAIL() << "expected strict hint failure on integer array subpath";
    } catch (const SegcoreError& err) {
        EXPECT_EQ(err.get_error_code(), ErrorCode::InvalidParameter);
        EXPECT_NE(std::string(err.what()).find("json_path_not_covered"),
                  std::string::npos)
            << err.what();
    }
}

TEST(MultiScalarIndexExprTest,
     TextMatchAndEntityTTLIgnoreStrictScalarIndexHints) {
    // 1. RequiredReader::TextMatch must ignore scalar index_hints even when
    //    the same VARCHAR/TEXT field carries a hinted scalar index.
    {
        constexpr FieldId kTextField(105);
        milvus::segcore::IndexCapabilityEntry ngram_entry{
            milvus::segcore::IndexKey{
                kTextField,
                milvus::segcore::IndexIdentity::PreBuiltIndex(7601)}};
        ngram_entry.family = "ngram";
        ngram_entry.value_type = DataType::VARCHAR;
        ngram_entry.caps = {.ngram_candidates = true};

        milvus::segcore::IndexCapabilityEntry text_entry{
            milvus::segcore::IndexKey{
                kTextField,
                milvus::segcore::IndexIdentity::PreBuiltIndex(7602)}};
        text_entry.family = "text";
        text_entry.value_type = DataType::VARCHAR;
        text_entry.caps = {.text_match = true, .exact = true};

        milvus::segcore::FieldIndexCapability caps(kTextField,
                                                   {ngram_entry, text_entry});
        std::vector<milvus::IndexHintOption> hints{
            {.field_id = kTextField.get(), .index_id = 7601}};

        milvus::exec::ExprIndexRequirement req;
        req.field_id = kTextField;
        req.reader = milvus::exec::RequiredReader::TextMatch;
        req.value_type = DataType::VARCHAR;
        req.index_hints = &hints;
        req.index_hint_strict = true;

        const auto decision = milvus::exec::DetermineExecPath(req, caps);
        EXPECT_EQ(decision.path, milvus::exec::ExprExecPath::TextIndex);
        ASSERT_TRUE(decision.key.has_value());
        EXPECT_EQ(*decision.key, text_entry.key);
    }

    // 2. Internal Entity-TTL expression compiled alongside a user filter must
    //    not inherit user index_hints / index_hint_strict when evaluated with
    //    OffsetVector input (IterativeFilterNode).
    {
        auto schema = std::make_shared<Schema>();
        auto pk_fid = schema->AddDebugField("pk", DataType::INT64);
        schema->set_primary_field_id(pk_fid);
        auto ttl_fid =
            schema->AddDebugField("ttl_ts", DataType::TIMESTAMPTZ, true);
        schema->set_ttl_field_id(ttl_fid);

        const std::vector<int64_t> pks = {1, 2, 3, 4};
        const std::vector<int64_t> ttls = {1000, 2000, 3000, 4000};
        const uint8_t valid = 0x0F;
        auto pk_data =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        pk_data->FillFieldData(pks.data(), pks.size());
        auto ttl_data =
            std::make_shared<FieldData<int64_t>>(DataType::TIMESTAMPTZ, true);
        ttl_data->FillFieldData(ttls.data(), &valid, ttls.size(), 0);

        RawFieldFiles files;
        auto segment = CreateSealedSegment(schema);
        segment->LoadFieldData(files.Prepare(pk_fid, {pk_data}));
        segment->LoadFieldData(files.Prepare(ttl_fid, {ttl_data}));
        InstallIndexWithId(*segment,
                           ttl_fid,
                           DataType::TIMESTAMPTZ,
                           7603,
                           BuildIndex(ttl_fid,
                                      DataType::TIMESTAMPTZ,
                                      milvus::index::ASCENDING_SORT,
                                      {ttl_data}));

        auto pk_gt = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(pk_fid, DataType::INT64),
            proto::plan::OpType::GreaterThan,
            Int64Literal(1));
        query::PlanOptions opts;
        opts.index_hints = {{.field_id = ttl_fid.get(), .index_id = 7603}};
        opts.index_hint_strict = true;

        std::vector<expr::TypedExprPtr> filters{pk_gt};
        auto qc = std::make_shared<exec::QueryContext>(
            DEAFULT_QUERY_ID,
            segment.get(),
            pks.size(),
            MAX_TIMESTAMP,
            0,
            0,
            opts,
            std::make_shared<exec::QueryConfig>(),
            nullptr,
            std::unordered_map<std::string,
                               std::shared_ptr<exec::BaseConfig>>{},
            /*entity_ttl_physical_time_us=*/2500);
        qc->set_read_schema(schema);
        auto ec = std::make_unique<exec::ExecContext>(qc.get());
        auto exprs = std::make_unique<exec::ExprSet>(
            filters, ec.get(), /*null_rejecting=*/true);

        exec::OffsetVector offsets{0, 1, 2, 3};
        exec::EvalCtx eval_ctx(ec.get(), &offsets);
        std::vector<VectorPtr> results;
        ASSERT_NO_THROW(exprs->Eval(0, 1, true, eval_ctx, results));
        auto col = milvus::test::GetColumnVectorForTest(results[0]);
        ExpectHitRows(
            col, {2, 3}, "pk > 1 AND ttl > 2500 with strict hint on ttl");
    }
}
