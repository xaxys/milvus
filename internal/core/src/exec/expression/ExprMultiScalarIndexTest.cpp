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
#include "pb/internal.pb.h"
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

int
ConditionCount(const proto::internal::ExecutionExplain& node) {
    if (node.group().operators_size() == 0) {
        return 0;
    }
    return node.group().operators(0).conditions_size();
}

const proto::internal::ExplainConditionStats&
ConditionAt(const proto::internal::ExecutionExplain& node, int idx) {
    return node.group().operators(0).conditions(idx);
}

std::map<std::string, int64_t>
CountersOf(const proto::internal::ExplainConditionStats& cond) {
    return {
        {"rows_in", cond.rows_in()},
        {"rows_out", cond.rows_out()},
        {"idx_scan", cond.idx_scan()},
        {"raw_scan", cond.raw_scan()},
        {"element_hits", cond.element_hits()},
    };
}

std::map<std::string, std::string>
AttributesOf(const proto::internal::ExplainConditionStats& cond) {
    std::map<std::string, std::string> out;
    if (cond.expr_id() != 0) {
        out["expr_id"] = std::to_string(cond.expr_id());
    }
    if (cond.field_id() != 0) {
        out["field_id"] = std::to_string(cond.field_id());
    }
    if (cond.skipped_segments() > 0) {
        out["skipped"] = "true";
    }
    if (cond.eval_orders_size() > 0) {
        out["eval_order"] = std::to_string(cond.eval_orders(0).eval_order());
    }
    if (cond.used_size() == 1) {
        const auto& u = cond.used(0);
        if (u.index_id() != 0 || u.family() == "pk") {
            out["index_id"] = std::to_string(u.index_id());
        }
        if (!u.family().empty()) {
            out["family"] = u.family();
        }
        if (!u.path().empty()) {
            out["path"] = u.path();
        }
    }
    if (cond.reasons_size() == 1) {
        out["decline_reason"] = cond.reasons(0).reason();
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
        EXPECT_EQ(node.group().segments(), 1);
        EXPECT_EQ(node.group().sealed_segments(), 1);
        EXPECT_EQ(node.group().growing_segments(), 0);
        EXPECT_EQ(node.group().rows_total(),
                  static_cast<int64_t>(kVarCharValues.size()));
        ASSERT_EQ(ConditionCount(node), 1);
        const auto& group = ConditionAt(node, 0);
        EXPECT_EQ(group.raw_scan_segments(), 1);
        const auto attributes = AttributesOf(group);
        EXPECT_EQ(attributes.at("field_id"), std::to_string(varchar_fid.get()));
        EXPECT_EQ(attributes.at("decline_reason"), "no_index");
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto& group = ConditionAt(node, 0);
        EXPECT_EQ(group.raw_scan_segments(), 0);
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
    ASSERT_EQ(ConditionCount(node), 1);
    const auto attributes = AttributesOf(ConditionAt(node, 0));
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
    ASSERT_EQ(ConditionCount(node), 2);
    bool saw_left = false;
    bool saw_right = false;
    for (int i = 0; i < ConditionCount(node); ++i) {
        const auto attributes = AttributesOf(ConditionAt(node, i));
        const auto counters = CountersOf(ConditionAt(node, i));
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
    options.index_hints = {
        {.field_id = json_fid.get(), .index_id = kPathAIndexId}};
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
    ASSERT_EQ(ConditionCount(node), 1);
    const auto attributes = AttributesOf(ConditionAt(node, 0));
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
    ASSERT_EQ(ConditionCount(node), 2)
        << "each expr_id keeps its own filter group";
    EXPECT_EQ(AttributesOf(ConditionAt(node, 0)).at("expr_id"), "71001");
    EXPECT_EQ(AttributesOf(ConditionAt(node, 1)).at("expr_id"), "71002");
    for (int i = 0; i < ConditionCount(node); ++i) {
        const auto attributes = AttributesOf(ConditionAt(node, i));
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
    ASSERT_EQ(ConditionCount(node), 2);

    std::map<std::string, const proto::internal::ExplainConditionStats*> by_id;
    for (int i = 0; i < ConditionCount(node); ++i) {
        const auto& child = ConditionAt(node, i);
        by_id[AttributesOf(child).at("expr_id")] = &child;
    }
    ASSERT_EQ(by_id.count("72001"), 1);
    ASSERT_EQ(by_id.count("72002"), 1);

    const auto attrs1 = AttributesOf(*by_id.at("72001"));
    const auto ctrs1 = CountersOf(*by_id.at("72001"));
    EXPECT_EQ(by_id.at("72001")->raw_scan_segments(), 0);
    EXPECT_EQ(attrs1.at("family"), "ngram");
    EXPECT_EQ(attrs1.at("index_id"), std::to_string(kNgramIndexId));
    EXPECT_EQ(attrs1.at("eval_order"), "1");
    EXPECT_EQ(ctrs1.at("rows_in"), 6);
    EXPECT_EQ(ctrs1.at("idx_scan"), 6);
    EXPECT_EQ(ctrs1.at("raw_scan"), 3);
    EXPECT_LE(ctrs1.at("raw_scan"), ctrs1.at("rows_in"));
    EXPECT_EQ(ctrs1.at("rows_out"), 2);

    const auto attrs2 = AttributesOf(*by_id.at("72002"));
    const auto ctrs2 = CountersOf(*by_id.at("72002"));
    EXPECT_EQ(by_id.at("72002")->raw_scan_segments(), 0);
    EXPECT_EQ(attrs2.at("family"), "ngram");
    EXPECT_EQ(attrs2.at("index_id"), std::to_string(kNgramIndexId));
    EXPECT_EQ(attrs2.at("eval_order"), "2");
    EXPECT_EQ(ctrs2.at("rows_in"), 2);
    EXPECT_EQ(ctrs2.at("idx_scan"), 2);
    EXPECT_EQ(ctrs2.at("raw_scan"), 2);
    EXPECT_LE(ctrs2.at("raw_scan"), ctrs2.at("rows_in"));
    EXPECT_EQ(ctrs2.at("rows_out"), 1);

    // When the first LIKE's Phase 2 rejects every candidate (e.g. "abczbcd" vs
    // "abcd_miss"), neither leaf may report positive hits or inverted rows_in.
    proto::plan::GenericValue miss_v1;
    miss_v1.set_string_val("abcz");
    auto only_row1 = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::InnerMatch,
        miss_v1,
        std::vector<proto::plan::GenericValue>());
    // "abcd" passes Phase 1 on row 1 ("abczbcd" has "abc"+"bcd") and fails Phase 2!
    auto zero_both = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, like1, only_row1);
    only_row1->SetExprId(72003);
    auto zero_explain =
        std::make_shared<milvus::exec::ExecutionExplainRecorder>(values.size());
    auto zero_res =
        EvaluateFilter(zero_both, segment.get(), values.size(), zero_explain);
    ExpectHitRows(zero_res, {}, "fused LIKE where Phase 2 rejects all");
    proto::internal::ExecutionExplain zero_node;
    ASSERT_TRUE(zero_node.ParseFromString(zero_explain->Finish()));
    ASSERT_EQ(ConditionCount(zero_node), 2);
    std::map<std::string, const proto::internal::ExplainConditionStats*>
        z_by_id;
    for (int i = 0; i < ConditionCount(zero_node); ++i) {
        const auto& child = ConditionAt(zero_node, i);
        z_by_id[AttributesOf(child).at("expr_id")] = &child;
    }
    EXPECT_EQ(CountersOf(*z_by_id.at("72001")).at("rows_out"), 0);
    EXPECT_EQ(CountersOf(*z_by_id.at("72003")).at("rows_in"), 0);
    EXPECT_EQ(CountersOf(*z_by_id.at("72003")).at("rows_out"), 0);
    EXPECT_GT(CountersOf(*z_by_id.at("72003")).at("idx_scan"), 0);
    EXPECT_EQ(CountersOf(*z_by_id.at("72003")).at("raw_scan"), 0);
    EXPECT_EQ(z_by_id.at("72003")->skipped_segments(), 0);
    EXPECT_EQ(AttributesOf(*z_by_id.at("72003")).at("family"), "ngram");
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
        ASSERT_EQ(ConditionCount(node), 3);

        std::map<std::string, const proto::internal::ExplainConditionStats*>
            by_id;
        for (int i = 0; i < ConditionCount(node); ++i) {
            const auto& child = ConditionAt(node, i);
            by_id[AttributesOf(child).at("expr_id")] = &child;
        }
        ASSERT_EQ(by_id.count("73001"), 1);
        ASSERT_EQ(by_id.count("73002"), 1);
        ASSERT_EQ(by_id.count("73003"), 1);

        EXPECT_EQ(AttributesOf(*by_id.at("73001")).at("eval_order"), "1");
        EXPECT_EQ(CountersOf(*by_id.at("73001")).at("rows_out"), 3);

        const auto g1_attrs = AttributesOf(*by_id.at("73002"));
        const auto g1_ctrs = CountersOf(*by_id.at("73002"));
        EXPECT_EQ(g1_attrs.at("decline_reason"), "no_index");
        EXPECT_EQ(g1_attrs.at("eval_order"), "2");
        EXPECT_EQ(g1_ctrs.at("rows_in"), 3);
        EXPECT_EQ(g1_ctrs.at("raw_scan"), 3);
        EXPECT_EQ(g1_ctrs.at("rows_out"), 1);

        const auto g2_attrs = AttributesOf(*by_id.at("73003"));
        const auto g2_ctrs = CountersOf(*by_id.at("73003"));
        EXPECT_EQ(g2_attrs.at("decline_reason"), "no_index");
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
        ASSERT_EQ(ConditionCount(node), 3);

        std::map<std::string, const proto::internal::ExplainConditionStats*>
            by_id;
        for (int i = 0; i < ConditionCount(node); ++i) {
            const auto& child = ConditionAt(node, i);
            by_id[AttributesOf(child).at("expr_id")] = &child;
        }
        ASSERT_EQ(by_id.count("73001"), 1);
        ASSERT_EQ(by_id.count("73002"), 1);
        ASSERT_EQ(by_id.count("73003"), 1);

        EXPECT_EQ(AttributesOf(*by_id.at("73001")).at("eval_order"), "1");

        const auto g1_attrs = AttributesOf(*by_id.at("73002"));
        const auto g1_ctrs = CountersOf(*by_id.at("73002"));
        EXPECT_EQ(by_id.at("73002")->raw_scan_segments(), 0);
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
        EXPECT_EQ(by_id.at("73003")->raw_scan_segments(), 0);
        EXPECT_EQ(g2_attrs.at("index_id"), std::to_string(kRTreeIndexId));
        EXPECT_EQ(g2_attrs.at("family"), "rtree");
        EXPECT_EQ(g2_attrs.at("eval_order"), "3");
        EXPECT_EQ(g2_ctrs.at("rows_in"), 1);
        EXPECT_EQ(g2_ctrs.at("idx_scan"), 1);
        // Only row 0 survived g1 exact refinement when g2 ran.
        EXPECT_EQ(g2_ctrs.at("raw_scan"), 1);
        EXPECT_LE(g2_ctrs.at("raw_scan"), g2_ctrs.at("rows_in"));
        EXPECT_EQ(g2_ctrs.at("rows_out"), 1);
    }

    // Case C1: POINT(2 8) falls inside both MBRs, outside gis_1 triangle.
    // Refine short-circuits at gis_1, and neither predicate may keep coarse
    // false positives in rows_out (nor may gis_2 report rows_in > gis_1.rows_out).
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
        ASSERT_EQ(ConditionCount(node), 2);
        std::map<std::string, const proto::internal::ExplainConditionStats*>
            c1_by_id;
        for (int i = 0; i < ConditionCount(node); ++i) {
            const auto& child = ConditionAt(node, i);
            c1_by_id[AttributesOf(child).at("expr_id")] = &child;
            EXPECT_EQ(CountersOf(child).at("rows_out"), 0);
        }
        EXPECT_EQ(CountersOf(*c1_by_id.at("73002")).at("rows_in"), 2);
        EXPECT_EQ(CountersOf(*c1_by_id.at("73003")).at("rows_in"), 0);
        EXPECT_GT(CountersOf(*c1_by_id.at("73003")).at("idx_scan"), 0);
        EXPECT_EQ(CountersOf(*c1_by_id.at("73003")).at("raw_scan"), 0);
        EXPECT_EQ(c1_by_id.at("73003")->skipped_segments(), 0);
        EXPECT_EQ(AttributesOf(*c1_by_id.at("73003")).at("family"), "rtree");
    }

    // Case C2: gis_1 triangle MBR covers both {POINT(8 2), POINT(2 8)},
    // whereas a tight gis_2 box covers only POINT(8 2). POINT(2 8) is pruned by
    // gis_2's coarse MBR before refinement, so gis_1 must still report exact
    // rows_out=1 rather than coarse=2, and gis_tight receives rows_in=1.
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
        ASSERT_EQ(ConditionCount(node), 2);
        std::map<std::string, const proto::internal::ExplainConditionStats*>
            by_id;
        for (int i = 0; i < ConditionCount(node); ++i) {
            const auto& child = ConditionAt(node, i);
            by_id[AttributesOf(child).at("expr_id")] = &child;
        }
        const auto g1_ctrs = CountersOf(*by_id.at("73002"));
        const auto gt_ctrs = CountersOf(*by_id.at("73004"));
        EXPECT_EQ(g1_ctrs.at("rows_in"), 2);
        EXPECT_EQ(g1_ctrs.at("raw_scan"), 1);
        EXPECT_EQ(g1_ctrs.at("rows_out"), 1);
        EXPECT_EQ(gt_ctrs.at("rows_in"), 1);
        EXPECT_EQ(gt_ctrs.at("raw_scan"), 1);
        EXPECT_EQ(gt_ctrs.at("rows_out"), 1);

        // Case C3a (top-level OR, not split-fused): gis_2 OR gis_1 on
        // {POINT(8 2), POINT(2 8)}. Without an AND parent, SplitFuseGISConjunct
        // leaves this as a regular OR conjunction; gis_2 satisfies 100% of rows
        // and short-circuits gis_1 before any R-Tree or Refine runs.
        auto or_expr = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::Or, gis_2, gis_1);
        auto or_explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(
                c2_wkbs.size());
        auto or_res =
            EvaluateFilter(or_expr, segment.get(), c2_wkbs.size(), or_explain);
        ExpectHitRows(or_res, {0, 1}, "GIS C3a: plain OR short-circuits");
        proto::internal::ExecutionExplain or_node;
        ASSERT_TRUE(or_node.ParseFromString(or_explain->Finish()));
        ASSERT_EQ(ConditionCount(or_node), 2);
        std::map<std::string, const proto::internal::ExplainConditionStats*>
            or_by_id;
        for (int i = 0; i < ConditionCount(or_node); ++i) {
            const auto& child = ConditionAt(or_node, i);
            or_by_id[AttributesOf(child).at("expr_id")] = &child;
        }
        EXPECT_EQ(CountersOf(*or_by_id.at("73003")).at("rows_in"), 2);
        EXPECT_EQ(CountersOf(*or_by_id.at("73003")).at("rows_out"), 2);
        EXPECT_EQ(CountersOf(*or_by_id.at("73002")).at("rows_in"), 0);
        EXPECT_EQ(CountersOf(*or_by_id.at("73002")).at("rows_out"), 0);
        EXPECT_EQ(or_by_id.at("73002")->skipped_segments(), 1);

        // Case C3b (OR child under AND, split-fused): int_lt AND (gis_2 OR gis_1).
        // SplitFuseGISConjunct hoists (gis_2 OR gis_1) into Coarse + Refine, so
        // gis_1 physically executes R-Tree Coarse (skipped_segments == 0,
        // idx_scan > 0, raw_scan == 0) even though gis_2 short-circuits Refine.
        const std::vector<int64_t> c3_ints = {10, 20};
        auto c3_int_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        c3_int_field->FillFieldData(c3_ints.data(), c3_ints.size());
        segment->LoadFieldData(files.Prepare(int_fid, {c3_int_field}));

        auto fused_or_expr = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, int_lt, or_expr);
        auto fused_or_explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(
                c2_wkbs.size());
        auto fused_or_res = EvaluateFilter(
            fused_or_expr, segment.get(), c2_wkbs.size(), fused_or_explain);
        ExpectHitRows(
            fused_or_res, {0, 1}, "GIS C3b: split-fused OR under AND");
        proto::internal::ExecutionExplain fused_or_node;
        ASSERT_TRUE(fused_or_node.ParseFromString(fused_or_explain->Finish()));
        ASSERT_EQ(ConditionCount(fused_or_node), 3);
        std::map<std::string, const proto::internal::ExplainConditionStats*>
            fo_by_id;
        for (int i = 0; i < ConditionCount(fused_or_node); ++i) {
            const auto& child = ConditionAt(fused_or_node, i);
            fo_by_id[AttributesOf(child).at("expr_id")] = &child;
        }
        EXPECT_EQ(CountersOf(*fo_by_id.at("73003")).at("rows_in"), 2);
        EXPECT_EQ(CountersOf(*fo_by_id.at("73003")).at("rows_out"), 2);
        EXPECT_EQ(CountersOf(*fo_by_id.at("73002")).at("rows_in"), 0);
        EXPECT_EQ(CountersOf(*fo_by_id.at("73002")).at("rows_out"), 0);
        EXPECT_GT(CountersOf(*fo_by_id.at("73002")).at("idx_scan"), 0);
        EXPECT_EQ(CountersOf(*fo_by_id.at("73002")).at("raw_scan"), 0);
        EXPECT_EQ(fo_by_id.at("73002")->skipped_segments(), 0);
        EXPECT_EQ(AttributesOf(*fo_by_id.at("73002")).at("family"), "rtree");
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
    ASSERT_EQ(ConditionCount(node), 2);

    std::map<std::string, std::string> orders;
    for (int i = 0; i < ConditionCount(node); ++i) {
        const auto attrs = AttributesOf(ConditionAt(node, i));
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto attrs = AttributesOf(ConditionAt(node, 0));
        const auto ctrs = CountersOf(ConditionAt(node, 0));
        EXPECT_EQ(attrs.at("expr_id"), "75001");
        EXPECT_EQ(attrs.at("field_id"), std::to_string(varchar_fid.get()));
        EXPECT_EQ(attrs.at("decline_reason"), "no_index");
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
        ASSERT_EQ(ConditionCount(node), 2);
        std::map<std::string, const proto::internal::ExplainConditionStats*>
            by_id;
        for (int i = 0; i < ConditionCount(node); ++i) {
            const auto& child = ConditionAt(node, i);
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto attrs = AttributesOf(ConditionAt(node, 0));
        const auto ctrs = CountersOf(ConditionAt(node, 0));
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto attrs = AttributesOf(ConditionAt(node, 0));
        const auto ctrs = CountersOf(ConditionAt(node, 0));
        EXPECT_EQ(attrs.at("expr_id"), "76001");
        EXPECT_EQ(attrs.at("decline_reason"), "no_index");
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto attrs = AttributesOf(ConditionAt(node, 0));
        const auto ctrs = CountersOf(ConditionAt(node, 0));
        EXPECT_EQ(attrs.at("expr_id"), "76002");
        EXPECT_EQ(attrs.find("family"), attrs.end());
        EXPECT_EQ(attrs.at("decline_reason"), "unusable");
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto attrs = AttributesOf(ConditionAt(node, 0));
        const auto ctrs = CountersOf(ConditionAt(node, 0));
        EXPECT_EQ(attrs.at("expr_id"), "76001");
        EXPECT_EQ(attrs.find("index_id"), attrs.end());
        EXPECT_EQ(attrs.at("decline_reason"), "unusable");
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
    ASSERT_EQ(ConditionCount(node), 1);
    EXPECT_EQ(ConditionAt(node, 0).raw_scan_segments(), 0);
    const auto attrs = AttributesOf(ConditionAt(node, 0));
    const auto ctrs = CountersOf(ConditionAt(node, 0));
    EXPECT_EQ(attrs.at("expr_id"), "78001");
    EXPECT_EQ(attrs.find("index_id"), attrs.end());
    EXPECT_EQ(attrs.at("decline_reason"), "not_nullable");
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
    ASSERT_EQ(ConditionCount(node), 1);
    const auto attrs = AttributesOf(ConditionAt(node, 0));
    const auto ctrs = CountersOf(ConditionAt(node, 0));
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
    ASSERT_EQ(ConditionCount(node), 2);

    bool found_b = false;
    for (int i = 0; i < ConditionCount(node); ++i) {
        const auto attrs = AttributesOf(ConditionAt(node, i));
        const auto ctrs = CountersOf(ConditionAt(node, i));
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto attrs = AttributesOf(ConditionAt(node, 0));
        const auto ctrs = CountersOf(ConditionAt(node, 0));
        EXPECT_EQ(attrs.at("expr_id"), "78201");
        EXPECT_EQ(attrs.at("decline_reason"), "no_index");
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto ctrs = CountersOf(ConditionAt(node, 0));
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
        EXPECT_EQ(ConditionCount(node), 0);
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto attrs = AttributesOf(ConditionAt(node, 0));
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
        ASSERT_EQ(ConditionCount(node), 2);
        std::map<std::string, std::map<std::string, std::string>> by_id;
        for (int i = 0; i < ConditionCount(node); ++i) {
            auto attrs = AttributesOf(ConditionAt(node, i));
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto& cond = ConditionAt(node, 0);
        EXPECT_EQ(cond.expr_id(), 78410);
        EXPECT_EQ(cond.rows_in(), row_count);
        EXPECT_EQ(cond.idx_scan(), row_count);
        EXPECT_EQ(cond.raw_scan(), row_count);
        EXPECT_EQ(cond.rows_out(), 2);
        EXPECT_EQ(cond.raw_scan_segments(), 1);
        ASSERT_EQ(cond.used_size(), 1);
        EXPECT_EQ(cond.used(0).index_id(), kNameMarisaIndexId);
        EXPECT_EQ(cond.used(0).family(), "marisa");
        EXPECT_EQ(cond.used(0).idx_scan(), row_count);
        EXPECT_EQ(cond.used(0).raw_scan(), 0);
        EXPECT_EQ(cond.used(0).rows_in(), 0);
        EXPECT_EQ(cond.used(0).rows_out(), 0);
        ASSERT_EQ(cond.reasons_size(), 1);
        EXPECT_EQ(cond.reasons(0).reason(), "no_index");
        EXPECT_EQ(cond.reasons(0).idx_scan(), 0);
        EXPECT_EQ(cond.reasons(0).raw_scan(), row_count);
        EXPECT_EQ(cond.reasons(0).rows_in(), 0);
        EXPECT_EQ(cond.reasons(0).rows_out(), 0);
    }

    // 2. Short-circuited CompareExpr (pfx == "nope" AND name < pfx) never
    // enters CompareExpr::Eval, yet Finish() emits its skipped condition
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
        ASSERT_EQ(ConditionCount(node), 2);
        std::map<std::string, const proto::internal::ExplainConditionStats*>
            by_id;
        for (int i = 0; i < ConditionCount(node); ++i) {
            const auto& child = ConditionAt(node, i);
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
        EXPECT_EQ(ConditionCount(node), 0);
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
        ASSERT_EQ(node.group().operators_size(), 1);
        const auto& op = node.group().operators(0);
        EXPECT_EQ(op.kind(), proto::internal::EXPLAIN_OP_FILTER);
        EXPECT_EQ(op.cached_segments(), 0);
        EXPECT_EQ(op.rows_in(), row_count);
        EXPECT_EQ(op.rows_out(), 3);
        ASSERT_EQ(op.conditions_size(), 1);
        const auto ctrs = CountersOf(op.conditions(0));
        EXPECT_EQ(ctrs.at("rows_in"), row_count);
        EXPECT_EQ(ctrs.at("idx_scan"), row_count);
        EXPECT_EQ(ctrs.at("rows_out"), 3);
    }

    // Warm run hits the whole-filter cache without scanning the index or
    // marking the leaf as skipped, while marking the filter operator cached
    // and attributing the bare leaf directly under reason="cached".
    {
        auto warm_explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        run_once(warm_explain);
        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(warm_explain->Finish()));
        ASSERT_EQ(node.group().operators_size(), 1);
        const auto& op = node.group().operators(0);
        EXPECT_EQ(op.kind(), proto::internal::EXPLAIN_OP_FILTER);
        EXPECT_EQ(op.cached_segments(), 1);
        EXPECT_EQ(op.rows_in(), row_count);
        EXPECT_EQ(op.rows_out(), 3);
        ASSERT_EQ(op.conditions_size(), 1);
        const auto& cond = op.conditions(0);
        EXPECT_EQ(cond.skipped_segments(), 0);
        EXPECT_EQ(cond.rows_in(), row_count);
        EXPECT_EQ(cond.rows_out(), 3);
        EXPECT_EQ(cond.idx_scan(), 0);
        EXPECT_EQ(cond.raw_scan(), 0);
        ASSERT_EQ(cond.reasons_size(), 1);
        EXPECT_EQ(cond.reasons(0).reason(), "cached");
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto attrs = AttributesOf(ConditionAt(node, 0));
        const auto ctrs = CountersOf(ConditionAt(node, 0));
        EXPECT_EQ(attrs.find("index_id"), attrs.end());
        EXPECT_EQ(attrs.at("decline_reason"), "unusable");
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto attrs = AttributesOf(ConditionAt(node, 0));
        const auto ctrs = CountersOf(ConditionAt(node, 0));
        EXPECT_EQ(attrs.find("index_id"), attrs.end());
        EXPECT_EQ(attrs.at("decline_reason"), "unusable");
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto attrs = AttributesOf(ConditionAt(node, 0));
        const auto ctrs = CountersOf(ConditionAt(node, 0));
        EXPECT_EQ(attrs.find("index_id"), attrs.end());
        EXPECT_EQ(attrs.at("decline_reason"), "unusable");
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto attrs = AttributesOf(ConditionAt(node, 0));
        const auto ctrs = CountersOf(ConditionAt(node, 0));
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
    ASSERT_EQ(seg_explain.group().operators_size(), 2);

    const auto& filter_op = seg_explain.group().operators(0);
    EXPECT_EQ(filter_op.kind(), proto::internal::EXPLAIN_OP_FILTER);
    EXPECT_EQ(filter_op.rows_in(), row_count);
    EXPECT_EQ(filter_op.rows_out(), 2);
    // Only the user leaf (91001) is emitted; internal TTL and Namespace leaves
    // are suppressed.
    ASSERT_EQ(filter_op.conditions_size(), 1);
    EXPECT_EQ(filter_op.conditions(0).expr_id(), 91001);

    const auto& mvcc_op = seg_explain.group().operators(1);
    EXPECT_EQ(mvcc_op.kind(), proto::internal::EXPLAIN_OP_MVCC);
    EXPECT_EQ(mvcc_op.rows_in(), 2);
    EXPECT_EQ(mvcc_op.rows_out(), 2);
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
        ASSERT_EQ(ConditionCount(node), 1);
        const auto ctrs = CountersOf(ConditionAt(node, 0));
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
        EXPECT_EQ(ConditionCount(node), 0);
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
        EXPECT_EQ(ConditionCount(node), 1);
        return std::make_pair(AttributesOf(ConditionAt(node, 0)),
                              CountersOf(ConditionAt(node, 0)));
    };

    // 1. NGRAM + INVERTED with 2-char literal (< min_gram=3) -> unsupported_literal.
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
        EXPECT_EQ(attrs.at("decline_reason"), "unsupported_literal");
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), n);
    }

    // 2a. NGRAM + INVERTED with valid 4-char literal, soft-hinting INVERTED
    // falls back to NGRAM in fallback mode rather than raw scan.
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
        EXPECT_EQ(attrs.at("index_id"), std::to_string(kNgramId));
        EXPECT_EQ(ctrs.at("idx_scan"), n);
    }

    // 2b. INVERTED-only VARCHAR on InnerMatch -> unsupported_op (no hint) and
    // hint_mismatch when soft-hinting an unloaded NGRAM index on the segment.
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
        ASSERT_EQ(ConditionCount(node), 1);
        EXPECT_EQ(AttributesOf(ConditionAt(node, 0)).at("decline_reason"),
                  "unsupported_op");

        milvus::query::PlanOptions unloaded_hint_opts;
        unloaded_hint_opts.index_hints = {{v_fid.get(), kNgramId}};
        unloaded_hint_opts.index_hint_strict = false;
        auto hint_explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(n);
        (void)EvaluateFilter(
            inner_like, inv_seg.get(), n, hint_explain, unloaded_hint_opts);
        proto::internal::ExecutionExplain hint_node;
        ASSERT_TRUE(hint_node.ParseFromString(hint_explain->Finish()));
        ASSERT_EQ(ConditionCount(hint_node), 1);
        EXPECT_EQ(AttributesOf(ConditionAt(hint_node, 0)).at("decline_reason"),
                  "hint_mismatch");
    }

    // 2c. JSON BinaryRange with >2^53 int64 bound -> unsupported_literal.
    {
        auto huge_range = std::make_shared<expr::BinaryRangeFilterExpr>(
            expr::ColumnInfo(json_fid, DataType::JSON, {"b"}),
            Int64Literal(1LL << 54),
            Int64Literal((1LL << 54) + 10),
            true,
            true);
        huge_range->SetExprId(93108);
        auto [attrs, _] = eval_reason(huge_range);
        EXPECT_EQ(attrs.at("decline_reason"), "unsupported_literal");
    }

    // 3. JSON /c (uncovered path) -> path_uncovered.
    {
        auto uncovered = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(json_fid, DataType::JSON, {"c"}),
            proto::plan::OpType::Equal,
            Int64Literal(1),
            std::vector<proto::plan::GenericValue>());
        uncovered->SetExprId(93103);
        auto [attrs, _] = eval_reason(uncovered);
        EXPECT_EQ(attrs.at("decline_reason"), "path_uncovered");
    }

    // 4. JSON /a (VARCHAR index) queried with INT64 literal -> type_mismatch
    // (must NOT be shadowed by /b's path_uncovered).
    {
        auto mismatch = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(json_fid, DataType::JSON, {"a"}),
            proto::plan::OpType::Equal,
            Int64Literal(1),
            std::vector<proto::plan::GenericValue>());
        mismatch->SetExprId(93104);
        auto [attrs, _] = eval_reason(mismatch);
        EXPECT_EQ(attrs.at("decline_reason"), "type_mismatch");
    }

    // 5. Indexed INT8 overflow (i8 > 200) -> constant_false omitted from reasons, 0 idx_scan & 0 raw_scan.
    {
        auto overflow_gt = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(i8_fid, DataType::INT8),
            proto::plan::OpType::GreaterThan,
            Int64Literal(200),
            std::vector<proto::plan::GenericValue>());
        overflow_gt->SetExprId(93105);
        auto [attrs, ctrs] = eval_reason(overflow_gt);
        EXPECT_EQ(attrs.count("index_id"), 0u);
        EXPECT_EQ(attrs.count("decline_reason"), 0u);
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), 0);
        EXPECT_EQ(ctrs.at("rows_out"), 0);
    }

    // 6. Indexed JSON json_contains_any(j['a'], []) -> constant_false omitted from reasons, 0 idx_scan & 0 raw_scan.
    {
        auto empty_contains = std::make_shared<expr::JsonContainsExpr>(
            expr::ColumnInfo(json_fid, DataType::JSON, {"a"}),
            proto::plan::JSONContainsExpr_JSONOp_ContainsAny,
            /*same_type=*/true,
            std::vector<proto::plan::GenericValue>{});
        empty_contains->SetExprId(93106);
        auto [attrs, ctrs] = eval_reason(empty_contains);
        EXPECT_EQ(attrs.count("index_id"), 0u);
        EXPECT_EQ(attrs.count("decline_reason"), 0u);
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
    ASSERT_EQ(ConditionCount(cold_node), 1);
    EXPECT_EQ(AttributesOf(ConditionAt(cold_node, 0)).at("decline_reason"),
              "no_index");
    EXPECT_EQ(CountersOf(ConditionAt(cold_node, 0)).at("raw_scan"), n);

    // Warm run hits RawExprCacheAdapter: decline_reason="cached", raw_scan=0.
    auto warm = eval_with_sub_cache();
    proto::internal::ExecutionExplain warm_node;
    ASSERT_TRUE(warm_node.ParseFromString(warm->Finish()));
    ASSERT_EQ(ConditionCount(warm_node), 1);
    EXPECT_EQ(ConditionAt(warm_node, 0).raw_scan_segments(), 0);
    const auto warm_attrs = AttributesOf(ConditionAt(warm_node, 0));
    const auto warm_ctrs = CountersOf(ConditionAt(warm_node, 0));
    EXPECT_EQ(warm_attrs.at("decline_reason"), "cached");
    EXPECT_EQ(warm_ctrs.at("idx_scan"), 0);
    EXPECT_EQ(warm_ctrs.at("raw_scan"), 0);
    EXPECT_EQ(warm_ctrs.at("rows_in"), n);
    EXPECT_EQ(warm_ctrs.at("rows_out"), 2);
}

TEST(MultiScalarIndexExprTest,
     ExplainIndexedAndGeometryCacheHitsAndSingleBatchElementHits) {
    using milvus::exec::ExprResCacheManager;
    auto& mgr = ExprResCacheManager::Instance();
    const bool prev_enabled = ExprResCacheManager::IsEnabled();
    ExprResCacheManager::SetEnabled(true);
    mgr.Clear();
    mgr.SetCapacityBytes(4ULL << 20);
    const bool prev_geo_cache =
        SegcoreConfig::default_config().get_enable_geometry_cache();
    auto restore = folly::makeGuard([&] {
        SegcoreConfig::default_config().set_enable_geometry_cache(
            prev_geo_cache);
        mgr.Clear();
        ExprResCacheManager::SetEnabled(prev_enabled);
    });

    // 1. Scalar index + JSON EXISTS index hit ExprCacheHelper::GetOrCompute:
    // cold run reports index usage; warm run reports decline_reason="cached"
    // with idx_scan=0, raw_scan=0.
    {
        auto schema = std::make_shared<Schema>();
        auto pk_fid = schema->AddDebugField("id", DataType::INT64);
        schema->set_primary_field_id(pk_fid);
        auto v_fid = schema->AddDebugField("v", DataType::VARCHAR);
        auto j_fid = schema->AddDebugField("j", DataType::JSON);

        const int64_t n = static_cast<int64_t>(kVarCharValues.size());
        std::vector<int64_t> pks(n);
        std::vector<std::string> jsons;
        for (int64_t i = 0; i < n; ++i) {
            pks[i] = i;
            jsons.push_back(i % 2 == 0 ? R"({"k": 1})" : R"({"other": 2})");
        }
        auto pk_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        pk_field->FillFieldData(pks.data(), n);
        auto v_field = milvus::test::expr_index::StringField(kVarCharValues);
        auto j_field = milvus::test::expr_index::JsonField(jsons);

        RawFieldFiles files;
        auto seg = CreateSealedSegment(schema);
        seg->LoadFieldData(files.Prepare(pk_fid, {pk_field}));
        seg->LoadFieldData(files.Prepare(v_fid, {v_field}));
        seg->LoadFieldData(files.Prepare(j_fid, {j_field}));

        constexpr int64_t kInvId = 94101;
        constexpr int64_t kJsonId = 94102;
        InstallIndexWithId(*seg,
                           v_fid,
                           DataType::VARCHAR,
                           kInvId,
                           BuildIndex(v_fid,
                                      DataType::VARCHAR,
                                      milvus::index::INVERTED_INDEX_TYPE,
                                      {v_field}));
        InstallIndexWithId(
            *seg,
            j_fid,
            DataType::JSON,
            kJsonId,
            BuildIndex(j_fid,
                       DataType::JSON,
                       milvus::index::INVERTED_INDEX_TYPE,
                       {j_field},
                       {{JSON_PATH, "/k"}, {JSON_CAST_TYPE, "DOUBLE"}}),
            {{JSON_PATH, "/k"}, {JSON_CAST_TYPE, "DOUBLE"}});

        auto run_leaf = [&](const expr::TypedExprPtr& leaf) {
            auto explain =
                std::make_shared<milvus::exec::ExecutionExplainRecorder>(n);
            auto qc = std::make_shared<milvus::exec::QueryContext>(
                DEAFULT_QUERY_ID, seg.get(), n, MAX_TIMESTAMP);
            qc->set_read_schema(schema);
            qc->set_read_snapshot(seg->CaptureReadSnapshot());
            qc->set_enable_sub_expr_cache_write(true);
            qc->SetExplain(explain);
            milvus::exec::ExecContext ec(qc.get());
            ec.SetExplain(explain);
            std::vector<expr::TypedExprPtr> filters{leaf};
            milvus::exec::ExprSet exprs(filters, &ec, /*null_rejecting=*/true);
            std::vector<VectorPtr> results;
            milvus::exec::EvalCtx eval_ctx(&ec);
            exprs.Eval(0, 1, true, eval_ctx, results);
            proto::internal::ExecutionExplain out;
            EXPECT_TRUE(out.ParseFromString(explain->Finish()));
            return out;
        };

        proto::plan::GenericValue target;
        target.set_string_val("prefix_beta");
        auto v_eq = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(v_fid, DataType::VARCHAR),
            proto::plan::OpType::Equal,
            target,
            std::vector<proto::plan::GenericValue>());
        v_eq->SetExprId(94111);

        auto cold_v = run_leaf(v_eq);
        ASSERT_EQ(ConditionCount(cold_v), 1);
        EXPECT_EQ(AttributesOf(ConditionAt(cold_v, 0)).at("index_id"),
                  std::to_string(kInvId));
        EXPECT_EQ(CountersOf(ConditionAt(cold_v, 0)).at("idx_scan"), n);

        auto warm_v = run_leaf(v_eq);
        ASSERT_EQ(ConditionCount(warm_v), 1);
        EXPECT_EQ(AttributesOf(ConditionAt(warm_v, 0)).count("index_id"), 0u);
        EXPECT_EQ(AttributesOf(ConditionAt(warm_v, 0)).at("decline_reason"),
                  "cached");
        EXPECT_EQ(CountersOf(ConditionAt(warm_v, 0)).at("idx_scan"), 0);
        EXPECT_EQ(CountersOf(ConditionAt(warm_v, 0)).at("raw_scan"), 0);
        EXPECT_EQ(CountersOf(ConditionAt(warm_v, 0)).at("rows_out"), 2);

        auto j_exists = std::make_shared<expr::ExistsExpr>(
            expr::ColumnInfo(j_fid, DataType::JSON, {"k"}));
        j_exists->SetExprId(94112);

        auto cold_j = run_leaf(j_exists);
        ASSERT_EQ(ConditionCount(cold_j), 1);
        EXPECT_EQ(AttributesOf(ConditionAt(cold_j, 0)).at("index_id"),
                  std::to_string(kJsonId));

        auto warm_j = run_leaf(j_exists);
        ASSERT_EQ(ConditionCount(warm_j), 1);
        EXPECT_EQ(AttributesOf(ConditionAt(warm_j, 0)).count("index_id"), 0u);
        EXPECT_EQ(AttributesOf(ConditionAt(warm_j, 0)).at("decline_reason"),
                  "cached");
        EXPECT_EQ(CountersOf(ConditionAt(warm_j, 0)).at("idx_scan"), 0);
        EXPECT_EQ(CountersOf(ConditionAt(warm_j, 0)).at("raw_scan"), 0);
        EXPECT_EQ(CountersOf(ConditionAt(warm_j, 0)).at("rows_out"), 5);
    }

    // 2. Single-batch nested array index (row_count <= 8192) records Batch 0
    // element_hits before ~LeafEvalGuard runs.
    {
        mgr.Clear();
        auto schema = std::make_shared<Schema>();
        auto pk_fid = schema->AddDebugField("id", DataType::INT64);
        schema->set_primary_field_id(pk_fid);
        auto sub_int_fid = schema->AddDebugArrayField(
            "struct_array[sub_int]", DataType::INT32, false);

        constexpr int64_t row_count = 3;
        auto insert_data = std::make_unique<InsertRecordProto>();
        std::vector<int64_t> ids = {0, 1, 2};
        auto id_array = CreateDataArrayFrom(
            ids.data(), nullptr, row_count, (*schema)[pk_fid]);
        insert_data->mutable_fields_data()->AddAllocated(id_array.release());

        auto* sub_field = insert_data->add_fields_data();
        sub_field->set_field_id(sub_int_fid.get());
        sub_field->set_field_name("struct_array[sub_int]");
        sub_field->set_type(proto::schema::DataType::Array);
        auto* array_data = sub_field->mutable_scalars()->mutable_array_data();
        array_data->set_element_type(proto::schema::DataType::Int32);
        auto append_row = [array_data](std::initializer_list<int32_t> values) {
            auto* row = array_data->add_data();
            for (auto v : values) {
                row->mutable_int_data()->add_data(v);
            }
        };
        // Row 0 has 2 matching elements, Row 1 has 1 matching element -> 3 element_hits, 2 rows_out.
        append_row({9100, 9200});
        append_row({9001, 2});
        append_row({5});
        insert_data->set_num_rows(row_count);

        auto arr_chunk =
            std::make_shared<FieldData<milvus::Array>>(DataType::ARRAY, false);
        std::vector<milvus::Array> arr_vals;
        arr_vals.reserve(array_data->data_size());
        for (int i = 0; i < array_data->data_size(); ++i) {
            arr_vals.emplace_back(array_data->data(i));
        }
        arr_chunk->FillFieldData(arr_vals.data(), row_count);

        GeneratedData generated;
        generated.schema_ = schema;
        generated.raw_ = insert_data.release();
        generated.row_ids_ = {0, 1, 2};
        generated.timestamps_ = {0, 1, 2};
        auto segment = CreateSealedWithFieldDataLoaded(schema, generated);

        constexpr int64_t kNestedIdxId = 94201;
        InstallIndexWithId(*segment,
                           sub_int_fid,
                           DataType::ARRAY,
                           kNestedIdxId,
                           BuildIndex(sub_int_fid,
                                      DataType::ARRAY,
                                      milvus::index::INVERTED_INDEX_TYPE,
                                      {arr_chunk},
                                      Config::object(),
                                      DataType::INT32,
                                      /*nested=*/true));

        expr::ColumnInfo sub_col(
            sub_int_fid, DataType::INT32, DataType::INT32, {}, false);
        sub_col.element_level_ = true;
        auto sub_ge = std::make_shared<expr::UnaryRangeFilterExpr>(
            sub_col,
            proto::plan::OpType::GreaterEqual,
            Int64Literal(9000),
            std::vector<proto::plan::GenericValue>());
        sub_ge->SetExprId(94211);
        auto match_expr = std::make_shared<expr::MatchExpr>(
            "struct_array", expr::MatchType::MatchAny, 0, sub_ge);

        auto run_match = [&](const expr::TypedExprPtr& match) {
            auto explain =
                std::make_shared<milvus::exec::ExecutionExplainRecorder>(
                    row_count);
            auto qc = std::make_shared<milvus::exec::QueryContext>(
                "explain_nested_cache_test",
                segment.get(),
                row_count,
                MAX_TIMESTAMP);
            qc->set_read_schema(schema);
            qc->set_read_snapshot(segment->CaptureReadSnapshot());
            qc->set_enable_sub_expr_cache_write(true);
            qc->SetExplain(explain);
            milvus::exec::ExecContext ec(qc.get());
            ec.SetExplain(explain);
            std::vector<expr::TypedExprPtr> filters{match};
            milvus::exec::ExprSet exprs(filters, &ec, /*null_rejecting=*/true);
            std::vector<VectorPtr> results;
            milvus::exec::EvalCtx eval_ctx(&ec);
            exprs.Eval(0, 1, true, eval_ctx, results);
            auto col = std::dynamic_pointer_cast<ColumnVector>(results[0]);
            proto::internal::ExecutionExplain out;
            EXPECT_TRUE(out.ParseFromString(explain->Finish()));
            return std::make_pair(std::move(col), std::move(out));
        };

        auto [res, node] = run_match(match_expr);
        ExpectHitRows(res, {0, 1}, "nested index Batch 0 element_hits");

        ASSERT_EQ(ConditionCount(node), 1);
        const auto attrs = AttributesOf(ConditionAt(node, 0));
        const auto ctrs = CountersOf(ConditionAt(node, 0));
        EXPECT_EQ(attrs.at("index_id"), std::to_string(kNestedIdxId));
        EXPECT_EQ(ctrs.at("rows_out"), 3);
        EXPECT_EQ(ctrs.at("element_hits"), 3);
    }

    // 3. Single GIS predicate with SimpleGeometryCache enabled (no R-Tree)
    // reports decline_reason="cached" and raw_scan=0.
    {
        SegcoreConfig::default_config().set_enable_geometry_cache(true);
        auto schema = std::make_shared<Schema>();
        auto pk_fid = schema->AddDebugField("id", DataType::INT64);
        schema->set_primary_field_id(pk_fid);
        auto geo_fid = schema->AddDebugField("geo", DataType::GEOMETRY);

        auto make_wkb = [](const char* wkt) {
            return Geometry(GetThreadLocalGEOSContext(), wkt).to_wkb_string();
        };
        const std::vector<std::string> wkbs = {
            make_wkb("POINT(1 1)"),
            make_wkb("POINT(50 50)"),
        };
        std::vector<int64_t> pks = {0, 1};
        auto pk_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        pk_field->FillFieldData(pks.data(), 2);
        auto geo_field =
            storage::CreateFieldData(DataType::GEOMETRY, DataType::NONE, false);
        geo_field->FillFieldData(wkbs.data(), wkbs.size());

        RawFieldFiles files;
        auto segment = CreateSealedSegment(schema);
        segment->LoadFieldData(files.Prepare(pk_fid, {pk_field}));
        segment->LoadFieldData(files.Prepare(geo_fid, {geo_field}));
        ASSERT_NE(segment->GetGeometryCache(geo_fid), nullptr);

        auto single_gis = std::make_shared<expr::GISFunctionFilterExpr>(
            expr::ColumnInfo(geo_fid, DataType::GEOMETRY),
            proto::plan::GISFunctionFilterExpr_GISOp_Intersects,
            "POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))");
        single_gis->SetExprId(94301);

        // Temporarily disable ExprResCacheManager so RawExprCacheAdapter does
        // not mask PhyGISFunctionFilterExpr's own SimpleGeometryCache path.
        ExprResCacheManager::SetEnabled(false);
        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(2);
        auto res = EvaluateFilter(single_gis, segment.get(), 2, explain);
        ExpectHitRows(res, {0}, "single GIS predicate via SimpleGeometryCache");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(ConditionCount(node), 1);
        const auto attrs = AttributesOf(ConditionAt(node, 0));
        const auto ctrs = CountersOf(ConditionAt(node, 0));
        EXPECT_EQ(attrs.at("decline_reason"), "cached");
        EXPECT_EQ(ctrs.at("idx_scan"), 0);
        EXPECT_EQ(ctrs.at("raw_scan"), 0);
        EXPECT_EQ(ctrs.at("rows_out"), 1);
    }
}

TEST(MultiScalarIndexExprTest,
     ExplainGrowingSegmentReasonAndInternalFilterFold) {
    const int64_t row_count = 4;
    const std::vector<int64_t> ids = {1, 2, 3, 4};
    const std::vector<int64_t> vals = {10, 20, 30, 40};
    std::vector<int64_t> row_ids = {0, 1, 2, 3};
    std::vector<uint64_t> timestamps = {1, 2, 3, 4};

    auto base_schema = std::make_shared<Schema>();
    auto pk_fid = base_schema->AddDebugField("id", DataType::INT64);
    base_schema->set_primary_field_id(pk_fid);
    auto val_fid = base_schema->AddDebugField("val", DataType::INT64);

    auto base_insert = std::make_shared<InsertRecordProto>();
    base_insert->mutable_fields_data()->AddAllocated(
        CreateDataArrayFrom(
            ids.data(), nullptr, row_count, (*base_schema)[pk_fid])
            .release());
    base_insert->mutable_fields_data()->AddAllocated(
        CreateDataArrayFrom(
            vals.data(), nullptr, row_count, (*base_schema)[val_fid])
            .release());
    base_insert->set_num_rows(row_count);

    auto growing = CreateGrowingSegment(base_schema, empty_index_meta);
    auto offset = growing->PreInsert(row_count);
    growing->Insert(
        offset, row_count, row_ids.data(), timestamps.data(), base_insert);

    // 1. Growing segment raw fallback reports growing_segments=1 and reason="growing".
    {
        auto val_ge = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(val_fid, DataType::INT64),
            proto::plan::OpType::GreaterEqual,
            Int64Literal(20),
            std::vector<proto::plan::GenericValue>());
        val_ge->SetExprId(95001);

        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        auto res = EvaluateFilter(val_ge, growing.get(), row_count, explain);
        ExpectHitRows(res, {1, 2, 3}, "growing raw fallback val >= 20");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        EXPECT_EQ(node.group().segments(), 1);
        EXPECT_EQ(node.group().sealed_segments(), 0);
        EXPECT_EQ(node.group().growing_segments(), 1);
        ASSERT_EQ(ConditionCount(node), 1);
        const auto& cond = ConditionAt(node, 0);
        EXPECT_EQ(cond.raw_scan_segments(), 1);
        ASSERT_EQ(cond.reasons_size(), 1);
        EXPECT_EQ(cond.reasons(0).reason(), "growing");
        EXPECT_EQ(cond.reasons(0).raw_scan(), row_count);
    }

    // 2. Growing segment PkIndex hit reports used(family="pk") rather than "growing".
    {
        auto pk_eq = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(pk_fid, DataType::INT64),
            proto::plan::OpType::Equal,
            Int64Literal(2),
            std::vector<proto::plan::GenericValue>());
        pk_eq->SetExprId(95002);

        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(row_count);
        auto res = EvaluateFilter(pk_eq, growing.get(), row_count, explain);
        ExpectHitRows(res, {1}, "growing PkIndex id == 2");

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        EXPECT_EQ(node.group().growing_segments(), 1);
        ASSERT_EQ(ConditionCount(node), 1);
        const auto& cond = ConditionAt(node, 0);
        EXPECT_EQ(cond.raw_scan_segments(), 0);
        EXPECT_EQ(cond.reasons_size(), 0);
        ASSERT_EQ(cond.used_size(), 1);
        EXPECT_EQ(cond.used(0).family(), "pk");
        EXPECT_EQ(cond.used(0).idx_scan(), row_count);
    }

    // 3. Internal-only FilterBitsNode (untagged AlwaysTrueExpr for Entity TTL)
    // followed by MvccNode folds directly into EXPLAIN_OP_MVCC with no empty
    // EXPLAIN_OP_FILTER operator emitted.
    {
        auto ttl_schema = std::make_shared<Schema>();
        auto ttl_pk_fid = ttl_schema->AddDebugField("id", DataType::INT64);
        ttl_schema->set_primary_field_id(ttl_pk_fid);
        auto ttl_fid = ttl_schema->AddDebugField(
            "ttl", DataType::INT64, /*nullable=*/true);
        ttl_schema->set_ttl_field_id(ttl_fid);

        // Row 0 expired (500 < 1000), rows 1..3 alive.
        const std::vector<int64_t> ttl_vals = {
            500LL, 9999999999LL, 9999999999LL, 9999999999LL};
        bool ttl_valid[4] = {true, true, true, false};

        auto ttl_insert = std::make_shared<InsertRecordProto>();
        ttl_insert->mutable_fields_data()->AddAllocated(
            CreateDataArrayFrom(
                ids.data(), nullptr, row_count, (*ttl_schema)[ttl_pk_fid])
                .release());
        ttl_insert->mutable_fields_data()->AddAllocated(
            CreateDataArrayFrom(
                ttl_vals.data(), ttl_valid, row_count, (*ttl_schema)[ttl_fid])
                .release());
        ttl_insert->set_num_rows(row_count);

        auto ttl_growing = CreateGrowingSegment(ttl_schema, empty_index_meta);
        auto ttl_off = ttl_growing->PreInsert(row_count);
        ttl_growing->Insert(
            ttl_off, row_count, row_ids.data(), timestamps.data(), ttl_insert);

        auto always_true = std::make_shared<expr::AlwaysTrueExpr>();
        auto filter_node = std::make_shared<plan::FilterBitsNode>(
            DEFAULT_PLANNODE_ID, always_true);
        auto mvcc_node = std::make_shared<plan::MvccNode>(
            "2", std::vector<plan::PlanNodePtr>{filter_node});

        auto retrieve_plan = std::make_unique<query::RetrievePlan>(ttl_schema);
        retrieve_plan->plan_node_ = std::make_unique<query::RetrievePlanNode>();
        retrieve_plan->plan_node_->plannodes_ = mvcc_node;
        retrieve_plan->plan_node_->limit_ = Unlimited;
        retrieve_plan->plan_node_->plan_options_.enable_explain = true;
        retrieve_plan->mode_ = query::RetrieveMode::Candidates;

        auto res = ttl_growing->Retrieve(nullptr,
                                         retrieve_plan.get(),
                                         MAX_TIMESTAMP,
                                         /*limit_size=*/4096,
                                         /*ignore_non_pk=*/true,
                                         folly::CancellationToken(),
                                         /*consistency_level=*/0,
                                         /*collection_ttl=*/0,
                                         /*entity_ttl_physical_time_us=*/1000);
        ASSERT_NE(res, nullptr);
        EXPECT_EQ(res->offset_size(), 3);

        proto::internal::ExecutionExplain seg_explain;
        ASSERT_TRUE(seg_explain.ParseFromString(res->explain()));
        EXPECT_EQ(seg_explain.group().growing_segments(), 1);
        ASSERT_EQ(seg_explain.group().operators_size(), 1);
        const auto& mvcc_op = seg_explain.group().operators(0);
        EXPECT_EQ(mvcc_op.kind(), proto::internal::EXPLAIN_OP_MVCC);
        EXPECT_EQ(mvcc_op.rows_in(), row_count);
        EXPECT_EQ(mvcc_op.rows_out(), 3);
        EXPECT_EQ(mvcc_op.conditions_size(), 0);
    }
}

TEST(MultiScalarIndexExprTest, ExplainBitmapInputMaskAndNullableValidity) {
    constexpr int64_t row_count = 4;
    const std::vector<int64_t> ids = {0, 1, 2, 3};
    // Row 0: 10, Row 1: 20, Row 2: NULL, Row 3: 30.
    const std::vector<int64_t> vals = {10, 20, 999, 30};
    bool valid[4] = {true, true, false, true};
    std::vector<int64_t> row_ids = {0, 1, 2, 3};
    std::vector<uint64_t> timestamps = {1, 2, 3, 4};

    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("id", DataType::INT64);
    schema->set_primary_field_id(pk_fid);
    auto val_fid =
        schema->AddDebugField("val", DataType::INT64, /*nullable=*/true);

    auto insert = std::make_shared<InsertRecordProto>();
    insert->mutable_fields_data()->AddAllocated(
        CreateDataArrayFrom(ids.data(), nullptr, row_count, (*schema)[pk_fid])
            .release());
    insert->mutable_fields_data()->AddAllocated(
        CreateDataArrayFrom(vals.data(), valid, row_count, (*schema)[val_fid])
            .release());
    insert->set_num_rows(row_count);

    auto growing = CreateGrowingSegment(schema, empty_index_meta);
    auto off = growing->PreInsert(row_count);
    growing->Insert(off, row_count, row_ids.data(), timestamps.data(), insert);

    // val != 10: row 0 -> FALSE, row 1 -> TRUE, row 2 (NULL) -> UNKNOWN, row 3 -> TRUE.
    // With bitmap_input selecting {row 0, row 1, row 2} (3 active rows),
    // rows_in must be 3 and rows_out must be 1 (row 1 only; NULL row 2 and
    // masked-out row 3 must not be counted as hits).
    auto ne_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(val_fid, DataType::INT64, {}, /*nullable=*/true),
        proto::plan::OpType::NotEqual,
        Int64Literal(10),
        std::vector<proto::plan::GenericValue>());
    ne_expr->SetExprId(96001);

    auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
        row_count,
        /*is_growing=*/true);
    auto qc = std::make_shared<milvus::exec::QueryContext>(
        DEAFULT_QUERY_ID, growing.get(), row_count, MAX_TIMESTAMP);
    qc->set_read_schema(schema);
    qc->SetExplain(explain);
    milvus::exec::ExecContext ec(qc.get());
    ec.SetExplain(explain);
    std::vector<expr::TypedExprPtr> filters{ne_expr};
    milvus::exec::ExprSet exprs(filters, &ec, /*null_rejecting=*/true);

    TargetBitmap mask(row_count, false);
    mask.set(0);
    mask.set(1);
    mask.set(2);
    milvus::exec::EvalCtx eval_ctx(&ec);
    eval_ctx.set_bitmap_input(std::move(mask));
    std::vector<VectorPtr> results;
    exprs.Eval(0, 1, true, eval_ctx, results);

    proto::internal::ExecutionExplain node;
    ASSERT_TRUE(node.ParseFromString(explain->Finish()));
    ASSERT_EQ(ConditionCount(node), 1);
    const auto ctrs = CountersOf(ConditionAt(node, 0));
    EXPECT_EQ(ctrs.at("rows_in"), 3);
    EXPECT_EQ(ctrs.at("rows_out"), 1);
    EXPECT_LE(ctrs.at("rows_out"), ctrs.at("rows_in"));
}

TEST(MultiScalarIndexExprTest,
     ExplainCompoundWholeFilterCacheHitAndPureIterativeZeroCandidates) {
    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("id", DataType::INT64);
    schema->set_primary_field_id(pk_fid);
    auto a_fid = schema->AddDebugField("a", DataType::INT64);
    auto b_fid = schema->AddDebugField("b", DataType::INT64);

    const std::vector<int64_t> ids = {0, 1, 2, 3};
    const std::vector<int64_t> a_vals = {10, 20, 30, 40};
    const std::vector<int64_t> b_vals = {5, 25, 15, 35};
    auto pk_data = std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    pk_data->FillFieldData(ids.data(), ids.size());
    auto a_data = std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    a_data->FillFieldData(a_vals.data(), a_vals.size());
    auto b_data = std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    b_data->FillFieldData(b_vals.data(), b_vals.size());

    RawFieldFiles files;
    auto segment = CreateSealedSegment(schema);
    segment->LoadFieldData(files.Prepare(pk_fid, {pk_data}));
    segment->LoadFieldData(files.Prepare(a_fid, {a_data}));
    segment->LoadFieldData(files.Prepare(b_fid, {b_data}));

    auto a_eq = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(a_fid, DataType::INT64),
        proto::plan::OpType::Equal,
        Int64Literal(20));
    a_eq->SetExprId(97001);
    auto b_gt = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(b_fid, DataType::INT64),
        proto::plan::OpType::GreaterThan,
        Int64Literal(30));
    b_gt->SetExprId(97002);
    auto or_expr = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::Or, a_eq, b_gt);

    // 1. Compound OR whole-filter cache hit records "cached" on all child leaves
    //    without fabricating per-leaf rows_in / rows_out or marking skipped.
    {
        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            ids.size(), /*is_growing=*/false);
        explain->RegisterOperator(0, milvus::ExplainOperatorKind::kFilter);
        explain->SetOperatorRows(0, 4, 2);
        explain->MarkOperatorCached(0);

        auto qc = std::make_shared<milvus::exec::QueryContext>(
            DEAFULT_QUERY_ID, segment.get(), ids.size(), MAX_TIMESTAMP);
        qc->set_read_schema(schema);
        qc->SetExplain(explain);
        milvus::exec::ExecContext ec(qc.get());
        ec.SetExplain(explain);
        ec.SetOperatorStep(0);
        std::vector<expr::TypedExprPtr> filters{or_expr};
        milvus::exec::ExprSet exprs(filters, &ec, /*null_rejecting=*/true);

        milvus::exec::RecordWholeFilterCacheHitExplain(
            exprs.exprs(), explain.get(), 4, 2);

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(ConditionCount(node), 2);
        for (int idx = 0; idx < 2; ++idx) {
            const auto& cond = ConditionAt(node, idx);
            EXPECT_EQ(cond.segments(), 1);
            EXPECT_EQ(cond.skipped_segments(), 0);
            EXPECT_EQ(cond.rows_in(), 0);
            EXPECT_EQ(cond.rows_out(), 0);
            ASSERT_EQ(cond.reasons_size(), 1);
            EXPECT_EQ(cond.reasons(0).reason(), "cached");
            EXPECT_EQ(cond.reasons(0).segments(), 1);
        }
    }

    // 2. Pure IterativeFilter with 0 candidate batches (no upstream pre-filter)
    //    marks child leaves entered so Finish() emits skipped_segments = 1.
    {
        auto explain = std::make_shared<milvus::exec::ExecutionExplainRecorder>(
            ids.size(), /*is_growing=*/false);
        explain->RegisterOperator(
            0, milvus::ExplainOperatorKind::kIterativeFilter);
        explain->SetOperatorRows(0, 0, 0);

        auto qc = std::make_shared<milvus::exec::QueryContext>(
            DEAFULT_QUERY_ID, segment.get(), ids.size(), MAX_TIMESTAMP);
        qc->set_read_schema(schema);
        qc->SetExplain(explain);
        milvus::exec::ExecContext ec(qc.get());
        ec.SetExplain(explain);
        ec.SetOperatorStep(0);
        std::vector<expr::TypedExprPtr> filters{or_expr};
        milvus::exec::ExprSet exprs(filters, &ec, /*null_rejecting=*/true);

        milvus::exec::MarkExplainEvalEntered(exprs.exprs(), explain.get());

        proto::internal::ExecutionExplain node;
        ASSERT_TRUE(node.ParseFromString(explain->Finish()));
        ASSERT_EQ(ConditionCount(node), 2);
        for (int idx = 0; idx < 2; ++idx) {
            const auto& cond = ConditionAt(node, idx);
            EXPECT_EQ(cond.segments(), 1);
            EXPECT_EQ(cond.skipped_segments(), 1);
        }
    }
}

TEST(MultiScalarIndexExprTest, ExplainLeafTimingAndSkippedPrefetchAttribution) {
    // 1. Recorder-level nanosecond rounding, multi-operand eval_time suppression
    // on individual operand sub-items, prefetch_wait_time clamping/folding, and
    // skipped-leaf prefetch_time.
    {
        milvus::exec::ExecutionExplainRecorder explain(1000);
        explain.RegisterOperator(0, milvus::ExplainOperatorKind::kFilter);
        explain.SetDefaultFilterStep(0);
        explain.SetEvalOrder(10, 1, FieldId(101));
        explain.SetEvalOrder(20, 2, FieldId(102));
        explain.MarkEvalEntered(10);
        explain.MarkEvalEntered(20);

        explain.RecordDecision(FieldId(101),
                               10,
                               /*used_index=*/true,
                               5001,
                               "sort",
                               false,
                               "",
                               "");
        explain.RecordDecision(FieldId(102),
                               10,
                               /*used_index=*/false,
                               0,
                               "",
                               false,
                               "",
                               "no_index_on_field");
        explain.AddRows(10, 1000, 0);
        explain.AddLeafEvalNs(10, 1500);
        explain.AddLeafPrefetchNs(10, 499);  // < 500 ns rounds to 0

        // Short-circuited leaf 20: never evaluated, no RecordDecision, but
        // background PrefetchAsync spent 8400 ns (8 us).
        explain.AddLeafPrefetchNs(20, 8400);
        explain.AddOperatorTimeNs(0, 12600);
        explain.AddOperatorPrefetchWaitNs(0, 7600);

        proto::internal::ExecutionExplain seg_node;
        ASSERT_TRUE(seg_node.ParseFromString(explain.Finish()));
        ASSERT_EQ(seg_node.group().operators_size(), 1);
        const auto& op_node = seg_node.group().operators(0);
        EXPECT_EQ(op_node.time().sum_us(), 13);
        EXPECT_EQ(op_node.prefetch_wait_time().sum_us(), 8);
        ASSERT_EQ(op_node.conditions_size(), 2);

        const auto& cond0 = op_node.conditions(0);
        EXPECT_EQ(cond0.expr_id(), 10);
        EXPECT_EQ(cond0.eval_time().sum_us(), 2);
        EXPECT_EQ(cond0.prefetch_time().sum_us(), 0);
        ASSERT_EQ(cond0.used_size(), 1);
        ASSERT_EQ(cond0.reasons_size(), 1);
        EXPECT_FALSE(cond0.used(0).has_eval_time());
        EXPECT_FALSE(cond0.reasons(0).has_eval_time());

        const auto& cond1 = op_node.conditions(1);
        EXPECT_EQ(cond1.expr_id(), 20);
        EXPECT_EQ(cond1.skipped_segments(), 1);
        EXPECT_FALSE(cond1.has_eval_time());
        EXPECT_EQ(cond1.prefetch_time().sum_us(), 8);
    }

    // 1b. Internal folded filter folds both time_ns and prefetch_wait_ns into
    // mvcc, and unattributed whole-filter cache hit separates unattributed time.
    {
        milvus::exec::ExecutionExplainRecorder folded(500);
        folded.RegisterOperator(
            0, milvus::exec::ExplainOperatorKind::kFilter, "", true);
        folded.RegisterOperator(1, milvus::exec::ExplainOperatorKind::kMvcc);
        folded.SetOperatorRows(0, 500, 480);
        folded.SetOperatorRows(1, 480, 470);
        folded.AddOperatorTimeNs(0, 3000);
        folded.AddOperatorPrefetchWaitNs(0, 2000);
        folded.AddOperatorTimeNs(1, 5000);
        folded.AddOperatorPrefetchWaitNs(1, 1000);

        proto::internal::ExecutionExplain f_out;
        ASSERT_TRUE(f_out.ParseFromString(folded.Finish()));
        ASSERT_EQ(f_out.group().operators_size(), 1);
        const auto& mvcc = f_out.group().operators(0);
        EXPECT_EQ(mvcc.kind(), proto::internal::EXPLAIN_OP_MVCC);
        EXPECT_EQ(mvcc.time().sum_us(), 8);
        EXPECT_EQ(mvcc.prefetch_wait_time().sum_us(), 3);

        milvus::exec::ExecutionExplainRecorder cached_op(500);
        cached_op.RegisterOperator(0, milvus::ExplainOperatorKind::kFilter);
        cached_op.SetOperatorRows(0, 500, 42);
        cached_op.MarkOperatorCached(0);
        cached_op.AddOperatorTimeNs(0, 9000);
        cached_op.AddOperatorPrefetchWaitNs(0, 5000);  // ignored when cached

        proto::internal::ExecutionExplain c_out;
        ASSERT_TRUE(c_out.ParseFromString(cached_op.Finish()));
        ASSERT_EQ(c_out.group().operators_size(), 1);
        const auto& cop = c_out.group().operators(0);
        EXPECT_TRUE(cop.has_time());
        EXPECT_EQ(cop.time().sum_us(), 9);
        EXPECT_EQ(cop.prefetch_wait_time().sum_us(), 0);
    }

    // 2. End-to-end Driver Retrieve with async prefetch records operator time,
    // prefetch_wait_time, and leaf eval_time/prefetch_time without data races.
    {
        auto schema = std::make_shared<Schema>();
        auto pk_fid = schema->AddDebugField("id", DataType::INT64);
        schema->set_primary_field_id(pk_fid);
        auto val_fid = schema->AddDebugField("val", DataType::INT64);

        const int64_t row_count = 256;
        std::vector<int64_t> ids(row_count);
        std::vector<int64_t> vals(row_count);
        for (int64_t i = 0; i < row_count; ++i) {
            ids[i] = i;
            vals[i] = i * 10;
        }
        std::vector<int64_t> ts(row_count, 1);
        auto ts_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        ts_field->FillFieldData(ts.data(), row_count);
        auto pk_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        pk_field->FillFieldData(ids.data(), row_count);
        auto val_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        val_field->FillFieldData(vals.data(), row_count);

        RawFieldFiles files;
        auto segment = CreateSealedSegment(schema);
        segment->LoadFieldData(files.Prepare(TimestampFieldID, {ts_field}));
        segment->LoadFieldData(files.Prepare(pk_fid, {pk_field}));
        segment->LoadFieldData(files.Prepare(val_fid, {val_field}));

        auto pred = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(val_fid, DataType::INT64),
            proto::plan::OpType::LessThan,
            Int64Literal(100),
            std::vector<proto::plan::GenericValue>());
        pred->SetExprId(93001);

        auto filter_node =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, pred);
        auto mvcc_node = std::make_shared<plan::MvccNode>(
            "2", std::vector<plan::PlanNodePtr>{filter_node});
        auto retrieve_plan = std::make_unique<query::RetrievePlan>(schema);
        retrieve_plan->plan_node_ = std::make_unique<query::RetrievePlanNode>();
        retrieve_plan->plan_node_->plannodes_ = mvcc_node;
        retrieve_plan->plan_node_->limit_ = Unlimited;
        retrieve_plan->plan_node_->plan_options_.enable_explain = true;
        retrieve_plan->mode_ = query::RetrieveMode::Candidates;

        auto ret = segment->Retrieve(nullptr,
                                     retrieve_plan.get(),
                                     MAX_TIMESTAMP,
                                     /*limit_size=*/4096,
                                     /*ignore_non_pk=*/true,
                                     folly::CancellationToken(),
                                     /*consistency_level=*/0,
                                     /*collection_ttl=*/0,
                                     /*entity_ttl_physical_time_us=*/0);
        ASSERT_NE(ret, nullptr);
        EXPECT_EQ(ret->offset_size(), 10);

        proto::internal::ExecutionExplain seg_node;
        ASSERT_TRUE(seg_node.ParseFromString(ret->explain()));
        ASSERT_EQ(seg_node.group().operators_size(), 2);
        const auto& f_op = seg_node.group().operators(0);
        EXPECT_EQ(f_op.kind(), proto::internal::EXPLAIN_OP_FILTER);
        ASSERT_TRUE(f_op.has_time());
        ASSERT_TRUE(f_op.has_prefetch_wait_time());
        EXPECT_EQ(f_op.prefetch_wait_time().count(), 1);
        EXPECT_LE(f_op.prefetch_wait_time().sum_us(), f_op.time().sum_us());
        ASSERT_EQ(f_op.conditions_size(), 1);
        const auto& f_cond = f_op.conditions(0);
        EXPECT_EQ(f_cond.rows_in(), row_count);
        EXPECT_EQ(f_cond.rows_out(), 10);
        ASSERT_TRUE(f_cond.has_eval_time());
        EXPECT_EQ(f_cond.eval_time().count(), 1);
        EXPECT_GE(f_cond.eval_time().sum_us(), 0);
    }

    // 3. Direct Eval() without prior WaitPrefetch/PrefetchAsync and with
    // explain disabled (nullptr) still runs DetermineExecPath() and selects the
    // scalar index on index-backed sealed segments.
    {
        auto schema = std::make_shared<Schema>();
        auto pk_fid = schema->AddDebugField("id", DataType::INT64);
        schema->set_primary_field_id(pk_fid);
        auto val_fid = schema->AddDebugField("val", DataType::INT64);

        const int64_t row_count = 128;
        std::vector<int64_t> ids(row_count);
        std::vector<int64_t> vals(row_count);
        for (int64_t i = 0; i < row_count; ++i) {
            ids[i] = i;
            vals[i] = i;
        }
        auto pk_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        pk_field->FillFieldData(ids.data(), row_count);
        auto val_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        val_field->FillFieldData(vals.data(), row_count);

        RawFieldFiles files;
        auto segment = CreateSealedSegment(schema);
        segment->LoadFieldData(files.Prepare(pk_fid, {pk_field}));
        InstallIndexWithId(*segment,
                           val_fid,
                           DataType::INT64,
                           94001,
                           BuildIndex(val_fid,
                                      DataType::INT64,
                                      milvus::index::ASCENDING_SORT,
                                      {val_field}));

        auto qc = std::make_shared<milvus::exec::QueryContext>(
            DEAFULT_QUERY_ID, segment.get(), row_count, MAX_TIMESTAMP);
        qc->set_read_schema(schema);
        qc->set_read_snapshot(segment->CaptureReadSnapshot());
        milvus::exec::ExecContext ec(qc.get());

        auto range_pred = std::make_shared<expr::BinaryRangeFilterExpr>(
            expr::ColumnInfo(val_fid, DataType::INT64),
            Int64Literal(10),
            Int64Literal(20),
            true,
            false);
        auto in_pred = std::make_shared<expr::TermFilterExpr>(
            expr::ColumnInfo(val_fid, DataType::INT64),
            std::vector<proto::plan::GenericValue>{Int64Literal(12),
                                                   Int64Literal(15)});
        for (const auto& p :
             std::vector<expr::TypedExprPtr>{range_pred, in_pred}) {
            std::vector<expr::TypedExprPtr> single{p};
            milvus::exec::ExprSet exprs_no_prefetch(single, &ec);
            milvus::exec::EvalCtx eval_ctx(&ec);
            std::vector<VectorPtr> out;
            exprs_no_prefetch.Eval(0, 1, true, eval_ctx, out);
            ASSERT_EQ(out.size(), 1);
            ASSERT_NE(out[0], nullptr);
        }
    }

    // 4. Synchronous NGRAM fusion through RawExprCacheAdapter transfers
    // WaitPrefetch() lookup/prepare duration to the unwrapped leaf rather than
    // dropping it when adapter->Eval() is bypassed, and empty offset_input keeps
    // short-circuited leaves skipped without eval_time.
    {
        auto schema = std::make_shared<Schema>();
        auto pk_fid = schema->AddDebugField("id", DataType::INT64);
        schema->set_primary_field_id(pk_fid);
        auto v_fid = schema->AddDebugField("v", DataType::VARCHAR);

        const int64_t n = static_cast<int64_t>(kVarCharValues.size());
        std::vector<int64_t> pks(n);
        for (int64_t i = 0; i < n; ++i) {
            pks[i] = i;
        }
        auto pk_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        pk_field->FillFieldData(pks.data(), n);
        auto v_field = milvus::test::expr_index::StringField(kVarCharValues);

        RawFieldFiles files;
        auto seg = CreateSealedSegment(schema);
        seg->LoadFieldData(files.Prepare(pk_fid, {pk_field}));
        seg->LoadFieldData(files.Prepare(v_fid, {v_field}));
        InstallIndexWithId(*seg,
                           v_fid,
                           DataType::VARCHAR,
                           95001,
                           BuildIndex(v_fid,
                                      DataType::VARCHAR,
                                      milvus::index::NGRAM_INDEX_TYPE,
                                      {v_field},
                                      {{milvus::index::MIN_GRAM, 3},
                                       {milvus::index::MAX_GRAM, 4}}));

        using milvus::exec::ExprResCacheManager;
        auto& mgr = ExprResCacheManager::Instance();
        const bool prev_enabled = ExprResCacheManager::IsEnabled();
        ExprResCacheManager::SetEnabled(true);
        mgr.Clear();
        mgr.SetCapacityBytes(1ULL << 20);
        auto cleanup = folly::makeGuard([&]() {
            mgr.Clear();
            ExprResCacheManager::SetEnabled(prev_enabled);
        });

        proto::plan::GenericValue pat1;
        pat1.set_string_val("alpha");
        auto like1 = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(v_fid, DataType::VARCHAR),
            proto::plan::OpType::InnerMatch,
            pat1,
            std::vector<proto::plan::GenericValue>());
        like1->SetExprId(95101);

        proto::plan::GenericValue pat2;
        pat2.set_string_val("beta");
        auto like2 = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(v_fid, DataType::VARCHAR),
            proto::plan::OpType::InnerMatch,
            pat2,
            std::vector<proto::plan::GenericValue>());
        like2->SetExprId(95102);

        auto conj = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, like1, like2);

        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(n);
        auto qc = std::make_shared<milvus::exec::QueryContext>(
            DEAFULT_QUERY_ID, seg.get(), n, MAX_TIMESTAMP);
        qc->set_read_schema(schema);
        qc->set_read_snapshot(seg->CaptureReadSnapshot());
        qc->set_enable_sub_expr_cache_write(true);
        qc->SetExplain(explain);
        milvus::exec::ExecContext ec(qc.get());
        ec.SetExplain(explain);
        std::vector<expr::TypedExprPtr> filters{conj};
        milvus::exec::ExprSet exprs_sync(filters, &ec, /*null_rejecting=*/true);
        exprs_sync.WaitPrefetch();
        milvus::exec::EvalCtx eval_ctx(&ec);
        auto res = milvus::exec::EvalExprSetOverAllBatches(
            exprs_sync, eval_ctx, n, "SyncFusedNgramPrepareTiming");
        (void)res;

        proto::internal::ExecutionExplain seg_node;
        ASSERT_TRUE(seg_node.ParseFromString(explain->Finish()));
        ASSERT_GE(ConditionCount(seg_node), 1);
        for (int i = 0; i < ConditionCount(seg_node); ++i) {
            const auto& c = ConditionAt(seg_node, i);
            if (c.skipped_segments() == 0) {
                EXPECT_TRUE(c.has_eval_time());
                EXPECT_GT(c.rows_in(), 0);
            }
        }

        auto raw_pred = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(pk_fid, DataType::INT64),
            proto::plan::OpType::GreaterThan,
            Int64Literal(0),
            std::vector<proto::plan::GenericValue>());
        raw_pred->SetExprId(95201);

        {
            auto prime_qc = std::make_shared<milvus::exec::QueryContext>(
                DEAFULT_QUERY_ID, seg.get(), n, MAX_TIMESTAMP);
            prime_qc->set_read_schema(schema);
            prime_qc->set_read_snapshot(seg->CaptureReadSnapshot());
            prime_qc->set_enable_sub_expr_cache_write(true);
            milvus::exec::ExecContext prime_ec(prime_qc.get());
            std::vector<expr::TypedExprPtr> single{raw_pred};
            milvus::exec::ExprSet prime_exprs(single, &prime_ec);
            milvus::exec::EvalCtx prime_ctx(&prime_ec);
            (void)milvus::exec::EvalExprSetOverAllBatches(
                prime_exprs, prime_ctx, n, "PrimeRawExprCache");
        }

        auto zero_explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(n);
        zero_explain->SetEvalOrder(95201, 1, pk_fid);
        auto zero_qc = std::make_shared<milvus::exec::QueryContext>(
            DEAFULT_QUERY_ID, seg.get(), n, MAX_TIMESTAMP);
        zero_qc->set_read_schema(schema);
        zero_qc->set_read_snapshot(seg->CaptureReadSnapshot());
        zero_qc->set_enable_sub_expr_cache_write(true);
        zero_qc->SetExplain(zero_explain);
        milvus::exec::ExecContext zero_ec(zero_qc.get());
        zero_ec.SetExplain(zero_explain);
        std::vector<expr::TypedExprPtr> zero_filters{raw_pred};
        milvus::exec::ExprSet zero_exprs(zero_filters, &zero_ec);
        zero_exprs.WaitPrefetch();
        milvus::exec::OffsetVector empty_offsets;
        milvus::exec::EvalCtx zero_ctx(&zero_ec, &empty_offsets);
        std::vector<VectorPtr> zero_out;
        zero_exprs.Eval(0, 1, true, zero_ctx, zero_out);

        proto::internal::ExecutionExplain zero_node;
        ASSERT_TRUE(zero_node.ParseFromString(zero_explain->Finish()));
        ASSERT_EQ(ConditionCount(zero_node), 1);
        const auto& z_cond = ConditionAt(zero_node, 0);
        EXPECT_EQ(z_cond.skipped_segments(), 1);
        EXPECT_FALSE(z_cond.has_eval_time());

        // Async PrefetchAsync attributes background lookup/DetermineExecPath
        // solely to prefetch_ns (leaving eval_ns == 0 prior to Eval), while
        // iterative offset_input Eval without PrefetchAsync captures
        // DetermineExecPath inside eval_ns.
        auto pool = std::make_shared<folly::CPUThreadPoolExecutor>(1);
        auto pf_explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(n);
        auto pf_qc = std::make_shared<milvus::exec::QueryContext>(
            DEAFULT_QUERY_ID, seg.get(), n, MAX_TIMESTAMP);
        pf_qc->set_read_schema(schema);
        pf_qc->set_read_snapshot(seg->CaptureReadSnapshot());
        pf_qc->SetExplain(pf_explain);
        milvus::exec::ExecContext pf_ec(pf_qc.get());
        pf_ec.SetExplain(pf_explain);
        auto uncached_pred = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(v_fid, DataType::VARCHAR),
            proto::plan::OpType::Equal,
            [] {
                proto::plan::GenericValue gv;
                gv.set_string_val("alpha");
                return gv;
            }(),
            std::vector<proto::plan::GenericValue>());
        uncached_pred->SetExprId(95301);
        std::vector<expr::TypedExprPtr> pf_filters{uncached_pred};
        milvus::exec::ExprSet pf_exprs(pf_filters, &pf_ec);
        pf_exprs.PrefetchAsync(pool);
        pf_exprs.WaitPrefetch();
        auto* pf_slot = pf_explain->EnsureLeafSlot(95301);
        EXPECT_GT(pf_slot->prefetch_ns.load(), 0);
        EXPECT_EQ(pf_slot->eval_ns.load(), 0);

        auto iter_explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(n);
        auto iter_qc = std::make_shared<milvus::exec::QueryContext>(
            DEAFULT_QUERY_ID, seg.get(), n, MAX_TIMESTAMP);
        iter_qc->set_read_schema(schema);
        iter_qc->set_read_snapshot(seg->CaptureReadSnapshot());
        iter_qc->SetExplain(iter_explain);
        milvus::exec::ExecContext iter_ec(iter_qc.get());
        iter_ec.SetExplain(iter_explain);
        milvus::exec::ExprSet iter_exprs(pf_filters, &iter_ec);
        milvus::exec::OffsetVector iter_offsets{0, 1};
        milvus::exec::EvalCtx iter_ctx(&iter_ec, &iter_offsets);
        std::vector<VectorPtr> iter_out;
        iter_exprs.Eval(0, 1, true, iter_ctx, iter_out);
        EXPECT_GT(iter_explain->EnsureLeafSlot(95301)->eval_ns.load(), 0);
    }
}

TEST(MultiScalarIndexExprTest, ZeroRowBatchesAndFusedShortCircuitTiming) {
    // 1. Zero-row batches (empty offset_input or all-false bitmap_input) on a
    // non-empty segment must not start leaf timers, flush sync-prepare time, or
    // turn skipped leaves into evaluated ones across SegmentExpr, CompareExpr,
    // AlwaysTrueExpr, and LogicalUnaryExpr (NOT).
    {
        auto schema = std::make_shared<Schema>();
        auto a_fid = schema->AddDebugField("a", DataType::INT64);
        auto b_fid = schema->AddDebugField("b", DataType::INT64);
        schema->set_primary_field_id(a_fid);

        const int64_t n = static_cast<int64_t>(kInt64Values.size());
        auto a_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        a_field->FillFieldData(kInt64Values.data(), n);
        auto b_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        b_field->FillFieldData(kInt64Values.data(), n);

        RawFieldFiles files;
        auto seg = CreateSealedSegment(schema);
        seg->LoadFieldData(files.Prepare(a_fid, {a_field}));
        seg->LoadFieldData(files.Prepare(b_fid, {b_field}));

        auto unary_pred = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(a_fid, DataType::INT64),
            proto::plan::OpType::GreaterThan,
            Int64Literal(50),
            std::vector<proto::plan::GenericValue>());
        unary_pred->SetExprId(96001);

        auto cmp_pred = std::make_shared<expr::CompareExpr>(
            a_fid, b_fid, DataType::INT64, DataType::INT64, OpType::Equal);
        cmp_pred->SetExprId(96002);

        auto always_true_pred = std::make_shared<expr::AlwaysTrueExpr>();
        always_true_pred->SetExprId(96003);

        auto not_inner = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(b_fid, DataType::INT64),
            proto::plan::OpType::Equal,
            Int64Literal(20),
            std::vector<proto::plan::GenericValue>());
        not_inner->SetExprId(96004);
        auto not_pred = std::make_shared<expr::LogicalUnaryExpr>(
            expr::LogicalUnaryExpr::OpType::LogicalNot, not_inner);

        struct Case {
            expr::TypedExprPtr expr;
            int64_t leaf_id;
            FieldId fid;
        };
        std::vector<Case> cases = {
            {unary_pred, 96001, a_fid},
            {cmp_pred, 96002, a_fid},
            {always_true_pred, 96003, a_fid},
            {not_pred, 96004, b_fid},
        };

        for (const auto& tc : cases) {
            for (bool use_empty_offsets : {true, false}) {
                auto explain =
                    std::make_shared<milvus::exec::ExecutionExplainRecorder>(n);
                explain->SetEvalOrder(tc.leaf_id, 1, tc.fid);
                auto qc = std::make_shared<milvus::exec::QueryContext>(
                    DEAFULT_QUERY_ID, seg.get(), n, MAX_TIMESTAMP);
                qc->set_read_schema(schema);
                qc->set_read_snapshot(seg->CaptureReadSnapshot());
                qc->SetExplain(explain);
                milvus::exec::ExecContext ec(qc.get());
                ec.SetExplain(explain);
                std::vector<expr::TypedExprPtr> single{tc.expr};
                milvus::exec::ExprSet expr_set(single, &ec);
                expr_set.WaitPrefetch();

                milvus::exec::OffsetVector empty_offsets;
                TargetBitmap empty_mask(n, false);
                milvus::exec::EvalCtx ctx =
                    use_empty_offsets
                        ? milvus::exec::EvalCtx(&ec, &empty_offsets)
                        : milvus::exec::EvalCtx(&ec);
                if (!use_empty_offsets) {
                    ctx.set_bitmap_input(empty_mask.view());
                }
                std::vector<VectorPtr> out;
                expr_set.Eval(0, 1, true, ctx, out);

                EXPECT_EQ(explain->EnsureLeafSlot(tc.leaf_id)->eval_ns.load(),
                          0)
                    << "leaf_id=" << tc.leaf_id
                    << " empty_offsets=" << use_empty_offsets;

                proto::internal::ExecutionExplain pb;
                ASSERT_TRUE(pb.ParseFromString(explain->Finish()));
                ASSERT_EQ(ConditionCount(pb), 1);
                const auto& cond = ConditionAt(pb, 0);
                EXPECT_EQ(cond.skipped_segments(), 1)
                    << "leaf_id=" << tc.leaf_id
                    << " empty_offsets=" << use_empty_offsets;
                EXPECT_FALSE(cond.has_eval_time())
                    << "leaf_id=" << tc.leaf_id
                    << " empty_offsets=" << use_empty_offsets;
            }
        }
    }

    // 2. Fused NGRAM LIKE conjunction attributes each predicate's own Phase 1
    // evaluation time and candidate narrowing directly to its own leaf slot,
    // while any trailing predicate after Phase 1 empties candidates remains
    // skipped without eval_time.
    {
        auto schema = std::make_shared<Schema>();
        auto pk_fid = schema->AddDebugField("id", DataType::INT64);
        schema->set_primary_field_id(pk_fid);
        auto v_fid = schema->AddDebugField("v", DataType::VARCHAR);

        const int64_t n = static_cast<int64_t>(kVarCharValues.size());
        std::vector<int64_t> pks(n);
        for (int64_t i = 0; i < n; ++i) {
            pks[i] = i;
        }
        auto pk_field =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        pk_field->FillFieldData(pks.data(), n);
        auto v_field = milvus::test::expr_index::StringField(kVarCharValues);

        RawFieldFiles files;
        auto seg = CreateSealedSegment(schema);
        seg->LoadFieldData(files.Prepare(pk_fid, {pk_field}));
        seg->LoadFieldData(files.Prepare(v_fid, {v_field}));
        InstallIndexWithId(*seg,
                           v_fid,
                           DataType::VARCHAR,
                           96100,
                           BuildIndex(v_fid,
                                      DataType::VARCHAR,
                                      milvus::index::NGRAM_INDEX_TYPE,
                                      {v_field},
                                      {{milvus::index::MIN_GRAM, 3},
                                       {milvus::index::MAX_GRAM, 4}}));

        proto::plan::GenericValue pat1;
        pat1.set_string_val("alpha");
        auto like1 = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(v_fid, DataType::VARCHAR),
            proto::plan::OpType::InnerMatch,
            pat1,
            std::vector<proto::plan::GenericValue>());
        like1->SetExprId(96101);

        proto::plan::GenericValue pat2;
        pat2.set_string_val("beta");
        auto like2 = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(v_fid, DataType::VARCHAR),
            proto::plan::OpType::InnerMatch,
            pat2,
            std::vector<proto::plan::GenericValue>());
        like2->SetExprId(96102);

        proto::plan::GenericValue pat3;
        pat3.set_string_val("gamma");
        auto like3 = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(v_fid, DataType::VARCHAR),
            proto::plan::OpType::InnerMatch,
            pat3,
            std::vector<proto::plan::GenericValue>());
        like3->SetExprId(96103);

        auto conj12 = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, like1, like2);
        auto conj = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, conj12, like3);

        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(n);
        auto res = EvaluateFilter(conj, seg.get(), n, explain);
        ExpectHitRows(res, {}, "disjoint fused NGRAM LIKE");

        EXPECT_GT(explain->EnsureLeafSlot(96101)->eval_ns.load(), 0);
        EXPECT_GT(explain->EnsureLeafSlot(96102)->eval_ns.load(), 0);
        EXPECT_EQ(explain->EnsureLeafSlot(96103)->eval_ns.load(), 0);

        proto::internal::ExecutionExplain pb;
        ASSERT_TRUE(pb.ParseFromString(explain->Finish()));
        ASSERT_EQ(ConditionCount(pb), 3);
        std::map<int64_t, const proto::internal::ExplainConditionStats*> by_id;
        for (int i = 0; i < ConditionCount(pb); ++i) {
            const auto& c = ConditionAt(pb, i);
            by_id[c.expr_id()] = &c;
        }
        ASSERT_EQ(by_id.count(96101), 1u);
        ASSERT_EQ(by_id.count(96102), 1u);
        ASSERT_EQ(by_id.count(96103), 1u);
        EXPECT_EQ(by_id[96101]->skipped_segments(), 0);
        EXPECT_TRUE(by_id[96101]->has_eval_time());
        EXPECT_EQ(by_id[96101]->rows_in(), n);
        EXPECT_GT(by_id[96101]->rows_out(), 0);
        EXPECT_EQ(by_id[96101]->raw_scan(), 0);

        EXPECT_EQ(by_id[96102]->skipped_segments(), 0);
        EXPECT_TRUE(by_id[96102]->has_eval_time());
        EXPECT_EQ(by_id[96102]->rows_in(), by_id[96101]->rows_out());
        EXPECT_EQ(by_id[96102]->rows_out(), 0);
        EXPECT_EQ(by_id[96102]->idx_scan(), by_id[96101]->rows_out());
        EXPECT_EQ(by_id[96102]->raw_scan(), 0);

        EXPECT_EQ(by_id[96103]->skipped_segments(), 1);
        EXPECT_FALSE(by_id[96103]->has_eval_time());
    }

    // 3. GIS split-fusion attributes each predicate's own R-Tree coarse
    // nanoseconds and candidate narrowing directly to its own leaf slot when an
    // R-Tree runs, while unindexed Refine short-circuit keeps unexecuted
    // trailing predicates skipped without eval_time.
    {
        const bool saved_gis =
            SegcoreConfig::default_config().get_enable_gis_split_fusion();
        SegcoreConfig::default_config().set_enable_gis_split_fusion(true);
        auto restore_gis = folly::makeGuard([&]() {
            SegcoreConfig::default_config().set_enable_gis_split_fusion(
                saved_gis);
        });

        auto schema = std::make_shared<Schema>();
        auto geo_fid = schema->AddDebugField("geo", DataType::GEOMETRY);
        auto make_wkb = [](const char* wkt) {
            return Geometry(GetThreadLocalGEOSContext(), wkt).to_wkb_string();
        };
        const std::vector<std::string> wkbs = {
            make_wkb("POINT(5 5)"),
            make_wkb("POINT(205 205)"),
        };
        auto geo_field =
            storage::CreateFieldData(DataType::GEOMETRY, DataType::NONE, false);
        geo_field->FillFieldData(wkbs.data(), wkbs.size());

        RawFieldFiles files;
        auto seg = CreateSealedSegment(schema);
        seg->LoadFieldData(files.Prepare(geo_fid, {geo_field}));
        InstallIndexWithId(*seg,
                           geo_fid,
                           DataType::GEOMETRY,
                           96200,
                           BuildIndex(geo_fid,
                                      DataType::GEOMETRY,
                                      milvus::index::RTREE_INDEX_TYPE,
                                      {geo_field}));

        auto gis1 = std::make_shared<expr::GISFunctionFilterExpr>(
            expr::ColumnInfo(geo_fid, DataType::GEOMETRY),
            proto::plan::GISFunctionFilterExpr_GISOp_Intersects,
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))");
        gis1->SetExprId(96201);

        auto gis2 = std::make_shared<expr::GISFunctionFilterExpr>(
            expr::ColumnInfo(geo_fid, DataType::GEOMETRY),
            proto::plan::GISFunctionFilterExpr_GISOp_Intersects,
            "POLYGON((100 100, 110 100, 110 110, 100 110, 100 100))");
        gis2->SetExprId(96202);

        auto conj = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, gis1, gis2);

        const int64_t n = static_cast<int64_t>(wkbs.size());
        auto explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(n);
        auto res = EvaluateFilter(conj, seg.get(), n, explain);
        ExpectHitRows(res, {}, "disjoint R-Tree coarse GIS split-fusion");

        EXPECT_GT(explain->EnsureLeafSlot(96201)->eval_ns.load(), 0);
        EXPECT_GT(explain->EnsureLeafSlot(96202)->eval_ns.load(), 0);

        proto::internal::ExecutionExplain pb;
        ASSERT_TRUE(pb.ParseFromString(explain->Finish()));
        ASSERT_EQ(ConditionCount(pb), 2);
        std::map<int64_t, const proto::internal::ExplainConditionStats*> by_id;
        for (int i = 0; i < ConditionCount(pb); ++i) {
            const auto& c = ConditionAt(pb, i);
            by_id[c.expr_id()] = &c;
        }
        ASSERT_EQ(by_id.count(96201), 1u);
        ASSERT_EQ(by_id.count(96202), 1u);
        EXPECT_EQ(by_id[96201]->skipped_segments(), 0);
        EXPECT_TRUE(by_id[96201]->has_eval_time());
        EXPECT_EQ(by_id[96201]->rows_in(), n);
        EXPECT_EQ(by_id[96201]->rows_out(), 1);
        EXPECT_EQ(by_id[96201]->idx_scan(), n);
        EXPECT_EQ(by_id[96201]->raw_scan(), 0);

        EXPECT_EQ(by_id[96202]->skipped_segments(), 0);
        EXPECT_TRUE(by_id[96202]->has_eval_time());
        EXPECT_EQ(by_id[96202]->rows_in(), 1);
        EXPECT_EQ(by_id[96202]->rows_out(), 0);
        EXPECT_EQ(by_id[96202]->idx_scan(), 1);
        EXPECT_EQ(by_id[96202]->raw_scan(), 0);

        // Unindexed segment where Refine short-circuits on gis1 (0 hits): gis2
        // ran neither R-Tree nor Refine, so it remains skipped without eval_time.
        const std::vector<std::string> raw_wkbs = {
            make_wkb("POINT(205 205)"),
        };
        auto raw_geo_field =
            storage::CreateFieldData(DataType::GEOMETRY, DataType::NONE, false);
        raw_geo_field->FillFieldData(raw_wkbs.data(), raw_wkbs.size());
        RawFieldFiles raw_files;
        auto raw_seg = CreateSealedSegment(schema);
        raw_seg->LoadFieldData(raw_files.Prepare(geo_fid, {raw_geo_field}));

        auto raw_explain =
            std::make_shared<milvus::exec::ExecutionExplainRecorder>(1);
        auto raw_res = EvaluateFilter(conj, raw_seg.get(), 1, raw_explain);
        ExpectHitRows(raw_res, {}, "unindexed GIS Refine short-circuit");
        EXPECT_GT(raw_explain->EnsureLeafSlot(96201)->eval_ns.load(), 0);
        EXPECT_EQ(raw_explain->EnsureLeafSlot(96202)->eval_ns.load(), 0);

        proto::internal::ExecutionExplain raw_pb;
        ASSERT_TRUE(raw_pb.ParseFromString(raw_explain->Finish()));
        ASSERT_EQ(ConditionCount(raw_pb), 2);
        std::map<int64_t, const proto::internal::ExplainConditionStats*>
            raw_by_id;
        for (int i = 0; i < ConditionCount(raw_pb); ++i) {
            const auto& c = ConditionAt(raw_pb, i);
            raw_by_id[c.expr_id()] = &c;
        }
        EXPECT_EQ(raw_by_id[96201]->skipped_segments(), 0);
        EXPECT_TRUE(raw_by_id[96201]->has_eval_time());
        EXPECT_EQ(raw_by_id[96202]->skipped_segments(), 1);
        EXPECT_FALSE(raw_by_id[96202]->has_eval_time());
    }
}
