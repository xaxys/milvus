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

// One field, several scalar indexes, end to end with real index objects:
// a sealed segment carries the raw column plus two real indexes of different
// types, both are loaded through LoadIndex(), and every filter is compared
// against the same filter evaluated over the raw column only.
//
// What this pins:
//   * a second scalar index on a field that already has one loads instead of
//     tripping the old AssertInfo(!has_index),
//   * both stay visible in GetScalarIndexCandidates() (size == 2),
//   * the leaf picks a candidate that can serve it -- Equal with INVERTED+NGRAM,
//     ranges and IN with STL_SORT+INVERTED -- and the rows it returns are the
//     rows the raw-data evaluation returns.

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ExprBatchTestUtils.h"
#include "ExprTestBase.h"
#include "index/InvertedIndexTantivy.h"
#include "index/NgramInvertedIndex.h"
#include "index/ScalarIndexSort.h"
#include "segcore/SegmentSealed.h"

namespace {

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

// The segment's raw column. segment_id only keys the binlog file, so every
// segment in this file gets its own.
segcore::SegmentSealedUPtr
CreateSegmentWithColumn(const SchemaPtr& schema,
                        FieldId field_id,
                        const FieldDataPtr& field_data,
                        int64_t segment_id) {
    auto segment = CreateSealedSegment(schema);
    auto cm = milvus::storage::RemoteChunkManagerSingleton::GetInstance()
                  .GetRemoteChunkManager();
    auto load_info = PrepareSingleFieldInsertBinlog(kCollectionID,
                                                    kPartitionID,
                                                    segment_id,
                                                    field_id.get(),
                                                    {field_data},
                                                    cm);
    segment->LoadFieldData(load_info);
    return segment;
}

void
LoadScalarIndexOnField(segcore::SegmentSealed* segment,
                       FieldId field_id,
                       DataType field_type,
                       int64_t index_id,
                       const std::string& index_type,
                       index::IndexBasePtr scalar_index,
                       const std::string& cache_key) {
    segcore::LoadIndexInfo load_index_info{};
    load_index_info.field_id = field_id.get();
    load_index_info.field_type = field_type;
    load_index_info.index_id = index_id;
    load_index_info.index_params = {{index::INDEX_TYPE, index_type}};
    load_index_info.cache_index =
        CreateTestCacheIndex(cache_key, std::move(scalar_index));
    segment->LoadIndex(load_index_info);
}

// The real INVERTED (tantivy) index over the VARCHAR column, built in memory
// with the same entry point the inverted-index unit tests use.
index::IndexBasePtr
MakeInvertedVarCharIndex() {
    auto built = std::make_unique<index::InvertedIndexTantivy<std::string>>();
    built->BuildWithRawDataForUT(
        kVarCharValues.size(), kVarCharValues.data(), Config());
    return built;
}

// The real NGRAM (tantivy ngram writer) index over the same column, so the LIKE
// below is served by ngram tokens, not by an index that happens to hold values.
index::IndexBasePtr
MakeNgramIndex(FieldId field_id, const FieldDataPtr& field_data) {
    auto ctx = storage::FileManagerContext();
    ctx.fieldDataMeta.field_schema.set_data_type(
        milvus::proto::schema::VarChar);
    ctx.fieldDataMeta.field_schema.set_fieldid(field_id.get());
    ctx.fieldDataMeta.field_id = field_id.get();
    auto built = std::make_unique<index::NgramInvertedIndex>(
        ctx,
        index::NgramParams{/*loading_index=*/false,
                           /*min_gram=*/2,
                           /*max_gram=*/4});
    built->BuildWithFieldData({field_data});
    built->finish();
    built->create_reader(milvus::index::SetBitsetSealed);
    return built;
}

// The real INVERTED (tantivy) index over the INT64 column.
index::IndexBasePtr
MakeInvertedInt64Index() {
    auto built = std::make_unique<index::InvertedIndexTantivy<int64_t>>();
    built->BuildWithRawDataForUT(
        kInt64Values.size(), kInt64Values.data(), Config());
    return built;
}

// The real STL_SORT index over the INT64 column.
index::IndexBasePtr
MakeStlSortIndex() {
    auto built = std::make_unique<index::ScalarIndexSort<int64_t>>();
    built->Build(kInt64Values.size(), kInt64Values.data(), nullptr);
    return built;
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
    auto varchar_field_data =
        storage::CreateFieldData(DataType::VARCHAR, DataType::NONE, false);
    varchar_field_data->FillFieldData(kVarCharValues.data(),
                                      kVarCharValues.size());

    // Oracle: the same column with no index at all.
    auto raw_segment =
        CreateSegmentWithColumn(schema, varchar_fid, varchar_field_data, 3101);
    // Feature under test: one field, two real scalar indexes, both loaded.
    auto indexed_segment =
        CreateSegmentWithColumn(schema, varchar_fid, varchar_field_data, 3102);
    LoadScalarIndexOnField(indexed_segment.get(),
                           varchar_fid,
                           DataType::VARCHAR,
                           kInvertedIndexId,
                           index::INVERTED_INDEX_TYPE,
                           MakeInvertedVarCharIndex(),
                           "multi-index-varchar-inverted");
    LoadScalarIndexOnField(indexed_segment.get(),
                           varchar_fid,
                           DataType::VARCHAR,
                           kNgramIndexId,
                           index::NGRAM_INDEX_TYPE,
                           MakeNgramIndex(varchar_fid, varchar_field_data),
                           "multi-index-varchar-ngram");

    // Silently dropping either load has to fail right here.
    auto candidates = indexed_segment->GetScalarIndexCandidates(varchar_fid);
    ASSERT_EQ(candidates.size(), 2u)
        << "INVERTED and NGRAM must both stay loaded on one field";
    EXPECT_EQ(candidates[0].index_id, kInvertedIndexId);
    EXPECT_FALSE(candidates[0].is_ngram);
    EXPECT_EQ(candidates[0].capability,
              segcore::ScalarIndexCapability::TermIndexed);
    EXPECT_EQ(candidates[1].index_id, kNgramIndexId);
    EXPECT_TRUE(candidates[1].is_ngram);
    EXPECT_EQ(candidates[1].capability,
              segcore::ScalarIndexCapability::PatternIndexed);

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

    // The LIKE leaf answers through the NGRAM channel, and that channel really
    // hands it this field's NGRAM index -- not the INVERTED sibling.
    milvus::OpContext op_ctx;
    auto pinned_ngram = indexed_segment->GetNgramIndex(&op_ctx, varchar_fid);
    ASSERT_NE(pinned_ngram.get(), nullptr);
    EXPECT_TRUE(pinned_ngram.get()->CanHandleLiteral(
        "prefix", proto::plan::OpType::PrefixMatch));
}

// Case 2: STL_SORT + INVERTED on one INT64 field. Open-ended ranges and `in`
// must return the raw-data rows. GetScalarIndexCandidates() sorts by indexID,
// so both indexID orders are run: whichever of the two the leaf lands on has to
// be able to serve the operator (neither may be handed a lookup shape it cannot
// answer instead of falling back to the raw column).
TEST(MultiScalarIndexExprTest,
     Int64StlSortAndInvertedOnOneFieldBothServeQueries) {
    auto schema = std::make_shared<Schema>();
    auto int64_fid = schema->AddDebugField("int64_field", DataType::INT64);
    auto int64_field_data =
        storage::CreateFieldData(DataType::INT64, DataType::NONE, false);
    int64_field_data->FillFieldData(kInt64Values.data(), kInt64Values.size());

    // Oracle: the same column with no index at all.
    auto raw_segment =
        CreateSegmentWithColumn(schema, int64_fid, int64_field_data, 3201);

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
        int64_t segment_id;  // keys the binlog file, so it must be unique
        int64_t stl_sort_id;
        int64_t inverted_id;
    };
    const std::vector<IndexIdOrder> orderings = {
        {3202, 3301, 3302},  // STL_SORT is the first candidate
        {3203, 3312, 3311},  // INVERTED is the first candidate
    };

    for (const auto& ordering : orderings) {
        auto indexed_segment = CreateSegmentWithColumn(
            schema, int64_fid, int64_field_data, ordering.segment_id);
        LoadScalarIndexOnField(
            indexed_segment.get(),
            int64_fid,
            DataType::INT64,
            ordering.stl_sort_id,
            index::ASCENDING_SORT,
            MakeStlSortIndex(),
            "multi-index-int64-stl-" + std::to_string(ordering.stl_sort_id));
        LoadScalarIndexOnField(indexed_segment.get(),
                               int64_fid,
                               DataType::INT64,
                               ordering.inverted_id,
                               index::INVERTED_INDEX_TYPE,
                               MakeInvertedInt64Index(),
                               "multi-index-int64-inverted-" +
                                   std::to_string(ordering.inverted_id));

        // Silently dropping either load has to fail right here.
        auto candidates = indexed_segment->GetScalarIndexCandidates(int64_fid);
        ASSERT_EQ(candidates.size(), 2u)
            << "STL_SORT and INVERTED must both stay loaded, segment "
            << ordering.segment_id;
        EXPECT_EQ(candidates[0].index_id,
                  std::min(ordering.stl_sort_id, ordering.inverted_id));
        EXPECT_EQ(candidates[1].index_id,
                  std::max(ordering.stl_sort_id, ordering.inverted_id));

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
