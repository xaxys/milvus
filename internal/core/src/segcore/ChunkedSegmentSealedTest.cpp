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

#include <fmt/core.h>
#include <folly/FBVector.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "NamedType/named_type_impl.hpp"
#include "NamedType/underlying_functionalities.hpp"
#include "bitset/bitset.h"
#include "bitset/detail/element_vectorized.h"
#include "cachinglayer/CacheSlot.h"
#include "cachinglayer/Manager.h"
#include "cachinglayer/Translator.h"
#include "common/BitsetView.h"
#include "common/Chunk.h"
#include "common/Consts.h"
#include "common/FieldData.h"
#include "common/FieldDataInterface.h"
#include "common/FieldMeta.h"
#include "common/IndexMeta.h"
#include "common/OffsetMapping.h"
#include "common/OpContext.h"
#include "common/QueryInfo.h"
#include "common/QueryResult.h"
#include "common/Schema.h"
#include "common/Span.h"
#include "common/Types.h"
#include "common/protobuf_utils.h"
#include "expr/ITypeExpr.h"
#include "exec/QueryContext.h"
#include "exec/expression/Expr.h"
#include "filemanager/InputStream.h"
#include "gtest/gtest.h"
#include "exec/expression/ExprBatchTestUtils.h"
#include "exec/expression/IndexPathSelection.h"
#include "segcore/test_utils/ConsumerIndexTestUtils.h"
#include "index/Meta.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "mmap/ChunkedColumn.h"
#include "pb/plan.pb.h"
#include "pb/schema.pb.h"
#include "plan/PlanNode.h"
#include "query/ExecPlanNodeVisitor.h"
#include "query/PlanImpl.h"
#include "query/SearchOnSealed.h"
#include "segcore/ChunkedSegmentSealedImpl.h"
#include "segcore/SegcoreConfig.h"
#include "segcore/SegmentSealed.h"
#include "segcore/Types.h"
#include "segcore/storagev1translator/ChunkTranslator.h"
#include "storage/FileManager.h"
#include "storage/RemoteChunkManagerSingleton.h"
#include "storage/Types.h"
#include "test_utils/DataGen.h"
#include "test_utils/cachinglayer_test_utils.h"
#include "test_utils/storage_test_utils.h"

struct DeferRelease {
    using functype = std::function<void()>;
    void
    AddDefer(const functype& closure) {
        closures.push_back(closure);
    }

    ~DeferRelease() {
        for (auto& closure : closures) {
            closure();
        }
    }

    std::vector<functype> closures;
};

using namespace milvus;

namespace {

class ScopedRejectRemoteVectorOutput {
 public:
    explicit ScopedRejectRemoteVectorOutput(bool enabled)
        : old_value_(segcore::SegcoreConfig::default_config()
                         .get_reject_remote_vector_output()) {
        segcore::SegcoreConfig::default_config()
            .set_reject_remote_vector_output(enabled);
    }

    ~ScopedRejectRemoteVectorOutput() {
        segcore::SegcoreConfig::default_config()
            .set_reject_remote_vector_output(old_value_);
    }

 private:
    bool old_value_;
};

SearchResult
MakeSearchResult(std::vector<int64_t> offsets) {
    SearchResult result;
    result.seg_offsets_ = std::move(offsets);
    result.distances_.resize(result.seg_offsets_.size(), 0.0F);
    return result;
}

struct NullableVectorArrayRow {
    bool valid = true;
    std::vector<float> values;
};

std::vector<char>
BuildNullableVectorArrayChunkBuffer(
    int64_t dim, const std::vector<NullableVectorArrayRow>& rows) {
    const auto row_count = static_cast<int64_t>(rows.size());
    const auto bitmap_bytes = (row_count + 7) / 8;
    const auto header_bytes = sizeof(uint32_t) * (row_count * 2 + 1);
    const auto payload_offset =
        static_cast<uint32_t>(bitmap_bytes + header_bytes);
    const auto vector_bytes = static_cast<uint32_t>(dim * sizeof(float));

    size_t payload_values = 0;
    for (const auto& row : rows) {
        AssertInfo(row.valid || row.values.empty(),
                   "null vector-array row must not carry payload");
        AssertInfo(row.values.size() % dim == 0,
                   "vector-array row payload is not aligned to dim");
        payload_values += row.values.size();
    }

    std::vector<char> buffer(bitmap_bytes + header_bytes +
                                 payload_values * sizeof(float) +
                                 MMAP_ARRAY_PADDING,
                             0);
    std::vector<uint32_t> header(static_cast<size_t>(row_count * 2 + 1));

    auto payload_cursor = payload_offset;
    auto* payload = buffer.data() + payload_offset;
    for (int64_t i = 0; i < row_count; ++i) {
        const auto& row = rows[i];
        if (row.valid) {
            buffer[i >> 3] |= 1U << (i & 0x07);
        }
        header[i * 2] = payload_cursor;
        header[i * 2 + 1] = static_cast<uint32_t>(row.values.size() / dim);
        if (!row.values.empty()) {
            const auto payload_bytes = row.values.size() * sizeof(float);
            memcpy(payload, row.values.data(), payload_bytes);
            payload += payload_bytes;
            payload_cursor +=
                static_cast<uint32_t>(row.values.size() / dim) * vector_bytes;
        }
    }
    header[row_count * 2] = payload_cursor;
    memcpy(buffer.data() + bitmap_bytes,
           header.data(),
           header.size() * sizeof(uint32_t));
    return buffer;
}

std::shared_ptr<ChunkedColumnInterface>
BuildNullableVectorArrayColumn(
    const FieldMeta& field_meta,
    int64_t dim,
    const std::vector<std::vector<NullableVectorArrayRow>>& chunk_rows,
    std::vector<std::vector<char>>& buffers) {
    std::vector<std::unique_ptr<Chunk>> chunks;
    std::vector<int64_t> rows_per_chunk;
    chunks.reserve(chunk_rows.size());
    rows_per_chunk.reserve(chunk_rows.size());
    buffers.reserve(chunk_rows.size());

    for (const auto& rows : chunk_rows) {
        buffers.emplace_back(BuildNullableVectorArrayChunkBuffer(dim, rows));
        rows_per_chunk.push_back(static_cast<int64_t>(rows.size()));
        auto chunk_mmap_guard =
            std::make_shared<ChunkMmapGuard>(nullptr, 0, "");
        chunks.emplace_back(std::make_unique<VectorArrayChunk>(
            dim,
            static_cast<int32_t>(rows.size()),
            buffers.back().data(),
            buffers.back().size(),
            field_meta.get_element_type(),
            chunk_mmap_guard,
            true));
    }

    auto translator = std::make_unique<TestChunkTranslator>(
        rows_per_chunk, "", std::move(chunks));
    auto slot =
        cachinglayer::Manager::GetInstance().CreateCacheSlot<milvus::Chunk>(
            std::move(translator), nullptr);
    return MakeChunkedColumnBase(
        DataType::VECTOR_ARRAY, std::move(slot), field_meta);
}

segcore::SegmentSealedUPtr
CreateColdVectorOutputSegment(const SchemaPtr& schema,
                              const FieldId& vector_field_id,
                              int64_t row_count) {
    auto dataset = segcore::DataGen(schema, row_count);
    auto cm = milvus::storage::RemoteChunkManagerSingleton::GetInstance()
                  .GetRemoteChunkManager();
    auto load_info = PrepareInsertBinlog(
        kCollectionID, kPartitionID, kSegmentID, dataset, cm);
    load_info.field_infos.at(vector_field_id.get()).warmup_policy = "disable";

    auto segment = segcore::CreateSealedSegment(schema);
    segment->LoadFieldData(load_info);
    return segment;
}

}  // namespace

TEST(test_chunk_segment, TestSearchOnSealed) {
    int dim = 16;
    int chunk_num = 3;
    int chunk_size = 100;

    DeferRelease defer;

    int total_row_count = chunk_num * chunk_size;
    int bitset_size = (total_row_count + 7) / 8;

    auto schema = std::make_shared<Schema>();
    auto fakevec_id = schema->AddDebugField(
        "fakevec", DataType::VECTOR_FLOAT, dim, knowhere::metric::COSINE);

    auto field_meta = schema->operator[](fakevec_id);

    std::vector<std::unique_ptr<Chunk>> chunks;
    std::vector<int64_t> num_rows_per_chunk;
    for (int i = 0; i < chunk_num; i++) {
        num_rows_per_chunk.push_back(chunk_size);
        auto dataset = segcore::DataGen(schema, chunk_size);
        auto data = dataset.get_col<float>(fakevec_id);
        auto buf_size = 4 * data.size();

        char* buf = new char[buf_size];
        defer.AddDefer([buf]() { delete[] buf; });
        memcpy(buf, data.data(), 4 * data.size());

        auto chunk_mmap_guard =
            std::make_shared<ChunkMmapGuard>(nullptr, 0, "");
        chunks.emplace_back(std::make_unique<FixedWidthChunk>(
            chunk_size, dim, buf, buf_size, 4, false, chunk_mmap_guard));
    }

    auto translator = std::make_unique<TestChunkTranslator>(
        num_rows_per_chunk, "", std::move(chunks));
    auto slot =
        cachinglayer::Manager::GetInstance().CreateCacheSlot<milvus::Chunk>(
            std::move(translator), nullptr);
    auto column = std::make_shared<ChunkedColumn>(std::move(slot), field_meta);

    SearchInfo search_info;
    auto search_conf = knowhere::Json{
        {knowhere::meta::METRIC_TYPE, knowhere::metric::COSINE},
    };
    search_info.search_params_ = search_conf;
    search_info.field_id_ = fakevec_id;
    search_info.metric_type_ = knowhere::metric::COSINE;
    // expect to return all rows
    search_info.topk_ = total_row_count;

    uint8_t* bitset_data = new uint8_t[bitset_size];
    defer.AddDefer([bitset_data]() { delete[] bitset_data; });
    std::fill(bitset_data, bitset_data + bitset_size, 0);
    BitsetView bv(bitset_data, total_row_count);

    auto query_ds = segcore::DataGen(schema, 1);
    auto col_query_data = query_ds.get_col<float>(fakevec_id);
    auto query_data = col_query_data.data();
    auto index_info = std::map<std::string, std::string>{};
    SearchResult search_result;
    milvus::OpContext op_context;
    query::SearchOnSealedColumn(*schema,
                                column.get(),
                                search_info,
                                index_info,
                                query_data,
                                nullptr,
                                1,
                                total_row_count,
                                bv,
                                &op_context,
                                search_result);

    std::set<int64_t> offsets;
    for (auto& offset : search_result.seg_offsets_) {
        if (offset != -1) {
            offsets.insert(offset);
        }
    }
    // check all rows are returned
    ASSERT_EQ(total_row_count, offsets.size());
    for (int i = 0; i < total_row_count; i++) {
        ASSERT_TRUE(offsets.find(i) != offsets.end());
    }

    // test with group by
    search_info.group_by_field_ids_ = {fakevec_id};
    std::fill(bitset_data, bitset_data + bitset_size, 0);
    query::SearchOnSealedColumn(*schema,
                                column.get(),
                                search_info,
                                index_info,
                                query_data,
                                nullptr,
                                1,
                                total_row_count,
                                bv,
                                &op_context,
                                search_result);

    ASSERT_EQ(1, search_result.vector_iterators_->size());

    auto iter = search_result.vector_iterators_->at(0);
    // collect all offsets
    offsets.clear();
    while (iter->HasNext()) {
        auto [offset, distance] = iter->Next().value();
        offsets.insert(offset);
    }

    ASSERT_EQ(total_row_count, offsets.size());
    for (int i = 0; i < total_row_count; i++) {
        ASSERT_TRUE(offsets.find(i) != offsets.end());
    }
}

TEST(test_chunk_segment, RejectRemoteVectorOutputFailsWhenVectorCellsAreCold) {
    constexpr int64_t row_count = 16;
    constexpr int64_t dim = 4;
    auto schema = std::make_shared<Schema>();
    auto pk_id = schema->AddDebugField("pk", DataType::INT64);
    auto vector_id = schema->AddDebugField(
        "vec", DataType::VECTOR_FLOAT, dim, knowhere::metric::L2);
    schema->set_primary_field_id(pk_id);

    auto segment = CreateColdVectorOutputSegment(schema, vector_id, row_count);
    auto* chunked =
        dynamic_cast<segcore::ChunkedSegmentSealedImpl*>(segment.get());
    ASSERT_NE(chunked, nullptr);

    query::Plan plan(schema);
    plan.target_entries_ = {vector_id};
    auto result = MakeSearchResult({0, 3, 7});

    ScopedRejectRemoteVectorOutput scoped_config(true);

    try {
        chunked->TestFillTargetEntry(&plan, result);
        FAIL() << "expected cold vector output to be rejected";
    } catch (const SegcoreError& err) {
        EXPECT_EQ(err.get_error_code(), RetrieveError);
        EXPECT_NE(std::string(err.what()).find("vector field"),
                  std::string::npos);
    }
}

TEST(test_chunk_segment,
     RejectRemoteVectorOutputFailsForRetrieveWhenVectorCellsAreCold) {
    constexpr int64_t row_count = 16;
    constexpr int64_t dim = 4;
    auto schema = std::make_shared<Schema>();
    auto pk_id = schema->AddDebugField("pk", DataType::INT64);
    auto vector_id = schema->AddDebugField(
        "vec", DataType::VECTOR_FLOAT, dim, knowhere::metric::L2);
    schema->set_primary_field_id(pk_id);

    auto segment = CreateColdVectorOutputSegment(schema, vector_id, row_count);
    auto* chunked =
        dynamic_cast<segcore::ChunkedSegmentSealedImpl*>(segment.get());
    ASSERT_NE(chunked, nullptr);

    query::RetrievePlan plan(schema);
    plan.field_ids_ = {vector_id};
    const int64_t offsets[] = {0, 3, 7};

    ScopedRejectRemoteVectorOutput scoped_config(true);

    try {
        auto results = chunked->Retrieve(
            nullptr,
            &plan,
            offsets,
            static_cast<int64_t>(sizeof(offsets) / sizeof(offsets[0])),
            folly::CancellationToken());
        (void)results;
        FAIL() << "expected cold vector output to be rejected";
    } catch (const SegcoreError& err) {
        EXPECT_EQ(err.get_error_code(), RetrieveError);
        EXPECT_NE(std::string(err.what()).find("vector field"),
                  std::string::npos);
    }
}

TEST(test_chunk_segment,
     RejectRemoteVectorOutputAllowsScalarAndDisabledConfig) {
    constexpr int64_t row_count = 16;
    constexpr int64_t dim = 4;
    auto schema = std::make_shared<Schema>();
    auto pk_id = schema->AddDebugField("pk", DataType::INT64);
    auto scalar_id = schema->AddDebugField("scalar", DataType::INT64);
    auto vector_id = schema->AddDebugField(
        "vec", DataType::VECTOR_FLOAT, dim, knowhere::metric::L2);
    schema->set_primary_field_id(pk_id);

    {
        auto segment =
            CreateColdVectorOutputSegment(schema, vector_id, row_count);
        auto* chunked =
            dynamic_cast<segcore::ChunkedSegmentSealedImpl*>(segment.get());
        ASSERT_NE(chunked, nullptr);

        query::Plan scalar_plan(schema);
        scalar_plan.target_entries_ = {scalar_id};
        auto scalar_result = MakeSearchResult({1, 2, 5});

        ScopedRejectRemoteVectorOutput scoped_config(true);
        ASSERT_NO_THROW(
            chunked->TestFillTargetEntry(&scalar_plan, scalar_result));
        ASSERT_EQ(scalar_result.output_fields_data_.count(scalar_id), 1);
    }

    {
        auto segment =
            CreateColdVectorOutputSegment(schema, vector_id, row_count);
        auto* chunked =
            dynamic_cast<segcore::ChunkedSegmentSealedImpl*>(segment.get());
        ASSERT_NE(chunked, nullptr);

        query::Plan vector_plan(schema);
        vector_plan.target_entries_ = {vector_id};
        auto vector_result = MakeSearchResult({1, 2, 5});

        ScopedRejectRemoteVectorOutput scoped_config(false);
        ASSERT_NO_THROW(
            chunked->TestFillTargetEntry(&vector_plan, vector_result));
        ASSERT_EQ(vector_result.output_fields_data_.count(vector_id), 1);
    }
}

// The result-fill phase must produce identical output whether it reads through
// the request-pinned snapshot (SearchResult::read_snapshot_) or the per-call
// segment fallback. This covers FillTargetEntry -> TryTakeForSearch /
// bulk_subscript_from_state reusing the pinned published state instead of
// re-capturing it per output field.
TEST(test_chunk_segment, PinnedFillTargetEntryMatchesUnpinned) {
    constexpr int64_t row_count = 16;
    constexpr int64_t dim = 4;
    auto schema = std::make_shared<Schema>();
    auto pk_id = schema->AddDebugField("pk", DataType::INT64);
    auto scalar_id = schema->AddDebugField("scalar", DataType::INT64);
    auto vector_id = schema->AddDebugField(
        "vec", DataType::VECTOR_FLOAT, dim, knowhere::metric::L2);
    schema->set_primary_field_id(pk_id);

    auto segment = CreateColdVectorOutputSegment(schema, vector_id, row_count);
    auto* chunked =
        dynamic_cast<segcore::ChunkedSegmentSealedImpl*>(segment.get());
    ASSERT_NE(chunked, nullptr);

    auto snapshot = chunked->CaptureReadSnapshot();
    ASSERT_NE(snapshot, nullptr);
    // The type-erased bridge must rebind the sealed snapshot to the concrete
    // published state, and reject null / non-sealed inputs.
    auto state = segcore::ChunkedSegmentSealedImpl::ToPublishedState(snapshot);
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(segcore::ChunkedSegmentSealedImpl::ToPublishedState(nullptr),
              nullptr);

    query::Plan plan(schema);
    plan.target_entries_ = {pk_id, scalar_id, vector_id};

    auto pinned = MakeSearchResult({0, 3, 7, 11});
    pinned.read_snapshot_ = snapshot;
    auto unpinned = MakeSearchResult({0, 3, 7, 11});

    // Vector field is cold in CreateColdVectorOutputSegment; disable the
    // reject-remote-vector-output gate so both paths exercise the same fill.
    ScopedRejectRemoteVectorOutput scoped_config(false);
    ASSERT_NO_THROW(chunked->TestFillTargetEntry(&plan, pinned));
    ASSERT_NO_THROW(chunked->TestFillTargetEntry(&plan, unpinned));

    for (auto field_id : plan.target_entries_) {
        ASSERT_EQ(pinned.output_fields_data_.count(field_id), 1);
        ASSERT_EQ(unpinned.output_fields_data_.count(field_id), 1);
        EXPECT_EQ(
            pinned.output_fields_data_.at(field_id)->SerializeAsString(),
            unpinned.output_fields_data_.at(field_id)->SerializeAsString())
            << "field " << field_id.get();
    }
    EXPECT_EQ(pinned.search_storage_cost_.scanned_total_bytes,
              unpinned.search_storage_cost_.scanned_total_bytes);
}

TEST(test_chunk_segment, ReopenSkipsFunctionOutputFieldWithoutData) {
    auto old_schema = std::make_shared<Schema>();
    old_schema->set_schema_version(1);
    auto pk = old_schema->AddDebugField("pk", DataType::INT64);
    old_schema->set_primary_field_id(pk);

    auto segment = segcore::CreateSealedSegment(old_schema);

    constexpr int64_t row_count = 5;
    std::vector<int64_t> pk_values(row_count);
    std::iota(pk_values.begin(), pk_values.end(), 0);

    auto field_data =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    field_data->FillFieldData(pk_values.data(), row_count);

    auto cm = milvus::storage::RemoteChunkManagerSingleton::GetInstance()
                  .GetRemoteChunkManager();
    auto load_info = PrepareSingleFieldInsertBinlog(
        kCollectionID, kPartitionID, kSegmentID, pk.get(), {field_data}, cm);
    segment->LoadFieldData(load_info);

    auto new_schema = std::make_shared<Schema>();
    new_schema->set_schema_version(2);
    new_schema->AddField(
        FieldName("pk"), pk, DataType::INT64, false, std::nullopt);
    new_schema->set_primary_field_id(pk);
    auto sparse = FieldId(pk.get() + 1);
    new_schema->AddField(FieldName("sparse_out"),
                         sparse,
                         DataType::VECTOR_SPARSE_U32_F32,
                         0,
                         std::nullopt,
                         false);
    new_schema->add_function_output_field_id(sparse);

    segment->Reopen(new_schema);
    EXPECT_FALSE(segment->FieldAccessible(sparse));
}

// #50783 defense-in-depth: GetFieldIndexMeta must throw (AssertInfo) rather than
// dereference end() for a missing field; assert() is compiled out under NDEBUG.
TEST(test_chunk_segment, GetFieldIndexMetaThrowsOnMissingField) {
    std::map<FieldId, FieldIndexMeta> field_metas;
    FieldId present(100);
    field_metas.emplace(
        present,
        FieldIndexMeta(
            present, {{"index_type", "IVF_FLAT"}, {"metric_type", "L2"}}, {}));
    CollectionIndexMeta meta(1024, std::move(field_metas));

    EXPECT_TRUE(meta.HasField(present));
    EXPECT_NO_THROW(meta.GetFieldIndexMeta(present));
    EXPECT_FALSE(meta.HasField(FieldId(present.get() + 1)));
    EXPECT_THROW(meta.GetFieldIndexMeta(FieldId(present.get() + 1)),
                 SegcoreError);
}

TEST(test_chunk_segment, MissingStructElementOffsetsReturnsEmptyForOldRows) {
    auto old_schema = std::make_shared<Schema>();
    old_schema->set_schema_version(1);
    auto pk = old_schema->AddDebugField("pk", DataType::INT64);
    old_schema->set_primary_field_id(pk);

    auto segment = segcore::CreateSealedSegment(old_schema);

    constexpr int64_t row_count = 5;
    std::vector<int64_t> pk_values(row_count);
    std::iota(pk_values.begin(), pk_values.end(), 0);

    auto field_data =
        std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
    field_data->FillFieldData(pk_values.data(), row_count);

    auto cm = milvus::storage::RemoteChunkManagerSingleton::GetInstance()
                  .GetRemoteChunkManager();
    auto load_info = PrepareSingleFieldInsertBinlog(
        kCollectionID, kPartitionID, kSegmentID, pk.get(), {field_data}, cm);
    segment->LoadFieldData(load_info);

    auto new_schema = std::make_shared<Schema>();
    new_schema->set_schema_version(2);
    new_schema->AddField(
        FieldName("pk"), pk, DataType::INT64, false, std::nullopt);
    new_schema->set_primary_field_id(pk);
    auto label = FieldId(pk.get() + 1);
    new_schema->AddField(FieldName("chunks[label]"),
                         label,
                         DataType::ARRAY,
                         DataType::VARCHAR,
                         true,
                         false);
    segment->Reopen(new_schema);

    auto offsets = segment->GetStructElementOffsets(label);
    ASSERT_NE(offsets, nullptr);
    EXPECT_EQ(offsets->GetRowCount(), row_count);
    EXPECT_EQ(offsets->GetTotalElementCount(), 0);
    for (int64_t i = 0; i <= row_count; ++i) {
        auto [start, end] = offsets->ElementIDRangeOfRow(i);
        EXPECT_EQ(start, 0);
        EXPECT_EQ(end, 0);
    }
}

// #52877: lazy reopen may skip the drop-only schema and see the old and new
// StructArrays as one same-name transition with disjoint child field IDs.
TEST(test_chunk_segment,
     ReopenInvalidatesOffsetsForReaddedStructArrayWithSameName) {
    auto old_schema = std::make_shared<Schema>();
    old_schema->set_schema_version(1);
    auto pk = old_schema->AddDebugField("pk", DataType::INT64);
    old_schema->set_primary_field_id(pk);
    auto old_label = old_schema->AddDebugArrayField(
        "chunks[label]", DataType::VARCHAR, true);
    auto old_score =
        old_schema->AddDebugArrayField("chunks[score]", DataType::INT32, true);

    constexpr int64_t row_count = 5;
    constexpr int64_t array_len = 3;
    auto dataset = segcore::DataGen(old_schema,
                                    row_count,
                                    /*seed=*/42,
                                    /*ts_offset=*/0,
                                    /*repeat_count=*/1,
                                    array_len);
    auto segment = CreateSealedWithFieldDataLoaded(old_schema, dataset);

    auto old_offsets = segment->GetStructElementOffsets(old_label);
    ASSERT_NE(old_offsets, nullptr);
    ASSERT_EQ(old_offsets.get(),
              segment->GetStructElementOffsets(old_score).get());
    // DataGen marks alternating rows valid for nullable scalar fields, starting
    // with row 0. Sealed binlog serialization drops payloads from null rows, so
    // only the three valid rows contribute elements to the shared offsets.
    constexpr int64_t valid_row_count = (row_count + 1) / 2;
    ASSERT_EQ(old_offsets->GetTotalElementCount(), valid_row_count * array_len);

    auto new_schema = std::make_shared<Schema>();
    new_schema->set_schema_version(2);
    new_schema->AddField(
        FieldName("pk"), pk, DataType::INT64, false, std::nullopt);
    new_schema->set_primary_field_id(pk);
    auto new_label = FieldId(old_score.get() + 1);
    auto new_score = FieldId(new_label.get() + 1);
    new_schema->AddField(FieldName("chunks[label]"),
                         new_label,
                         DataType::ARRAY,
                         DataType::VARCHAR,
                         true,
                         false);
    new_schema->AddField(FieldName("chunks[score]"),
                         new_score,
                         DataType::ARRAY,
                         DataType::INT32,
                         true,
                         false);

    // Model lazy reopen skipping the intermediate drop-only schema: the struct
    // name is unchanged, while every child field ID belongs to a new generation.
    segment->Reopen(new_schema);

    auto new_offsets = segment->GetStructElementOffsets(new_label);
    ASSERT_NE(new_offsets, nullptr);
    EXPECT_NE(new_offsets.get(), old_offsets.get());
    EXPECT_EQ(new_offsets.get(),
              segment->GetStructElementOffsets(new_score).get());
    EXPECT_EQ(new_offsets->GetRowCount(), row_count);
    EXPECT_EQ(new_offsets->GetTotalElementCount(), 0);
    EXPECT_EQ(segment->GetStructElementOffsets(old_label), nullptr);
    EXPECT_EQ(segment->GetStructElementOffsets(old_score), nullptr);
}

TEST(test_chunk_segment, SearchOnSealedColumnBruteForceUsesOriginalTopk) {
    int dim = 16;
    int chunk_num = 2;
    int chunk_size = 64;

    DeferRelease defer;

    int total_row_count = chunk_num * chunk_size;
    int bitset_size = (total_row_count + 7) / 8;

    auto schema = std::make_shared<Schema>();
    auto fakevec_id = schema->AddDebugField(
        "fakevec", DataType::VECTOR_FLOAT, dim, knowhere::metric::L2);
    auto field_meta = schema->operator[](fakevec_id);

    std::vector<std::unique_ptr<Chunk>> chunks;
    std::vector<int64_t> num_rows_per_chunk;
    for (int i = 0; i < chunk_num; i++) {
        num_rows_per_chunk.push_back(chunk_size);
        auto dataset = segcore::DataGen(schema, chunk_size);
        auto data = dataset.get_col<float>(fakevec_id);
        auto buf_size = static_cast<int>(sizeof(float) * data.size());

        char* buf = new char[buf_size];
        defer.AddDefer([buf]() { delete[] buf; });
        memcpy(buf, data.data(), buf_size);

        auto chunk_mmap_guard =
            std::make_shared<ChunkMmapGuard>(nullptr, 0, "");
        chunks.emplace_back(
            std::make_unique<FixedWidthChunk>(chunk_size,
                                              dim,
                                              buf,
                                              buf_size,
                                              sizeof(float),
                                              false,
                                              chunk_mmap_guard));
    }

    auto translator = std::make_unique<TestChunkTranslator>(
        num_rows_per_chunk, "", std::move(chunks));
    auto slot =
        cachinglayer::Manager::GetInstance().CreateCacheSlot<milvus::Chunk>(
            std::move(translator), nullptr);
    auto column = std::make_shared<ChunkedColumn>(std::move(slot), field_meta);

    SearchInfo search_info;
    search_info.search_params_ = knowhere::Json{
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
    };
    search_info.field_id_ = fakevec_id;
    search_info.metric_type_ = knowhere::metric::L2;
    search_info.topk_ = 5;
    search_info.global_refine_enable_ = true;
    search_info.search_topk_ratio_ = 3.0;
    search_info.refine_topk_ratio_ = 2.0;

    uint8_t* bitset_data = new uint8_t[bitset_size];
    defer.AddDefer([bitset_data]() { delete[] bitset_data; });
    std::fill(bitset_data, bitset_data + bitset_size, 0);
    BitsetView bv(bitset_data, total_row_count);

    auto query_ds = segcore::DataGen(schema, 1);
    auto col_query_data = query_ds.get_col<float>(fakevec_id);
    auto query_data = col_query_data.data();
    auto index_info = std::map<std::string, std::string>{};
    SearchResult search_result;
    milvus::OpContext op_context;
    query::SearchOnSealedColumn(*schema,
                                column.get(),
                                search_info,
                                index_info,
                                query_data,
                                nullptr,
                                1,
                                total_row_count,
                                bv,
                                &op_context,
                                search_result);

    ASSERT_EQ(search_result.unity_topK_, search_info.topk_);
    ASSERT_EQ(search_result.seg_offsets_.size(), search_info.topk_);
    ASSERT_EQ(search_result.distances_.size(), search_info.topk_);
}

TEST(test_chunk_segment,
     SearchOnSealedColumnNullableVectorArrayUsesLogicalRowOffsets) {
    constexpr int64_t dim = 2;
    constexpr int64_t row_count = 6;

    auto schema = std::make_shared<Schema>();
    auto vec_id =
        schema->AddDebugVectorArrayField("profile[embedding]",
                                         DataType::VECTOR_FLOAT,
                                         dim,
                                         knowhere::metric::MAX_SIM_COSINE,
                                         true);
    auto field_meta = schema->operator[](vec_id);

    std::vector<std::vector<char>> buffers;
    auto column = BuildNullableVectorArrayColumn(field_meta,
                                                 dim,
                                                 {{{false, {}},
                                                   {true, {}},
                                                   {true, {1.0F, 0.0F}},
                                                   {true, {0.0F, 1.0F}},
                                                   {true, {}},
                                                   {true, {-1.0F, 0.0F}}}},
                                                 buffers);
    ASSERT_FALSE(column->GetOffsetMapping().IsEnabled());
    auto offsets_pw = column->VectorArrayOffsets(nullptr, 0);
    const std::array<size_t, row_count + 1> expected_offsets{
        0, 0, 0, 1, 2, 2, 3};
    for (int64_t i = 0; i <= row_count; ++i) {
        ASSERT_EQ(offsets_pw.get()[i], expected_offsets[i]);
    }

    SearchInfo search_info;
    search_info.search_params_ = knowhere::Json{
        {knowhere::meta::METRIC_TYPE, knowhere::metric::MAX_SIM_COSINE}};
    search_info.field_id_ = vec_id;
    search_info.metric_type_ = knowhere::metric::MAX_SIM_COSINE;
    search_info.topk_ = 3;

    const std::array<float, dim> query{1.0F, 0.0F};
    const std::array<size_t, 2> query_offsets{0, 1};
    const std::map<std::string, std::string> index_info;
    milvus::OpContext op_context;

    auto run_search = [&](const BitsetView& bitset) {
        SearchResult result;
        query::SearchOnSealedColumn(*schema,
                                    column.get(),
                                    search_info,
                                    index_info,
                                    query.data(),
                                    query_offsets.data(),
                                    1,
                                    row_count,
                                    bitset,
                                    &op_context,
                                    result);
        return result;
    };

    auto result = run_search(BitsetView{});
    ASSERT_EQ(result.seg_offsets_.size(), 3);
    ASSERT_EQ(result.distances_.size(), 3);
    EXPECT_EQ(result.seg_offsets_[0], 2);
    EXPECT_EQ(result.seg_offsets_[1], 3);
    EXPECT_EQ(result.seg_offsets_[2], 5);
    EXPECT_FLOAT_EQ(result.distances_[0], 1.0F);
    EXPECT_FLOAT_EQ(result.distances_[1], 0.0F);
    EXPECT_FLOAT_EQ(result.distances_[2], -1.0F);

    // Bitsets are also in logical row space for sealed raw VECTOR_ARRAY.
    std::array<uint8_t, 1> filtered_bits{1U << 3};
    auto filtered_result =
        run_search(BitsetView(filtered_bits.data(), row_count));
    ASSERT_EQ(filtered_result.seg_offsets_.size(), 3);
    EXPECT_EQ(filtered_result.seg_offsets_[0], 2);
    EXPECT_EQ(filtered_result.seg_offsets_[1], 5);
    EXPECT_EQ(filtered_result.seg_offsets_[2], INVALID_SEG_OFFSET);
    EXPECT_FLOAT_EQ(filtered_result.distances_[0], 1.0F);
    EXPECT_FLOAT_EQ(filtered_result.distances_[1], -1.0F);
    EXPECT_TRUE(column->GetOffsetMapping().IsEnabled());
    EXPECT_EQ(column->GetOffsetMapping().GetValidCount(), 5);
    const std::array<size_t, 6> expected_valid_offsets{0, 0, 1, 2, 2, 3};
    const auto& valid_offsets = column->GetValidArrayOffsetsInChunk(0);
    ASSERT_EQ(valid_offsets.size(), expected_valid_offsets.size());
    for (size_t i = 0; i < expected_valid_offsets.size(); ++i) {
        EXPECT_EQ(valid_offsets[i], expected_valid_offsets[i]);
    }
}

TEST(test_chunk_segment,
     SearchOnSealedColumnNullableVectorArrayUsesCachedChunkOffsets) {
    constexpr int64_t dim = 2;
    constexpr int64_t row_count = 7;

    auto schema = std::make_shared<Schema>();
    auto vec_id =
        schema->AddDebugVectorArrayField("profile[embedding]",
                                         DataType::VECTOR_FLOAT,
                                         dim,
                                         knowhere::metric::MAX_SIM_COSINE,
                                         true);
    auto field_meta = schema->operator[](vec_id);

    std::vector<std::vector<char>> buffers;
    auto column = BuildNullableVectorArrayColumn(
        field_meta,
        dim,
        {{{false, {}}, {true, {}}, {true, {1.0F, 0.0F}}},
         {{false, {}},
          {true, {0.0F, 1.0F}},
          {true, {}},
          {true, {-1.0F, 0.0F}}}},
        buffers);

    SearchInfo search_info;
    search_info.search_params_ = knowhere::Json{
        {knowhere::meta::METRIC_TYPE, knowhere::metric::MAX_SIM_COSINE}};
    search_info.field_id_ = vec_id;
    search_info.metric_type_ = knowhere::metric::MAX_SIM_COSINE;
    search_info.topk_ = 3;

    const std::array<float, dim> query{1.0F, 0.0F};
    const std::array<size_t, 2> query_offsets{0, 1};
    const std::map<std::string, std::string> index_info;
    milvus::OpContext op_context;

    auto run_search = [&](const BitsetView& bitset) {
        SearchResult result;
        query::SearchOnSealedColumn(*schema,
                                    column.get(),
                                    search_info,
                                    index_info,
                                    query.data(),
                                    query_offsets.data(),
                                    1,
                                    row_count,
                                    bitset,
                                    &op_context,
                                    result);
        return result;
    };

    auto result = run_search(BitsetView{});
    ASSERT_EQ(result.seg_offsets_.size(), 3);
    EXPECT_EQ(result.seg_offsets_[0], 2);
    EXPECT_EQ(result.seg_offsets_[1], 4);
    EXPECT_EQ(result.seg_offsets_[2], 6);

    std::array<uint8_t, 1> filtered_bits{1U << 4};
    auto filtered_result =
        run_search(BitsetView(filtered_bits.data(), row_count));
    ASSERT_EQ(filtered_result.seg_offsets_.size(), 3);
    EXPECT_EQ(filtered_result.seg_offsets_[0], 2);
    EXPECT_EQ(filtered_result.seg_offsets_[1], 6);
    EXPECT_EQ(filtered_result.seg_offsets_[2], INVALID_SEG_OFFSET);

    const std::array<size_t, 3> chunk0_offsets{0, 0, 1};
    const std::array<size_t, 4> chunk1_offsets{0, 1, 1, 2};
    const auto& valid_offsets0 = column->GetValidArrayOffsetsInChunk(0);
    const auto& valid_offsets1 = column->GetValidArrayOffsetsInChunk(1);
    ASSERT_EQ(valid_offsets0.size(), chunk0_offsets.size());
    ASSERT_EQ(valid_offsets1.size(), chunk1_offsets.size());
    for (size_t i = 0; i < chunk0_offsets.size(); ++i) {
        EXPECT_EQ(valid_offsets0[i], chunk0_offsets[i]);
    }
    for (size_t i = 0; i < chunk1_offsets.size(); ++i) {
        EXPECT_EQ(valid_offsets1[i], chunk1_offsets[i]);
    }
}

// Test search on nullable vector field with all null vectors
// This test verifies that SearchOnSealedColumn returns empty results
// instead of crashing when all vectors are null
TEST(test_chunk_segment, TestSearchOnSealedWithAllNullVectors) {
    int dim = 16;
    int chunk_num = 2;
    int chunk_size = 100;

    DeferRelease defer;

    int total_row_count = chunk_num * chunk_size;
    int bitset_size = (total_row_count + 7) / 8;

    auto schema = std::make_shared<Schema>();
    // Create nullable vector field
    auto fakevec_id = schema->AddDebugField(
        "fakevec", DataType::VECTOR_FLOAT, dim, knowhere::metric::COSINE, true);

    auto field_meta = schema->operator[](fakevec_id);
    ASSERT_TRUE(field_meta.is_nullable());

    std::vector<std::unique_ptr<Chunk>> chunks;
    std::vector<int64_t> num_rows_per_chunk;

    for (int i = 0; i < chunk_num; i++) {
        num_rows_per_chunk.push_back(chunk_size);

        // Calculate buffer size: null_bitmap + vector_data
        int null_bitmap_bytes = (chunk_size + 7) / 8;
        int vector_data_size = chunk_size * dim * sizeof(float);
        int buf_size = null_bitmap_bytes + vector_data_size;

        char* buf = new char[buf_size];
        defer.AddDefer([buf]() { delete[] buf; });

        // Set null bitmap to all zeros (all vectors are null)
        std::fill(buf, buf + null_bitmap_bytes, 0);

        // Fill vector data with zeros (doesn't matter since all are null)
        std::fill(buf + null_bitmap_bytes, buf + buf_size, 0);

        auto chunk_mmap_guard =
            std::make_shared<ChunkMmapGuard>(nullptr, 0, "");
        chunks.emplace_back(
            std::make_unique<FixedWidthChunk>(chunk_size,
                                              dim,
                                              buf,
                                              buf_size,
                                              sizeof(float),
                                              true,
                                              chunk_mmap_guard));
    }

    auto translator = std::make_unique<TestChunkTranslator>(
        num_rows_per_chunk, "", std::move(chunks));
    auto slot =
        cachinglayer::Manager::GetInstance().CreateCacheSlot<milvus::Chunk>(
            std::move(translator), nullptr);
    auto column = std::make_shared<ChunkedColumn>(std::move(slot), field_meta);

    // Build valid row ids to initialize offset_mapping for nullable vectors
    column->BuildValidRowIds(nullptr);

    // Verify offset_mapping is enabled and valid count is 0
    const auto& offset_mapping = column->GetOffsetMapping();
    ASSERT_TRUE(offset_mapping.IsEnabled());
    ASSERT_EQ(offset_mapping.GetValidCount(), 0);

    SearchInfo search_info;
    auto search_conf = knowhere::Json{
        {knowhere::meta::METRIC_TYPE, knowhere::metric::COSINE},
    };
    search_info.search_params_ = search_conf;
    search_info.field_id_ = fakevec_id;
    search_info.metric_type_ = knowhere::metric::COSINE;
    search_info.topk_ = 10;

    uint8_t* bitset_data = new uint8_t[bitset_size];
    defer.AddDefer([bitset_data]() { delete[] bitset_data; });
    std::fill(bitset_data, bitset_data + bitset_size, 0);
    BitsetView bv(bitset_data, total_row_count);

    // Generate query vector
    std::vector<float> query_data(dim);
    for (int i = 0; i < dim; i++) {
        query_data[i] = static_cast<float>(i) / dim;
    }
    auto index_info = std::map<std::string, std::string>{};
    SearchResult search_result;
    milvus::OpContext op_context;

    // This should NOT crash and should return empty results
    query::SearchOnSealedColumn(*schema,
                                column.get(),
                                search_info,
                                index_info,
                                query_data.data(),
                                nullptr,
                                1,
                                total_row_count,
                                bv,
                                &op_context,
                                search_result);

    // Verify that all offsets are -1 (invalid) since no valid vectors exist
    ASSERT_EQ(search_result.seg_offsets_.size(), search_info.topk_);
    for (auto& offset : search_result.seg_offsets_) {
        ASSERT_EQ(offset, INVALID_SEG_OFFSET);
    }
    ASSERT_EQ(search_result.total_nq_, 1);
    ASSERT_EQ(search_result.unity_topK_, search_info.topk_);
}

// Test search iterator on nullable vector field with all null vectors
TEST(test_chunk_segment, TestSearchIteratorOnSealedWithAllNullVectors) {
    int dim = 16;
    int chunk_num = 2;
    int chunk_size = 100;

    DeferRelease defer;

    int total_row_count = chunk_num * chunk_size;
    int bitset_size = (total_row_count + 7) / 8;

    auto schema = std::make_shared<Schema>();
    auto fakevec_id = schema->AddDebugField(
        "fakevec", DataType::VECTOR_FLOAT, dim, knowhere::metric::L2, true);

    auto field_meta = schema->operator[](fakevec_id);

    std::vector<std::unique_ptr<Chunk>> chunks;
    std::vector<int64_t> num_rows_per_chunk;

    for (int i = 0; i < chunk_num; i++) {
        num_rows_per_chunk.push_back(chunk_size);

        int null_bitmap_bytes = (chunk_size + 7) / 8;
        int vector_data_size = chunk_size * dim * sizeof(float);
        int buf_size = null_bitmap_bytes + vector_data_size;

        char* buf = new char[buf_size];
        defer.AddDefer([buf]() { delete[] buf; });

        // All vectors are null
        std::fill(buf, buf + null_bitmap_bytes, 0);
        std::fill(buf + null_bitmap_bytes, buf + buf_size, 0);

        auto chunk_mmap_guard =
            std::make_shared<ChunkMmapGuard>(nullptr, 0, "");
        chunks.emplace_back(
            std::make_unique<FixedWidthChunk>(chunk_size,
                                              dim,
                                              buf,
                                              buf_size,
                                              sizeof(float),
                                              true,
                                              chunk_mmap_guard));
    }

    auto translator = std::make_unique<TestChunkTranslator>(
        num_rows_per_chunk, "", std::move(chunks));
    auto slot =
        cachinglayer::Manager::GetInstance().CreateCacheSlot<milvus::Chunk>(
            std::move(translator), nullptr);
    auto column = std::make_shared<ChunkedColumn>(std::move(slot), field_meta);

    // Build valid row ids to initialize offset_mapping for nullable vectors
    column->BuildValidRowIds(nullptr);

    SearchInfo search_info;
    auto search_conf = knowhere::Json{
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
    };
    search_info.search_params_ = search_conf;
    search_info.field_id_ = fakevec_id;
    search_info.metric_type_ = knowhere::metric::L2;
    search_info.topk_ = 10;
    // Enable iterator_v2
    search_info.iterator_v2_info_ = SearchIteratorV2Info{"", 10};

    uint8_t* bitset_data = new uint8_t[bitset_size];
    defer.AddDefer([bitset_data]() { delete[] bitset_data; });
    std::fill(bitset_data, bitset_data + bitset_size, 0);
    BitsetView bv(bitset_data, total_row_count);

    std::vector<float> query_data(dim);
    for (int i = 0; i < dim; i++) {
        query_data[i] = static_cast<float>(i) / dim;
    }
    auto index_info = std::map<std::string, std::string>{};
    SearchResult search_result;
    milvus::OpContext op_context;

    // This should NOT crash - search iterator with all null vectors
    query::SearchOnSealedColumn(*schema,
                                column.get(),
                                search_info,
                                index_info,
                                query_data.data(),
                                nullptr,
                                1,
                                total_row_count,
                                bv,
                                &op_context,
                                search_result);

    // Verify empty results
    ASSERT_EQ(search_result.seg_offsets_.size(), search_info.topk_);
    for (auto& offset : search_result.seg_offsets_) {
        ASSERT_EQ(offset, INVALID_SEG_OFFSET);
    }
}

// Test search iterator on nullable vector field with partial null vectors.
// This test verifies:
// 1. CachedSearchIterator uses valid_count_per_chunk (not total row count)
//    as chunk_size, preventing out-of-bounds reads
// 2. Knowhere BF receives each chunk's physical->logical id window and returns
//    logical offsets directly.
TEST(test_chunk_segment, TestSearchIteratorOnSealedWithPartialNullVectors) {
    int dim = 16;
    int chunk_num = 2;
    int chunk_rows = 100;      // logical rows per chunk
    int valid_per_chunk = 50;  // even rows valid, odd null

    DeferRelease defer;

    int total_row_count = chunk_num * chunk_rows;
    int total_valid = chunk_num * valid_per_chunk;
    int bitset_size = (total_row_count + 7) / 8;

    auto schema = std::make_shared<Schema>();
    auto fakevec_id = schema->AddDebugField(
        "fakevec", DataType::VECTOR_FLOAT, dim, knowhere::metric::L2, true);

    auto field_meta = schema->operator[](fakevec_id);
    ASSERT_TRUE(field_meta.is_nullable());

    // Generate base vectors for valid rows only
    auto base_schema = std::make_shared<Schema>();
    base_schema->AddDebugField(
        "fakevec", DataType::VECTOR_FLOAT, dim, knowhere::metric::L2);
    auto base_dataset = segcore::DataGen(base_schema, total_valid);
    auto base_data = base_dataset.get_col<float>(
        base_schema->get_field_id(FieldName("fakevec")));

    std::vector<std::unique_ptr<Chunk>> chunks;
    std::vector<int64_t> num_rows_per_chunk;
    // Track which logical rows are valid for verification
    std::vector<bool> valid_rows(total_row_count, false);

    for (int i = 0; i < chunk_num; i++) {
        num_rows_per_chunk.push_back(chunk_rows);

        int null_bitmap_bytes = (chunk_rows + 7) / 8;
        // Data section: only valid vectors stored contiguously
        // (matching NullableVectorChunkWriter behavior)
        int vector_data_size = valid_per_chunk * dim * sizeof(float);
        int buf_size = null_bitmap_bytes + vector_data_size;

        char* buf = new char[buf_size];
        defer.AddDefer([buf]() { delete[] buf; });

        // Set null bitmap: even rows valid, odd rows null (50% null)
        std::fill(buf, buf + null_bitmap_bytes, 0);
        for (int j = 0; j < chunk_rows; j++) {
            if (j % 2 == 0) {
                // Set bit to 1 (valid)
                buf[j >> 3] |= (1 << (j & 0x07));
                valid_rows[i * chunk_rows + j] = true;
            }
        }

        // Copy valid vectors contiguously into data section
        memcpy(buf + null_bitmap_bytes,
               base_data.data() + i * valid_per_chunk * dim,
               vector_data_size);

        auto chunk_mmap_guard =
            std::make_shared<ChunkMmapGuard>(nullptr, 0, "");
        chunks.emplace_back(
            std::make_unique<FixedWidthChunk>(chunk_rows,
                                              dim,
                                              buf,
                                              buf_size,
                                              sizeof(float),
                                              true,
                                              chunk_mmap_guard));
    }

    auto translator = std::make_unique<TestChunkTranslator>(
        num_rows_per_chunk, "", std::move(chunks));
    auto slot =
        cachinglayer::Manager::GetInstance().CreateCacheSlot<milvus::Chunk>(
            std::move(translator), nullptr);
    auto column = std::make_shared<ChunkedColumn>(std::move(slot), field_meta);

    // Build valid row ids to initialize offset_mapping
    column->BuildValidRowIds(nullptr);

    const auto& offset_mapping = column->GetOffsetMapping();
    ASSERT_TRUE(offset_mapping.IsEnabled());
    // 50% valid: 50 per chunk * 2 chunks = 100
    ASSERT_EQ(offset_mapping.GetValidCount(), total_valid);

    for (int i = 0; i < chunk_num; i++) {
        ASSERT_EQ(column->GetValidCountInChunk(i), valid_per_chunk);
    }

    SearchInfo search_info;
    auto search_conf = knowhere::Json{
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
    };
    search_info.search_params_ = search_conf;
    search_info.field_id_ = fakevec_id;
    search_info.metric_type_ = knowhere::metric::L2;
    search_info.topk_ = 10;
    // Enable iterator_v2 to exercise CachedSearchIterator path
    search_info.iterator_v2_info_ = SearchIteratorV2Info{"", 10};

    uint8_t* bitset_data = new uint8_t[bitset_size];
    defer.AddDefer([bitset_data]() { delete[] bitset_data; });
    std::fill(bitset_data, bitset_data + bitset_size, 0);
    BitsetView bv(bitset_data, total_row_count);

    // Use a valid vector as query
    std::vector<float> query_data(base_data.begin(), base_data.begin() + dim);
    auto index_info = std::map<std::string, std::string>{};
    SearchResult search_result;
    milvus::OpContext op_context;

    query::SearchOnSealedColumn(*schema,
                                column.get(),
                                search_info,
                                index_info,
                                query_data.data(),
                                nullptr,
                                1,
                                total_row_count,
                                bv,
                                &op_context,
                                search_result);

    ASSERT_EQ(search_result.seg_offsets_.size(), search_info.topk_);
    ASSERT_EQ(search_result.distances_.size(), search_info.topk_);

    // Verify returned offsets are logical offsets (in [0, total_row_count))
    // and correspond to valid (non-null) rows
    int valid_count = 0;
    for (auto& offset : search_result.seg_offsets_) {
        if (offset == INVALID_SEG_OFFSET) {
            continue;
        }
        valid_count++;
        // Offset must be a logical offset in valid range
        ASSERT_GE(offset, 0);
        ASSERT_LT(offset, total_row_count);
        // Offset must refer to a valid (non-null) row
        ASSERT_TRUE(valid_rows[offset])
            << "offset " << offset << " should be a valid (non-null) row";
    }
    // We should get some valid results
    ASSERT_GT(valid_count, 0);

    // Test brute force (non-iterator) path for comparison
    SearchInfo search_info_bf = search_info;
    search_info_bf.iterator_v2_info_ = std::nullopt;
    SearchResult bf_result;
    query::SearchOnSealedColumn(*schema,
                                column.get(),
                                search_info_bf,
                                index_info,
                                query_data.data(),
                                nullptr,
                                1,
                                total_row_count,
                                bv,
                                &op_context,
                                bf_result);

    // Both paths should return the same offsets (same ordering)
    ASSERT_EQ(search_result.seg_offsets_.size(), bf_result.seg_offsets_.size());
    for (size_t i = 0; i < search_result.seg_offsets_.size(); i++) {
        EXPECT_EQ(search_result.seg_offsets_[i], bf_result.seg_offsets_[i])
            << "Mismatch at index " << i;
    }
}

class TestChunkSegment : public testing::TestWithParam<bool> {
 protected:
    void
    SetUp() override {
        bool pk_is_string = GetParam();
        auto schema = segcore::GenChunkedSegmentTestSchema(pk_is_string);
        segment = segcore::CreateSealedSegment(
            schema,
            nullptr,
            -1,
            segcore::SegcoreConfig::default_config(),
            true);
        test_data_count = 10000;

        std::vector<FieldId> field_ids = {
            schema->get_field_id(FieldName("int64")),
            schema->get_field_id(FieldName("pk")),
            TimestampFieldID,
            schema->get_field_id(FieldName("string1")),
            schema->get_field_id(FieldName("string2"))};
        fields = {{"int64", schema->get_field_id(FieldName("int64"))},
                  {"pk", schema->get_field_id(FieldName("pk"))},
                  {"ts", TimestampFieldID},
                  {"string1", schema->get_field_id(FieldName("string1"))},
                  {"string2", schema->get_field_id(FieldName("string2"))}};

        int start_id = 0;
        chunk_num = 2;

        std::vector<std::string> str_data;
        for (int i = 0; i < test_data_count * chunk_num; i++) {
            str_data.push_back("test" + std::to_string(i));
        }
        std::sort(str_data.begin(), str_data.end());

        auto cm = milvus::storage::RemoteChunkManagerSingleton::GetInstance()
                      .GetRemoteChunkManager();

        std::unordered_map<FieldId, std::vector<FieldDataPtr>> field_data_map;

        // generate data
        for (int chunk_id = 0; chunk_id < chunk_num;
             chunk_id++, start_id += test_data_count) {
            std::vector<int64_t> test_data(test_data_count);
            std::iota(test_data.begin(), test_data.end(), start_id);

            for (int i = 0; i < field_ids.size(); i++) {
                // if i < 3, this is system field, thus must be int64.
                // other fields include a pk field and 2 string fields. pk field is string if pk_is_string is true.
                auto datatype =
                    i < 3 && (field_ids[i] !=
                                  schema->get_field_id(FieldName("pk")) ||
                              !pk_is_string)
                        ? DataType::INT64
                        : DataType::VARCHAR;
                FieldDataPtr field_data{nullptr};

                if (datatype == DataType::INT64) {
                    field_data =
                        std::make_shared<FieldData<int64_t>>(datatype, false);
                    field_data->FillFieldData(test_data.data(),
                                              test_data_count);
                } else {
                    field_data = std::make_shared<FieldData<std::string>>(
                        datatype, false);
                    field_data->FillFieldData(str_data.data() + start_id,
                                              test_data_count);
                }
                auto fid = field_ids[i];
                field_data_map[fid].push_back(field_data);
            }
        }
        for (auto& [fid, field_datas] : field_data_map) {
            auto load_info = PrepareSingleFieldInsertBinlog(kCollectionID,
                                                            kPartitionID,
                                                            kSegmentID,
                                                            fid.get(),
                                                            field_datas,
                                                            cm);
            segment->LoadFieldData(load_info);
        }
    }

    segcore::SegmentSealedUPtr segment;
    int chunk_num;
    int test_data_count;
    std::unordered_map<std::string, FieldId> fields;
};

INSTANTIATE_TEST_SUITE_P(TestChunkSegment, TestChunkSegment, testing::Bool());

TEST_P(TestChunkSegment, TestSkipNextTermExpr) {
    bool pk_is_string = GetParam();
    // only need to test int64 case
    if (pk_is_string) {
        return;
    }

    // test segment with 2 chunks and expr is: int64 >= 10000 and pk in (10001, 10002, 10003, 10004, 10005)
    proto::plan::GenericValue v1;
    v1.set_int64_val(10000);
    auto first_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(fields.at("int64"), DataType::INT64),
        proto::plan::OpType::GreaterEqual,
        v1);

    std::vector<proto::plan::GenericValue> v2;
    for (int i = 1; i <= 5; ++i) {
        proto::plan::GenericValue v;
        v.set_int64_val(i + 10000);
        v2.push_back(v);
    }
    auto second_expr = std::make_shared<expr::TermFilterExpr>(
        expr::ColumnInfo(fields.at("pk"), DataType::INT64), v2);
    auto and_expr = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, first_expr, second_expr);
    auto plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, and_expr);
    auto final = query::ExecuteQueryExpr(
        plan, segment.get(), chunk_num * test_data_count, MAX_TIMESTAMP);
    ASSERT_EQ(5, final.count());
    for (int i = 10001; i <= 10005; ++i) {
        ASSERT_EQ(true, final[i]) << "i: " << i;
    }
}

TEST_P(TestChunkSegment, TestTermExpr) {
    bool pk_is_string = GetParam();
    // query int64 expr
    std::vector<proto::plan::GenericValue> filter_data;
    for (int i = 1; i <= 10; ++i) {
        proto::plan::GenericValue v;
        v.set_int64_val(i);
        filter_data.push_back(v);
    }
    auto term_filter_expr = std::make_shared<expr::TermFilterExpr>(
        expr::ColumnInfo(fields.at("int64"), DataType::INT64), filter_data);
    BitsetType final;
    auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                       term_filter_expr);
    final = query::ExecuteQueryExpr(
        plan, segment.get(), chunk_num * test_data_count, MAX_TIMESTAMP);
    ASSERT_EQ(10, final.count());

    std::vector<proto::plan::GenericValue> filter_str_data;
    for (int i = 1; i <= 10; ++i) {
        proto::plan::GenericValue v;
        v.set_string_val("test" + std::to_string(i));
        filter_str_data.push_back(v);
    }
    // query pk expr
    auto pk_term_filter_expr = std::make_shared<expr::TermFilterExpr>(
        expr::ColumnInfo(fields.at("pk"),
                         pk_is_string ? DataType::VARCHAR : DataType::INT64),
        pk_is_string ? filter_str_data : filter_data);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                  pk_term_filter_expr);
    final = query::ExecuteQueryExpr(
        plan, segment.get(), chunk_num * test_data_count, MAX_TIMESTAMP);
    ASSERT_EQ(10, final.count());

    // query pk in second chunk
    std::vector<proto::plan::GenericValue> filter_data2;
    proto::plan::GenericValue v;
    if (pk_is_string) {
        v.set_string_val("test" + std::to_string(test_data_count + 1));
    } else {
        v.set_int64_val(test_data_count + 1);
    }
    filter_data2.push_back(v);

    pk_term_filter_expr = std::make_shared<expr::TermFilterExpr>(
        expr::ColumnInfo(fields.at("pk"),
                         pk_is_string ? DataType::VARCHAR : DataType::INT64),
        filter_data2);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                  pk_term_filter_expr);
    final = query::ExecuteQueryExpr(
        plan, segment.get(), chunk_num * test_data_count, MAX_TIMESTAMP);
    ASSERT_EQ(1, final.count());
}

TEST_P(TestChunkSegment, TestCompareExpr) {
    srand(time(NULL));
    bool pk_is_string = GetParam();
    DataType pk_data_type = pk_is_string ? DataType::VARCHAR : DataType::INT64;
    auto expr = std::make_shared<expr::CompareExpr>(
        pk_is_string ? fields.at("string1") : fields.at("int64"),
        fields.at("pk"),
        pk_data_type,
        pk_data_type,
        proto::plan::OpType::Equal);
    auto plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    BitsetType final = query::ExecuteQueryExpr(
        plan, segment.get(), chunk_num * test_data_count, MAX_TIMESTAMP);
    ASSERT_EQ(chunk_num * test_data_count, final.count());

    expr = std::make_shared<expr::CompareExpr>(fields.at("string1"),
                                               fields.at("string2"),
                                               DataType::VARCHAR,
                                               DataType::VARCHAR,
                                               proto::plan::OpType::Equal);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = query::ExecuteQueryExpr(
        plan, segment.get(), chunk_num * test_data_count, MAX_TIMESTAMP);
    ASSERT_EQ(chunk_num * test_data_count, final.count());

    // test with inverted index
    auto fid = fields.at("int64");
    std::vector<int64_t> data(test_data_count * chunk_num);
    for (int i = 0; i < chunk_num; i++) {
        auto pw = segment->chunk_data<int64_t>(nullptr, fid, i);
        auto d = pw.get();
        std::copy(d.data(),
                  d.data() + test_data_count,
                  data.begin() + i * test_data_count);
    }

    auto opened =
        test::consumer::BuildScalarReader<int64_t>(fid,
                                                   DataType::INT64,
                                                   index::INVERTED_INDEX_TYPE,
                                                   data.size(),
                                                   data.data());
    auto load_index_info = test::consumer::MakeLoadIndexInfo(
        std::move(opened), DataType::INT64, fid.get());
    segment->LoadIndex(load_index_info);

    expr = std::make_shared<expr::CompareExpr>(
        pk_is_string ? fields.at("string1") : fields.at("int64"),
        fields.at("pk"),
        pk_data_type,
        pk_data_type,
        proto::plan::OpType::Equal);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = query::ExecuteQueryExpr(
        plan, segment.get(), chunk_num * test_data_count, MAX_TIMESTAMP);
    ASSERT_EQ(chunk_num * test_data_count, final.count());
}

TEST_P(TestChunkSegment, TestPkRange) {
    using namespace milvus::segcore;
    using namespace milvus::test::consumer;
    bool pk_is_string = GetParam();
    auto segment_impl = dynamic_cast<ChunkedSegmentSealedImpl*>(segment.get());
    ASSERT_NE(segment_impl, nullptr);

    // Test cases for sorted PK
    BitsetType bitset_sorted(chunk_num * test_data_count);
    BitsetTypeView bitset_sorted_view(bitset_sorted);

    // Test Equal operation
    if (pk_is_string) {
        segment_impl->pk_range(nullptr,
                               proto::plan::OpType::Equal,
                               PkType("test1"),
                               bitset_sorted_view);
        EXPECT_EQ(1, bitset_sorted_view.count());
    } else {
        segment_impl->pk_range(
            nullptr, proto::plan::OpType::Equal, PkType(1), bitset_sorted_view);
        EXPECT_EQ(1, bitset_sorted_view.count());
    }

    // Test LessEqual operation
    bitset_sorted.reset();
    if (pk_is_string) {
        segment_impl->pk_range(nullptr,
                               proto::plan::OpType::LessEqual,
                               PkType("test100"),
                               bitset_sorted_view);
        // only 'test0', 'test1', 'test10' are less than 'test100'
        EXPECT_EQ(bitset_sorted_view.count(), 4);
    } else {
        segment_impl->pk_range(nullptr,
                               proto::plan::OpType::LessEqual,
                               PkType(100),
                               bitset_sorted_view);
        EXPECT_EQ(bitset_sorted_view.count(), 101);
    }

    bitset_sorted.reset();
    if (pk_is_string) {
        segment_impl->pk_range(nullptr,
                               proto::plan::OpType::Equal,
                               PkType(std::string("non_existent_pk")),
                               bitset_sorted_view);
        EXPECT_EQ(0, bitset_sorted_view.count());
    } else {
        segment_impl->pk_range(nullptr,
                               proto::plan::OpType::Equal,
                               PkType(int64_t(999999)),
                               bitset_sorted_view);
        EXPECT_EQ(0, bitset_sorted_view.count());
    }
}

TEST(TestTTLFieldFilter, TestMaskWithTTLField) {
    using namespace milvus::segcore;
    using namespace milvus::test::consumer;

    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("pk", DataType::INT64, false);
    schema->AddField(FieldName("ts"),
                     TimestampFieldID,
                     DataType::INT64,
                     false,
                     std::nullopt);
    auto ttl_fid =
        schema->AddDebugField("ttl_field", DataType::TIMESTAMPTZ, false);
    schema->set_primary_field_id(pk_fid);
    schema->set_ttl_field_id(ttl_fid);

    auto segment = CreateSealedSegment(
        schema, nullptr, -1, SegcoreConfig::default_config(), true);

    int test_data_count = 100;
    auto cm = milvus::storage::RemoteChunkManagerSingleton::GetInstance()
                  .GetRemoteChunkManager();

    std::vector<int64_t> pk_data(test_data_count);
    std::iota(pk_data.begin(), pk_data.end(), 0);

    uint64_t base_ts = 1000000000ULL << 18;
    std::vector<int64_t> ts_data(test_data_count);
    for (int i = 0; i < test_data_count; i++) {
        ts_data[i] = static_cast<int64_t>(base_ts + i);
    }

    uint64_t base_physical_us = (base_ts >> 18) * 1000;
    std::vector<int64_t> ttl_data(test_data_count);
    for (int i = 0; i < test_data_count; i++) {
        if (i < test_data_count / 2) {
            ttl_data[i] = static_cast<int64_t>(base_physical_us - 10);
        } else {
            ttl_data[i] = static_cast<int64_t>(base_physical_us + 10);
        }
    }

    {
        auto field_data =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        field_data->FillFieldData(pk_data.data(), test_data_count);
        std::vector<FieldDataPtr> field_datas = {field_data};
        auto load_info = PrepareSingleFieldInsertBinlog(kCollectionID,
                                                        kPartitionID,
                                                        kSegmentID,
                                                        pk_fid.get(),
                                                        field_datas,
                                                        cm);
        segment->LoadFieldData(load_info);
    }

    {
        auto field_data =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        field_data->FillFieldData(ts_data.data(), test_data_count);
        std::vector<FieldDataPtr> field_datas = {field_data};
        auto load_info = PrepareSingleFieldInsertBinlog(kCollectionID,
                                                        kPartitionID,
                                                        kSegmentID,
                                                        TimestampFieldID.get(),
                                                        field_datas,
                                                        cm);
        segment->LoadFieldData(load_info);
    }

    {
        auto field_data =
            std::make_shared<FieldData<int64_t>>(DataType::TIMESTAMPTZ, false);
        field_data->FillFieldData(ttl_data.data(), test_data_count);
        std::vector<FieldDataPtr> field_datas = {field_data};
        auto load_info = PrepareSingleFieldInsertBinlog(kCollectionID,
                                                        kPartitionID,
                                                        kSegmentID,
                                                        ttl_fid.get(),
                                                        field_datas,
                                                        cm);
        segment->LoadFieldData(load_info);
    }

    // Test TTL field filtering using CompileExpressions pathway
    Timestamp query_ts = base_ts + test_data_count;
    int64_t active_count = segment->get_active_count(query_ts);

    // Create an expression list with AlwaysTrueExpr - CompileExpressions will
    // automatically add TTL field filtering expression
    auto always_true_expr = std::make_shared<expr::AlwaysTrueExpr>();
    auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                       always_true_expr);

    // Execute query expression - this will trigger CompileExpressions which
    // automatically adds TTL field filtering expression
    BitsetType bitset =
        query::ExecuteQueryExpr(plan, segment.get(), active_count, query_ts);

    // Note: ExecuteQueryExpr already flips the bitset, so bitset[i] = true
    // means the row matches (not expired), bitset[i] = false means expired
    BitsetTypeView bitset_view(bitset);

    // Verify results:
    // After ExecuteQueryExpr, bitset[i] = true means row matches (not expired)
    // bitset[i] = false means row is filtered out (expired)
    // - i < test_data_count / 2: expired (TTL < current time), should be expired (bitset_view[i] = false)
    // - i >= test_data_count / 2: not expired, should NOT be expired (bitset_view[i] = true)
    int expired_count = 0;
    for (int i = 0; i < test_data_count; i++) {
        if (!bitset_view[i]) {
            expired_count++;
        }
    }

    EXPECT_EQ(expired_count, test_data_count / 2);
    for (int i = 0; i < test_data_count / 2; i++) {
        EXPECT_FALSE(bitset_view[i]) << "Row " << i << " should be expired";
    }
    for (int i = test_data_count / 2; i < test_data_count; i++) {
        EXPECT_TRUE(bitset_view[i]) << "Row " << i << " should not be expired";
    }
}

TEST(TestTTLFieldFilter, TestMaskWithNullableTTLField) {
    using namespace milvus::segcore;
    using namespace milvus::test::consumer;

    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("pk", DataType::INT64, false);
    schema->AddField(FieldName("ts"),
                     TimestampFieldID,
                     DataType::INT64,
                     false,
                     std::nullopt);
    auto ttl_fid =
        schema->AddDebugField("ttl_field", DataType::TIMESTAMPTZ, true);
    schema->set_primary_field_id(pk_fid);
    schema->set_ttl_field_id(ttl_fid);

    auto segment = CreateSealedSegment(
        schema, nullptr, -1, SegcoreConfig::default_config(), true);

    int test_data_count = 100;
    auto cm = milvus::storage::RemoteChunkManagerSingleton::GetInstance()
                  .GetRemoteChunkManager();

    std::vector<int64_t> pk_data(test_data_count);
    std::iota(pk_data.begin(), pk_data.end(), 0);

    uint64_t base_ts = 1000000000ULL << 18;
    std::vector<int64_t> ts_data(test_data_count);
    for (int i = 0; i < test_data_count; i++) {
        ts_data[i] = static_cast<int64_t>(base_ts + i);
    }

    uint64_t base_physical_us = (base_ts >> 18) * 1000;
    std::vector<int64_t> ttl_data(test_data_count);
    // Arrow validity bitmap uses 1 bit per value (LSB-first within each byte).
    std::vector<uint8_t> valid_data((test_data_count + 7) / 8, 0);
    for (int i = 0; i < test_data_count; i++) {
        const size_t byte_idx = static_cast<size_t>(i) >> 3;
        const uint8_t bit_mask = static_cast<uint8_t>(1u << (i & 7));
        if (i % 4 == 0) {
            ttl_data[i] = 0;
            valid_data[byte_idx] &= static_cast<uint8_t>(~bit_mask);
        } else if (i % 4 == 1) {
            ttl_data[i] = static_cast<int64_t>(base_physical_us - 10);
            valid_data[byte_idx] |= bit_mask;
        } else {
            ttl_data[i] = static_cast<int64_t>(base_physical_us + 10);
            valid_data[byte_idx] |= bit_mask;
        }
    }

    {
        auto field_data =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        field_data->FillFieldData(pk_data.data(), test_data_count);
        std::vector<FieldDataPtr> field_datas = {field_data};
        auto load_info = PrepareSingleFieldInsertBinlog(kCollectionID,
                                                        kPartitionID,
                                                        kSegmentID,
                                                        pk_fid.get(),
                                                        field_datas,
                                                        cm);
        segment->LoadFieldData(load_info);
    }

    {
        auto field_data =
            std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        field_data->FillFieldData(ts_data.data(), test_data_count);
        std::vector<FieldDataPtr> field_datas = {field_data};
        auto load_info = PrepareSingleFieldInsertBinlog(kCollectionID,
                                                        kPartitionID,
                                                        kSegmentID,
                                                        TimestampFieldID.get(),
                                                        field_datas,
                                                        cm);
        segment->LoadFieldData(load_info);
    }

    {
        auto field_data =
            std::make_shared<FieldData<int64_t>>(DataType::TIMESTAMPTZ, true);
        field_data->FillFieldData(
            ttl_data.data(), valid_data.data(), test_data_count, 0);
        std::vector<FieldDataPtr> field_datas = {field_data};
        auto load_info = PrepareSingleFieldInsertBinlog(kCollectionID,
                                                        kPartitionID,
                                                        kSegmentID,
                                                        ttl_fid.get(),
                                                        field_datas,
                                                        cm);
        segment->LoadFieldData(load_info);
    }

    // Test TTL field filtering using CompileExpressions pathway
    Timestamp query_ts = base_ts + test_data_count;
    int64_t active_count = segment->get_active_count(query_ts);

    // Create an expression list with AlwaysTrueExpr - CompileExpressions will
    // automatically add TTL field filtering expression
    auto always_true_expr = std::make_shared<expr::AlwaysTrueExpr>();
    auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                       always_true_expr);

    // Execute query expression - this will trigger CompileExpressions which
    // automatically adds TTL field filtering expression
    BitsetType bitset =
        query::ExecuteQueryExpr(plan, segment.get(), active_count, query_ts);

    // Note: ExecuteQueryExpr already flips the bitset, so bitset[i] = true
    // means the row matches (not expired), bitset[i] = false means expired
    BitsetTypeView bitset_view(bitset);

    // Verify results:
    // After ExecuteQueryExpr, bitset[i] = true means row matches (not expired)
    // bitset[i] = false means row is filtered out (expired)
    // - i % 4 == 0: null, should NOT be expired (bitset_view[i] = true)
    // - i % 4 == 1: expired (TTL < current time), should be expired (bitset_view[i] = false)
    // - i % 4 == 2 or 3: not expired, should NOT be expired (bitset_view[i] = true)
    int expired_count = 0;
    for (int i = 0; i < test_data_count; i++) {
        if (i % 4 == 0) {
            // Null value should not be expired (should match)
            EXPECT_TRUE(bitset_view[i])
                << "Row " << i << " (null) should not be expired";
        } else if (i % 4 == 1) {
            // Should be expired (should be filtered out)
            EXPECT_FALSE(bitset_view[i]) << "Row " << i << " should be expired";
            expired_count++;
        } else {
            // Should not be expired (should match)
            EXPECT_TRUE(bitset_view[i])
                << "Row " << i << " should not be expired";
        }
    }

    EXPECT_EQ(expired_count, test_data_count / 4);
}

// ---------------------------------------------------------------------------
// One field, several scalar indexes.
//
// A field's indexes live in one inventory keyed by (field, indexID), so a second
// index on a field neither replaces nor hides its sibling. These cases pin the
// container and selection contracts that makes that safe:
//   * two indexes on one field both stay loaded, both are visible as capability
//     metadata, and each one can be pinned on its own;
//   * selection picks by static capability: an exact operator takes the predicate
//     reader, a pattern operator takes the NGRAM candidate, and a field that
//     cannot serve the operator falls back to the raw column;
//   * the field-level "the index provides raw data" flag reports false while a
//     field carries several indexes, so readers fall back to the raw column
//     instead of reverse-looking-up through an index that may have no raw data.
// ---------------------------------------------------------------------------
namespace {

// Installs one already-opened reader under an explicit indexID, so a test can
// decide which index is the field's smallest and which one is loaded second.
milvus::segcore::LoadIndexInfo
MakeMultiIndexLoadInfo(milvus::test::consumer::OpenedIndex opened,
                       DataType field_type,
                       FieldId field_id,
                       int64_t index_id,
                       std::map<std::string, std::string> extra_params = {}) {
    auto info = milvus::test::consumer::MakeLoadIndexInfo(
        std::move(opened), field_type, field_id.get());
    info.index_id = index_id;
    for (auto& [key, value] : extra_params) {
        info.index_params.emplace(std::move(key), std::move(value));
    }
    return info;
}

void
LoadMultiIndex(milvus::segcore::SegmentSealed& segment,
               milvus::test::consumer::OpenedIndex opened,
               DataType field_type,
               FieldId field_id,
               int64_t index_id,
               std::map<std::string, std::string> extra_params = {}) {
    auto info = MakeMultiIndexLoadInfo(std::move(opened),
                                       field_type,
                                       field_id,
                                       index_id,
                                       std::move(extra_params));
    segment.LoadIndex(info);
}

// A segment that records which inventory entry the expression layer pinned, so a
// test can assert the selection decision itself and not only the result rows.
class RecordingScalarIndexSegment
    : public milvus::segcore::ChunkedSegmentSealedImpl {
 public:
    RecordingScalarIndexSegment(SchemaPtr schema, int64_t segment_id)
        : ChunkedSegmentSealedImpl(
              std::move(schema),
              milvus::empty_index_meta,
              milvus::segcore::SegcoreConfig::default_config(),
              segment_id,
              /*is_sorted_by_pk=*/false) {
    }

    milvus::segcore::IndexPin
    PinIndex(milvus::OpContext* op_ctx,
             const milvus::segcore::IndexKey& key) const override {
        pinned_keys_.push_back(key);
        return ChunkedSegmentSealedImpl::PinIndex(op_ctx, key);
    }

    // Only the entries the expression layer pinned, in pin order.
    mutable std::vector<milvus::segcore::IndexKey> pinned_keys_;
};

constexpr int64_t kInvertedIndexId = 4001;
constexpr int64_t kNgramIndexId = 4002;

}  // namespace

// T3: two scalar indexes on one field both load, both are reported as loaded,
// and each one can be pinned on its own.
//
// This case used to be named MultiScalarIndexOnSameFieldAsserts and asserted
// that loading a second index on a field trips AssertInfo(!has_index)
// (ChunkedSegmentSealedImpl's LoadScalarIndex before this change). With the
// inventory keyed by (field, indexID) the second index trips nothing and no
// longer replaces its sibling: the two coexist as independent entries.
//
// The production path is LoadDiff -> LoadBatchIndexes -> committer.Commit ->
// LoadIndex(is_replace=false); both entries share LoadScalarIndex, and
// SegmentLoadInfoTest.GetLoadDiffWithTwoScalarIndexesOnSameField proves the two
// land in the same indexes_to_load[field] vector.
TEST_P(TestChunkSegment, MultiScalarIndexOnSameFieldLoads) {
    using namespace milvus::segcore;
    auto seg = dynamic_cast<ChunkedSegmentSealedImpl*>(segment.get());
    ASSERT_NE(seg, nullptr);
    auto fid = fields.at("int64");

    std::vector<int64_t> data(test_data_count * chunk_num);
    for (int i = 0; i < chunk_num; i++) {
        auto pw = seg->chunk_data<int64_t>(nullptr, fid, i);
        auto d = pw.get();
        std::copy(d.data(),
                  d.data() + test_data_count,
                  data.begin() + i * test_data_count);
    }

    // indexID -> the reader object itself; the assertions below pin down which
    // one was handed back.
    std::map<int64_t, const index::IIndexReaderBase*> raw_readers;
    auto load_inverted_index = [&](int64_t index_id) {
        auto opened = milvus::test::consumer::BuildScalarReader<int64_t>(
            fid,
            DataType::INT64,
            index::INVERTED_INDEX_TYPE,
            data.size(),
            data.data());
        raw_readers[index_id] = opened.reader.get();
        LoadMultiIndex(*seg, std::move(opened), DataType::INT64, fid, index_id);
    };

    // first index: loads normally, the field turns index_ready
    ASSERT_NO_THROW(load_inverted_index(1));
    ASSERT_TRUE(seg->HasIndex(fid));

    // second index (same field, different indexID): AssertInfo(!has_index) is gone
    ASSERT_NO_THROW(load_inverted_index(2));

    // Two entries in the inventory, reported ascending by index identity so the
    // container's hash order plays no part.
    const auto capability = seg->IndexCapability(fid);
    ASSERT_EQ(capability.entries().size(), 2u);
    EXPECT_EQ(capability.entries()[0].key,
              (IndexKey{fid, IndexIdentity::PreBuiltIndex(1)}));
    EXPECT_EQ(capability.entries()[1].key,
              (IndexKey{fid, IndexIdentity::PreBuiltIndex(2)}));
    for (const auto& entry : capability.entries()) {
        EXPECT_EQ(entry.family, index::families::kInverted);
        EXPECT_TRUE(entry.json_path.empty());
        EXPECT_TRUE(entry.caps.predicate);
        EXPECT_FALSE(entry.caps.ngram_candidates);
    }
    EXPECT_TRUE(seg->HasIndex(fid));

    // Both indexes can still be pinned, and each pins its own reader.
    milvus::OpContext op_ctx;
    auto pinned_1 =
        seg->PinIndex(&op_ctx, IndexKey{fid, IndexIdentity::PreBuiltIndex(1)});
    ASSERT_FALSE(pinned_1.empty());
    EXPECT_EQ(pinned_1.get(), raw_readers.at(1));
    auto pinned_2 =
        seg->PinIndex(&op_ctx, IndexKey{fid, IndexIdentity::PreBuiltIndex(2)});
    ASSERT_FALSE(pinned_2.empty());
    EXPECT_EQ(pinned_2.get(), raw_readers.at(2));

    // Loading the same indexID again is a no-op, not a second entry: a reopen
    // that re-lists the index, or a retried load, must not disturb the field.
    ASSERT_NO_THROW(load_inverted_index(2));
    EXPECT_EQ(seg->IndexCapability(fid).entries().size(), 2u);
}

// D9 + T8: two casts on one JSON path are two different identities (the identity
// is the indexID, not the path), so both load and each can be pinned on its own.
// One of them is the flat router (cast JSON) and the other a typed cast on the
// same path, which is exactly the pair that must never be confused.
//
// Also covers I1/I3: a non-NGRAM JSON path index must NOT set the field's
// index_ready_bitset -- HasIndex() feeds plan sampling and has to stay narrow
// (see ScalarIndexCandidate.h's FoldsIntoIndexReadyBit / CountsAsJsonIndex).
TEST(MultiScalarIndexOnSameField, JsonSamePathTwoCastTypesBothLoadAndPin) {
    using namespace milvus::segcore;
    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("pk", DataType::INT64);
    auto json_fid = schema->AddDebugField("json_field", DataType::JSON);
    schema->set_primary_field_id(pk_fid);

    auto segment = CreateSealedSegment(
        schema, nullptr, -1, SegcoreConfig::default_config(), false);
    auto seg = dynamic_cast<ChunkedSegmentSealedImpl*>(segment.get());
    ASSERT_NE(seg, nullptr);

    const std::vector<std::string> json_rows = {
        R"({"a": 1})", R"({"a": 2})", R"({"a": 3})", R"({"a": 4})"};
    auto json_field = milvus::test::expr_index::JsonField(json_rows);

    auto load_json_cast =
        [&](int64_t index_id,
            const std::string& cast) -> const index::IIndexReaderBase* {
        auto opened = milvus::test::expr_index::BuildIndex(
            json_fid,
            DataType::JSON,
            index::INVERTED_INDEX_TYPE,
            {json_field},
            {{JSON_PATH, "/a"}, {JSON_CAST_TYPE, cast}});
        auto* raw = opened.reader.get();
        LoadMultiIndex(*seg,
                       std::move(opened),
                       DataType::JSON,
                       json_fid,
                       index_id,
                       {{JSON_PATH, "/a"}, {JSON_CAST_TYPE, cast}});
        return raw;
    };

    auto* double_reader = load_json_cast(11, "DOUBLE");
    auto* flat_reader = load_json_cast(12, "JSON");

    const auto capability = seg->IndexCapability(json_fid);
    ASSERT_EQ(capability.entries().size(), 2u)
        << "two casts on one path are two identities (different indexIDs)";
    EXPECT_EQ(capability.entries()[0].json_path, "/a");
    EXPECT_EQ(capability.entries()[0].json_cast_type.data_type(),
              JsonCastType::DataType::DOUBLE);
    EXPECT_EQ(capability.entries()[1].json_path, "/a");
    EXPECT_EQ(capability.entries()[1].json_cast_type.data_type(),
              JsonCastType::DataType::JSON);

    // Both pin individually, and each pins its own reader (not the sibling cast).
    milvus::OpContext op_ctx;
    auto pinned_double = seg->PinIndex(
        &op_ctx, IndexKey{json_fid, IndexIdentity::PreBuiltIndex(11)});
    ASSERT_FALSE(pinned_double.empty());
    EXPECT_EQ(pinned_double.get(), double_reader);
    auto pinned_flat = seg->PinIndex(
        &op_ctx, IndexKey{json_fid, IndexIdentity::PreBuiltIndex(12)});
    ASSERT_FALSE(pinned_flat.empty());
    EXPECT_EQ(pinned_flat.get(), flat_reader);

    // I2: HasJsonIndex recognises "a non-NGRAM JSON path index".
    EXPECT_TRUE(seg->HasJsonIndex(json_fid));
    // I1/I3: a JSON path index (non-NGRAM) does not go into index_ready_bitset.
    EXPECT_FALSE(seg->HasIndex(json_fid));
}

// Regression: with an NGRAM index and a typed index on the same JSON path, an
// exact operator must select the typed one.
//
// A JSON NGRAM's cast type is VARCHAR, so a plain string comparison looks
// compatible and a term query over n-gram tokens would return wrong rows
// silently (a row containing "xabcx" matches == "abc", and a literal longer than
// the max gram never matches). Selection is metadata-only, so the rule is a
// capability bit: an NGRAM entry is a candidate family (caps.exact == false) and
// only pattern operators ask for it.
TEST(MultiScalarIndexOnSameField, JsonPinSkipsNgramAndPicksTyped) {
    using namespace milvus::segcore;
    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("pk", DataType::INT64);
    auto json_fid = schema->AddDebugField("json_field", DataType::JSON);
    schema->set_primary_field_id(pk_fid);

    auto segment = CreateSealedSegment(
        schema, nullptr, -1, SegcoreConfig::default_config(), false);
    auto seg = dynamic_cast<ChunkedSegmentSealedImpl*>(segment.get());
    ASSERT_NE(seg, nullptr);

    const std::vector<std::string> json_rows = {R"({"a": "abc"})",
                                                R"({"a": "xabcx"})",
                                                R"({"a": "abd"})",
                                                R"({"a": "ab"})"};
    auto json_field = milvus::test::expr_index::JsonField(json_rows);

    auto load = [&](int64_t index_id,
                    const std::string& index_type,
                    std::map<std::string, std::string> extra_params)
        -> const index::IIndexReaderBase* {
        // The JSON identity params are fixed for this field; anything the caller adds
        // (NGRAM's min_gram/max_gram, for one) has to reach the builder or the reader
        // cannot be constructed at all.
        std::map<std::string, std::string> params = {
            {JSON_PATH, "/a"}, {JSON_CAST_TYPE, "VARCHAR"}};
        params.insert(extra_params.begin(), extra_params.end());
        auto opened = milvus::test::expr_index::BuildIndex(
            json_fid, DataType::JSON, index_type, {json_field}, params);
        auto* raw = opened.reader.get();
        LoadMultiIndex(*seg,
                       std::move(opened),
                       DataType::JSON,
                       json_fid,
                       index_id,
                       std::move(extra_params));
        return raw;
    };

    // NGRAM has the smaller indexID: without the capability rule it would win the
    // tie for an exact operator.
    auto* ngram_reader = load(21,
                              index::NGRAM_INDEX_TYPE,
                              {{JSON_PATH, "/a"},
                               {JSON_CAST_TYPE, "VARCHAR"},
                               {index::MIN_GRAM, "2"},
                               {index::MAX_GRAM, "4"}});
    auto* typed_reader = load(22,
                              index::INVERTED_INDEX_TYPE,
                              {{JSON_PATH, "/a"}, {JSON_CAST_TYPE, "VARCHAR"}});
    ASSERT_NE(ngram_reader, typed_reader);

    const auto capability = seg->IndexCapability(json_fid);
    ASSERT_EQ(capability.entries().size(), 2u);
    EXPECT_TRUE(capability.entries()[0].caps.ngram_candidates);
    EXPECT_FALSE(capability.entries()[0].caps.exact);
    EXPECT_TRUE(capability.entries()[1].caps.predicate);

    // An exact operator on this path must resolve to the typed index.
    exec::ExprIndexRequirement eq_req{
        .field_id = json_fid,
        .reader = exec::RequiredReader::Predicate,
        .value_type = DataType::VARCHAR,
        .is_json_field = true,
        .json_path = "/a",
        .accepts_candidates = false,
    };
    auto eq_decision = exec::DetermineExecPath(eq_req, capability);
    ASSERT_TRUE(eq_decision.key.has_value());
    EXPECT_EQ(*eq_decision.key,
              (IndexKey{json_fid, IndexIdentity::PreBuiltIndex(22)}));
    milvus::OpContext op_ctx;
    auto pinned_eq = seg->PinIndex(&op_ctx, *eq_decision.key);
    ASSERT_FALSE(pinned_eq.empty());
    EXPECT_EQ(pinned_eq.get(), typed_reader)
        << "equality/range must hit the typed index; an NGRAM hit returns "
           "wrong "
           "rows silently";

    // A pattern operator on the same path does take the NGRAM candidate.
    exec::ExprIndexRequirement prefix_req{
        .field_id = json_fid,
        .reader = exec::RequiredReader::Ngram,
        .value_type = DataType::VARCHAR,
        .is_json_field = true,
        .json_path = "/a",
        .accepts_candidates = true,
    };
    auto prefix_decision = exec::DetermineExecPath(prefix_req, capability);
    ASSERT_TRUE(prefix_decision.key.has_value());
    EXPECT_EQ(*prefix_decision.key,
              (IndexKey{json_fid, IndexIdentity::PreBuiltIndex(21)}));
}

// Regression (the observable half of the conservative raw-data flag): the
// field-level flag has to report false while one field carries several indexes.
//
// The flag is per-field and written by whichever index was loaded last, while a
// reader selects and pins one entry -- the two can differ, and a true would let
// the reader reverse-look-up through an index that has no raw data and fail. With
// several indexes it reports false, so readers fall back to the raw column.
TEST(MultiScalarIndexOnSameField, RawDataFlagStaysFalseWithSeveralIndexes) {
    using namespace milvus::segcore;
    auto schema = std::make_shared<Schema>();
    auto pk_fid = schema->AddDebugField("pk", DataType::INT64);
    auto int_fid = schema->AddDebugField("int64_field", DataType::INT64);
    schema->set_primary_field_id(pk_fid);

    auto segment = CreateSealedSegment(
        schema, nullptr, -1, SegcoreConfig::default_config(), false);
    auto seg = dynamic_cast<ChunkedSegmentSealedImpl*>(segment.get());
    ASSERT_NE(seg, nullptr);

    const std::vector<int64_t> values = {1, 2, 3, 4};
    auto load = [&](int64_t index_id) {
        auto opened = milvus::test::consumer::BuildScalarReader<int64_t>(
            int_fid,
            DataType::INT64,
            index::INVERTED_INDEX_TYPE,
            values.size(),
            values.data());
        auto info = MakeMultiIndexLoadInfo(
            std::move(opened), DataType::INT64, int_fid, index_id);
        // Declare the index as carrying raw data explicitly: otherwise the
        // field-level flag is false anyway and this case loses its power (it is
        // about the flag not turning true while several indexes exist).
        LoadResourceRequest request{};
        request.has_raw_data = true;
        info.load_resource_request = request;
        seg->LoadIndex(info);
    };

    // Single index: the flag reports what that index says.
    load(31);
    EXPECT_TRUE(seg->IndexHasRawData(int_fid));

    // Once the second index arrives the field-level flag loses its single meaning
    // and must be false.
    load(32);
    EXPECT_TRUE(seg->HasIndex(int_fid));
    EXPECT_FALSE(seg->IndexHasRawData(int_fid))
        << "with several indexes it must report false, or a reader may "
           "reverse-look-up through an index without raw data";
    // Both predicates have to answer the same thing: the load/skip path (the
    // column-group skip, FillDefaultValueFields) and the read path
    // (bulk_subscript's fallback) read the same semantics. IndexHasRawData is the
    // guarded predicate they all go through -- "HasIndexRawDataFromState("
    // appears only at its definition and inside that guard.
    //
    // HasRawData(field) is deliberately not compared here: with no published
    // index-ready bit it returns true by contract ("the column is the raw data"),
    // a different layer from this case's staged-only LoadIndex.

    // Both indexes are still there and both pin (the conservative flag hurt
    // neither).
    milvus::OpContext op_ctx;
    EXPECT_FALSE(
        seg->PinIndex(&op_ctx,
                      IndexKey{int_fid, IndexIdentity::PreBuiltIndex(31)})
            .empty());
    EXPECT_FALSE(
        seg->PinIndex(&op_ctx,
                      IndexKey{int_fid, IndexIdentity::PreBuiltIndex(32)})
            .empty());
}

// T4/T5/T6: which of a field's several indexes does a filter leaf take? The
// choice is metadata-only (exec::DetermineExecPath over index::ReaderCaps) and
// the pinned reader is not reachable from the expression, so a recording segment
// captures "which inventory entry did the selection layer pin" -- that is the
// layer's outward contract (IndexCapability(field) -> PinIndex(key)).
TEST(MultiScalarIndexOnSameField, SelectionPicksByCapabilityPerLeaf) {
    using namespace milvus::segcore;
    auto schema = std::make_shared<Schema>();
    auto varchar_fid = schema->AddDebugField("varchar", DataType::VARCHAR);

    const std::vector<std::string> rows = {
        "prefix_a", "prefix_b", "other", "prefix_c", "other_d"};
    auto field = milvus::test::expr_index::StringField(rows);

    milvus::test::expr_index::RawFieldFiles raw_files;
    auto segment =
        std::make_unique<RecordingScalarIndexSegment>(schema, /*segment_id=*/1);
    segment->LoadFieldData(raw_files.Prepare(varchar_fid, {field}));

    LoadMultiIndex(
        *segment,
        milvus::test::expr_index::BuildIndex(varchar_fid,
                                             DataType::VARCHAR,
                                             index::INVERTED_INDEX_TYPE,
                                             {field}),
        DataType::VARCHAR,
        varchar_fid,
        kInvertedIndexId);
    LoadMultiIndex(*segment,
                   milvus::test::expr_index::BuildIndex(
                       varchar_fid,
                       DataType::VARCHAR,
                       index::NGRAM_INDEX_TYPE,
                       {field},
                       {{index::MIN_GRAM, 2}, {index::MAX_GRAM, 4}}),
                   DataType::VARCHAR,
                   varchar_fid,
                   kNgramIndexId);

    // The leaf really has a choice: both entries are reported, ascending by
    // index identity.
    const auto capability = segment->IndexCapability(varchar_fid);
    ASSERT_EQ(capability.entries().size(), 2u);
    EXPECT_EQ(capability.entries()[0].key,
              (IndexKey{varchar_fid,
                        IndexIdentity::PreBuiltIndex(kInvertedIndexId)}));
    EXPECT_TRUE(capability.entries()[0].caps.predicate);
    EXPECT_EQ(
        capability.entries()[1].key,
        (IndexKey{varchar_fid, IndexIdentity::PreBuiltIndex(kNgramIndexId)}));
    EXPECT_TRUE(capability.entries()[1].caps.ngram_candidates);

    // T4: equality is not a pattern operator, so the NGRAM candidate is not even
    // a candidate; INVERTED is pinned and there is no fallback to raw data.
    proto::plan::GenericValue equal_value;
    equal_value.set_string_val("prefix_a");
    auto equal_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::Equal,
        equal_value,
        std::vector<proto::plan::GenericValue>());
    segment->pinned_keys_.clear();
    EXPECT_TRUE(milvus::test::CanExprExecuteAllAtOnce(
        equal_expr, segment.get(), rows.size()))
        << "Equal must go through an index, not the raw column";
    ASSERT_EQ(segment->pinned_keys_.size(), 1u)
        << "only the index actually used is pinned";
    EXPECT_EQ(
        segment->pinned_keys_.front(),
        (IndexKey{varchar_fid, IndexIdentity::PreBuiltIndex(kInvertedIndexId)}))
        << "Equal must not pin the NGRAM sibling on the same field";

    auto equal_plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, equal_expr);
    auto equal_result = query::ExecuteQueryExpr(
        equal_plan, segment.get(), rows.size(), MAX_TIMESTAMP);
    EXPECT_EQ(equal_result.count(), 1);
    EXPECT_TRUE(equal_result[0]);

    // T5: LIKE 'prefix%' becomes PrefixMatch plus the literal "prefix", and that
    // is exactly what an NGRAM candidate answers -- so the NGRAM entry is pinned
    // and the INVERTED sibling is not.
    proto::plan::GenericValue prefix_value;
    prefix_value.set_string_val("prefix");
    auto like_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::PrefixMatch,
        prefix_value,
        std::vector<proto::plan::GenericValue>());
    segment->pinned_keys_.clear();
    EXPECT_TRUE(milvus::test::CanExprExecuteAllAtOnce(
        like_expr, segment.get(), rows.size()));
    ASSERT_EQ(segment->pinned_keys_.size(), 1u);
    EXPECT_EQ(
        segment->pinned_keys_.front(),
        (IndexKey{varchar_fid, IndexIdentity::PreBuiltIndex(kNgramIndexId)}))
        << "LIKE must take the NGRAM candidate, not the INVERTED sibling";

    auto like_plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, like_expr);
    auto like_result = query::ExecuteQueryExpr(
        like_plan, segment.get(), rows.size(), MAX_TIMESTAMP);
    // The NGRAM candidate is a SUPERSET, so every row it returns has to be
    // rechecked; the answer must still be exactly the raw-column answer.
    EXPECT_EQ(like_result.count(), 3);
    EXPECT_TRUE(like_result[0]);
    EXPECT_TRUE(like_result[1]);
    EXPECT_FALSE(like_result[2]);
    EXPECT_TRUE(like_result[3]);
    EXPECT_FALSE(like_result[4]);
}

// T6: a field with only an NGRAM index and an equality leaf. An NGRAM index
// serves pattern leaves only, so the leaf must fall back to the raw column rather
// than borrow it.
TEST(MultiScalarIndexOnSameField, EqualNeverUsesAnNgramOnlyField) {
    using namespace milvus::segcore;
    auto schema = std::make_shared<Schema>();
    auto varchar_fid = schema->AddDebugField("varchar", DataType::VARCHAR);

    const std::vector<std::string> rows = {"prefix_a", "other"};
    auto field = milvus::test::expr_index::StringField(rows);

    milvus::test::expr_index::RawFieldFiles raw_files;
    auto segment =
        std::make_unique<RecordingScalarIndexSegment>(schema, /*segment_id=*/2);
    segment->LoadFieldData(raw_files.Prepare(varchar_fid, {field}));
    LoadMultiIndex(*segment,
                   milvus::test::expr_index::BuildIndex(
                       varchar_fid,
                       DataType::VARCHAR,
                       index::NGRAM_INDEX_TYPE,
                       {field},
                       {{index::MIN_GRAM, 2}, {index::MAX_GRAM, 4}}),
                   DataType::VARCHAR,
                   varchar_fid,
                   kNgramIndexId);

    const auto capability = segment->IndexCapability(varchar_fid);
    ASSERT_EQ(capability.entries().size(), 1u);
    EXPECT_TRUE(capability.entries()[0].caps.ngram_candidates);
    // A whole-field NGRAM index still folds into the field's readiness bit (a
    // JSON NGRAM one does too; only JSON path indexes without NGRAM stay out), so
    // the leaf does consider the index path before metadata rules it out.
    EXPECT_TRUE(segment->HasIndex(varchar_fid));

    // The segment-side half of the rule: metadata alone says no entry can serve
    // an equality predicate.
    exec::ExprIndexRequirement eq_req{
        .field_id = varchar_fid,
        .reader = exec::RequiredReader::Predicate,
        .value_type = DataType::VARCHAR,
        .is_json_field = false,
        .accepts_candidates = false,
    };
    EXPECT_FALSE(exec::DetermineExecPath(eq_req, capability).key.has_value());

    proto::plan::GenericValue equal_value;
    equal_value.set_string_val("prefix_a");
    auto equal_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(varchar_fid, DataType::VARCHAR),
        proto::plan::OpType::Equal,
        equal_value,
        std::vector<proto::plan::GenericValue>());
    segment->pinned_keys_.clear();
    EXPECT_FALSE(milvus::test::CanExprExecuteAllAtOnce(
        equal_expr, segment.get(), rows.size()))
        << "NGRAM cannot serve Equal => fall back to RawData";
    EXPECT_TRUE(segment->pinned_keys_.empty())
        << "Equal must not even try to pin the NGRAM index";

    auto equal_plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, equal_expr);
    auto equal_result = query::ExecuteQueryExpr(
        equal_plan, segment.get(), rows.size(), MAX_TIMESTAMP);
    EXPECT_EQ(equal_result.count(), 1);
    EXPECT_TRUE(equal_result[0]);
}
