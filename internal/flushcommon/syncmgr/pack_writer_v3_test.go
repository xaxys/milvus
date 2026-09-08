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

package syncmgr

import (
	"context"
	"fmt"
	"math"
	"path"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/flushcommon/metacache"
	"github.com/milvus-io/milvus/internal/flushcommon/metacache/pkoracle"
	"github.com/milvus-io/milvus/internal/mocks/flushcommon/mock_util"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagecommon"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/internal/util/initcore"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/objectstorage"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func TestPackWriterV3Suite(t *testing.T) {
	suite.Run(t, new(PackWriterV3Suite))
}

type PackWriterV3Suite struct {
	suite.Suite

	ctx          context.Context
	mockID       atomic.Int64
	logIDAlloc   allocator.Interface
	mockBinlogIO *mock_util.MockBinlogIO

	schema        *schemapb.CollectionSchema
	cm            storage.ChunkManager
	rootPath      string
	maxRowNum     int64
	chunkSize     uint64
	currentSplit  []storagecommon.ColumnGroup
	storageConfig *indexpb.StorageConfig
}

func (s *PackWriterV3Suite) SetupTest() {
	s.ctx = context.Background()
	s.logIDAlloc = allocator.NewLocalAllocator(1, math.MaxInt64)
	s.rootPath = s.T().TempDir()
	initcore.CleanArrowFileSystem()
	initcore.InitLocalArrowFileSystem(s.rootPath)
	paramtable.Get().Init(paramtable.NewBaseTable())
	paramtable.Get().Save(paramtable.Get().MinioCfg.RootPath.Key, "/tmp")
	paramtable.Get().Save(paramtable.Get().CommonCfg.StorageType.Key, "local")
	// force use v3
	paramtable.Get().Save(paramtable.Get().CommonCfg.UseLoonFFI.Key, "true")

	s.schema = &schemapb.CollectionSchema{
		Name: "sync_task_test_col",
		Fields: []*schemapb.FieldSchema{
			{FieldID: common.RowIDField, DataType: schemapb.DataType_Int64},
			{FieldID: common.TimeStampField, DataType: schemapb.DataType_Int64},
			{
				FieldID:      100,
				Name:         "pk",
				DataType:     schemapb.DataType_Int64,
				IsPrimaryKey: true,
			},
			{
				FieldID:  101,
				Name:     "vector",
				DataType: schemapb.DataType_FloatVector,
				TypeParams: []*commonpb.KeyValuePair{
					{Key: common.DimKey, Value: "128"},
				},
			},
		},
		StructArrayFields: []*schemapb.StructArrayFieldSchema{
			{
				FieldID: 102,
				Name:    "struct_array",
				Fields: []*schemapb.FieldSchema{
					{
						FieldID:     103,
						Name:        "vector_array",
						DataType:    schemapb.DataType_ArrayOfVector,
						ElementType: schemapb.DataType_FloatVector,
						TypeParams: []*commonpb.KeyValuePair{
							{Key: common.DimKey, Value: "128"},
						},
					},
				},
			},
		},
	}
	allFields := typeutil.GetAllFieldSchemas(s.schema)
	s.currentSplit = storagecommon.SplitColumns(allFields, map[int64]storagecommon.ColumnStats{}, storagecommon.NewSelectedDataTypePolicy(), storagecommon.NewRemanentShortPolicy(-1))
	s.cm = storage.NewLocalChunkManager(objectstorage.RootPath(s.rootPath))
	s.storageConfig = &indexpb.StorageConfig{
		StorageType: "local",
		RootPath:    s.rootPath,
	}
}

func (s *PackWriterV3Suite) TearDownTest() {
	paramtable.Get().Reset(paramtable.Get().CommonCfg.UseLoonFFI.Key)
	paramtable.Get().Reset(paramtable.Get().MinioCfg.RootPath.Key)
	paramtable.Get().Reset(paramtable.Get().CommonCfg.StorageType.Key)
}

func (s *PackWriterV3Suite) TestPackWriterV3_Write() {
	collectionID := int64(123)
	partitionID := int64(456)
	segmentID := int64(789)
	channelName := fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)
	rows := 10

	bfs := pkoracle.NewBloomFilterSet()

	k := metautil.JoinIDPath(collectionID, partitionID, segmentID)
	basePath := path.Join(common.SegmentInsertLogPath, k)
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{
		ManifestPath: manifestPath,
	}, bfs, nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(1000)(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSchema(mock.Anything).Return(s.schema).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()
	mc.EXPECT().GetSegmentsBy(mock.Anything, mock.Anything).Return([]*metacache.SegmentInfo{seg}).Maybe()
	mc.EXPECT().UpdateSegments(mock.Anything, mock.Anything).Run(func(action metacache.SegmentAction, filters ...metacache.SegmentFilter) {
		action(seg)
	}).Return().Maybe()

	deletes := &storage.DeleteData{}
	for i := 0; i < rows; i++ {
		pk := storage.NewInt64PrimaryKey(int64(i + 1))
		ts := uint64(100 + i)
		deletes.Append(pk, ts)
	}

	pack := new(SyncPack).WithCollectionID(collectionID).WithPartitionID(partitionID).WithSegmentID(segmentID).WithChannelName(channelName).WithInsertData(genInsertData(rows, s.schema)).WithDeleteData(deletes)

	parquetSplit := storagecommon.FillColumnGroupFormats(s.currentSplit, "parquet")
	bw := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, parquetSplit, manifestPath)

	gotInserts, _, _, _, writtenManifestPath, _, _, err := bw.Write(context.Background(), pack)
	s.NoError(err)
	s.Equal(gotInserts[0].Binlogs[0].GetEntriesNum(), int64(rows))
	s.Equal("parquet", gotInserts[0].GetFormat())
	writtenBasePath, revision, err := packed.UnmarshalManifestPath(writtenManifestPath)
	s.NoError(err)
	s.Equal(basePath, writtenBasePath)
	s.Greater(revision, int64(0))
}

func (s *PackWriterV3Suite) TestPackWriterV3_UsesManifestFormatAfterConfigSwitch() {
	params := paramtable.Get()
	s.Require().NoError(params.Save(params.DataNodeCfg.StorageFormat.Key, "parquet"))
	s.T().Cleanup(func() {
		_ = params.Reset(params.DataNodeCfg.StorageFormat.Key)
	})

	collectionID := int64(123)
	partitionID := int64(456)
	segmentID := int64(789)
	channelName := fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)

	bfs := pkoracle.NewBloomFilterSet()
	k := metautil.JoinIDPath(collectionID, partitionID, segmentID)
	basePath := path.Join(common.SegmentInsertLogPath, k)
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{
		ManifestPath: manifestPath,
	}, bfs, nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(1000)(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSchema(mock.Anything).Return(s.schema).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()
	mc.EXPECT().GetSegmentsBy(mock.Anything, mock.Anything).Return([]*metacache.SegmentInfo{seg}).Maybe()
	mc.EXPECT().UpdateSegments(mock.Anything, mock.Anything).Run(func(action metacache.SegmentAction, filters ...metacache.SegmentFilter) {
		action(seg)
	}).Return().Maybe()

	parquetSplit := storagecommon.FillColumnGroupFormats(s.currentSplit, "parquet")
	bw := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, parquetSplit, manifestPath)

	firstPack := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(channelName).
		WithInsertData(genInsertData(5, s.schema))
	_, _, _, _, firstManifestPath, _, _, err := bw.Write(context.Background(), firstPack)
	s.Require().NoError(err)

	format, err := packed.ResolveManifestSingleWriterFormat(firstManifestPath, s.storageConfig, nil, "")
	s.Require().NoError(err)
	s.Equal("parquet", format)

	s.Require().NoError(params.Save(params.DataNodeCfg.StorageFormat.Key, "vortex"))
	bw.initialManifestPath = firstManifestPath
	writerFormat, schemaBasedFormats, err := bw.resolveInsertWriterFormats()
	s.Require().NoError(err)
	s.Equal("vortex", writerFormat)
	s.Require().Len(schemaBasedFormats, len(parquetSplit))
	for _, schemaBasedFormat := range schemaBasedFormats {
		s.Equal("parquet", schemaBasedFormat)
	}

	secondPack := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(channelName).
		WithInsertData(genInsertData(5, s.schema))
	_, _, _, _, secondManifestPath, _, _, err := bw.Write(context.Background(), secondPack)
	s.Require().NoError(err)

	format, err = packed.ResolveManifestSingleWriterFormat(secondManifestPath, s.storageConfig, nil, "")
	s.Require().NoError(err)
	s.Equal("parquet", format)
}

func (s *PackWriterV3Suite) TestResolveInsertWriterFormatsUsesColumnGroupFormatsWithoutReadingManifest() {
	params := paramtable.Get()
	s.Require().NoError(params.Save(params.DataNodeCfg.StorageFormat.Key, "vortex"))
	s.T().Cleanup(func() {
		_ = params.Reset(params.DataNodeCfg.StorageFormat.Key)
	})

	columnGroups := []storagecommon.ColumnGroup{
		{GroupID: 0, Columns: []int{0, 2}, Fields: []int64{common.RowIDField, 100}, Format: "parquet"},
		{GroupID: 101, Columns: []int{1}, Fields: []int64{101}, Format: "vortex"},
	}
	bw := NewBulkPackWriterV3(nil, s.schema, s.cm, s.logIDAlloc,
		packed.DefaultWriteBufferSize, 0, s.storageConfig, columnGroups,
		packed.MarshalManifestPath("files/missing-manifest-segment", 42))
	bw.initialManifestPath = bw.manifestPath

	writerFormat, schemaBasedFormats, err := bw.resolveInsertWriterFormats()
	s.Require().NoError(err)
	s.Equal("vortex", writerFormat)
	s.Equal([]string{"parquet", "vortex"}, schemaBasedFormats)
}

func (s *PackWriterV3Suite) TestResolveInsertWriterFormatsRequiresFormatsForExistingManifest() {
	params := paramtable.Get()
	s.Require().NoError(params.Save(params.DataNodeCfg.StorageFormat.Key, "vortex"))
	s.T().Cleanup(func() {
		_ = params.Reset(params.DataNodeCfg.StorageFormat.Key)
	})

	columnGroups := []storagecommon.ColumnGroup{
		{GroupID: 0, Columns: []int{0, 2}, Fields: []int64{common.RowIDField, 100}},
	}
	bw := NewBulkPackWriterV3(nil, s.schema, s.cm, s.logIDAlloc,
		packed.DefaultWriteBufferSize, 0, s.storageConfig, columnGroups,
		packed.MarshalManifestPath("files/existing-manifest-segment", 42))
	bw.initialManifestPath = bw.manifestPath

	_, _, err := bw.resolveInsertWriterFormats()
	s.Require().Error(err)
	s.Contains(err.Error(), "missing format")
}

func (s *PackWriterV3Suite) TestWriteEmptyInsertData() {
	s.logIDAlloc = allocator.NewLocalAllocator(1, 1)
	collectionID := int64(123)
	partitionID := int64(456)
	segmentID := int64(789)
	channelName := fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)

	k := metautil.JoinIDPath(collectionID, partitionID, segmentID)
	basePath := path.Join(common.SegmentInsertLogPath, k)
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	bfs := pkoracle.NewBloomFilterSet()
	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{ManifestPath: manifestPath}, bfs, nil, metacache.NewEmptySegmentStats())
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().GetSchema(mock.Anything).Return(s.schema).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()

	pack := new(SyncPack).WithCollectionID(collectionID).WithPartitionID(partitionID).WithSegmentID(segmentID).WithChannelName(channelName)
	bw := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)

	_, _, _, _, _, _, _, err := bw.Write(context.Background(), pack)
	s.NoError(err)
}

func (s *PackWriterV3Suite) TestNoPkField() {
	s.schema = &schemapb.CollectionSchema{
		Name: "no pk field",
		Fields: []*schemapb.FieldSchema{
			{FieldID: common.RowIDField, DataType: schemapb.DataType_Int64},
			{FieldID: common.TimeStampField, DataType: schemapb.DataType_Int64},
		},
	}
	s.logIDAlloc = allocator.NewLocalAllocator(1, math.MaxInt64)
	collectionID := int64(123)
	partitionID := int64(456)
	segmentID := int64(789)

	k := metautil.JoinIDPath(collectionID, partitionID, segmentID)
	basePath := path.Join(common.SegmentInsertLogPath, k)
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	channelName := fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().GetSchema(mock.Anything).Return(s.schema).Maybe()

	buf, _ := storage.NewInsertData(s.schema)
	data := make(map[storage.FieldID]any)
	data[common.RowIDField] = int64(1)
	data[common.TimeStampField] = int64(1)
	buf.Append(data)

	pack := new(SyncPack).WithCollectionID(collectionID).WithPartitionID(partitionID).WithSegmentID(segmentID).WithChannelName(channelName).WithInsertData([]*storage.InsertData{buf})
	bw := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)

	_, _, _, _, _, _, _, err := bw.Write(context.Background(), pack)
	s.Error(err)
}

func (s *PackWriterV3Suite) TestWriteInsertDataError() {
	s.logIDAlloc = allocator.NewLocalAllocator(1, math.MaxInt64)
	collectionID := int64(123)
	partitionID := int64(456)
	segmentID := int64(789)
	channelName := fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)

	k := metautil.JoinIDPath(collectionID, partitionID, segmentID)
	basePath := path.Join(common.SegmentInsertLogPath, k)
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().GetSchema(mock.Anything).Return(s.schema).Maybe()

	buf, _ := storage.NewInsertData(s.schema)
	data := make(map[storage.FieldID]any)
	data[common.RowIDField] = int64(1)
	buf.Append(data)

	pack := new(SyncPack).WithCollectionID(collectionID).WithPartitionID(partitionID).WithSegmentID(segmentID).WithChannelName(channelName).WithInsertData([]*storage.InsertData{buf})
	bw := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)

	_, _, _, _, _, _, _, err := bw.Write(context.Background(), pack)
	s.Error(err)
}

func (s *PackWriterV3Suite) TestInvalidManifestPath() {
	s.logIDAlloc = allocator.NewLocalAllocator(1, math.MaxInt64)
	collectionID := int64(123)
	partitionID := int64(456)
	segmentID := int64(789)
	channelName := fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)
	rows := 10

	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().GetSchema(mock.Anything).Return(s.schema).Maybe()

	// Use an invalid manifest path (not a valid JSON)
	invalidManifestPath := "invalid-manifest-path"

	pack := new(SyncPack).WithCollectionID(collectionID).WithPartitionID(partitionID).WithSegmentID(segmentID).WithChannelName(channelName).WithInsertData(genInsertData(rows, s.schema))
	bw := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, invalidManifestPath)

	_, _, _, _, _, _, _, err := bw.Write(context.Background(), pack)
	s.Error(err)
}

func (s *PackWriterV3Suite) TestWriteWithDeleteData() {
	collectionID := int64(123)
	partitionID := int64(456)
	segmentID := int64(789)
	channelName := fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)
	rows := 10

	bfs := pkoracle.NewBloomFilterSet()

	k := metautil.JoinIDPath(collectionID, partitionID, segmentID)
	basePath := path.Join(common.SegmentInsertLogPath, k)
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{
		ManifestPath: manifestPath,
	}, bfs, nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(1000)(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSchema(mock.Anything).Return(s.schema).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()
	mc.EXPECT().GetSegmentsBy(mock.Anything, mock.Anything).Return([]*metacache.SegmentInfo{seg}).Maybe()
	mc.EXPECT().UpdateSegments(mock.Anything, mock.Anything).Run(func(action metacache.SegmentAction, filters ...metacache.SegmentFilter) {
		action(seg)
	}).Return().Maybe()

	deletes := &storage.DeleteData{}
	for i := 0; i < rows; i++ {
		pk := storage.NewInt64PrimaryKey(int64(i + 1))
		ts := uint64(100 + i)
		deletes.Append(pk, ts)
	}

	// Test with only delete data (no inserts)
	pack := new(SyncPack).WithCollectionID(collectionID).WithPartitionID(partitionID).WithSegmentID(segmentID).WithChannelName(channelName).WithDeleteData(deletes)

	bw := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)

	gotInserts, gotDeletes, _, _, writtenManifestPath, _, _, err := bw.Write(context.Background(), pack)
	s.NoError(err)
	s.Equal(0, len(gotInserts)) // No insert binlogs when only deletes
	// For V3, delta summary is returned for compaction trigger (no path, only stats)
	s.NotNil(gotDeletes)
	s.Equal(1, len(gotDeletes.GetBinlogs()))
	s.EqualValues(int64(rows), gotDeletes.GetBinlogs()[0].GetEntriesNum())
	s.NotZero(gotDeletes.GetBinlogs()[0].GetLogID())
	s.Empty(gotDeletes.GetBinlogs()[0].GetLogPath())
	// Verify manifest was updated (version should be > -1)
	_, revision, err := packed.UnmarshalManifestPath(writtenManifestPath)
	s.NoError(err)
	s.Greater(revision, int64(-1))
}

func (s *PackWriterV3Suite) TestV3InheritsV2Fields() {
	collectionID := int64(123)
	partitionID := int64(456)
	segmentID := int64(789)

	k := metautil.JoinIDPath(collectionID, partitionID, segmentID)
	basePath := path.Join(common.SegmentInsertLogPath, k)
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	mc := metacache.NewMockMetaCache(s.T())

	// Create V3 writer and verify it has access to V2 fields
	bw := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, packed.DefaultMultiPartUploadSize, s.storageConfig, s.currentSplit, manifestPath)

	// Verify V3 can access fields from embedded V2
	s.Equal(s.schema, bw.schema)
	s.Equal(s.cm, bw.chunkManager)
	s.Equal(s.logIDAlloc, bw.allocator)
	s.EqualValues(packed.DefaultWriteBufferSize, bw.bufferSize) // 0 was passed for bufferSize (default write buffer)
	s.EqualValues(packed.DefaultMultiPartUploadSize, bw.multiPartUploadSize)
	s.Equal(s.currentSplit, bw.columnGroups)
	s.Equal(manifestPath, bw.manifestPath)
}

// genInsertDataWithPKOffset generates insert data with PKs starting at pkOffset+1.
func genInsertDataWithPKOffset(size int, pkOffset int, schema *schemapb.CollectionSchema) []*storage.InsertData {
	buf, _ := storage.NewInsertData(schema)
	for i := 0; i < size; i++ {
		data := make(map[storage.FieldID]any)
		data[common.RowIDField] = int64(pkOffset + i + 1)
		data[common.TimeStampField] = int64(pkOffset + i + 1)
		data[100] = int64(pkOffset + i + 1) // pk field

		vector := make([]float32, 128)
		for j := range vector {
			vector[j] = float32(i+j) * 0.01
		}
		data[101] = vector

		vectorData := make([]float32, 2*128)
		for j := range vectorData {
			vectorData[j] = float32(i+j) * 0.001
		}
		vectorArray := &schemapb.VectorField{
			Dim: 128,
			Data: &schemapb.VectorField_FloatVector{
				FloatVector: &schemapb.FloatArray{Data: vectorData},
			},
		}
		data[103] = vectorArray
		buf.Append(data)
	}
	return []*storage.InsertData{buf}
}

// TestMultiBatchStatsAccumulation verifies that bloom filter and BM25 stat files
// from multiple batches accumulate in the manifest rather than being replaced.
// This is the regression test for the bug where loon_transaction_update_stat
// replaced the previous batch's stat entry, leaving only the last batch's
// bloom filter in the manifest.
func (s *PackWriterV3Suite) TestMultiBatchStatsAccumulation() {
	collectionID := int64(123)
	partitionID := int64(456)
	segmentID := int64(10001) // unique ID to avoid manifest collision with other tests
	channelName := fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)
	batchRows := 5

	bfs := pkoracle.NewBloomFilterSet()

	k := metautil.JoinIDPath(collectionID, partitionID, segmentID)
	basePath := path.Join(common.SegmentInsertLogPath, k)
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{
		ManifestPath: manifestPath,
	}, bfs, nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(1000)(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSchema(mock.Anything).Return(s.schema).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()
	mc.EXPECT().GetSegmentsBy(mock.Anything, mock.Anything).Return([]*metacache.SegmentInfo{seg}).Maybe()
	mc.EXPECT().UpdateSegments(mock.Anything, mock.Anything).Run(func(action metacache.SegmentAction, filters ...metacache.SegmentFilter) {
		action(seg)
	}).Return().Maybe()

	bfKey := fmt.Sprintf("bloom_filter.%d", 100) // pk field ID

	// Batch 1: PKs 1..5
	pack1 := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(channelName).
		WithInsertData(genInsertDataWithPKOffset(batchRows, 0, s.schema)).
		WithBatchRows(int64(batchRows))

	bw1 := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)
	_, _, _, _, manifest1, _, _, err := bw1.Write(context.Background(), pack1)
	s.Require().NoError(err)
	currentSplit := storagecommon.FillColumnGroupFormats(s.currentSplit, paramtable.Get().DataNodeCfg.StorageFormat.GetValue())

	stats1, err := packed.GetManifestStats(manifest1, s.storageConfig)
	s.Require().NoError(err)
	count1 := len(stats1[bfKey].Paths)
	s.Greater(count1, 0, "batch 1 should produce bloom filter files")

	// Batch 2: PKs 6..10, starting from the manifest written by batch 1
	pack2 := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(channelName).
		WithInsertData(genInsertDataWithPKOffset(batchRows, batchRows, s.schema)).
		WithBatchRows(int64(batchRows))

	bw2 := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, currentSplit, manifest1)
	_, _, _, _, manifest2, _, _, err := bw2.Write(context.Background(), pack2)
	s.Require().NoError(err)

	stats2, err := packed.GetManifestStats(manifest2, s.storageConfig)
	s.Require().NoError(err)
	count2 := len(stats2[bfKey].Paths)
	s.Greater(count2, count1, "batch 2 should accumulate more bloom filter files than batch 1")

	// Batch 3: PKs 11..15
	pack3 := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(channelName).
		WithInsertData(genInsertDataWithPKOffset(batchRows, batchRows*2, s.schema)).
		WithBatchRows(int64(batchRows))

	bw3 := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, currentSplit, manifest2)
	_, _, _, _, manifest3, _, _, err := bw3.Write(context.Background(), pack3)
	s.Require().NoError(err)

	stats3, err := packed.GetManifestStats(manifest3, s.storageConfig)
	s.Require().NoError(err)
	count3 := len(stats3[bfKey].Paths)
	s.Greater(count3, count2, "batch 3 should accumulate more bloom filter files than batch 2")
	s.NotEmpty(stats3[bfKey].Metadata["memory_size"], "memory_size metadata should be set")
}

// TestFinalFlushWritesCompoundOnly verifies the H22/H23 contract on a brand
// new non-L0 final flush: no allocator call is needed for a batch bloom object,
// the manifest publishes only the compound path, and size accounting reflects
// the object the resolver will actually load.
func (s *PackWriterV3Suite) TestFinalFlushWritesCompoundOnly() {
	collectionID := int64(124)
	partitionID := int64(457)
	segmentID := int64(10003)
	rows := 10
	basePath := path.Join(common.SegmentInsertLogPath,
		metautil.JoinIDPath(collectionID, partitionID, segmentID))
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{ManifestPath: manifestPath},
		pkoracle.NewBloomFilterSet(), nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(int64(rows))(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()
	alloc := allocator.NewMockAllocator(s.T())

	pack := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)).
		WithInsertData(genInsertDataWithPKOffset(rows, 0, s.schema)).
		WithBatchRows(int64(rows)).
		WithLevel(datapb.SegmentLevel_L1).
		WithFlush()

	bw := NewBulkPackWriterV3(mc, s.schema, s.cm, alloc,
		packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)
	_, _, _, _, writtenManifest, _, segmentStats, err := bw.Write(context.Background(), pack)
	s.Require().NoError(err)

	manifestStats, err := packed.GetManifestStats(writtenManifest, s.storageConfig)
	s.Require().NoError(err)
	bloom := manifestStats["bloom_filter.100"]
	s.Require().Len(bloom.Paths, 1)
	s.Equal(storage.CompoundStatsType.LogIdx(), path.Base(bloom.Paths[0]))
	s.Equal("bloom_filter.100", path.Base(path.Dir(path.Dir(bloom.Paths[0]))))
	s.Regexp(`^compound-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`,
		path.Base(path.Dir(bloom.Paths[0])))

	compoundBlob, err := packed.ReadFile(s.storageConfig, bloom.Paths[0])
	s.Require().NoError(err)
	memorySize, err := strconv.ParseInt(bloom.Metadata["memory_size"], 10, 64)
	s.Require().NoError(err)
	s.EqualValues(len(compoundBlob), memorySize)
	s.Equal(memorySize, segmentStats.GetStatsBinlogSize())

	decoded, err := storage.DeserializeBloomFilterStats(
		bloom.Paths, []*storage.Blob{{Value: compoundBlob}})
	s.Require().NoError(err)
	s.Require().Len(decoded, 1)
	assertInt64PKMembership(s.T(), decoded, 1, int64(rows))
}

// TestFinalCompoundPathsIsolateHandoffWriters simulates a new channel owner
// writing first and an old owner finishing late from the same initial manifest.
// Their compound objects must have different keys, so the late write cannot
// change the bytes already published by the new owner.
func (s *PackWriterV3Suite) TestFinalCompoundPathsIsolateHandoffWriters() {
	collectionID := int64(129)
	partitionID := int64(462)
	segmentID := int64(10009)
	basePath := path.Join(common.SegmentInsertLogPath,
		metautil.JoinIDPath(collectionID, partitionID, segmentID))
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{ManifestPath: manifestPath},
		pkoracle.NewBloomFilterSet(), nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(10)(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()

	newFinalPack := func(rows, offset int) *SyncPack {
		return new(SyncPack).
			WithCollectionID(collectionID).
			WithPartitionID(partitionID).
			WithSegmentID(segmentID).
			WithChannelName(fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)).
			WithInsertData(genInsertDataWithPKOffset(rows, offset, s.schema)).
			WithBatchRows(int64(rows)).
			WithLevel(datapb.SegmentLevel_L1).
			WithFlush()
	}

	newOwner := NewBulkPackWriterV3(mc, s.schema, s.cm, allocator.NewMockAllocator(s.T()),
		packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)
	newOwner.initialManifestPath = manifestPath
	newEntries, err := newOwner.writeStats(context.Background(), newFinalPack(7, 100), basePath)
	s.Require().NoError(err)
	s.Require().Len(newEntries, 1)
	s.Require().Len(newEntries[0].Files, 1)
	newPath := newEntries[0].Files[0]
	newBytesBeforeLateWrite, err := packed.ReadFile(s.storageConfig, newPath)
	s.Require().NoError(err)

	// This represents the old channel owner's background task completing after
	// the new owner has already written its compound object.
	oldOwner := NewBulkPackWriterV3(mc, s.schema, s.cm, allocator.NewMockAllocator(s.T()),
		packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)
	oldOwner.initialManifestPath = manifestPath
	oldEntries, err := oldOwner.writeStats(context.Background(), newFinalPack(3, 0), basePath)
	s.Require().NoError(err)
	s.Require().Len(oldEntries, 1)
	s.Require().Len(oldEntries[0].Files, 1)
	oldPath := oldEntries[0].Files[0]
	s.NotEqual(newPath, oldPath)

	newBytesAfterLateWrite, err := packed.ReadFile(s.storageConfig, newPath)
	s.Require().NoError(err)
	s.Equal(newBytesBeforeLateWrite, newBytesAfterLateWrite,
		"a late old writer must not overwrite the new owner's compound bloom")
	oldBytes, err := packed.ReadFile(s.storageConfig, oldPath)
	s.Require().NoError(err)
	s.NotEqual(newBytesAfterLateWrite, oldBytes)

	newStats, err := storage.DeserializeBloomFilterStats(
		[]string{newPath}, []*storage.Blob{{Value: newBytesAfterLateWrite}})
	s.Require().NoError(err)
	assertInt64PKMembership(s.T(), newStats, 101, 107)
	oldStats, err := storage.DeserializeBloomFilterStats(
		[]string{oldPath}, []*storage.Blob{{Value: oldBytes}})
	s.Require().NoError(err)
	assertInt64PKMembership(s.T(), oldStats, 1, 3)
}

func (s *PackWriterV3Suite) TestEmptyFinalMigratesLegacySharedCompoundPath() {
	collectionID := int64(130)
	partitionID := int64(463)
	segmentID := int64(10010)
	rows := 5
	basePath := path.Join(common.SegmentInsertLogPath,
		metautil.JoinIDPath(collectionID, partitionID, segmentID))
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{ManifestPath: manifestPath},
		pkoracle.NewBloomFilterSet(), nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(int64(rows))(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()

	sourcePack := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)).
		WithInsertData(genInsertDataWithPKOffset(rows, 0, s.schema)).
		WithBatchRows(int64(rows)).
		WithLevel(datapb.SegmentLevel_L1).
		WithFlush()
	serializer, err := NewStorageSerializer(mc, s.schema)
	s.Require().NoError(err)
	pkStats, err := serializer.buildPrimaryKeyStats(sourcePack)
	s.Require().NoError(err)
	legacyBlob, err := serializer.serializeMergedPkStatsList(
		[]*storage.PrimaryKeyStats{pkStats}, int64(rows))
	s.Require().NoError(err)

	legacyPath := path.Join(basePath, "_stats/bloom_filter.100", storage.CompoundStatsType.LogIdx())
	s.False(isIsolatedCompoundBloomPath(legacyPath))
	s.Require().NoError(packed.WriteFile(s.storageConfig, legacyPath, legacyBlob.Value))
	legacyManifest, err := packed.CommitManifestUpdates(basePath, packed.ManifestEarliest, s.storageConfig,
		&packed.ManifestUpdates{Stats: []packed.StatEntry{{
			Key:      "bloom_filter.100",
			Files:    []string{legacyPath},
			Metadata: map[string]string{"memory_size": strconv.Itoa(len(legacyBlob.Value))},
		}}})
	s.Require().NoError(err)

	emptyFinal := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)).
		WithLevel(datapb.SegmentLevel_L1).
		WithFlush()
	writer := NewBulkPackWriterV3(mc, s.schema, s.cm, allocator.NewMockAllocator(s.T()),
		packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, legacyManifest)
	writer.initialManifestPath = legacyManifest
	entries, err := writer.writeStats(context.Background(), emptyFinal, basePath)
	s.Require().NoError(err)
	s.Require().Len(entries, 1)
	s.Require().Len(entries[0].Files, 1)
	migratedPath := entries[0].Files[0]
	s.NotEqual(legacyPath, migratedPath)
	s.True(isIsolatedCompoundBloomPath(migratedPath))

	migratedBytes, err := packed.ReadFile(s.storageConfig, migratedPath)
	s.Require().NoError(err)
	migratedStats, err := storage.DeserializeBloomFilterStats(
		[]string{migratedPath}, []*storage.Blob{{Value: migratedBytes}})
	s.Require().NoError(err)
	assertInt64PKMembership(s.T(), migratedStats, 1, int64(rows))
}

func (s *PackWriterV3Suite) TestL0FinalFlushKeepsBatchBloom() {
	collectionID := int64(126)
	partitionID := int64(459)
	segmentID := int64(10005)
	rows := 4
	basePath := path.Join(common.SegmentInsertLogPath,
		metautil.JoinIDPath(collectionID, partitionID, segmentID))
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)
	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{ManifestPath: manifestPath},
		pkoracle.NewBloomFilterSet(), nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(int64(rows))(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()

	pack := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)).
		WithInsertData(genInsertDataWithPKOffset(rows, 0, s.schema)).
		WithBatchRows(int64(rows)).
		WithLevel(datapb.SegmentLevel_L0).
		WithFlush()

	bw := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc,
		packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)
	_, _, _, _, writtenManifest, _, _, err := bw.Write(context.Background(), pack)
	s.Require().NoError(err)
	manifestStats, err := packed.GetManifestStats(writtenManifest, s.storageConfig)
	s.Require().NoError(err)
	bloom := manifestStats["bloom_filter.100"]
	s.Require().Len(bloom.Paths, 1)
	s.NotEqual(storage.CompoundStatsType.LogIdx(), path.Base(bloom.Paths[0]))
}

func (s *PackWriterV3Suite) TestFinalFlushCompoundOnlyWithStringPK() {
	s.schema = &schemapb.CollectionSchema{
		Name: "string_pk",
		Fields: []*schemapb.FieldSchema{
			{FieldID: common.RowIDField, DataType: schemapb.DataType_Int64},
			{FieldID: common.TimeStampField, DataType: schemapb.DataType_Int64},
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_VarChar, IsPrimaryKey: true},
			{
				FieldID: 101, Name: "vector", DataType: schemapb.DataType_FloatVector,
				TypeParams: []*commonpb.KeyValuePair{{Key: common.DimKey, Value: "8"}},
			},
		},
	}
	s.currentSplit = storagecommon.SplitColumns(typeutil.GetAllFieldSchemas(s.schema),
		map[int64]storagecommon.ColumnStats{}, storagecommon.NewSelectedDataTypePolicy(), storagecommon.NewRemanentShortPolicy(-1))

	collectionID := int64(127)
	partitionID := int64(460)
	segmentID := int64(10006)
	rows := 3
	basePath := path.Join(common.SegmentInsertLogPath,
		metautil.JoinIDPath(collectionID, partitionID, segmentID))
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)
	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{ManifestPath: manifestPath},
		pkoracle.NewBloomFilterSet(), nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(int64(rows))(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()
	alloc := allocator.NewMockAllocator(s.T())

	pack := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)).
		WithInsertData(s.genStringPKInsertData(rows, s.schema)).
		WithBatchRows(int64(rows)).
		WithLevel(datapb.SegmentLevel_L1).
		WithFlush()
	bw := NewBulkPackWriterV3(mc, s.schema, s.cm, alloc,
		packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)
	_, _, _, _, writtenManifest, _, _, err := bw.Write(context.Background(), pack)
	s.Require().NoError(err)
	decoded := s.readCompoundPKStats(writtenManifest)
	s.Require().Len(decoded, 1)
	bfs := pkoracle.NewBloomFilterSet()
	bfs.Roll(decoded...)
	for i := 0; i < rows; i++ {
		pk := fmt.Sprintf("pk-%d", i)
		s.True(bfs.PkExists(storage.NewLocationsCache(storage.NewVarCharPrimaryKey(pk))),
			"compound bloom missing PK %s", pk)
	}
}

func (s *PackWriterV3Suite) TestEmptyFinalFlushCompactsCommittedBatchHistory() {
	collectionID := int64(128)
	partitionID := int64(461)
	segmentID := int64(10008)
	rows := 5
	basePath := path.Join(common.SegmentInsertLogPath,
		metautil.JoinIDPath(collectionID, partitionID, segmentID))
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)
	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{ManifestPath: manifestPath},
		pkoracle.NewBloomFilterSet(), nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(int64(rows))(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()

	incremental := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)).
		WithInsertData(genInsertDataWithPKOffset(rows, 0, s.schema)).
		WithBatchRows(int64(rows)).
		WithLevel(datapb.SegmentLevel_L1)
	bw1 := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc,
		packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)
	_, _, _, _, manifest1, _, _, err := bw1.Write(context.Background(), incremental)
	s.Require().NoError(err)
	metacache.SetStatistics(bw1.PreparedStats())(seg)
	metacache.UpdateManifestPath(manifest1)(seg)

	finalPack := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)).
		WithLevel(datapb.SegmentLevel_L1).
		WithFlush()
	currentSplit := storagecommon.FillColumnGroupFormats(s.currentSplit, paramtable.Get().DataNodeCfg.StorageFormat.GetValue())
	bw2 := NewBulkPackWriterV3(mc, s.schema, s.cm, allocator.NewMockAllocator(s.T()),
		packed.DefaultWriteBufferSize, 0, s.storageConfig, currentSplit, manifest1)
	_, _, _, _, manifest2, _, segmentStats, err := bw2.Write(context.Background(), finalPack)
	s.Require().NoError(err)
	decoded := s.readCompoundPKStats(manifest2)
	s.Require().Len(decoded, 1)
	assertInt64PKMembership(s.T(), decoded, 1, int64(rows))
	manifestFootprint, err := packed.StatsBinlogSizeFromManifest(manifest2, s.storageConfig)
	s.Require().NoError(err)
	s.Equal(manifestFootprint, segmentStats.GetStatsBinlogSize())
}

// TestFinalFlushRecoversMissingStatsFootprint covers a rolling upgrade from a
// pre-Statistics DataNode. Its committed V3 manifest contains the prior batch
// bloom, but DataCoord's legacy array fallback persisted StatsBinlogSize=0
// because V3 does not publish stats-log arrays.
func (s *PackWriterV3Suite) TestFinalFlushRecoversMissingStatsFootprint() {
	collectionID := int64(131)
	partitionID := int64(464)
	segmentID := int64(10011)
	rows := 5
	basePath := path.Join(common.SegmentInsertLogPath,
		metautil.JoinIDPath(collectionID, partitionID, segmentID))
	earliestManifest := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	initialSegment := metacache.NewSegmentInfo(&datapb.SegmentInfo{ManifestPath: earliestManifest},
		pkoracle.NewBloomFilterSet(), nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(int64(rows))(initialSegment)
	initialMetaCache := metacache.NewMockMetaCache(s.T())
	initialMetaCache.EXPECT().Collection().Return(collectionID).Maybe()
	initialMetaCache.EXPECT().GetSegmentByID(segmentID).Return(initialSegment, true).Maybe()

	incremental := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)).
		WithInsertData(genInsertDataWithPKOffset(rows, 0, s.schema)).
		WithBatchRows(int64(rows)).
		WithLevel(datapb.SegmentLevel_L1)
	initialWriter := NewBulkPackWriterV3(initialMetaCache, s.schema, s.cm, s.logIDAlloc,
		packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, earliestManifest)
	_, _, _, _, batchManifest, _, _, err := initialWriter.Write(context.Background(), incremental)
	s.Require().NoError(err)
	priorFootprint, err := packed.StatsBinlogSizeFromManifest(batchManifest, s.storageConfig)
	s.Require().NoError(err)
	s.Positive(priorFootprint)

	// Simulate recovery from the SegmentInfo written by the old DataNode. Do
	// not install initialWriter.PreparedStats(): the persisted Statistics has a
	// zero StatsBinlogSize despite the batch bloom in batchManifest.
	recoveredStats := &datapb.Statistics{StatsBinlogSize: 0}
	recoveredSegment := metacache.NewSegmentInfo(&datapb.SegmentInfo{
		ID:             segmentID,
		PartitionID:    partitionID,
		NumOfRows:      int64(rows),
		Level:          datapb.SegmentLevel_L1,
		StorageVersion: storage.StorageV3,
		ManifestPath:   batchManifest,
		Stats:          recoveredStats,
	}, pkoracle.NewBloomFilterSet(), nil, metacache.NewSegmentStatsFromStats(recoveredStats, int64(rows)))
	metacache.UpdateBufferedRows(int64(rows))(recoveredSegment)
	recoveredMetaCache := metacache.NewMockMetaCache(s.T())
	recoveredMetaCache.EXPECT().Collection().Return(collectionID).Maybe()
	recoveredMetaCache.EXPECT().GetSegmentByID(segmentID).Return(recoveredSegment, true).Maybe()

	finalPack := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)).
		WithInsertData(genInsertDataWithPKOffset(rows, rows, s.schema)).
		WithBatchRows(int64(rows)).
		WithLevel(datapb.SegmentLevel_L1).
		WithFlush()
	currentSplit := storagecommon.FillColumnGroupFormats(s.currentSplit, paramtable.Get().DataNodeCfg.StorageFormat.GetValue())
	finalWriter := NewBulkPackWriterV3(recoveredMetaCache, s.schema, s.cm, allocator.NewMockAllocator(s.T()),
		packed.DefaultWriteBufferSize, 0, s.storageConfig, currentSplit, batchManifest)
	_, _, _, _, finalManifest, _, segmentStats, err := finalWriter.Write(context.Background(), finalPack)
	s.Require().NoError(err)

	manifestFootprint, err := packed.StatsBinlogSizeFromManifest(finalManifest, s.storageConfig)
	s.Require().NoError(err)
	s.Equal(manifestFootprint, segmentStats.GetStatsBinlogSize())
	s.Require().NotNil(finalWriter.PreparedStats())
	s.Equal(manifestFootprint, finalWriter.PreparedStats().Publish().GetStatsBinlogSize(),
		"the collector installed after DataCoord ack must keep the repaired baseline")
	decoded := s.readCompoundPKStats(finalManifest)
	s.Require().Len(decoded, 2)
	assertInt64PKMembership(s.T(), decoded, 1, int64(rows*2))
}

// TestFinalFlushCompoundCarriesCommittedHistory verifies that the compound is
// built from the committed manifest's prior batch blobs plus the current
// in-memory stats. It also covers a subsequent final flush whose input
// manifest is already compound-only.
func (s *PackWriterV3Suite) TestFinalFlushCompoundCarriesCommittedHistory() {
	collectionID := int64(125)
	partitionID := int64(458)
	segmentID := int64(10004)
	batchRows := 5
	basePath := path.Join(common.SegmentInsertLogPath,
		metautil.JoinIDPath(collectionID, partitionID, segmentID))
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{ManifestPath: manifestPath},
		pkoracle.NewBloomFilterSet(), nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(int64(batchRows * 3))(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()

	newPack := func(offset int, final bool) *SyncPack {
		pack := new(SyncPack).
			WithCollectionID(collectionID).
			WithPartitionID(partitionID).
			WithSegmentID(segmentID).
			WithChannelName(fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)).
			WithInsertData(genInsertDataWithPKOffset(batchRows, offset, s.schema)).
			WithBatchRows(int64(batchRows)).
			WithLevel(datapb.SegmentLevel_L1)
		if final {
			pack.WithFlush()
		}
		return pack
	}

	bw1 := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc,
		packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)
	_, _, _, _, manifest1, _, _, err := bw1.Write(context.Background(), newPack(0, false))
	s.Require().NoError(err)
	metacache.SetStatistics(bw1.PreparedStats())(seg)
	metacache.UpdateManifestPath(manifest1)(seg)
	currentSplit := storagecommon.FillColumnGroupFormats(s.currentSplit, paramtable.Get().DataNodeCfg.StorageFormat.GetValue())

	bw2 := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc,
		packed.DefaultWriteBufferSize, 0, s.storageConfig, currentSplit, manifest1)
	_, _, _, _, manifest2, _, segmentStats2, err := bw2.Write(context.Background(), newPack(batchRows, true))
	s.Require().NoError(err)
	decoded := s.readCompoundPKStats(manifest2)
	s.Require().Len(decoded, 2)
	assertInt64PKMembership(s.T(), decoded, 1, int64(batchRows*2))
	manifestFootprint, err := packed.StatsBinlogSizeFromManifest(manifest2, s.storageConfig)
	s.Require().NoError(err)
	s.Equal(manifestFootprint, segmentStats2.GetStatsBinlogSize(),
		"final replacement must update the cumulative footprint, not add all historical PUT bytes")
	metacache.SetStatistics(bw2.PreparedStats())(seg)
	metacache.UpdateManifestPath(manifest2)(seg)

	// Once compound-only is committed, an incremental batch would be invisible
	// because readers prioritize the compound. Reject this lifecycle violation
	// instead of producing an ambiguous compound+batch manifest.
	bwInvalid := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc,
		packed.DefaultWriteBufferSize, 0, s.storageConfig, currentSplit, manifest2)
	_, _, _, _, _, _, _, err = bwInvalid.Write(context.Background(), newPack(batchRows*2, false))
	s.Require().Error(err)
	s.ErrorIs(err, merr.ErrDataIntegrity)

	// The second final starts from a compound-only committed manifest. The
	// reader expands that compound back to its per-batch stats before appending
	// the current batch, so no prior PK is lost.
	bw3 := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc,
		packed.DefaultWriteBufferSize, 0, s.storageConfig, currentSplit, manifest2)
	_, _, _, _, manifest3, _, _, err := bw3.Write(context.Background(), newPack(batchRows*2, true))
	s.Require().NoError(err)
	decoded = s.readCompoundPKStats(manifest3)
	s.Require().Len(decoded, 3)
	assertInt64PKMembership(s.T(), decoded, 1, int64(batchRows*3))
}

func (s *PackWriterV3Suite) readCompoundPKStats(manifestPath string) []*storage.PrimaryKeyStats {
	manifestStats, err := packed.GetManifestStats(manifestPath, s.storageConfig)
	s.Require().NoError(err)
	bloom := manifestStats["bloom_filter.100"]
	s.Require().Len(bloom.Paths, 1)
	s.Equal(storage.CompoundStatsType.LogIdx(), path.Base(bloom.Paths[0]))
	value, err := packed.ReadFile(s.storageConfig, bloom.Paths[0])
	s.Require().NoError(err)
	decoded, err := storage.DeserializeBloomFilterStats(bloom.Paths, []*storage.Blob{{Value: value}})
	s.Require().NoError(err)
	return decoded
}

func assertInt64PKMembership(t *testing.T, stats []*storage.PrimaryKeyStats, from, to int64) {
	t.Helper()
	bfs := pkoracle.NewBloomFilterSet()
	bfs.Roll(stats...)
	for id := from; id <= to; id++ {
		location := storage.NewLocationsCache(storage.NewInt64PrimaryKey(id))
		require.True(t, bfs.PkExists(location), "compound bloom missing PK %d", id)
	}
}

func (s *PackWriterV3Suite) genStringPKInsertData(size int, schema *schemapb.CollectionSchema) []*storage.InsertData {
	buf, err := storage.NewInsertData(schema)
	s.Require().NoError(err)
	for i := 0; i < size; i++ {
		data := map[storage.FieldID]any{
			common.RowIDField:     int64(i + 1),
			common.TimeStampField: int64(i + 1),
			100:                   fmt.Sprintf("pk-%d", i),
			101:                   make([]float32, 8),
		}
		s.Require().NoError(buf.Append(data))
	}
	return []*storage.InsertData{buf}
}

// TestWrite_PropagatesPriorStatsReadError guards against silently committing a
// truncated compound bloom blob. When the prior-batch stats read fails on a
// flush over an existing manifest, Write must return the error so the sync
// retries — dropping the prior per-batch paths under loon replace semantics
// would commit a compound holding only this sync's PKs, losing prior-batch PKs
// for delete-routing / PK-pruning.
func (s *PackWriterV3Suite) TestWrite_PropagatesPriorStatsReadError() {
	collectionID := int64(123)
	partitionID := int64(456)
	segmentID := int64(10007) // unique ID to avoid manifest collision with other tests
	channelName := fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)
	batchRows := 5

	bfs := pkoracle.NewBloomFilterSet()
	k := metautil.JoinIDPath(collectionID, partitionID, segmentID)
	basePath := path.Join(common.SegmentInsertLogPath, k)
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{ManifestPath: manifestPath}, bfs, nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(1000)(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSchema(mock.Anything).Return(s.schema).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()
	mc.EXPECT().GetSegmentsBy(mock.Anything, mock.Anything).Return([]*metacache.SegmentInfo{seg}).Maybe()
	mc.EXPECT().UpdateSegments(mock.Anything, mock.Anything).Run(func(action metacache.SegmentAction, filters ...metacache.SegmentFilter) {
		action(seg)
	}).Return().Maybe()

	// Batch 1 creates a real manifest (version past earliest) carrying prior
	// per-batch bloom paths.
	pack1 := new(SyncPack).
		WithCollectionID(collectionID).WithPartitionID(partitionID).WithSegmentID(segmentID).
		WithChannelName(channelName).
		WithInsertData(genInsertDataWithPKOffset(batchRows, 0, s.schema)).
		WithBatchRows(int64(batchRows))
	bw1 := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)
	_, _, _, _, manifest1, _, _, err := bw1.Write(context.Background(), pack1)
	s.Require().NoError(err)
	currentSplit := storagecommon.FillColumnGroupFormats(s.currentSplit, paramtable.Get().DataNodeCfg.StorageFormat.GetValue())

	// Fault-inject a transient prior-stats read failure on the flush.
	patched := mockey.Mock(packed.GetManifestStats).
		Return(nil, errors.New("transient manifest read failure")).Build()
	defer patched.UnPatch()

	pack2 := new(SyncPack).
		WithCollectionID(collectionID).WithPartitionID(partitionID).WithSegmentID(segmentID).
		WithChannelName(channelName).
		WithInsertData(genInsertDataWithPKOffset(batchRows, batchRows, s.schema)).
		WithFlush()
	bw2 := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, currentSplit, manifest1)
	_, _, _, _, _, _, _, err = bw2.Write(context.Background(), pack2)
	s.Require().Error(err, "flush must fail when prior-batch stats read fails, not commit a truncated compound blob")
}

// TestPerBatchStatPathsExcludesCompound guards the flush-merge fix: when a
// segment in Flushing state flushes over more than one sync, the pre-commit
// manifest already holds the prior flush's compound (merged) blob. That compound
// must be excluded when assembling the prior-stats-for-merge, otherwise the PK
// merge short-circuits on it (DeserializeBloomFilterStats) and the BM25 merge
// additively re-counts it (double-count). perBatchStatPaths keeps only per-batch
// blobs, which are the complete, non-redundant set.
func (s *PackWriterV3Suite) TestPerBatchStatPathsExcludesCompound() {
	base := "files/insert_log/1/10/200/_stats/bloom_filter.100"
	// Per-batch ids are allocator-assigned (large); use values that cannot
	// collide with CompoundStatsType.LogIdx() (== "1").
	perBatch1 := base + "/10"
	perBatch2 := base + "/20"
	compound := base + "/" + storage.CompoundStatsType.LogIdx()
	s.Require().NotEqual(perBatch1, compound)

	// Compound dropped, per-batch order preserved.
	s.Equal([]string{perBatch1, perBatch2}, perBatchStatPaths([]string{perBatch1, compound, perBatch2}))
	// Compound-only input yields nothing to merge from.
	s.Empty(perBatchStatPaths([]string{compound}))
	// Per-batch-only input is unchanged (the normal first-flush case).
	s.Equal([]string{perBatch1}, perBatchStatPaths([]string{perBatch1}))
	// Empty input.
	s.Empty(perBatchStatPaths(nil))

	// Bloom merge uses all batches for a mixed layout, but falls back to the
	// compound when H22 has already replaced the manifest entry.
	s.Equal([]string{perBatch1, perBatch2}, bloomMergeSourcePaths([]string{perBatch1, compound, perBatch2}))
	s.Equal([]string{compound}, bloomMergeSourcePaths([]string{compound}))
	s.True(hasCompoundStatPath([]string{perBatch1, compound}))
	s.False(hasCompoundStatPath([]string{perBatch1, perBatch2}))
}

// TestMultiBatchBM25StatsAccumulation verifies that BM25 stat files from
// multiple batches accumulate in the manifest.
func (s *PackWriterV3Suite) TestMultiBatchBM25StatsAccumulation() {
	collectionID := int64(123)
	partitionID := int64(456)
	segmentID := int64(10002) // unique ID to avoid manifest collision with other tests
	channelName := fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)
	batchRows := 5

	bfs := pkoracle.NewBloomFilterSet()

	k := metautil.JoinIDPath(collectionID, partitionID, segmentID)
	basePath := path.Join(common.SegmentInsertLogPath, k)
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{
		ManifestPath: manifestPath,
	}, bfs, nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(1000)(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSchema(mock.Anything).Return(s.schema).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()
	mc.EXPECT().GetSegmentsBy(mock.Anything, mock.Anything).Return([]*metacache.SegmentInfo{seg}).Maybe()
	mc.EXPECT().UpdateSegments(mock.Anything, mock.Anything).Run(func(action metacache.SegmentAction, filters ...metacache.SegmentFilter) {
		action(seg)
	}).Return().Maybe()

	bm25FieldID := int64(200)
	makeBM25Stats := func() map[int64]*storage.BM25Stats {
		stats := storage.NewBM25Stats()
		stats.Append(map[uint32]float32{1: 1.0, 2: 2.0})
		return map[int64]*storage.BM25Stats{bm25FieldID: stats}
	}

	bm25Key := fmt.Sprintf("bm25.%d", bm25FieldID)

	// Batch 1
	pack1 := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(channelName).
		WithInsertData(genInsertDataWithPKOffset(batchRows, 0, s.schema)).
		WithBatchRows(int64(batchRows)).
		WithBM25Stats(makeBM25Stats())

	bw1 := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)
	_, _, _, _, manifest1, _, _, err := bw1.Write(context.Background(), pack1)
	s.Require().NoError(err)
	currentSplit := storagecommon.FillColumnGroupFormats(s.currentSplit, paramtable.Get().DataNodeCfg.StorageFormat.GetValue())

	stats1, err := packed.GetManifestStats(manifest1, s.storageConfig)
	s.Require().NoError(err)
	count1 := len(stats1[bm25Key].Paths)
	s.Greater(count1, 0, "batch 1 should produce bm25 stat files")

	// Batch 2
	pack2 := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(channelName).
		WithInsertData(genInsertDataWithPKOffset(batchRows, batchRows, s.schema)).
		WithBatchRows(int64(batchRows)).
		WithBM25Stats(makeBM25Stats())

	bw2 := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc, packed.DefaultWriteBufferSize, 0, s.storageConfig, currentSplit, manifest1)
	_, _, _, _, manifest2, _, _, err := bw2.Write(context.Background(), pack2)
	s.Require().NoError(err)

	stats2, err := packed.GetManifestStats(manifest2, s.storageConfig)
	s.Require().NoError(err)
	count2 := len(stats2[bm25Key].Paths)
	s.Greater(count2, count1, "batch 2 should accumulate more bm25 files than batch 1")
	s.NotEmpty(stats2[bm25Key].Metadata["memory_size"], "memory_size metadata should be set")
}

// TestWrite_SingleVersionBumpAcrossSections verifies that one Write call
// bumps the manifest version exactly once even when inserts, stats, delta,
// and bm25 are all present — the atomicity guarantee this refactor adds.
func (s *PackWriterV3Suite) TestWrite_SingleVersionBumpAcrossSections() {
	collectionID := int64(123)
	partitionID := int64(456)
	segmentID := int64(789)
	channelName := fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)
	rows := 10

	bfs := pkoracle.NewBloomFilterSet()
	k := metautil.JoinIDPath(collectionID, partitionID, segmentID)
	basePath := path.Join(common.SegmentInsertLogPath, k)
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)

	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{ManifestPath: manifestPath}, bfs, nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(1000)(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSchema(mock.Anything).Return(s.schema).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()
	mc.EXPECT().GetSegmentsBy(mock.Anything, mock.Anything).Return([]*metacache.SegmentInfo{seg}).Maybe()
	mc.EXPECT().UpdateSegments(mock.Anything, mock.Anything).
		Run(func(action metacache.SegmentAction, filters ...metacache.SegmentFilter) { action(seg) }).
		Return().Maybe()

	deletes := &storage.DeleteData{}
	for i := 0; i < rows; i++ {
		deletes.Append(storage.NewInt64PrimaryKey(int64(i+1)), uint64(100+i))
	}

	pack := new(SyncPack).
		WithCollectionID(collectionID).
		WithPartitionID(partitionID).
		WithSegmentID(segmentID).
		WithChannelName(channelName).
		WithInsertData(genInsertData(rows, s.schema)).
		WithDeleteData(deletes).
		WithFlush()

	_, baseVer, err := packed.UnmarshalManifestPath(manifestPath)
	s.Require().NoError(err)

	bw := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc,
		packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)

	_, _, _, _, writtenManifestPath, _, _, err := bw.Write(context.Background(), pack)
	s.Require().NoError(err)
	_, newVer, err := packed.UnmarshalManifestPath(writtenManifestPath)
	s.Require().NoError(err)
	s.Equalf(baseVer+1, newVer,
		"Write must produce exactly one version bump; baseVer=%d newVer=%d", baseVer, newVer)

	bfKey := fmt.Sprintf("bloom_filter.%d", int64(100))
	stats, err := packed.GetManifestStats(writtenManifestPath, s.storageConfig)
	s.Require().NoError(err)
	bfStat := stats[bfKey]
	var compoundPath string
	for _, statPath := range bfStat.Paths {
		if path.Base(statPath) == strconv.FormatInt(int64(storage.CompoundStatsType), 10) {
			compoundPath = statPath
			break
		}
	}
	s.Require().NotEmpty(compoundPath, "flush should publish a compound bloom filter")
	compoundBlob, err := packed.ReadFile(s.storageConfig, compoundPath)
	s.Require().NoError(err)
	metadataMemorySize, err := strconv.ParseInt(bfStat.Metadata["memory_size"], 10, 64)
	s.Require().NoError(err)
	s.Equal(int64(len(compoundBlob)), metadataMemorySize)

	resolvedMemorySize, err := packed.NewStatsResolver(writtenManifestPath, s.storageConfig).BloomFilterMemorySize(100)
	s.Require().NoError(err)
	s.Equal(int64(len(compoundBlob)), resolvedMemorySize)
}

// TestWrite_RetryDoesNotLeakVersionBumps verifies that when a transient
// commit failure forces the retry loop to re-run, the eventual successful
// commit produces only one version bump, not one per attempt.
func (s *PackWriterV3Suite) TestWrite_RetryDoesNotLeakVersionBumps() {
	collectionID := int64(123)
	partitionID := int64(456)
	segmentID := int64(789)
	channelName := fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", collectionID)
	rows := 4

	bfs := pkoracle.NewBloomFilterSet()
	k := metautil.JoinIDPath(collectionID, partitionID, segmentID)
	basePath := path.Join(common.SegmentInsertLogPath, k)
	manifestPath := packed.MarshalManifestPath(basePath, packed.ManifestEarliest)
	seg := metacache.NewSegmentInfo(&datapb.SegmentInfo{ManifestPath: manifestPath}, bfs, nil, metacache.NewEmptySegmentStats())
	metacache.UpdateNumOfRows(1000)(seg)
	mc := metacache.NewMockMetaCache(s.T())
	mc.EXPECT().Collection().Return(collectionID).Maybe()
	mc.EXPECT().GetSchema(mock.Anything).Return(s.schema).Maybe()
	mc.EXPECT().GetSegmentByID(segmentID).Return(seg, true).Maybe()
	mc.EXPECT().GetSegmentsBy(mock.Anything, mock.Anything).Return([]*metacache.SegmentInfo{seg}).Maybe()
	mc.EXPECT().UpdateSegments(mock.Anything, mock.Anything).
		Run(func(action metacache.SegmentAction, filters ...metacache.SegmentFilter) { action(seg) }).
		Return().Maybe()

	pack := new(SyncPack).
		WithCollectionID(collectionID).WithPartitionID(partitionID).WithSegmentID(segmentID).
		WithChannelName(channelName).WithInsertData(genInsertData(rows, s.schema))

	_, baseVer, err := packed.UnmarshalManifestPath(manifestPath)
	s.Require().NoError(err)

	var calls int32
	var origin func(string, int64, *indexpb.StorageConfig, *packed.ManifestUpdates) (string, error)
	patched := mockey.Mock(packed.CommitManifestUpdates).
		To(func(basePath string, baseVersion int64, cfg *indexpb.StorageConfig, updates *packed.ManifestUpdates) (string, error) {
			if atomic.AddInt32(&calls, 1) == 1 {
				return "", packed.ErrLoonTransient
			}
			return origin(basePath, baseVersion, cfg, updates)
		}).
		Origin(&origin).
		Build()
	defer patched.UnPatch()

	bw := NewBulkPackWriterV3(mc, s.schema, s.cm, s.logIDAlloc,
		packed.DefaultWriteBufferSize, 0, s.storageConfig, s.currentSplit, manifestPath)
	_, _, _, _, newManifestPath, _, _, err := bw.Write(context.Background(), pack)
	s.Require().NoError(err)
	_, newVer, err := packed.UnmarshalManifestPath(newManifestPath)
	s.Require().NoError(err)
	s.Equal(baseVer+1, newVer, "retry must not produce extra version bumps")
	s.Equal(int32(2), atomic.LoadInt32(&calls), "exactly one retry expected")
}
