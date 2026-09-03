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

package datanode

import (
	"context"
	"testing"

	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// TestNormalizeReshardBatchKeepsFunctionOutput pins that normalization leaves
// function output columns untouched: the reshard reader (full schema) reads a
// user-supplied column and runReshardFunctions fills or overwrites the rest,
// so fragments stay uniform without normalize knowing about functions.
func TestNormalizeReshardBatchKeepsFunctionOutput(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "embedding_out", DataType: schemapb.DataType_Int64, IsFunctionOutput: true},
		},
	}
	data, err := storage.NewInsertDataWithFunctionOutputField(schema)
	require.NoError(t, err)
	data.Data[100] = &storage.Int64FieldData{Data: []int64{1, 2}}
	data.Data[101] = &storage.Int64FieldData{Data: []int64{10, 20}}

	var offset int64
	source := &datapb.SourceFileSpec{
		File: &internalpb.ImportFile{Id: 1, PreAllocatedAutoIds: &commonpb.IDRange{Begin: 100, End: 102}},
	}
	require.NoError(t, normalizeReshardBatch(source, schema, data, 2, &offset))

	_, has := data.Data[101]
	require.True(t, has)
	require.Equal(t, []int64{10, 20}, data.Data[101].(*storage.Int64FieldData).Data)
	require.Equal(t, []int64{1, 2}, data.Data[100].(*storage.Int64FieldData).Data)
	require.Equal(t, []int64{100, 101}, data.Data[common.RowIDField].(*storage.Int64FieldData).Data)
}

// TestNormalizeReshardBatchKeepsBackupFunctionOutput pins the backup branch:
// backup fragments carry function outputs per the source binlog reader, so they
// must survive normalization untouched.
func TestNormalizeReshardBatchKeepsBackupFunctionOutput(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "embedding_out", DataType: schemapb.DataType_Int64, IsFunctionOutput: true},
		},
	}
	data, err := storage.NewInsertDataWithFunctionOutputField(schema)
	require.NoError(t, err)
	data.Data[100] = &storage.Int64FieldData{Data: []int64{1, 2}}
	data.Data[101] = &storage.Int64FieldData{Data: []int64{10, 20}}

	source := &datapb.SourceFileSpec{
		FileType: datapb.ImportFileType_BackupBinlog,
		File:     &internalpb.ImportFile{Id: 1},
	}
	require.NoError(t, normalizeReshardBatch(source, schema, data, 2, nil))

	_, has := data.Data[101]
	require.True(t, has)
	require.Equal(t, []int64{10, 20}, data.Data[101].(*storage.Int64FieldData).Data)
}

// TestRunReshardFunctionsGeneratesBM25AndMinHash pins the V2-equivalent
// placement: reshard runs every function over the batch, so missing BM25 and
// MinHash output columns are generated before hash routing and fragment write.
func TestRunReshardFunctionsGeneratesBM25AndMinHash(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "text", DataType: schemapb.DataType_VarChar},
			{FieldID: 102, Name: "sparse", DataType: schemapb.DataType_SparseFloatVector, IsFunctionOutput: true},
			{FieldID: 103, Name: "mh", DataType: schemapb.DataType_BinaryVector, IsFunctionOutput: true,
				TypeParams: []*commonpb.KeyValuePair{{Key: "dim", Value: "64"}}},
		},
		Functions: []*schemapb.FunctionSchema{
			{Name: "bm", Type: schemapb.FunctionType_BM25, InputFieldIds: []int64{101}, OutputFieldIds: []int64{102}},
			{Name: "mh", Type: schemapb.FunctionType_MinHash, InputFieldIds: []int64{101}, OutputFieldIds: []int64{103}},
		},
	}
	data, err := storage.NewInsertDataWithFunctionOutputField(schema)
	require.NoError(t, err)
	data.Data[100] = &storage.Int64FieldData{Data: []int64{1, 2}}
	data.Data[101] = &storage.StringFieldData{Data: []string{"milvus vector database", "milvus again"}}

	require.NoError(t, runReshardFunctions(context.Background(), schema, data))

	sparse, ok := data.Data[102].(*storage.SparseFloatVectorFieldData)
	require.True(t, ok)
	require.Equal(t, 2, sparse.RowNum())
	mh, ok := data.Data[103].(*storage.BinaryVectorFieldData)
	require.True(t, ok)
	require.Equal(t, 2, mh.RowNum())
	require.Equal(t, 64, mh.Dim)
}

// TestRunReshardFunctionsOverwritesUserMinHash pins the V2 alignment for a
// user-supplied MinHash column: V2's RunAll unconditionally recomputes MinHash,
// so reshard must overwrite the user column with the deterministic recomputation
// instead of preserving or dropping it.
func TestRunReshardFunctionsOverwritesUserMinHash(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Properties: []*commonpb.KeyValuePair{
			{Key: common.CollectionAllowInsertNonBM25FunctionOutputs, Value: "true"},
		},
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "text", DataType: schemapb.DataType_VarChar},
			{FieldID: 103, Name: "mh", DataType: schemapb.DataType_BinaryVector, IsFunctionOutput: true,
				TypeParams: []*commonpb.KeyValuePair{{Key: "dim", Value: "32"}}},
		},
		Functions: []*schemapb.FunctionSchema{
			{Name: "mh", Type: schemapb.FunctionType_MinHash, InputFieldIds: []int64{101}, OutputFieldIds: []int64{103}},
		},
	}
	data, err := storage.NewInsertDataWithFunctionOutputField(schema)
	require.NoError(t, err)
	data.Data[100] = &storage.Int64FieldData{Data: []int64{1, 2}}
	data.Data[101] = &storage.StringFieldData{Data: []string{"milvus vector database", "milvus again"}}
	// user-supplied junk signatures: all-zero bytes
	data.Data[103] = &storage.BinaryVectorFieldData{Data: make([]byte, 8), Dim: 32}

	require.NoError(t, runReshardFunctions(context.Background(), schema, data))

	mh, ok := data.Data[103].(*storage.BinaryVectorFieldData)
	require.True(t, ok)
	require.Equal(t, 2, mh.RowNum())
	require.Equal(t, 32, mh.Dim)
	require.NotEqual(t, make([]byte, 8), mh.Data[:8])
}

type captureRecordWriter struct {
	records []storage.Record
}

func (w *captureRecordWriter) Write(r storage.Record) error {
	r.Retain()
	w.records = append(w.records, r)
	return nil
}

func (w *captureRecordWriter) GetWrittenUncompressed() uint64 { return 0 }

func (w *captureRecordWriter) Close() error { return nil }

// TestImportV3FinalWriterPassesFunctionOutputs pins that the final transform
// runs no functions: every function output column is already in the fragments,
// so the writer only materializes timestamps and forwards the record.
func TestImportV3FinalWriterPassesFunctionOutputs(t *testing.T) {
	targetSchema := typeutil.AppendSystemFields(&schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "text", DataType: schemapb.DataType_VarChar},
			{FieldID: 102, Name: "sparse", DataType: schemapb.DataType_SparseFloatVector, IsFunctionOutput: true},
		},
		Functions: []*schemapb.FunctionSchema{
			{Name: "bm", Type: schemapb.FunctionType_BM25, InputFieldIds: []int64{101}, OutputFieldIds: []int64{102}},
		},
	})
	// ordinary-import temp schema: target user fields + RowID (what
	// buildImportV3TempSchema produces for a non-backup job)
	tempSchema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "text", DataType: schemapb.DataType_VarChar},
			{FieldID: 102, Name: "sparse", DataType: schemapb.DataType_SparseFloatVector, IsFunctionOutput: true},
			{FieldID: common.RowIDField, Name: common.RowIDFieldName, DataType: schemapb.DataType_Int64},
		},
		Functions: []*schemapb.FunctionSchema{
			{Name: "bm", Type: schemapb.FunctionType_BM25, InputFieldIds: []int64{101}, OutputFieldIds: []int64{102}},
		},
	}

	// fragment batch: pk/text/sparse/RowID, no timestamp
	fragmentData := map[int64]storage.FieldData{
		100:               &storage.Int64FieldData{Data: []int64{1, 2}},
		101:               &storage.StringFieldData{Data: []string{"milvus vector database", "milvus again"}},
		102:               &storage.SparseFloatVectorFieldData{SparseFloatArray: schemapb.SparseFloatArray{Dim: 8, Contents: [][]byte{{1, 0, 0, 0, 2, 0, 0, 0}, {1, 0, 0, 0, 3, 0, 0, 0}}}},
		common.RowIDField: &storage.Int64FieldData{Data: []int64{7, 8}},
	}
	recordData := &storage.InsertData{Data: fragmentData}
	reader, err := storage.NewInsertDataRecordReader(recordData, tempSchema)
	require.NoError(t, err)
	record, err := reader.Next()
	require.NoError(t, err)

	output := &captureRecordWriter{}
	writer := newImportV3FinalWriter(context.Background(), output, tempSchema, targetSchema, 12345, false)
	require.NoError(t, writer.Write(record))
	require.NoError(t, reader.Close())
	require.Len(t, output.records, 1)

	final := output.records[0]
	require.Equal(t, 2, final.Len())
	tsCol := final.Column(common.TimeStampField)
	require.Equal(t, 2, tsCol.Len())
	require.Equal(t, int64(12345), tsCol.(*array.Int64).Value(0))
	sparseCol := final.Column(102)
	require.Equal(t, 2, sparseCol.Len())
}

// TestImportV3FinalWriterBackupKeepsSourceTimestamps pins the backup branch:
// fragments carry the source binlog timestamps, so the final merge must pass
// them through untouched instead of overwriting them with the import data ts.
func TestImportV3FinalWriterBackupKeepsSourceTimestamps(t *testing.T) {
	targetSchema := typeutil.AppendSystemFields(&schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		},
	})
	tempSchema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: common.RowIDField, Name: common.RowIDFieldName, DataType: schemapb.DataType_Int64},
			{FieldID: common.TimeStampField, Name: common.TimeStampFieldName, DataType: schemapb.DataType_Int64},
		},
	}

	fragmentData := map[int64]storage.FieldData{
		100:                   &storage.Int64FieldData{Data: []int64{1, 2}},
		common.RowIDField:     &storage.Int64FieldData{Data: []int64{7, 8}},
		common.TimeStampField: &storage.Int64FieldData{Data: []int64{111, 222}},
	}
	recordData := &storage.InsertData{Data: fragmentData}
	reader, err := storage.NewInsertDataRecordReader(recordData, tempSchema)
	require.NoError(t, err)
	record, err := reader.Next()
	require.NoError(t, err)

	output := &captureRecordWriter{}
	writer := newImportV3FinalWriter(context.Background(), output, tempSchema, targetSchema, 12345, true)
	require.NoError(t, writer.Write(record))
	require.NoError(t, reader.Close())
	require.Len(t, output.records, 1)

	tsCol := output.records[0].Column(common.TimeStampField)
	require.Equal(t, 2, tsCol.Len())
	require.Equal(t, int64(111), tsCol.(*array.Int64).Value(0))
	require.Equal(t, int64(222), tsCol.(*array.Int64).Value(1))
}

// TestImportV3FinalWriterBackupRejectsBadTimestampColumn covers the two backup
// validation branches. Both are defensive: a well-formed backup plan carries
// timestamps in its fragments, so the triggers are constructed by violating
// the schema/fragment contract the way a corrupt plan would.
func TestImportV3FinalWriterBackupRejectsBadTimestampColumn(t *testing.T) {
	t.Run("timestamp field absent from target schema", func(t *testing.T) {
		// RecordToInsertData only initializes fields listed in the target
		// schema, so a target schema without the timestamp system field is the
		// one shape that leaves data.Data[TimeStampField] nil.
		targetSchema := &schemapb.CollectionSchema{
			Fields: []*schemapb.FieldSchema{
				{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			},
		}
		tempSchema := &schemapb.CollectionSchema{
			Fields: []*schemapb.FieldSchema{
				{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			},
		}
		recordData := &storage.InsertData{Data: map[int64]storage.FieldData{
			100: &storage.Int64FieldData{Data: []int64{1}},
		}}
		reader, err := storage.NewInsertDataRecordReader(recordData, tempSchema)
		require.NoError(t, err)
		record, err := reader.Next()
		require.NoError(t, err)

		writer := newImportV3FinalWriter(context.Background(), &captureRecordWriter{}, tempSchema, targetSchema, 12345, true)
		err = writer.Write(record)
		require.NoError(t, reader.Close())
		require.ErrorContains(t, err, "backup timestamp is missing")
	})

	t.Run("fragment without timestamp column", func(t *testing.T) {
		// The target schema carries the timestamp field but the fragment does
		// not, so RecordToInsertData leaves it empty: RowNum 0 against N data
		// rows trips the mismatch check.
		targetSchema := typeutil.AppendSystemFields(&schemapb.CollectionSchema{
			Fields: []*schemapb.FieldSchema{
				{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			},
		})
		tempSchema := &schemapb.CollectionSchema{
			Fields: []*schemapb.FieldSchema{
				{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
				{FieldID: common.RowIDField, Name: common.RowIDFieldName, DataType: schemapb.DataType_Int64},
			},
		}
		recordData := &storage.InsertData{Data: map[int64]storage.FieldData{
			100:               &storage.Int64FieldData{Data: []int64{1, 2}},
			common.RowIDField: &storage.Int64FieldData{Data: []int64{7, 8}},
		}}
		reader, err := storage.NewInsertDataRecordReader(recordData, tempSchema)
		require.NoError(t, err)
		record, err := reader.Next()
		require.NoError(t, err)

		writer := newImportV3FinalWriter(context.Background(), &captureRecordWriter{}, tempSchema, targetSchema, 12345, true)
		err = writer.Write(record)
		require.NoError(t, reader.Close())
		require.ErrorContains(t, err, "backup timestamp rows mismatch")
	})
}

func TestReshardSpillThreshold(t *testing.T) {
	// budget 640MB, fragment input 256MB, 3x16MB read buffers.
	require.Equal(t, int64(640<<20-256<<20-3*(16<<20)), reshardSpillThreshold(640<<20, 256<<20, 16<<20))
	// Degenerate budget leaves no room: spill on any resident bytes.
	require.LessOrEqual(t, reshardSpillThreshold(16<<20, 64<<20, 16<<20), int64(0))
}
