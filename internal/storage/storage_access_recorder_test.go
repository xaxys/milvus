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

package storage

import (
	"bytes"
	"context"
	"io"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/util/storageaccess"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
)

type testStorageAccessReader struct {
	*bytes.Reader
}

func (r *testStorageAccessReader) Close() error {
	return nil
}

func (r *testStorageAccessReader) Size() (int64, error) {
	return r.Reader.Size(), nil
}

func TestStorageAccessReaderRecordsOneLogicalRead(t *testing.T) {
	collector := storageaccess.NewCollector()
	ctx := storageaccess.WithCollector(context.Background(), collector)
	reader := wrapStorageAccessReader(ctx, &testStorageAccessReader{Reader: bytes.NewReader([]byte("storage"))}, timerecord.NewTimeRecorder("testReader"))

	data, err := io.ReadAll(reader)
	require.NoError(t, err)
	require.Equal(t, []byte("storage"), data)
	require.NoError(t, reader.Close())

	stats := collector.Snapshot()
	require.NotNil(t, stats)
	require.EqualValues(t, 1, stats.RequestCount)
	require.EqualValues(t, len(data), stats.Bytes)
	require.Len(t, stats.OpStats, 1)
	require.Equal(t, storageaccess.OpRead, stats.OpStats[0].OpType)
}
