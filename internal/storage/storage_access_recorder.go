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
	"context"
	"io"
	"sync"
	"time"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus/internal/util/storageaccess"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
)

const (
	StorageAccessOpRead   = storageaccess.OpRead
	StorageAccessOpWrite  = storageaccess.OpWrite
	StorageAccessOpStat   = storageaccess.OpStat
	StorageAccessOpList   = storageaccess.OpList
	StorageAccessOpDelete = storageaccess.OpDelete
	StorageAccessOpCopy   = storageaccess.OpCopy
)

func recordStorageAccess(ctx context.Context, opType string, bytes int64, err error, recorder *timerecord.TimeRecorder) time.Duration {
	if recorder == nil {
		return 0
	}
	elapsed := recorder.ElapseSpan()
	storageaccess.RecordAccess(ctx, opType, bytes, err, elapsed)
	return elapsed
}

func wrapStorageAccessReader(ctx context.Context, reader FileReader, recorder *timerecord.TimeRecorder) FileReader {
	if reader == nil {
		return nil
	}
	if recorder == nil {
		recorder = timerecord.NewTimeRecorder("storageReader")
	}
	return &storageAccessFileReader{ctx: ctx, reader: reader, recorder: recorder}
}

type storageAccessFileReader struct {
	ctx      context.Context
	reader   FileReader
	recorder *timerecord.TimeRecorder

	mu       sync.Mutex
	bytes    int64
	readErr  error
	recorded bool
}

func (r *storageAccessFileReader) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	r.recordResult(n, err)
	return n, err
}

func (r *storageAccessFileReader) ReadAt(p []byte, off int64) (int, error) {
	n, err := r.reader.ReadAt(p, off)
	r.recordResult(n, err)
	return n, err
}

func (r *storageAccessFileReader) Seek(offset int64, whence int) (int64, error) {
	position, err := r.reader.Seek(offset, whence)
	if err != nil {
		r.recordResult(0, err)
	}
	return position, err
}

func (r *storageAccessFileReader) Close() error {
	err := r.reader.Close()
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.recorded {
		resultErr := r.readErr
		if resultErr == nil {
			resultErr = err
		}
		recordStorageAccess(r.ctx, StorageAccessOpRead, r.bytes, resultErr, r.recorder)
		r.recorded = true
	}
	return err
}

func (r *storageAccessFileReader) Size() (int64, error) {
	recorder := timerecord.NewTimeRecorder("storageReaderSize")
	size, err := r.reader.Size()
	recordStorageAccess(r.ctx, StorageAccessOpStat, 0, err, recorder)
	return size, err
}

func (r *storageAccessFileReader) recordResult(n int, err error) {
	if n == 0 && (err == nil || errors.Is(err, io.EOF)) {
		return
	}
	if n > 0 && errors.Is(err, io.EOF) {
		err = nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if n > 0 {
		r.bytes += int64(n)
	}
	if err != nil && r.readErr == nil {
		r.readErr = err
	}
}
