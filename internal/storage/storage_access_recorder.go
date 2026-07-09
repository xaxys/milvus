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
	"time"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus/internal/util/storageaccess"
)

const (
	StorageAccessOpRead       = storageaccess.OpRead
	StorageAccessOpWrite      = storageaccess.OpWrite
	StorageAccessOpStat       = storageaccess.OpStat
	StorageAccessOpList       = storageaccess.OpList
	StorageAccessOpDelete     = storageaccess.OpDelete
	StorageAccessOpCopy       = storageaccess.OpCopy
	StorageAccessOpCreateDir  = storageaccess.OpCreateDir
	StorageAccessOpDeleteDir  = storageaccess.OpDeleteDir
	StorageAccessOpDeleteFile = storageaccess.OpDeleteFile
	StorageAccessOpMove       = storageaccess.OpMove
)

func recordStorageAccess(ctx context.Context, opType string, bytes int64, err error, start time.Time) {
	RecordStorageAccess(ctx, opType, bytes, err, start)
}

// RecordStorageAccess records one storage access observation to Prometheus and
// to the task-local collector stored in ctx, if present.
func RecordStorageAccess(ctx context.Context, opType string, bytes int64, err error, start time.Time) {
	storageaccess.RecordAccess(ctx, opType, bytes, err, start)
}

func wrapStorageAccessReader(ctx context.Context, reader FileReader) FileReader {
	if reader == nil {
		return nil
	}
	return &storageAccessFileReader{
		ctx:    ctx,
		reader: reader,
	}
}

type storageAccessFileReader struct {
	ctx    context.Context
	reader FileReader
}

func (r *storageAccessFileReader) Read(p []byte) (int, error) {
	start := time.Now()
	n, err := r.reader.Read(p)
	r.recordRead(n, err, start)
	return n, err
}

func (r *storageAccessFileReader) ReadAt(p []byte, off int64) (int, error) {
	start := time.Now()
	n, err := r.reader.ReadAt(p, off)
	r.recordRead(n, err, start)
	return n, err
}

func (r *storageAccessFileReader) Seek(offset int64, whence int) (int64, error) {
	start := time.Now()
	pos, err := r.reader.Seek(offset, whence)
	recordStorageAccess(r.ctx, StorageAccessOpRead, 0, err, start)
	return pos, err
}

func (r *storageAccessFileReader) Close() error {
	start := time.Now()
	err := r.reader.Close()
	if err != nil {
		recordStorageAccess(r.ctx, StorageAccessOpRead, 0, err, start)
	}
	return err
}

func (r *storageAccessFileReader) Size() (int64, error) {
	start := time.Now()
	size, err := r.reader.Size()
	recordStorageAccess(r.ctx, StorageAccessOpStat, 0, err, start)
	return size, err
}

func (r *storageAccessFileReader) recordRead(n int, err error, start time.Time) {
	if n == 0 && (err == nil || errors.Is(err, io.EOF)) {
		return
	}
	if n > 0 && errors.Is(err, io.EOF) {
		err = nil
	}
	recordStorageAccess(r.ctx, StorageAccessOpRead, int64(n), err, start)
}
