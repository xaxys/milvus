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

package importutilv2

import (
	"context"
	"sync"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/internal/util/hookutil"
	"github.com/milvus-io/milvus/internal/util/importutilv2/binlog"
	"github.com/milvus-io/milvus/internal/util/importutilv2/csv"
	"github.com/milvus-io/milvus/internal/util/importutilv2/json"
	"github.com/milvus-io/milvus/internal/util/importutilv2/numpy"
	"github.com/milvus-io/milvus/internal/util/importutilv2/parquet"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

//go:generate mockery --name=Reader --structname=MockReader --output=./  --filename=mock_reader.go --with-expecter --inpackage
type Reader interface {
	// Size returns the size of the underlying file/files in bytes.
	// It returns an error if the size cannot be determined.
	Size() (int64, error)

	// Read reads data from the underlying file/files.
	// It returns the storage.InsertData and an error, if any.
	Read() (*storage.InsertData, error)

	// Close closes the underlying file reader.
	Close()
}

// ReaderFactory belongs to one task execution. Only resolved source dependencies
// are shared; each NewReader call owns its stream, decoder and request context.
// Options and storage configuration must remain immutable during execution.
type ReaderFactory struct {
	cm      storage.ChunkManager
	cfg     *indexpb.StorageConfig
	options Options
	storage func() (readerStorage, error)
}

type readerStorage struct {
	cm         storage.ChunkManager
	cfg        *indexpb.StorageConfig
	encryption binlog.SourceEncryption
}

func NewReaderFactory(ctx context.Context, cm storage.ChunkManager, cfg *indexpb.StorageConfig, options Options) *ReaderFactory {
	return &ReaderFactory{
		cm: cm, cfg: cfg, options: options,
		storage: sync.OnceValues(func() (readerStorage, error) {
			// Use the task context, never the first file's context. Cache a
			// resolution error for this attempt too; a retry creates a new factory.
			if err := ctx.Err(); err != nil {
				return readerStorage{}, err
			}
			uri, _ := funcutil.GetAttrByKeyFromRepeatedKV(SnapshotSourceURI, options)
			sourceCM, sourceCfg, err := ResolveSnapshotImportStorage(ctx, cm, cfg, uri, options)
			if err != nil {
				return readerStorage{}, err
			}
			resolved := readerStorage{cm: sourceCM, cfg: sourceCfg}
			if ezk, _ := GetEZK(options); ezk != "" {
				resolved.encryption.Encrypted = true
				ezID, err := hookutil.GetEzIDByImportEzk(ezk)
				if err != nil {
					return readerStorage{}, err
				}
				resolved.encryption.PluginContext, err = hookutil.GetCPluginContextByEzID(ezID)
				if err != nil {
					return readerStorage{}, err
				}
			}
			return resolved, nil
		}),
	}
}

func (f *ReaderFactory) NewReader(ctx context.Context, schema *schemapb.CollectionSchema, file *internalpb.ImportFile,
	bufferSize int, deleteBudget int64, shared *binlog.SnapshotL0Deletes,
) (Reader, error) {
	if shared != nil && !SnapshotSourceUsesSharedL0(file.GetSnapshotSource()) {
		return nil, merr.WrapErrServiceInternalMsg("shared L0 reader requires a task-shared snapshot descriptor and prepared bitmaps")
	}
	if err := ValidateSnapshotSourceOptions(f.options); err != nil {
		return nil, err
	}
	if file.GetSnapshotSource() != nil || HasExternalSource(f.options) {
		if err := ValidateSnapshotImportFiles([]*internalpb.ImportFile{file}, f.options); err != nil {
			return nil, err
		}
	}
	resolved := readerStorage{cm: f.cm, cfg: f.cfg}
	if IsSnapshotSource(f.options) {
		// Validate the descriptor before accessing storage or the key retriever.
		// Same-bucket and no-L0 snapshots use this path too. Ordinary import
		// retains its existing storage and encryption initialization.
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var err error
		resolved, err = f.storage()
		if err != nil {
			return nil, err
		}
	}
	return newReader(ctx, resolved.cm, schema, file, f.options, bufferSize, resolved.cfg, deleteBudget, shared, resolved.encryption)
}

// PrepareSnapshotDeletes is task-local and shared by PreImport and Import. The
// existing private-map reservation becomes the bitmap pool; neither the wire
// task nor coordinator grouping/slot calculation needs a new field or knob.
func (f *ReaderFactory) PrepareSnapshotDeletes(ctx context.Context, schema *schemapb.CollectionSchema,
	files []*internalpb.ImportFile, source *internalpb.SnapshotImportL0Source, deleteBudget, bitmapBudget int64,
) (*binlog.SnapshotL0Deletes, error) {
	var sources []*internalpb.SnapshotImportL0Source
	if source != nil {
		sources = []*internalpb.SnapshotImportL0Source{source}
	}
	return f.PrepareSnapshotDeletesMulti(ctx, schema, files, sources, deleteBudget, bitmapBudget)
}

// PrepareSnapshotDeletesMulti is the multi-scope form used by Import V3, where
// one reshard task may pack files from several source scopes to avoid creating
// one tiny task and fragment set per source channel/partition. Each inventory
// is folded into only the masks of its own scope.
func (f *ReaderFactory) PrepareSnapshotDeletesMulti(ctx context.Context, schema *schemapb.CollectionSchema,
	files []*internalpb.ImportFile, sources []*internalpb.SnapshotImportL0Source, deleteBudget, bitmapBudget int64,
) (*binlog.SnapshotL0Deletes, error) {
	if err := ValidateSnapshotImportPlan(files, f.options, sources); err != nil {
		return nil, err
	}
	start, end, err := ParseTimeRange(f.options)
	if err != nil {
		return nil, err
	}
	resolved, err := f.storage()
	if err != nil {
		return nil, err
	}
	validate, err := SnapshotPathValidator(f.options, resolved.cm)
	if err != nil {
		return nil, err
	}
	fileSources := make([]*internalpb.SnapshotImportSource, len(files))
	for i, file := range files {
		fileSources[i] = file.GetSnapshotSource()
	}
	return binlog.BuildSnapshotDeleteMasks(ctx, resolved.cm, schema, resolved.cfg, fileSources, sources,
		start, end, deleteBudget, bitmapBudget, resolved.encryption, validate)
}

// SnapshotFileSize sizes one exact snapshot manifest without reading rows or
// deltalogs. Import V3's preimport is size-only for backup/snapshot jobs, so it
// must not pay the row-reader initialization that loads segment-local deletes.
func (f *ReaderFactory) SnapshotFileSize(ctx context.Context, schema *schemapb.CollectionSchema, file *internalpb.ImportFile) (int64, error) {
	if err := ValidateSnapshotImportFiles([]*internalpb.ImportFile{file}, f.options); err != nil {
		return 0, err
	}
	if IsSnapshotPreparation([]*internalpb.ImportFile{file}) {
		return 0, merr.WrapErrServiceInternalMsg("snapshot preparation must finish before worker dispatch")
	}
	source := file.GetSnapshotSource()
	manifestPath := source.GetManifestPath()
	if manifestPath == "" {
		if len(file.GetPaths()) != 1 {
			return 0, merr.WrapErrImportFailedMsg("snapshot-source import file requires exactly one manifest path")
		}
		manifestPath = file.GetPaths()[0]
	}
	base, version, err := packed.UnmarshalManifestPath(manifestPath)
	if err != nil || base == "" || version == packed.ManifestLatest {
		return 0, merr.WrapErrImportFailedMsg("snapshot-source import requires an exact StorageV3 manifest version")
	}
	resolved, err := f.storage()
	if err != nil {
		return 0, err
	}
	validate, err := SnapshotPathValidator(f.options, resolved.cm)
	if err != nil {
		return 0, err
	}
	if validate != nil {
		if err := validate(base); err != nil {
			return 0, err
		}
	}
	files, lobSize, err := binlog.StorageV3ManifestFiles(ctx, resolved.cm, resolved.cfg, schema, manifestPath, validate)
	if err != nil {
		return 0, err
	}
	size, err := storage.GetFilesSize(ctx, files, resolved.cm)
	if err != nil {
		return 0, err
	}
	return size + lobSize, nil
}

// NewReader requires the caller's reserved positive delete budget for typed
// snapshot sources. Other inputs do not use that budget and pass zero.
func NewReader(ctx context.Context,
	cm storage.ChunkManager,
	schema *schemapb.CollectionSchema,
	importFile *internalpb.ImportFile,
	options Options,
	bufferSize int,
	storageConfig *indexpb.StorageConfig,
	snapshotDeleteBudget int64,
) (Reader, error) {
	return NewReaderFactory(ctx, cm, storageConfig, options).NewReader(ctx, schema, importFile, bufferSize, snapshotDeleteBudget, nil)
}

func newReader(ctx context.Context, cm storage.ChunkManager, schema *schemapb.CollectionSchema,
	importFile *internalpb.ImportFile, options Options, bufferSize int, storageConfig *indexpb.StorageConfig,
	snapshotDeleteBudget int64, shared *binlog.SnapshotL0Deletes,
	encryption binlog.SourceEncryption,
) (Reader, error) {
	source := importFile.GetSnapshotSource()
	if shared != nil {
		// Adapt only the validated reader view. Never expand L0 paths back
		// into a segment or mutate the persisted descriptor: the reader must
		// borrow the task's prepared bitmap instead of opening L0 again.
		source = proto.Clone(source).(*internalpb.SnapshotImportSource)
		source.Version -= 4
		source.SourceChannel = ""
		source.SourcePartitionId = 0
	}
	if IsBackup(options) {
		tsStart, tsEnd, err := ParseTimeRange(options)
		if err != nil {
			return nil, err
		}
		paths := importFile.GetPaths()
		if IsSnapshotSource(options) {
			validate, err := SnapshotPathValidator(options, cm)
			if err != nil {
				return nil, err
			}
			if source != nil {
				if shared != nil {
					return binlog.NewStorageV3ManifestReaderWithSharedL0(ctx, cm, schema, storageConfig,
						source.GetManifestPath(), tsStart, tsEnd, bufferSize, encryption, source, snapshotDeleteBudget, shared, validate)
				}
				return binlog.NewStorageV3ManifestReader(
					ctx, cm, schema, storageConfig, source.GetManifestPath(), tsStart, tsEnd, bufferSize, encryption, source, snapshotDeleteBudget, validate,
				)
			}
			// DataCoord broadcasts a metadata URI, then expands and persists
			// exact StorageV3 manifests during snapshot preparation before dispatch.
			// Do not consult storage_version: snapshot metadata is the source of
			// truth and ValidateSnapshotSourceOptions rejects that ambiguity.
			if len(paths) != 1 {
				return nil, merr.WrapErrImportFailedMsg(
					"snapshot-source import file requires exactly one manifest path",
				)
			}
			return binlog.NewStorageV3ManifestReader(
				ctx, cm, schema, storageConfig, paths[0], tsStart, tsEnd, bufferSize, encryption, nil, 0, validate,
			)
		}
		storageVersion, err := GetStorageVersion(options)
		if err != nil {
			return nil, err
		}
		importEz, _ := GetEZK(options)
		return binlog.NewReader(ctx, cm, schema, storageConfig, storageVersion, paths, tsStart, tsEnd, bufferSize, importEz, nil)
	}

	fileType, err := GetFileType(importFile)
	if err != nil {
		return nil, err
	}
	switch fileType {
	case FileTypeJSON:
		return json.NewReader(ctx, cm, schema, importFile.GetPaths()[0], bufferSize)
	case FileTypeJSONLines:
		return json.NewLinesReader(ctx, cm, schema, importFile.GetPaths()[0], bufferSize)
	case FileTypeNumpy:
		return numpy.NewReader(ctx, cm, schema, importFile.GetPaths(), bufferSize)
	case FileTypeParquet:
		return parquet.NewReader(ctx, cm, schema, importFile.GetPaths()[0], bufferSize)
	case FileTypeCsv:
		sep, err := GetCSVSep(options)
		if err != nil {
			return nil, err
		}
		nullkey, err := GetCSVNullKey(options)
		if err != nil {
			return nil, err
		}
		return csv.NewReader(ctx, cm, schema, importFile.GetPaths()[0], bufferSize, sep, nullkey)
	}
	return nil, merr.WrapErrImportFailed("unexpected import file")
}
