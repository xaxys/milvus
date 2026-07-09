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

package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
)

const (
	DataGetLabel    = "get"
	DataPutLabel    = "put"
	DataRemoveLabel = "remove"
	DataWalkLabel   = "walk"
	DataStatLabel   = "stat"

	persistentDataOpType = "persistent_data_op_type"
)

var (
	PersistentDataKvSize = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "kv_size",
			Help:      "kv size stats",
			Buckets:   buckets,
		}, []string{persistentDataOpType})

	PersistentDataRequestLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "request_latency",
			Help:      "request latency on the client side ",
			Buckets:   buckets,
		}, []string{persistentDataOpType})

	PersistentDataOpCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "op_count",
			Help:      "count of persistent data operation",
		}, []string{persistentDataOpType, statusLabelName})

	// Filesystem metrics (default filesystem only) - common across all nodes
	FilesystemReadCount = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "filesystem_read_count",
			Help:      "number of filesystem read operations",
		}, []string{
			filesystemKeyLabelName,
		})

	FilesystemWriteCount = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "filesystem_write_count",
			Help:      "number of filesystem write operations",
		}, []string{
			filesystemKeyLabelName,
		})

	FilesystemReadBytes = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "filesystem_read_bytes",
			Help:      "total bytes read from filesystem",
		}, []string{
			filesystemKeyLabelName,
		})

	FilesystemWriteBytes = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "filesystem_write_bytes",
			Help:      "total bytes written to filesystem",
		}, []string{
			filesystemKeyLabelName,
		})

	FilesystemGetFileInfoCount = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "filesystem_get_file_info_count",
			Help:      "number of get file info operations",
		}, []string{
			filesystemKeyLabelName,
		})

	FilesystemCreateDirCount = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "filesystem_create_dir_count",
			Help:      "number of create dir filesystem operations",
		}, []string{
			filesystemKeyLabelName,
		})

	FilesystemDeleteDirCount = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "filesystem_delete_dir_count",
			Help:      "number of delete dir filesystem operations",
		}, []string{
			filesystemKeyLabelName,
		})

	FilesystemDeleteFileCount = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "filesystem_delete_file_count",
			Help:      "number of delete file filesystem operations",
		}, []string{
			filesystemKeyLabelName,
		})

	FilesystemMoveCount = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "filesystem_move_count",
			Help:      "number of move filesystem operations",
		}, []string{
			filesystemKeyLabelName,
		})

	FilesystemCopyFileCount = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "filesystem_copy_file_count",
			Help:      "number of copy file filesystem operations",
		}, []string{
			filesystemKeyLabelName,
		})

	FilesystemFailedCount = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "filesystem_failed_count",
			Help:      "number of failed filesystem operations",
		}, []string{
			filesystemKeyLabelName,
		})

	FilesystemMultiPartUploadCreated = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "filesystem_multi_part_upload_created",
			Help:      "number of multi-part uploads created",
		}, []string{
			filesystemKeyLabelName,
		})

	FilesystemMultiPartUploadFinished = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "filesystem_multi_part_upload_finished",
			Help:      "number of multi-part uploads finished",
		}, []string{
			filesystemKeyLabelName,
		})

	StorageAccessOpCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "access_op_count",
			Help:      "count of storage access operations",
		}, []string{
			storageAccessOpType,
			statusLabelName,
		})

	StorageAccessRequestLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "access_request_latency",
			Help:      "latency of storage access operations in milliseconds",
			Buckets:   StorageAccessLatencyBucketsMs,
		}, []string{
			storageAccessOpType,
			statusLabelName,
		})

	StorageAccessBytes = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: milvusNamespace,
			Subsystem: "storage",
			Name:      "access_bytes",
			Help:      "bytes transferred by storage access operations",
		}, []string{
			storageAccessOpType,
			statusLabelName,
		})
)

// RegisterStorageMetrics registers storage metrics
func RegisterStorageMetrics(registry *prometheus.Registry) {
	registry.MustRegister(PersistentDataKvSize)
	registry.MustRegister(PersistentDataRequestLatency)
	registry.MustRegister(PersistentDataOpCounter)

	// filesystem metrics
	registry.MustRegister(FilesystemReadCount)
	registry.MustRegister(FilesystemWriteCount)
	registry.MustRegister(FilesystemReadBytes)
	registry.MustRegister(FilesystemWriteBytes)
	registry.MustRegister(FilesystemGetFileInfoCount)
	registry.MustRegister(FilesystemCreateDirCount)
	registry.MustRegister(FilesystemDeleteDirCount)
	registry.MustRegister(FilesystemDeleteFileCount)
	registry.MustRegister(FilesystemMoveCount)
	registry.MustRegister(FilesystemCopyFileCount)
	registry.MustRegister(FilesystemFailedCount)
	registry.MustRegister(FilesystemMultiPartUploadCreated)
	registry.MustRegister(FilesystemMultiPartUploadFinished)
	registry.MustRegister(StorageAccessOpCount)
	registry.MustRegister(StorageAccessRequestLatency)
	registry.MustRegister(StorageAccessBytes)
}

// PublishFilesystemMetrics publishes filesystem metrics (common across all nodes)
func PublishFilesystemMetrics(fs string, readCount, writeCount, readBytes, writeBytes, getFileInfoCount, createDirCount, deleteDirCount, deleteFileCount, moveCount, copyFileCount, failedCount, multiPartUploadCreated, multiPartUploadFinished int64) {
	labels := prometheus.Labels{
		filesystemKeyLabelName: fs,
	}

	FilesystemReadCount.With(labels).Set(float64(readCount))
	FilesystemWriteCount.With(labels).Set(float64(writeCount))
	FilesystemReadBytes.With(labels).Set(float64(readBytes))
	FilesystemWriteBytes.With(labels).Set(float64(writeBytes))
	FilesystemGetFileInfoCount.With(labels).Set(float64(getFileInfoCount))
	FilesystemCreateDirCount.With(labels).Set(float64(createDirCount))
	FilesystemDeleteDirCount.With(labels).Set(float64(deleteDirCount))
	FilesystemDeleteFileCount.With(labels).Set(float64(deleteFileCount))
	FilesystemMoveCount.With(labels).Set(float64(moveCount))
	FilesystemCopyFileCount.With(labels).Set(float64(copyFileCount))
	FilesystemFailedCount.With(labels).Set(float64(failedCount))
	FilesystemMultiPartUploadCreated.With(labels).Set(float64(multiPartUploadCreated))
	FilesystemMultiPartUploadFinished.With(labels).Set(float64(multiPartUploadFinished))
}
