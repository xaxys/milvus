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

package importv2

import (
	"context"
	"sync"

	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/util/hardware"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

var (
	globalMemoryAllocator     MemoryAllocator
	globalMemoryAllocatorOnce sync.Once
)

// GetMemoryAllocator returns the global memory allocator instance
func GetMemoryAllocator() MemoryAllocator {
	globalMemoryAllocatorOnce.Do(func() {
		globalMemoryAllocator = NewMemoryAllocator(int64(hardware.GetMemoryCount()))
	})
	return globalMemoryAllocator
}

// MemoryReservation represents a held import memory reservation.
type MemoryReservation interface {
	Size() int64
	Release()
}

// MemoryAllocator handles memory allocation and deallocation for import tasks
type MemoryAllocator interface {
	// BlockingAllocate blocks until memory is available and then allocates
	// This method will block until memory becomes available
	BlockingAllocate(taskID int64, size int64)

	// TryReserve allocates memory if it is immediately available.
	TryReserve(taskID int64, size int64) (MemoryReservation, bool)

	// Reserve blocks until memory is available and returns a releasable reservation.
	Reserve(taskID int64, size int64) MemoryReservation

	// ForceReserve reserves memory even if it exceeds the limit.
	ForceReserve(taskID int64, size int64) MemoryReservation

	// Release releases memory of the specified size
	Release(taskID int64, size int64)
}

type memoryReservation struct {
	allocator *memoryAllocator
	taskID    int64
	size      int64
	once      sync.Once
}

func (r *memoryReservation) Size() int64 {
	return r.size
}

func (r *memoryReservation) Release() {
	r.once.Do(func() {
		if r.allocator != nil && r.size > 0 {
			r.allocator.Release(r.taskID, r.size)
		}
	})
}

type noopMemoryReservation struct {
	size int64
}

func (r *noopMemoryReservation) Size() int64 {
	return r.size
}

func (r *noopMemoryReservation) Release() {}

type memoryAllocator struct {
	systemTotalMemory int64
	usedMemory        int64
	mutex             sync.RWMutex
	cond              *sync.Cond
}

func newNoopMemoryReservation(size int64) MemoryReservation {
	return &noopMemoryReservation{size: size}
}

// NewMemoryAllocator creates a new MemoryAllocator instance
func NewMemoryAllocator(systemTotalMemory int64) MemoryAllocator {
	mlog.Info(context.TODO(), "new import memory allocator", mlog.Int64("systemTotalMemory", systemTotalMemory))
	ma := &memoryAllocator{
		systemTotalMemory: systemTotalMemory,
		usedMemory:        0,
	}
	ma.cond = sync.NewCond(&ma.mutex)
	return ma
}

func (ma *memoryAllocator) memoryLimit() int64 {
	percentage := paramtable.Get().DataNodeCfg.ImportMemoryLimitPercentage.GetAsFloat()
	return int64(float64(ma.systemTotalMemory) * percentage / 100.0)
}

func (ma *memoryAllocator) canAllocate(size int64) bool {
	return ma.usedMemory+size <= ma.memoryLimit()
}

func (ma *memoryAllocator) reserveLocked(taskID int64, size int64) MemoryReservation {
	if size <= 0 {
		return newNoopMemoryReservation(size)
	}
	ma.usedMemory += size
	memoryLimit := ma.memoryLimit()
	mlog.Info(context.TODO(), "memory allocated successfully",
		mlog.FieldTaskID(taskID),
		mlog.Int64("allocatedSize", size),
		mlog.Int64("usedMemory", ma.usedMemory),
		mlog.Int64("availableMemory", memoryLimit-ma.usedMemory))
	return &memoryReservation{
		allocator: ma,
		taskID:    taskID,
		size:      size,
	}
}

// BlockingAllocate blocks until memory is available and then allocates
func (ma *memoryAllocator) BlockingAllocate(taskID int64, size int64) {
	ma.Reserve(taskID, size)
}

func (ma *memoryAllocator) TryReserve(taskID int64, size int64) (MemoryReservation, bool) {
	if size <= 0 {
		return newNoopMemoryReservation(size), true
	}
	ma.mutex.Lock()
	defer ma.mutex.Unlock()
	if !ma.canAllocate(size) {
		return nil, false
	}
	return ma.reserveLocked(taskID, size), true
}

func (ma *memoryAllocator) Reserve(taskID int64, size int64) MemoryReservation {
	if size <= 0 {
		return newNoopMemoryReservation(size)
	}
	ma.mutex.Lock()
	defer ma.mutex.Unlock()

	// Wait until enough memory is available
	for !ma.canAllocate(size) {
		memoryLimit := ma.memoryLimit()
		mlog.Warn(context.TODO(), "task waiting for memory allocation...",
			mlog.FieldTaskID(taskID),
			mlog.Int64("requestedSize", size),
			mlog.Int64("usedMemory", ma.usedMemory),
			mlog.Int64("availableMemory", memoryLimit-ma.usedMemory))

		ma.cond.Wait()
	}

	return ma.reserveLocked(taskID, size)
}

func (ma *memoryAllocator) ForceReserve(taskID int64, size int64) MemoryReservation {
	if size <= 0 {
		return newNoopMemoryReservation(size)
	}
	ma.mutex.Lock()
	defer ma.mutex.Unlock()
	return ma.reserveLocked(taskID, size)
}

// Release releases memory of the specified size
func (ma *memoryAllocator) Release(taskID int64, size int64) {
	ma.mutex.Lock()
	defer ma.mutex.Unlock()

	ma.usedMemory -= size
	if ma.usedMemory < 0 {
		ma.usedMemory = 0 // Prevent negative memory usage
		mlog.Warn(context.TODO(), "memory release resulted in negative usage, reset to 0",
			mlog.FieldTaskID(taskID),
			mlog.Int64("releaseSize", size))
	}

	mlog.Info(context.TODO(), "memory released successfully",
		mlog.FieldTaskID(taskID),
		mlog.Int64("releasedSize", size),
		mlog.Int64("usedMemory", ma.usedMemory))

	// Wake up waiting tasks after memory is released
	ma.cond.Broadcast()
}
