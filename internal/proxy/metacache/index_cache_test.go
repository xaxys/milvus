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

package metacache

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/retry"
)

func TestMetaCacheCollectionIndexes(t *testing.T) {
	ctx := context.Background()
	const collID = int64(42)

	mix := mocks.NewMockMixCoordClient(t)
	mc, err := NewMetaCache(mix)
	require.NoError(t, err)

	mix.EXPECT().ListIndexes(mock.Anything, &indexpb.ListIndexesRequest{CollectionID: collID}).
		Return(&indexpb.ListIndexesResponse{
			Status: merr.Success(),
			IndexInfos: []*indexpb.IndexInfo{
				{CollectionID: collID, FieldID: 100, IndexID: 1001, IndexName: "idx_a"},
			},
		}, nil).Once()

	first, err := mc.GetCollectionIndexes(ctx, collID)
	require.NoError(t, err)
	require.NotNil(t, first.ByName["idx_a"])
	require.NotNil(t, first.ByID[1001])

	second, err := mc.GetCollectionIndexes(ctx, collID)
	require.NoError(t, err)
	assert.Same(t, first, second)

	// Collection-level invalidation evicts indexCache even when m.collections[collID] is empty.
	mc.InvalidateCollectionMeta(ctx, "default", "", collID, false)

	mix.EXPECT().ListIndexes(mock.Anything, &indexpb.ListIndexesRequest{CollectionID: collID}).
		Return(&indexpb.ListIndexesResponse{
			Status: merr.Success(),
			IndexInfos: []*indexpb.IndexInfo{
				{CollectionID: collID, FieldID: 100, IndexID: 1002, IndexName: "idx_b"},
			},
		}, nil).Once()

	third, err := mc.GetCollectionIndexes(ctx, collID)
	require.NoError(t, err)
	assert.Nil(t, third.ByName["idx_a"])
	assert.NotNil(t, third.ByName["idx_b"])
	assert.NotNil(t, third.ByID[1002])

	// Index-only invalidation evicts indexCache and maps ErrIndexNotFound to empty snapshot.
	mc.InvalidateCollectionIndexes(ctx, collID)
	mix.EXPECT().ListIndexes(mock.Anything, &indexpb.ListIndexesRequest{CollectionID: collID}).
		Return(&indexpb.ListIndexesResponse{
			Status: merr.Status(merr.ErrIndexNotFound),
		}, nil).Once()

	empty, err := mc.GetCollectionIndexes(ctx, collID)
	require.NoError(t, err)
	assert.Empty(t, empty.Infos)
	assert.Empty(t, empty.ByName)
	assert.Empty(t, empty.ByID)
}

func TestMetaCacheCollectionIndexesIsolatesCallerCancelAndRetryContext(t *testing.T) {
	const collID = int64(77)
	mix := mocks.NewMockMixCoordClient(t)
	mc, err := NewMetaCache(mix)
	require.NoError(t, err)

	type callerCtxKey struct{}
	entered := make(chan struct{})
	release := make(chan struct{})
	var sawCleanFillCtx bool
	mix.EXPECT().ListIndexes(mock.Anything, &indexpb.ListIndexesRequest{CollectionID: collID}).
		RunAndReturn(func(fillCtx context.Context, _ *indexpb.ListIndexesRequest, _ ...grpc.CallOption) (*indexpb.ListIndexesResponse, error) {
			// The shared singleflight fill must NOT inherit caller-specific
			// MaxAttempts=1 or arbitrary context values.
			_, hasAttempts := retry.MaxAttemptsFromContext(fillCtx)
			sawCleanFillCtx = !hasAttempts && fillCtx.Value(callerCtxKey{}) == nil
			close(entered)
			<-release
			return &indexpb.ListIndexesResponse{
				Status: merr.Success(),
				IndexInfos: []*indexpb.IndexInfo{
					{CollectionID: collID, FieldID: 100, IndexID: 2001, IndexName: "idx_shared"},
				},
			}, nil
		}).Once()

	shortCtx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	shortCtx = retry.WithMaxAttemptsContext(shortCtx, 1)
	shortCtx = context.WithValue(shortCtx, callerCtxKey{}, "poison")

	shortErrCh := make(chan error, 1)
	go func() {
		_, err := mc.GetCollectionIndexes(shortCtx, collID)
		shortErrCh <- err
	}()

	<-entered

	type normalRes struct {
		snap *CollectionIndexes
		err  error
	}
	normalCh := make(chan normalRes, 1)
	go func() {
		snap, err := mc.GetCollectionIndexes(context.Background(), collID)
		normalCh <- normalRes{snap: snap, err: err}
	}()

	// Short-deadline caller returns promptly with DeadlineExceeded without poisoning normal caller.
	require.ErrorIs(t, <-shortErrCh, context.DeadlineExceeded)
	close(release)

	got := <-normalCh
	require.NoError(t, got.err)
	require.NotNil(t, got.snap.ByName["idx_shared"])
	assert.True(t, sawCleanFillCtx, "shared singleflight fill must not inherit caller retry/fail-fast context values")
}
