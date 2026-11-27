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

package ddl

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/datacoord/broker"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/pkg/v3/proto/proxypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func TestInvalidateCollectionIndexMeta(t *testing.T) {
	paramtable.Init()
	ctx := context.Background()

	mix := mocks.NewMixCoord(t)
	var seen []*proxypb.InvalidateCollMetaCacheRequest
	mix.EXPECT().InvalidateCollectionMetaCache(
		mock.Anything,
		mock.Anything,
	).RunAndReturn(func(_ context.Context, req *proxypb.InvalidateCollMetaCacheRequest) (*commonpb.Status, error) {
		seen = append(seen, req)
		return merr.Success(), nil
	})

	cb := &Callbacks{Deps: Deps{
		Broker: broker.NewCoordinatorBroker(mix),
	}}
	require.NoError(t, cb.invalidateCollectionIndexMeta(ctx, 100, commonpb.MsgType_CreateIndex))
	require.NoError(t, cb.invalidateCollectionIndexMeta(ctx, 100, commonpb.MsgType_AlterIndex))
	require.NoError(t, cb.invalidateCollectionIndexMeta(ctx, 100, commonpb.MsgType_DropIndex))

	require.Len(t, seen, 3)
	assert.Equal(t, commonpb.MsgType_CreateIndex, seen[0].GetBase().GetMsgType())
	assert.Equal(t, int64(100), seen[0].GetCollectionID())
	assert.Equal(t, commonpb.MsgType_AlterIndex, seen[1].GetBase().GetMsgType())
	assert.Equal(t, commonpb.MsgType_DropIndex, seen[2].GetBase().GetMsgType())

	errMix := mocks.NewMixCoord(t)
	errMix.EXPECT().InvalidateCollectionMetaCache(mock.Anything, mock.Anything).Return(
		merr.Status(merr.WrapErrServiceNotReadyMsg("transient rpc failure")), nil)
	errCb := &Callbacks{Deps: Deps{
		Broker: broker.NewCoordinatorBroker(errMix),
	}}
	require.Error(t, errCb.invalidateCollectionIndexMeta(ctx, 100, commonpb.MsgType_CreateIndex))
}
