// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package proxy

import (
	"context"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/metadata"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus/internal/storageprofile"
	"github.com/milvus-io/milvus/pkg/v3/util/interceptor"
)

func TestBeginProxyStorageProfileRequiresTrustedInternalRequest(t *testing.T) {
	enabled := Params.StorageProfileCfg.Enabled.GetValue()
	allowExplicit := Params.StorageProfileCfg.RequestAllowExplicit.GetValue()
	defer func() {
		require.NoError(t, Params.Save(Params.StorageProfileCfg.Enabled.Key, enabled))
		require.NoError(t, Params.Save(Params.StorageProfileCfg.RequestAllowExplicit.Key, allowExplicit))
	}()
	require.NoError(t, Params.Save(Params.StorageProfileCfg.Enabled.Key, "true"))
	require.NoError(t, Params.Save(Params.StorageProfileCfg.RequestAllowExplicit.Key, "true"))

	untrustedCtx := metadata.NewIncomingContext(context.Background(), metadata.Pairs(
		storageprofile.ExplicitRequestHeader, "summary",
	))
	_, untrustedScope, untrustedLevel, _ := beginProxyStorageProfile(untrustedCtx, storageprofile.WorkloadKindSearch)
	defer untrustedScope.Finish()
	assert.Equal(t, storageprofile.StorageProfileDisabled, untrustedLevel)

	trustedCtx := metadata.NewIncomingContext(context.Background(), metadata.Pairs(
		storageprofile.ExplicitRequestHeader, "summary",
		interceptor.ServerIDKey, "1",
	))
	bound, scope, level, scopeID := beginProxyStorageProfile(trustedCtx, storageprofile.WorkloadKindSearch)
	defer scope.Finish()

	assert.Equal(t, storageprofile.StorageProfileSummary, level)
	assert.Equal(t, storageprofile.StorageProfileSummary, storageprofile.ProfileLevelFromContext(bound))
	assert.NotEmpty(t, scopeID)
	assert.Equal(t, scopeID, storageprofile.AttributionFromContext(bound).RequestID)
}

func TestSearchStartsStorageProfileBeforeSearchByPKLookup(t *testing.T) {
	mockey.PatchConvey("storage profile covers search-by-PK lookup and early return", t, func() {
		enabled := Params.StorageProfileCfg.Enabled.GetValue()
		allowExplicit := Params.StorageProfileCfg.RequestAllowExplicit.GetValue()
		defer func() {
			require.NoError(t, Params.Save(Params.StorageProfileCfg.Enabled.Key, enabled))
			require.NoError(t, Params.Save(Params.StorageProfileCfg.RequestAllowExplicit.Key, allowExplicit))
		}()
		require.NoError(t, Params.Save(Params.StorageProfileCfg.Enabled.Key, "true"))
		require.NoError(t, Params.Save(Params.StorageProfileCfg.RequestAllowExplicit.Key, "true"))

		var nestedLevel storageprofile.StorageProfileLevel
		mockey.Mock((*Proxy).handleIfSearchByPK).To(func(_ *Proxy, ctx context.Context, request *milvuspb.SearchRequest) ([]bool, error) {
			nestedLevel = storageprofile.ProfileLevelFromContext(ctx)
			request.Nq = 0
			return []bool{false}, nil
		}).Build()

		node := &Proxy{}
		node.UpdateStateCode(commonpb.StateCode_Healthy)
		ctx := metadata.NewIncomingContext(context.Background(), metadata.Pairs(
			storageprofile.ExplicitRequestHeader, "summary",
			interceptor.ServerIDKey, "1",
		))

		result, _, _, _, err := node.search(ctx, &milvuspb.SearchRequest{Nq: 1}, false, false)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Equal(t, commonpb.ErrorCode_Success, result.GetStatus().GetErrorCode())
		assert.Equal(t, storageprofile.StorageProfileSummary, nestedLevel)
	})
}
