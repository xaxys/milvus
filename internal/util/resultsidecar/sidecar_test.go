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

package resultsidecar

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
)

func TestMergeForwardsOpaquePayloadsAndMarksMissingInput(t *testing.T) {
	first := New("storage_profile", 1, []byte("first"))
	second := New("future_diagnostic", 7, []byte("second"))

	merged := Merge(first, nil, second)
	require.NotNil(t, merged)
	assert.True(t, merged.GetIncomplete())
	require.Len(t, merged.GetItems(), 2)
	assert.Equal(t, "first", string(merged.GetItems()[0].GetPayload()))
	assert.Equal(t, "second", string(merged.GetItems()[1].GetPayload()))
}

func TestMergeDropsMalformedAndOversizedItems(t *testing.T) {
	valid := New("storage_profile", 1, []byte("valid"))
	malformed := &internalpb.ResultSidecars{Items: []*internalpb.ResultSidecar{{Type: "storage_profile", SchemaVersion: 1}}}
	oversized := &internalpb.ResultSidecars{Items: []*internalpb.ResultSidecar{{
		Type:          "storage_profile",
		SchemaVersion: 1,
		Payload:       make([]byte, MaxPayloadSize+1),
	}}}

	merged := Merge(valid, malformed, oversized)
	require.NotNil(t, merged)
	assert.True(t, merged.GetIncomplete())
	require.Len(t, merged.GetItems(), 1)
	assert.Equal(t, "valid", string(merged.GetItems()[0].GetPayload()))
}

func TestSelectIgnoresUnknownTypesAndRejectsUnknownSelectedVersion(t *testing.T) {
	sidecars := Merge(
		New("storage_profile", 1, []byte("known")),
		New("future_diagnostic", 9, []byte("opaque")),
		New("storage_profile", 2, []byte("future-profile")),
	)

	payloads, incomplete := Select(sidecars, "storage_profile", 1)
	assert.True(t, incomplete)
	require.Len(t, payloads, 1)
	assert.Equal(t, "known", string(payloads[0]))
}

func TestSidecarsUseASeparateWireFieldFromLegacyStorageProfile(t *testing.T) {
	result := &internalpb.SearchResults{
		StorageProfile: []byte("legacy-json"),
		Sidecars:       New("storage_profile", 1, []byte("sidecar-json")),
	}
	encoded, err := proto.Marshal(result)
	require.NoError(t, err)

	decoded := &internalpb.SearchResults{}
	require.NoError(t, proto.Unmarshal(encoded, decoded))
	assert.Equal(t, "legacy-json", string(decoded.GetStorageProfile()))
	payloads, incomplete := Select(decoded.GetSidecars(), "storage_profile", 1)
	assert.False(t, incomplete)
	require.Len(t, payloads, 1)
	assert.Equal(t, "sidecar-json", string(payloads[0]))
}
