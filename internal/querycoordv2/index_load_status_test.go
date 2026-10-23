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

package querycoordv2

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
)

func TestClassifyIndexLoad(t *testing.T) {
	t.Run("no replica means the index is not there", func(t *testing.T) {
		state, reason := classifyIndexLoad(stateEvidence{})
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_NOT_LOADED, state)
		assert.Contains(t, reason, "no replica")
	})

	t.Run("no segment carries it", func(t *testing.T) {
		state, reason := classifyIndexLoad(stateEvidence{replicas: 2, segments: 10})
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_NOT_LOADED, state)
		assert.Contains(t, reason, "no loaded segment")
	})

	t.Run("every replica and segment carries it", func(t *testing.T) {
		state, reason := classifyIndexLoad(stateEvidence{replicas: 2, readyReps: 2, segments: 10, readySegs: 10})
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_LOADED, state)
		assert.Empty(t, reason)
	})

	t.Run("a replica is short of it", func(t *testing.T) {
		state, reason := classifyIndexLoad(stateEvidence{replicas: 2, readyReps: 1, segments: 10, readySegs: 10})
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_PARTIALLY_LOADED, state)
		assert.Contains(t, reason, "some replicas")
	})

	t.Run("all replicas covered but the distribution still holds uncovered segments", func(t *testing.T) {
		state, reason := classifyIndexLoad(stateEvidence{replicas: 2, readyReps: 2, segments: 10, readySegs: 8})
		assert.Equal(t, querypb.IndexLoadState_INDEX_LOAD_PARTIALLY_LOADED, state)
		assert.Contains(t, reason, "some loaded segments")
	})
}
