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

import "github.com/milvus-io/milvus/pkg/v3/proto/internalpb"

const (
	MaxPayloadSize = 64 << 10
	MaxTotalSize   = 1 << 20
)

// New returns a single-item sidecar collection when the envelope is valid.
// Invalid diagnostic data is dropped so it can never fail the business path.
func New(typeName string, schemaVersion uint32, payload []byte) *internalpb.ResultSidecars {
	item := &internalpb.ResultSidecar{
		Type:          typeName,
		SchemaVersion: schemaVersion,
		Payload:       payload,
	}
	if !valid(item) {
		return nil
	}
	return &internalpb.ResultSidecars{Items: []*internalpb.ResultSidecar{item}}
}

// Merge forwards opaque sidecars without interpreting type-specific payloads.
// It enforces independent per-item and aggregate budgets and records any loss
// with the generic incomplete bit for the final consumer.
func Merge(inputs ...*internalpb.ResultSidecars) *internalpb.ResultSidecars {
	output := &internalpb.ResultSidecars{}
	totalSize := 0
	for _, input := range inputs {
		if input == nil {
			output.Incomplete = true
			continue
		}
		output.Incomplete = output.Incomplete || input.GetIncomplete()
		for _, item := range input.GetItems() {
			if !valid(item) || totalSize+len(item.GetPayload()) > MaxTotalSize {
				output.Incomplete = true
				continue
			}
			totalSize += len(item.GetPayload())
			output.Items = append(output.Items, item)
		}
	}
	if len(output.Items) == 0 {
		return nil
	}
	return output
}

// Select returns payloads for one sidecar type and schema. Unknown sidecar
// types remain opaque; a version mismatch for the selected type is incomplete.
func Select(sidecars *internalpb.ResultSidecars, typeName string, schemaVersion uint32) ([][]byte, bool) {
	if sidecars == nil {
		return nil, true
	}
	payloads := make([][]byte, 0, len(sidecars.GetItems()))
	incomplete := sidecars.GetIncomplete()
	for _, item := range sidecars.GetItems() {
		if item.GetType() != typeName {
			continue
		}
		if !valid(item) || item.GetSchemaVersion() != schemaVersion {
			incomplete = true
			continue
		}
		payloads = append(payloads, item.GetPayload())
	}
	return payloads, incomplete
}

func valid(item *internalpb.ResultSidecar) bool {
	return item != nil && item.GetType() != "" && item.GetSchemaVersion() != 0 &&
		len(item.GetPayload()) > 0 && len(item.GetPayload()) <= MaxPayloadSize
}
