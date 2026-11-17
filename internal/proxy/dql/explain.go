// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

package dql

import (
	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

const (
	// explainKey is the client status entry that carries the rendered explain.
	explainKey     = "explain"
	reportValueKey = "report_value"
)

// attachExplain writes rendered explain output onto status while keeping the
// compatibility invariant that any non-nil ExtraInfo map carries report_value.
func attachExplain(status **commonpb.Status, rendered string) {
	if rendered == "" {
		return
	}
	if *status == nil {
		*status = merr.Success()
	}
	if (*status).ExtraInfo == nil {
		(*status).ExtraInfo = make(map[string]string, 2)
	}
	if _, ok := (*status).ExtraInfo[reportValueKey]; !ok {
		(*status).ExtraInfo[reportValueKey] = "0"
	}
	(*status).ExtraInfo[explainKey] = rendered
}
