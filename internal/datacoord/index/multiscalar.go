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

package index

// The multi-index-per-field switch gates every path that can create a second scalar
// index on one field: the CreateIndex policy and the snapshot restore path.
import (
	"context"

	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func CheckMultiScalarIndexEnabled(fieldID int64) error {
	item := &paramtable.Get().DataCoordCfg.EnableMultiScalarIndexPerField
	if item.GetAsBool() {
		return nil
	}
	errMsg := "CreateIndex failed: creating multiple indexes on same field is not supported"
	mlog.Warn(context.TODO(), errMsg, mlog.Int64("fieldID", fieldID))
	return merr.WrapErrParameterInvalidMsg("%s", errMsg)
}

// CheckMultiScalarIndexEngineVersion is the version half: a second index on a field may
// only be created once every QueryNode reports a scalar index engine version that can
// load such a segment (an older one asserts in LoadScalarIndex). Unknown counts as the
// oldest cluster, mirroring snapshot_manager's nil guard. Retriable on purpose: the
// answer changes as a rolling upgrade progresses.
func CheckMultiScalarIndexEngineVersion(fieldID int64, resolved int32) error {
	if resolved >= common.MinScalarIndexVersionForScalarMultiIndex {
		return nil
	}
	return merr.WrapErrServiceNotReadyMsg(
		"creating a second index on field %d requires scalar index engine version >= %d, "+
			"current resolved version: %d (a rolling upgrade may still be in progress)",
		fieldID, common.MinScalarIndexVersionForScalarMultiIndex, resolved)
}

// CheckMultiScalarIndexPerField gates the creation of a second index on one field. The
// switch and the engine version are checked separately, and both helpers are also used
// by the paths that broadcast CreateIndex themselves (snapshot restore), which is why
// they are package-level.
func CheckMultiScalarIndexPerField(fieldID int64, resolved int32) error {
	if err := CheckMultiScalarIndexEnabled(fieldID); err != nil {
		return err
	}
	return CheckMultiScalarIndexEngineVersion(fieldID, resolved)
}

// checkIndexCreationPolicy rejects the shapes of CreateIndex that the segment
// runtime cannot serve, before any metadata is written.
//
// Two shapes are rejected, and both need the field to already carry an index:
//
//   - a second index on a VECTOR field. The segment keeps one vector index per
//     field, search does not carry an indexID, and QueryNode rejects a second one
//     outright (internal/querynodev2/services.go), so this stays rejected
//     unconditionally.
//
//   - a second index under the same (field, json_path) SCALAR identity. This one
//     is gated by dataCoord.enableMultiScalarIndexPerField, which is three-valued:
//     "auto" (the default) allows it only once the cluster reports scalar index
//     engine version >= common.MinScalarIndexVersionForScalarMultiIndex -- i.e. once
//     every QueryNode can load such a segment, which also means the capability turns
//     itself off again if a QueryNode is rolled back; "false" refuses it outright
//     (the escape hatch, and the safe value while older QueryNodes are online);
//     "true" forces it on and bypasses the readiness check. An older QueryNode keeps
//     at most one scalar index per field and its
//     ChunkedSegmentSealedImpl::LoadScalarIndex asserts while loading a segment
//     that carries two. See checkMultiScalarIndexPerField for the details.
//
// JSON indexes under DIFFERENT paths are the pre-existing multi-path capability,
// which an older QueryNode loads correctly, so they are deliberately allowed
// without any check.
