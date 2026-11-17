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

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// MaxScalarIndexesPerField bounds how many scalar indexes one field (or one JSON
// path on a JSON field) may carry.
const MaxScalarIndexesPerField = 10

// CheckMultiScalarIndexPerField gates the creation of a second scalar index on one field
// (shared by Server.CreateIndex and snapshot restore):
//   - "true" (either operator override or once VersionGateSwitcher flips etcd after all
//     online components reach the gate version): allowed.
//   - "auto" before the gate flips (!item.IsSetByUser()): refused with retriable
//     ErrServiceNotReady so callers can retry once the rolling upgrade completes.
//   - explicit "false": refused with non-retriable ErrParameterInvalid, preserving the
//     pre-feature behavior.
func CheckMultiScalarIndexPerField(fieldID int64) error {
	item := &paramtable.Get().DataCoordCfg.EnableMultiScalarIndexPerField
	if item.GetAsBool() {
		return nil
	}
	if !item.IsSetByUser() {
		return merr.WrapErrServiceNotReadyMsg(
			"creating a second index on field %d is disabled by the cluster software upgrade gate "+
				"(a rolling upgrade may still be in progress)",
			fieldID)
	}
	errMsg := "CreateIndex failed: creating multiple indexes on same field is not supported"
	mlog.Warn(context.TODO(), errMsg, mlog.Int64("fieldID", fieldID))
	return merr.WrapErrParameterInvalidMsg("%s", errMsg)
}

// ValidateRestoredIndexesPolicy validates that restored snapshot indexes satisfy
// the single-vector-index invariant, per-(field, json_path) scalar cap, and
// multi-scalar-index software-upgrade gate against the target collection schema.
func ValidateRestoredIndexesPolicy(
	schema *schemapb.CollectionSchema,
	indexes []*indexpb.IndexInfo,
) error {
	if len(indexes) == 0 {
		return nil
	}
	if schema == nil {
		return merr.WrapErrServiceInternalMsg("restore snapshot failed: collection schema is nil")
	}
	type indexIdentity struct {
		fieldID  int64
		jsonPath string
	}
	fieldsByID := make(map[int64]*schemapb.FieldSchema, len(indexes))
	indexesPerIdentity := make(map[indexIdentity]int, len(indexes))
	scalarGroups := make(map[indexIdentity][]*indexpb.IndexInfo, len(indexes))
	seenNames := make(map[string]struct{}, len(indexes))
	for _, indexInfo := range indexes {
		if name := indexInfo.GetIndexName(); name != "" {
			if _, dup := seenNames[name]; dup {
				return merr.WrapErrParameterInvalidMsg(
					"restore snapshot failed: duplicate index name %q", name)
			}
			seenNames[name] = struct{}{}
		}
		fid := indexInfo.GetFieldID()
		field, seen := fieldsByID[fid]
		if !seen {
			field = typeutil.GetField(schema, fid)
			fieldsByID[fid] = field
		}
		var path string
		if field != nil && typeutil.IsJSONType(field.GetDataType()) {
			path, _ = funcutil.GetAttrByKeyFromRepeatedKV(common.JSONPathKey, indexInfo.GetIndexParams())
		}
		id := indexIdentity{fid, path}
		indexesPerIdentity[id]++
		if field == nil || !typeutil.IsVectorType(field.GetDataType()) {
			scalarGroups[id] = append(scalarGroups[id], indexInfo)
		}
	}
	for identity, count := range indexesPerIdentity {
		field := fieldsByID[identity.fieldID]
		if field != nil && typeutil.IsVectorType(field.GetDataType()) {
			if count > 1 {
				return merr.WrapErrParameterInvalidMsg(
					"restore snapshot failed: creating multiple indexes on vector field %d is not supported",
					identity.fieldID)
			}
			continue
		}
		if count > MaxScalarIndexesPerField {
			return merr.WrapErrParameterInvalidMsg(
				"restore snapshot failed: field %d has %d indexes, the limit is %d",
				identity.fieldID, count, MaxScalarIndexesPerField)
		}
		if count < 2 {
			continue
		}
		if err := CheckMultiScalarIndexPerField(identity.fieldID); err != nil {
			return merr.Wrapf(err, "failed to validate indexes on field %d", identity.fieldID)
		}
		isJSON := field != nil && typeutil.IsJSONType(field.GetDataType())
		group := scalarGroups[identity]
		for i := 0; i < len(group); i++ {
			existing := &model.Index{
				CollectionID:    group[i].GetCollectionID(),
				FieldID:         group[i].GetFieldID(),
				IndexID:         group[i].GetIndexID(),
				IndexName:       group[i].GetIndexName(),
				IsAutoIndex:     group[i].GetIsAutoIndex(),
				TypeParams:      group[i].GetTypeParams(),
				IndexParams:     group[i].GetIndexParams(),
				UserIndexParams: group[i].GetUserIndexParams(),
			}
			for j := i + 1; j < len(group); j++ {
				probe := &indexpb.CreateIndexRequest{
					CollectionID:    group[j].GetCollectionID(),
					FieldID:         group[j].GetFieldID(),
					IndexName:       group[j].GetIndexName(),
					IsAutoIndex:     group[j].GetIsAutoIndex(),
					TypeParams:      group[j].GetTypeParams(),
					IndexParams:     group[j].GetIndexParams(),
					UserIndexParams: group[j].GetUserIndexParams(),
				}
				if SameScalarIndexDefinition(existing, probe, isJSON) {
					return merr.WrapErrParameterInvalidMsg(
						"restore snapshot failed: field %d has duplicate scalar index definition (%q and %q)",
						identity.fieldID, group[i].GetIndexName(), group[j].GetIndexName())
				}
			}
		}
	}
	return nil
}
