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

package dql

import (
	"context"
	"strings"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/proxy/taskmodel"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

const (
	// IndexHintKey names the indexes a query asks for, in priority order:
	// "index_hint" = "idx_a,idx_b".
	IndexHintKey = "index_hint"
	// IndexHintModeKey decides what happens when a hinted index cannot be used:
	// "fallback" (default) selects by rule, "strict" fails the query.
	IndexHintModeKey = "index_hint_mode"

	indexHintModeStrict = "strict"
)

// indexHintNames returns the hinted index names and whether strict mode was asked
// for. An absent or empty hint yields no names and no error.
func indexHintNames(params []*commonpb.KeyValuePair) ([]string, bool, error) {
	hint, err := funcutil.GetAttrByKeyFromRepeatedKV(IndexHintKey, params)
	if err != nil || strings.TrimSpace(hint) == "" {
		return nil, false, nil
	}
	mode, _ := funcutil.GetAttrByKeyFromRepeatedKV(IndexHintModeKey, params)
	mode = strings.ToLower(strings.TrimSpace(mode))
	if mode != "" && mode != "fallback" && mode != indexHintModeStrict {
		return nil, false, merr.WrapErrParameterInvalidMsg(
			"invalid %s %q: expected \"fallback\" or %q", IndexHintModeKey, mode, indexHintModeStrict)
	}

	names := make([]string, 0, 4)
	for _, name := range strings.Split(hint, ",") {
		if name = strings.TrimSpace(name); name != "" {
			names = append(names, name)
		}
	}
	return names, mode == indexHintModeStrict, nil
}

// applyIndexHint resolves the hinted index names to (field, index) pairs and attaches
// them to the plan, preserving the user's priority order. Resolution happens here
// because only the coordinator knows the metadata; the selection itself happens in
// segcore, which matches the pairs against each field's loaded indexes.
//
// A name that does not exist fails the request in strict mode and is dropped with a
// warning otherwise. Carrying the field id keeps a hint harmless for fields it does
// not name.
func applyIndexHint(
	ctx context.Context,
	node taskmodel.TaskNode,
	collectionID int64,
	plan *planpb.PlanNode,
	params []*commonpb.KeyValuePair,
) error {
	names, strict, err := indexHintNames(params)
	if err != nil || len(names) == 0 || plan == nil {
		return err
	}

	resp, err := node.MixCoord().DescribeIndex(ctx, &indexpb.DescribeIndexRequest{
		CollectionID: collectionID,
	})
	if err == nil {
		err = merr.Error(resp.GetStatus())
	}
	if err != nil {
		// A lookup failure is not a reason to fail the query: the hint only picks
		// between indexes that the segment would consider anyway. Keep the cause's
		// code (so a retriable failure stays retriable) when the caller insisted.
		if strict {
			return merr.Wrapf(err, "failed to resolve %s", IndexHintKey)
		}
		mlog.Warn(ctx, "index hint ignored: cannot resolve index names",
			mlog.String("hint", IndexHintKey), mlog.Err(err))
		return nil
	}

	byName := make(map[string]*indexpb.IndexInfo, len(resp.GetIndexInfos()))
	for _, info := range resp.GetIndexInfos() {
		byName[info.GetIndexName()] = info
	}

	hints := make([]*planpb.IndexHint, 0, len(names))
	for _, name := range names {
		info, ok := byName[name]
		if !ok {
			if strict {
				return merr.WrapErrParameterInvalidMsg("%s names unknown index %q", IndexHintKey, name)
			}
			mlog.Warn(ctx, "index hint ignored: unknown index name",
				mlog.String("index_name", name))
			continue
		}
		hints = append(hints, &planpb.IndexHint{
			FieldId: info.GetFieldID(),
			IndexId: info.GetIndexID(),
		})
	}
	if len(hints) == 0 {
		return nil
	}

	if plan.GetPlanOptions() == nil {
		plan.PlanOptions = &planpb.PlanOption{}
	}
	plan.PlanOptions.IndexHints = hints
	plan.PlanOptions.IndexHintStrict = strict
	return nil
}
