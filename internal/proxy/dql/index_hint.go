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

	"github.com/tidwall/gjson"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/proxy/metacache"
	"github.com/milvus-io/milvus/internal/proxy/taskmodel"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
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

type parsedIndexHint struct {
	names   []string
	strict  bool
	hasHint bool
	hasMode bool
}

// lookupStringParamWithNestedFallback resolves key from top-level KV first and
// falls back to nested "params" JSON only when the top-level key is absent.
func lookupStringParamWithNestedFallback(key string, params []*commonpb.KeyValuePair) (string, bool, error) {
	if topVal, err := funcutil.GetAttrByKeyFromRepeatedKV(key, params); err == nil {
		return topVal, true, nil
	}
	if rawParams, err := funcutil.GetAttrByKeyFromRepeatedKV(ParamsKey, params); err == nil && strings.TrimSpace(rawParams) != "" {
		res := gjson.Get(rawParams, key)
		if res.Exists() && res.Type != gjson.Null {
			if res.Type != gjson.String {
				return "", false, merr.WrapErrParameterInvalidMsg("%s in %s must be a string", key, ParamsKey)
			}
			return res.String(), true, nil
		}
	}
	return "", false, nil
}

func parseIndexHintParams(params []*commonpb.KeyValuePair) (parsedIndexHint, error) {
	hint, hasHint, err := lookupStringParamWithNestedFallback(IndexHintKey, params)
	if err != nil {
		return parsedIndexHint{}, err
	}
	mode, hasMode, err := lookupStringParamWithNestedFallback(IndexHintModeKey, params)
	if err != nil {
		return parsedIndexHint{}, err
	}
	strict := false
	if hasMode {
		mode = strings.ToLower(strings.TrimSpace(mode))
		if mode != "" && mode != "fallback" && mode != indexHintModeStrict {
			return parsedIndexHint{}, merr.WrapErrParameterInvalidMsg(
				"invalid %s %q: expected \"fallback\" or %q", IndexHintModeKey, mode, indexHintModeStrict)
		}
		strict = mode == indexHintModeStrict
	}
	var names []string
	if hasHint && strings.TrimSpace(hint) != "" {
		names = make([]string, 0, 4)
		for _, name := range strings.Split(hint, ",") {
			if name = strings.TrimSpace(name); name != "" {
				names = append(names, name)
			}
		}
	}
	return parsedIndexHint{
		names:   names,
		strict:  strict,
		hasHint: len(names) > 0,
		hasMode: hasMode && mode != "",
	}, nil
}

func mergeIndexHint(outer, sub parsedIndexHint) ([]string, bool) {
	names := sub.names
	if outer.hasHint {
		names = outer.names
	}
	strict := sub.strict
	if outer.hasMode {
		strict = outer.strict
	}
	return names, strict
}

// indexHintNames returns the hinted index names and whether strict mode was asked
// for. An absent or empty hint yields no names and no error.
func indexHintNames(params []*commonpb.KeyValuePair) ([]string, bool, error) {
	parsed, err := parseIndexHintParams(params)
	if err != nil {
		return nil, false, err
	}
	return parsed.names, parsed.strict, nil
}

func resolveCollectionIndexes(
	ctx context.Context,
	node taskmodel.TaskNode,
	collectionID int64,
) (*metacache.CollectionIndexes, error) {
	if mc := node.GetMetaCache(); mc != nil {
		return mc.GetCollectionIndexes(ctx, collectionID)
	}
	return metacache.FetchCollectionIndexes(ctx, node.MixCoord(), collectionID)
}

// applyIndexHint resolves the hinted index names to (field, index) entries and
// attaches them to the plan, preserving the user's priority order.
func applyIndexHint(
	ctx context.Context,
	node taskmodel.TaskNode,
	collectionID int64,
	schema *schemapb.CollectionSchema,
	plan *planpb.PlanNode,
	params []*commonpb.KeyValuePair,
) error {
	names, strict, err := indexHintNames(params)
	if err != nil {
		return err
	}
	return applyResolvedIndexHint(ctx, node, collectionID, schema, plan, names, strict)
}

func applyResolvedIndexHint(
	ctx context.Context,
	node taskmodel.TaskNode,
	collectionID int64,
	schema *schemapb.CollectionSchema,
	plan *planpb.PlanNode,
	names []string,
	strict bool,
) error {
	if len(names) == 0 || plan == nil {
		return nil
	}

	var vectorFields map[int64]bool
	if schema != nil {
		for _, field := range typeutil.GetAllFieldSchemas(schema) {
			if typeutil.IsVectorType(field.GetDataType()) {
				if vectorFields == nil {
					vectorFields = make(map[int64]bool)
				}
				vectorFields[field.GetFieldID()] = true
			}
		}
	}

	colIndexes, err := resolveCollectionIndexes(ctx, node, collectionID)
	if err != nil {
		// A lookup failure is not a reason to fail a fallback query: keep the
		// cause's code (so a retriable failure stays retriable) when strict.
		if strict {
			return merr.Wrapf(err, "failed to resolve %s", IndexHintKey)
		}
		mlog.Warn(ctx, "index hint ignored: cannot resolve index names",
			mlog.String("hint", IndexHintKey), mlog.Err(err))
		return nil
	}

	byName := colIndexes.ByName
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
		if vectorFields[info.GetFieldID()] {
			if strict {
				return merr.WrapErrParameterInvalidMsg("%s names vector index %q; only scalar indexes are supported", IndexHintKey, name)
			}
			mlog.Warn(ctx, "index hint ignored: vector index is not supported",
				mlog.String("index_name", name))
			continue
		}
		if !common.IsIndexInfoVisible(info.GetIndexParams(), info.GetUserIndexParams()) {
			if strict {
				return merr.WrapErrParameterInvalidMsg("%s names invisible index %q", IndexHintKey, name)
			}
			mlog.Warn(ctx, "index hint ignored: index is not visible",
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
