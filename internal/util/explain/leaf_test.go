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

package explain

import (
	"encoding/json"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
)

// segmentNode is one segment subtree as a query node reports it: rows_total on
// the segment, one filter_group child per decision.
func segmentNode(id string, rowsTotal int64, groups ...*internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	return &internalpb.ExecutionExplain{
		Kind:     KindSegment,
		Name:     id,
		Counters: map[string]int64{"rows_total": rowsTotal},
		Children: groups,
	}
}

// indexGroup is a leaf that answered through an index.
func indexGroup(fieldID, exprID, indexID int64, family string, rowsMatched, elementHits int64) *internalpb.ExecutionExplain {
	counters := map[string]int64{
		"leaves":       1,
		"rows_matched": rowsMatched,
		"rows_scanned": 0,
	}
	if elementHits > 0 {
		counters["element_hits"] = elementHits
	}
	return &internalpb.ExecutionExplain{
		Kind: KindFilterGroup,
		Attributes: map[string]string{
			"field_id": strconv.FormatInt(fieldID, 10),
			"expr_id":  strconv.FormatInt(exprID, 10),
			"index_id": strconv.FormatInt(indexID, 10),
			"family":   family,
			"nested":   "false",
		},
		Counters: counters,
	}
}

// rawGroup is a leaf that fell back to scanning the column.
func rawGroup(fieldID, exprID int64, reason string, rowsScanned int64) *internalpb.ExecutionExplain {
	attributes := map[string]string{
		"field_id": strconv.FormatInt(fieldID, 10),
		"expr_id":  strconv.FormatInt(exprID, 10),
		"nested":   "false",
	}
	if reason != "" {
		attributes["decline_reason"] = reason
	}
	return &internalpb.ExecutionExplain{
		Kind:       KindFilterGroup,
		Attributes: attributes,
		Counters:   map[string]int64{"leaves": 1, "rows_matched": 0, "rows_scanned": rowsScanned},
	}
}

// threeSegments is the shape of the leaf view: one condition that always finds
// its index, one that finds it on two segments and falls back on the third, and
// one that has no index at all.
func threeSegments() []*internalpb.ExecutionExplain {
	return []*internalpb.ExecutionExplain{
		segmentNode("1", 50000,
			indexGroup(101, 3, 459123456789100001, "ngram", 30, 0),
			indexGroup(102, 6, 459123456789100002, "bitmap", 4, 0),
			rawGroup(103, 4, "no_index_on_field", 50000),
		),
		segmentNode("2", 50000,
			indexGroup(101, 3, 459123456789100001, "ngram", 30, 0),
			indexGroup(102, 6, 459123456789100002, "bitmap", 4, 0),
			rawGroup(103, 4, "no_index_on_field", 50000),
		),
		segmentNode("3", 50000,
			indexGroup(101, 3, 459123456789100001, "ngram", 30, 0),
			rawGroup(102, 6, "index_unusable", 50000),
			rawGroup(103, 4, "no_index_on_field", 50000),
		),
	}
}

func threeLeaves() map[int64]LeafInfo {
	return map[int64]LeafInfo{
		3: {Condition: `title contains "%降噪%"`, Type: "contains", FieldID: 101},
		4: {Condition: "price in (100, +inf)", Type: "range", FieldID: 103},
		6: {Condition: `brand == "Acme"`, Type: "equal", FieldID: 102},
	}
}

func TestSummarizeMixedDecisions(t *testing.T) {
	summary := Summarize(threeSegments(), threeLeaves())

	assert.Equal(t, "summary", summary.Level)
	assert.Equal(t, 3, summary.Segments)
	assert.Equal(t, int64(150000), summary.RowsTotal)

	require.Equal(t, []Condition{
		{
			ExprID:      3,
			Condition:   `title contains "%降噪%"`,
			Type:        "contains",
			FieldID:     101,
			Segments:    3,
			RowsMatched: 90,
			Used: []IndexUse{{
				Kind:        "scalar_index",
				IndexID:     459123456789100001,
				Family:      "ngram",
				Segments:    3,
				RowsMatched: 90,
			}},
			Reasons: []ReasonCount{},
		},
		{
			ExprID:          4,
			Condition:       "price in (100, +inf)",
			Type:            "range",
			FieldID:         103,
			Segments:        3,
			RowsScanned:     150000,
			Used:            []IndexUse{},
			RawScanSegments: 3,
			Reasons:         []ReasonCount{{Reason: "no_index_on_field", Segments: 3}},
		},
		{
			ExprID:          6,
			Condition:       `brand == "Acme"`,
			Type:            "equal",
			FieldID:         102,
			Segments:        3,
			RowsMatched:     8,
			RowsScanned:     50000,
			Used:            []IndexUse{{Kind: "scalar_index", IndexID: 459123456789100002, Family: "bitmap", Segments: 2, RowsMatched: 8}},
			RawScanSegments: 1,
			Reasons:         []ReasonCount{{Reason: "index_unusable", Segments: 1}},
		},
	}, summary.Conditions)
}

func TestSummarizeCountsSegmentOnce(t *testing.T) {
	// One segment that a query node reports twice (replica, or one report per
	// reduce pass), and that reports the same decision twice inside one report.
	report := func() *internalpb.ExecutionExplain {
		return segmentNode("7", 1000,
			indexGroup(102, 6, 11, "bitmap", 40, 0),
			indexGroup(102, 6, 11, "bitmap", 40, 0),
			rawGroup(103, 4, "no_index_on_field", 900),
		)
	}

	summary := Summarize([]*internalpb.ExecutionExplain{report(), report()}, threeLeaves())

	assert.Equal(t, 1, summary.Segments)
	assert.Equal(t, int64(1000), summary.RowsTotal)
	require.Len(t, summary.Conditions, 2)

	price := summary.Conditions[0]
	assert.Equal(t, int64(4), price.ExprID)
	assert.Equal(t, 1, price.Segments)
	assert.Equal(t, int64(900), price.RowsScanned)
	assert.Equal(t, 1, price.RawScanSegments)
	assert.Equal(t, []ReasonCount{{Reason: "no_index_on_field", Segments: 1}}, price.Reasons)

	brand := summary.Conditions[1]
	assert.Equal(t, int64(6), brand.ExprID)
	assert.Equal(t, 1, brand.Segments)
	assert.Equal(t, int64(40), brand.RowsMatched)
	require.Len(t, brand.Used, 1)
	assert.Equal(t, 1, brand.Used[0].Segments)
}

func TestSummarizeUnknownLeaf(t *testing.T) {
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 100, rawGroup(101, 9, "", 100)),
	}, threeLeaves())

	require.Len(t, summary.Conditions, 1)
	condition := summary.Conditions[0]
	assert.Equal(t, int64(9), condition.ExprID)
	assert.Equal(t, "field_101", condition.Condition, "an expr the plan table does not know falls back to its field")
	assert.Equal(t, "unknown", condition.Type)
	assert.Equal(t, int64(101), condition.FieldID)
	assert.Equal(t, []ReasonCount{{Reason: "unknown", Segments: 1}}, condition.Reasons,
		"a raw fallback that names no reason is unknown")
	assert.Contains(t, RenderSummaryText(summary), "{decline_reason=unknown}\n")
}

func TestSummarizeOrdersConditionsAndIndexes(t *testing.T) {
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 100,
			indexGroup(105, 0, 30, "sort", 1, 0),
			indexGroup(101, 5, 20, "sort", 4, 0),
			indexGroup(101, 5, 20, "inverted", 2, 0),
			indexGroup(101, 5, 10, "bitmap", 3, 0),
			rawGroup(105, 2, "index_unusable", 10),
		),
	}, nil)

	require.Len(t, summary.Conditions, 3)
	assert.Equal(t, []int64{2, 5, 0}, []int64{
		summary.Conditions[0].ExprID,
		summary.Conditions[1].ExprID,
		summary.Conditions[2].ExprID,
	}, "exprs with an id come first, ascending; exprs without an id come last")

	used := summary.Conditions[1].Used
	require.Len(t, used, 3)
	assert.Equal(t, []int64{10, 20, 20}, []int64{used[0].IndexID, used[1].IndexID, used[2].IndexID})
	assert.Equal(t, []string{"bitmap", "inverted", "sort"}, []string{used[0].Family, used[1].Family, used[2].Family},
		"one index id, several families: the family breaks the tie")

	assert.Equal(t, "field_105", summary.Conditions[2].Condition)
	assert.Equal(t, "unknown", summary.Conditions[2].Type)
}

func TestSummarizeKeepsDecisionsApart(t *testing.T) {
	// Two leaves of one expr on one segment: one at the root of a json column and
	// one nested inside it. They are two decisions, so both count, but they answer
	// with the same index, so the index line merges them back.
	leaf := func(nested string, rowsMatched int64) *internalpb.ExecutionExplain {
		return &internalpb.ExecutionExplain{
			Kind: KindFilterGroup,
			Attributes: map[string]string{
				"field_id": "101",
				"expr_id":  "3",
				"index_id": "10",
				"family":   "bitmap",
				"nested":   nested,
			},
			Counters: map[string]int64{"leaves": 1, "rows_matched": rowsMatched},
		}
	}

	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 100, leaf("false", 2), leaf("true", 3)),
	}, threeLeaves())

	require.Len(t, summary.Conditions, 1)
	condition := summary.Conditions[0]
	assert.Equal(t, int64(5), condition.RowsMatched)
	require.Len(t, condition.Used, 1)
	assert.Equal(t, 1, condition.Used[0].Segments)
	assert.Equal(t, int64(5), condition.Used[0].RowsMatched)
}

func TestSummarizeElementHits(t *testing.T) {
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 10, indexGroup(101, 3, 11, "inverted", 2, 5)),
		segmentNode("2", 20, indexGroup(101, 3, 11, "inverted", 3, 7)),
	}, map[int64]LeafInfo{3: {Condition: `j["a"] == 1`, Type: "json_contains", FieldID: 101}})

	require.Len(t, summary.Conditions, 1)
	condition := summary.Conditions[0]
	assert.Equal(t, int64(5), condition.RowsMatched)
	assert.Equal(t, int64(12), condition.ElementHits, "element hits stay in the element domain")
	require.Len(t, condition.Used, 1)
	assert.Equal(t, int64(12), condition.Used[0].ElementHits)

	assert.Equal(t, ""+
		"query [rows_total=30, segments=2]\n"+
		`  expr j["a"] == 1 [element_hits=12, rows_matched=5, rows_scanned=0, segments=2] {expr_id=3, field_id=101, type=json_contains}`+"\n"+
		"    index 11 inverted [element_hits=12, rows_matched=5, rows_scanned=0, segments=2]\n",
		RenderSummaryText(summary))

	encoded, err := RenderSummaryJSON(summary)
	require.NoError(t, err)
	assert.Contains(t, encoded, `"element_hits":12`)
}

func TestSummarizeEmpty(t *testing.T) {
	summary := Summarize(nil, nil)

	require.NotNil(t, summary)
	assert.Equal(t, "summary", summary.Level)
	assert.Equal(t, 0, summary.Segments)
	assert.Equal(t, int64(0), summary.RowsTotal)
	assert.NotNil(t, summary.Conditions)
	assert.Empty(t, summary.Conditions)

	encoded, err := RenderSummaryJSON(summary)
	require.NoError(t, err)
	assert.JSONEq(t, `{"level":"summary","segments":0,"rows_total":0,"conditions":[]}`, encoded)
	assert.Equal(t, "query [rows_total=0, segments=0]\n", RenderSummaryText(summary))
	assert.Equal(t, "", RenderSummaryText(nil))
}

func TestSummarizeSkipsEmptyNodes(t *testing.T) {
	summary := Summarize([]*internalpb.ExecutionExplain{
		nil,
		segmentNode("1", 10, nil, &internalpb.ExecutionExplain{Kind: KindQueryNode, Name: "1"}),
	}, nil)

	assert.Equal(t, 1, summary.Segments)
	assert.Equal(t, int64(10), summary.RowsTotal)
	assert.Empty(t, summary.Conditions)
}

func TestRenderSummaryText(t *testing.T) {
	assert.Equal(t, ""+
		"query [rows_total=150000, segments=3]\n"+
		`  expr title contains "%降噪%" [rows_matched=90, rows_scanned=0, segments=3] {expr_id=3, field_id=101, type=contains}`+"\n"+
		"    index 459123456789100001 ngram [rows_matched=90, rows_scanned=0, segments=3]\n"+
		"  expr price in (100, +inf) [rows_matched=0, rows_scanned=150000, segments=3] {expr_id=4, field_id=103, type=range}\n"+
		"    raw [rows_matched=0, rows_scanned=150000, segments=3] {decline_reason=no_index_on_field}\n"+
		`  expr brand == "Acme" [rows_matched=8, rows_scanned=50000, segments=3] {expr_id=6, field_id=102, type=equal}`+"\n"+
		"    index 459123456789100002 bitmap [rows_matched=8, rows_scanned=0, segments=2]\n"+
		"    raw [rows_matched=0, rows_scanned=50000, segments=1] {decline_reason=index_unusable}\n",
		RenderSummaryText(Summarize(threeSegments(), threeLeaves())))
}

func TestRenderSummaryTextOrdersReasons(t *testing.T) {
	// Hand-built: the renderer sorts conditions, reasons and exprs without an id
	// on its own. Every reason of a condition carries what its raw leaves scanned
	// in total, because the reason breakdown only carries segment counts.
	summary := &Summary{
		Level:     "summary",
		Segments:  2,
		RowsTotal: 900,
		Conditions: []Condition{
			{
				ExprID:          0,
				Condition:       "field_106",
				Type:            "unknown",
				FieldID:         106,
				Segments:        1,
				RowsScanned:     300,
				Used:            []IndexUse{},
				RawScanSegments: 1,
				Reasons:         []ReasonCount{{Reason: "no_index_on_field", Segments: 1}},
			},
			{
				ExprID:          7,
				Condition:       "v > 1",
				Type:            "range",
				FieldID:         105,
				Segments:        2,
				RowsScanned:     600,
				Used:            []IndexUse{},
				RawScanSegments: 2,
				Reasons: []ReasonCount{
					{Reason: "no_index_on_field", Segments: 1},
					{Reason: "index_unusable", Segments: 1},
				},
			},
		},
	}

	assert.Equal(t, ""+
		"query [rows_total=900, segments=2]\n"+
		"  expr v > 1 [rows_matched=0, rows_scanned=600, segments=2] {expr_id=7, field_id=105, type=range}\n"+
		"    raw [rows_matched=0, rows_scanned=600, segments=1] {decline_reason=index_unusable}\n"+
		"    raw [rows_matched=0, rows_scanned=600, segments=1] {decline_reason=no_index_on_field}\n"+
		"  expr field_106 [rows_matched=0, rows_scanned=300, segments=1] {expr_id=0, field_id=106, type=unknown}\n"+
		"    raw [rows_matched=0, rows_scanned=300, segments=1] {decline_reason=no_index_on_field}\n",
		RenderSummaryText(summary))
}

func TestRenderSummaryJSON(t *testing.T) {
	summary := Summarize(threeSegments(), threeLeaves())

	encoded, err := RenderSummaryJSON(summary)
	require.NoError(t, err)

	var decoded struct {
		Level      string `json:"level"`
		Segments   int    `json:"segments"`
		RowsTotal  int64  `json:"rows_total"`
		Conditions []struct {
			ExprID          int64         `json:"expr_id"`
			Condition       string        `json:"condition"`
			Type            string        `json:"type"`
			FieldID         int64         `json:"field_id"`
			Segments        int           `json:"segments"`
			RowsMatched     int64         `json:"rows_matched"`
			RowsScanned     int64         `json:"rows_scanned"`
			Used            []IndexUse    `json:"used"`
			RawScanSegments int           `json:"raw_scan_segments"`
			Reasons         []ReasonCount `json:"reasons"`
		} `json:"conditions"`
	}
	require.NoError(t, json.Unmarshal([]byte(encoded), &decoded))

	assert.Equal(t, "summary", decoded.Level)
	assert.Equal(t, 3, decoded.Segments)
	assert.Equal(t, int64(150000), decoded.RowsTotal)
	require.Len(t, decoded.Conditions, 3)

	brand := decoded.Conditions[2]
	assert.Equal(t, int64(6), brand.ExprID)
	assert.Equal(t, `brand == "Acme"`, brand.Condition)
	assert.Equal(t, "equal", brand.Type)
	assert.Equal(t, int64(102), brand.FieldID)
	assert.Equal(t, 3, brand.Segments)
	assert.Equal(t, int64(8), brand.RowsMatched)
	assert.Equal(t, int64(50000), brand.RowsScanned)
	assert.Equal(t, 1, brand.RawScanSegments)
	assert.Equal(t, []IndexUse{{
		Kind:        "scalar_index",
		IndexID:     459123456789100002,
		Family:      "bitmap",
		Segments:    2,
		RowsMatched: 8,
	}}, brand.Used)
	assert.Equal(t, []ReasonCount{{Reason: "index_unusable", Segments: 1}}, brand.Reasons)

	// Empty lists are [] and not null, and a zero counter is left out.
	assert.Contains(t, encoded, `"used":[]`)
	assert.Contains(t, encoded, `"reasons":[]`)
	assert.NotContains(t, encoded, "null")
	assert.NotContains(t, encoded, "element_hits")

	var raw map[string]any
	require.NoError(t, json.Unmarshal([]byte(encoded), &raw))
	conditions, ok := raw["conditions"].([]any)
	require.True(t, ok)
	require.Len(t, conditions, 3)
	title, ok := conditions[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, []any{}, title["reasons"])
	price, ok := conditions[1].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, []any{}, price["used"])
}
