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

// indexCounters is the counters of a leaf the index answered: the index covers
// every input row, and rawScan is what it read back from the columns for
// candidates (0 for a plain index).
func indexCounters(rowsIn, rowsOut, rawScan int64) map[string]int64 {
	return map[string]int64{
		"leaves":   1,
		"rows_in":  rowsIn,
		"idx_scan": rowsIn,
		"raw_scan": rawScan,
		"rows_out": rowsOut,
	}
}

// rawCounters is the counters of a leaf that scanned the column: it covers
// nothing with an index and reads every row it receives.
func rawCounters(rowsIn, rowsOut int64) map[string]int64 {
	return map[string]int64{
		"leaves":   1,
		"rows_in":  rowsIn,
		"idx_scan": 0,
		"raw_scan": rowsIn,
		"rows_out": rowsOut,
	}
}

// elementHits adds the element-domain hits a nested index reports.
func elementHits(counters map[string]int64, hits int64) map[string]int64 {
	counters["element_hits"] = hits
	return counters
}

// indexGroup is a leaf that answered through an index.
func indexGroup(fieldID, exprID, indexID int64, family string, counters map[string]int64) *internalpb.ExecutionExplain {
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
func rawGroup(fieldID, exprID int64, reason string, counters map[string]int64) *internalpb.ExecutionExplain {
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
		Counters:   counters,
	}
}

// legacyIndexGroup is an index leaf as a query node that predates the four axes
// reports it: rows_matched only, no rows_in and no idx_scan.
func legacyIndexGroup(fieldID, exprID, indexID int64, family string, rowsMatched int64) *internalpb.ExecutionExplain {
	return indexGroup(fieldID, exprID, indexID, family, map[string]int64{
		"leaves":       1,
		"rows_matched": rowsMatched,
		"rows_scanned": 0,
	})
}

// legacyRawGroup is a raw leaf of the same older contract.
func legacyRawGroup(fieldID, exprID int64, reason string, rowsScanned int64) *internalpb.ExecutionExplain {
	return rawGroup(fieldID, exprID, reason, map[string]int64{
		"leaves":       1,
		"rows_matched": 0,
		"rows_scanned": rowsScanned,
	})
}

// threeSegments is the shape of the leaf view: one condition that always finds
// its index, one that finds it on two segments and falls back on the third, and
// one that has no index at all.
func threeSegments() []*internalpb.ExecutionExplain {
	return []*internalpb.ExecutionExplain{
		segmentNode("1", 50000,
			indexGroup(101, 3, 459123456789100001, "ngram", indexCounters(50000, 30, 0)),
			indexGroup(102, 6, 459123456789100002, "bitmap", indexCounters(50000, 100, 0)),
			rawGroup(103, 4, "no_index_on_field", rawCounters(50000, 5842)),
		),
		segmentNode("2", 50000,
			indexGroup(101, 3, 459123456789100001, "ngram", indexCounters(50000, 30, 0)),
			indexGroup(102, 6, 459123456789100002, "bitmap", indexCounters(50000, 100, 0)),
			rawGroup(103, 4, "no_index_on_field", rawCounters(50000, 5842)),
		),
		segmentNode("3", 50000,
			indexGroup(101, 3, 459123456789100001, "ngram", indexCounters(50000, 30, 0)),
			rawGroup(102, 6, "index_unusable", rawCounters(50000, 37)),
			rawGroup(103, 4, "no_index_on_field", rawCounters(50000, 5842)),
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
	assert.Equal(t, int64(450000), summary.RowsIn, "the root sums the input rows of its conditions")
	assert.Equal(t, int64(17853), summary.RowsOut, "the root sums the rows its conditions matched")

	require.Equal(t, []Condition{
		{
			ExprID:    3,
			Condition: `title contains "%降噪%"`,
			Type:      "contains",
			FieldID:   101,
			Segments:  3,
			IdxScan:   150000,
			RawScan:   0,
			RowsIn:    150000,
			RowsOut:   90,
			Used: []IndexUse{{
				Kind:     "scalar_index",
				IndexID:  459123456789100001,
				Family:   "ngram",
				Segments: 3,
				IdxScan:  150000,
				RawScan:  0,
				RowsIn:   150000,
				RowsOut:  90,
			}},
			Reasons: []ReasonCount{},
		},
		{
			ExprID:          4,
			Condition:       "price in (100, +inf)",
			Type:            "range",
			FieldID:         103,
			Segments:        3,
			IdxScan:         0,
			RawScan:         150000,
			RowsIn:          150000,
			RowsOut:         17526,
			Used:            []IndexUse{},
			RawScanSegments: 3,
			Reasons: []ReasonCount{{
				Reason:   "no_index_on_field",
				Segments: 3,
				RawScan:  150000,
				RowsIn:   150000,
				RowsOut:  17526,
			}},
		},
		{
			ExprID:    6,
			Condition: `brand == "Acme"`,
			Type:      "equal",
			FieldID:   102,
			Segments:  3,
			IdxScan:   100000,
			RawScan:   50000,
			RowsIn:    150000,
			RowsOut:   237,
			Used: []IndexUse{{
				Kind:     "scalar_index",
				IndexID:  459123456789100002,
				Family:   "bitmap",
				Segments: 2,
				IdxScan:  100000,
				RawScan:  0,
				RowsIn:   100000,
				RowsOut:  200,
			}},
			RawScanSegments: 1,
			Reasons: []ReasonCount{{
				Reason:   "index_unusable",
				Segments: 1,
				RawScan:  50000,
				RowsIn:   50000,
				RowsOut:  37,
			}},
		},
	}, summary.Conditions)
}

func TestSummarizeCountsSegmentOnce(t *testing.T) {
	// One segment that a query node reports twice (replica, or one report per
	// reduce pass), and that reports the same decision twice inside one report.
	report := func() *internalpb.ExecutionExplain {
		return segmentNode("7", 1000,
			indexGroup(102, 6, 11, "bitmap", indexCounters(1000, 40, 0)),
			indexGroup(102, 6, 11, "bitmap", indexCounters(1000, 40, 0)),
			rawGroup(103, 4, "no_index_on_field", rawCounters(1000, 12)),
		)
	}

	summary := Summarize([]*internalpb.ExecutionExplain{report(), report()}, threeLeaves())

	assert.Equal(t, 1, summary.Segments)
	assert.Equal(t, int64(2000), summary.RowsIn, "every counter of a repeated decision is taken once")
	assert.Equal(t, int64(52), summary.RowsOut)
	require.Len(t, summary.Conditions, 2)

	price := summary.Conditions[0]
	assert.Equal(t, int64(4), price.ExprID)
	assert.Equal(t, 1, price.Segments)
	assert.Equal(t, int64(0), price.IdxScan)
	assert.Equal(t, int64(1000), price.RawScan)
	assert.Equal(t, int64(1000), price.RowsIn)
	assert.Equal(t, int64(12), price.RowsOut)
	assert.Equal(t, 1, price.RawScanSegments)
	assert.Equal(t, []ReasonCount{{
		Reason:   "no_index_on_field",
		Segments: 1,
		RawScan:  1000,
		RowsIn:   1000,
		RowsOut:  12,
	}}, price.Reasons)

	brand := summary.Conditions[1]
	assert.Equal(t, int64(6), brand.ExprID)
	assert.Equal(t, 1, brand.Segments)
	assert.Equal(t, int64(1000), brand.IdxScan)
	assert.Equal(t, int64(0), brand.RawScan)
	assert.Equal(t, int64(1000), brand.RowsIn)
	assert.Equal(t, int64(40), brand.RowsOut)
	require.Len(t, brand.Used, 1)
	assert.Equal(t, 1, brand.Used[0].Segments)
	assert.Equal(t, int64(1000), brand.Used[0].RowsIn)
	assert.Equal(t, int64(40), brand.Used[0].RowsOut)
}

func TestSummarizeLegacyCounters(t *testing.T) {
	// A query node that predates the four axes reports rows_matched /
	// rows_scanned only. An index leaf of that contract knew neither its input
	// rows nor that its rows came from the index; a raw leaf scanned everything
	// it received.
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 1000,
			legacyIndexGroup(102, 6, 11, "bitmap", 40),
			legacyRawGroup(103, 4, "no_index_on_field", 900),
		),
	}, threeLeaves())

	assert.Equal(t, 1, summary.Segments)
	assert.Equal(t, int64(940), summary.RowsIn)
	assert.Equal(t, int64(40), summary.RowsOut)
	require.Len(t, summary.Conditions, 2)

	price := summary.Conditions[0]
	assert.Equal(t, int64(4), price.ExprID)
	assert.Equal(t, int64(0), price.IdxScan, "a raw leaf scans no index")
	assert.Equal(t, int64(900), price.RawScan)
	assert.Equal(t, int64(900), price.RowsIn, "a raw leaf receives what it scans")
	assert.Equal(t, int64(0), price.RowsOut)
	require.Len(t, price.Reasons, 1)

	brand := summary.Conditions[1]
	assert.Equal(t, int64(6), brand.ExprID)
	assert.Equal(t, int64(40), brand.IdxScan, "an old index leaf covered what it matched")
	assert.Equal(t, int64(0), brand.RawScan)
	assert.Equal(t, int64(40), brand.RowsIn)
	assert.Equal(t, int64(40), brand.RowsOut)
	assert.Equal(t, int64(0), brand.ElementHits, "an old leaf without element_hits reports none")
	require.Len(t, brand.Used, 1)
	assert.Equal(t, int64(40), brand.Used[0].IdxScan)
	assert.Equal(t, int64(0), brand.Used[0].RawScan)
	assert.Equal(t, int64(40), brand.Used[0].RowsIn)
	assert.Equal(t, int64(40), brand.Used[0].RowsOut)
}

func TestSummarizeIndexCandidateLookup(t *testing.T) {
	// An index that reads candidates back from the columns scans raw rows on top
	// of covering its input; a raw decision of the same expr sits beside it.
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 100000,
			indexGroup(101, 3, 11, "ngram", indexCounters(50000, 60, 800)),
			rawGroup(101, 3, "index_unusable", rawCounters(50000, 37)),
		),
	}, threeLeaves())

	require.Len(t, summary.Conditions, 1)
	condition := summary.Conditions[0]
	assert.Equal(t, int64(50000), condition.IdxScan)
	assert.Equal(t, int64(50800), condition.RawScan, "a candidate lookup scans on top of the index path")
	assert.Equal(t, int64(100000), condition.RowsIn)
	assert.Equal(t, int64(97), condition.RowsOut)
	require.Len(t, condition.Used, 1)
	assert.Equal(t, int64(800), condition.Used[0].RawScan)

	assert.Equal(t, ""+
		"query [rows_in=100000, rows_out=97, segments=1]\n"+
		`  expr title contains "%降噪%" [idx_scan=50000, raw_scan=50800, rows_in=100000, rows_out=97, segments=1] {expr_id=3, field_id=101, type=contains}`+"\n"+
		"    index 11 ngram [idx_scan=50000, raw_scan=800, rows_in=50000, rows_out=60, segments=1]\n"+
		"    raw [idx_scan=0, raw_scan=50000, rows_in=50000, rows_out=37, segments=1] {decline_reason=index_unusable}\n",
		RenderSummaryText(summary), "the raw line carries what the raw leaves processed, not the index lookups")
}

func TestSummarizeUnknownLeaf(t *testing.T) {
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 100, rawGroup(101, 9, "", rawCounters(100, 3))),
	}, threeLeaves())

	require.Len(t, summary.Conditions, 1)
	condition := summary.Conditions[0]
	assert.Equal(t, int64(9), condition.ExprID)
	assert.Equal(t, "field_101", condition.Condition, "an expr the plan table does not know falls back to its field")
	assert.Equal(t, "unknown", condition.Type)
	assert.Equal(t, int64(101), condition.FieldID)
	assert.Equal(t, []ReasonCount{{
		Reason:   "unknown",
		Segments: 1,
		RawScan:  100,
		RowsIn:   100,
		RowsOut:  3,
	}}, condition.Reasons,
		"a raw fallback that names no reason is unknown")
	assert.Contains(t, RenderSummaryText(summary), "{decline_reason=unknown}\n")
}

func TestSummarizeOrdersConditionsAndIndexes(t *testing.T) {
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 100,
			indexGroup(105, 0, 30, "sort", indexCounters(100, 1, 0)),
			indexGroup(101, 5, 20, "sort", indexCounters(100, 4, 0)),
			indexGroup(101, 5, 20, "inverted", indexCounters(100, 2, 0)),
			indexGroup(101, 5, 10, "bitmap", indexCounters(100, 3, 0)),
			rawGroup(105, 2, "index_unusable", rawCounters(100, 10)),
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
	leaf := func(nested string, rows int64) *internalpb.ExecutionExplain {
		return &internalpb.ExecutionExplain{
			Kind: KindFilterGroup,
			Attributes: map[string]string{
				"field_id": "101",
				"expr_id":  "3",
				"index_id": "10",
				"family":   "bitmap",
				"nested":   nested,
			},
			Counters: indexCounters(rows, rows, 0),
		}
	}

	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 100, leaf("false", 2), leaf("true", 3)),
	}, threeLeaves())

	require.Len(t, summary.Conditions, 1)
	condition := summary.Conditions[0]
	assert.Equal(t, int64(5), condition.RowsIn)
	assert.Equal(t, int64(5), condition.IdxScan)
	assert.Equal(t, int64(5), condition.RowsOut)
	require.Len(t, condition.Used, 1)
	assert.Equal(t, 1, condition.Used[0].Segments)
	assert.Equal(t, int64(5), condition.Used[0].RowsIn)
	assert.Equal(t, int64(5), condition.Used[0].RowsOut)
}

func TestSummarizeElementHits(t *testing.T) {
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 10, indexGroup(101, 3, 11, "inverted", elementHits(indexCounters(2, 2, 0), 5))),
		segmentNode("2", 20, indexGroup(101, 3, 11, "inverted", elementHits(indexCounters(3, 3, 0), 7))),
	}, map[int64]LeafInfo{3: {Condition: `j["a"] == 1`, Type: "json_contains", FieldID: 101}})

	require.Len(t, summary.Conditions, 1)
	condition := summary.Conditions[0]
	assert.Equal(t, int64(5), condition.RowsIn)
	assert.Equal(t, int64(5), condition.RowsOut)
	assert.Equal(t, int64(12), condition.ElementHits, "element hits stay in the element domain")
	require.Len(t, condition.Used, 1)
	assert.Equal(t, int64(12), condition.Used[0].ElementHits)

	assert.Equal(t, ""+
		"query [rows_in=5, rows_out=5, segments=2]\n"+
		`  expr j["a"] == 1 [element_hits=12, idx_scan=5, raw_scan=0, rows_in=5, rows_out=5, segments=2] {expr_id=3, field_id=101, type=json_contains}`+"\n"+
		"    index 11 inverted [element_hits=12, idx_scan=5, raw_scan=0, rows_in=5, rows_out=5, segments=2]\n",
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
	assert.Equal(t, int64(0), summary.RowsIn)
	assert.Equal(t, int64(0), summary.RowsOut)
	assert.NotNil(t, summary.Conditions)
	assert.Empty(t, summary.Conditions)

	encoded, err := RenderSummaryJSON(summary)
	require.NoError(t, err)
	assert.JSONEq(t, `{"level":"summary","segments":0,"rows_in":0,"rows_out":0,"conditions":[]}`, encoded)
	assert.NotContains(t, encoded, "rows_total", "the leaf view does not repeat the segment sizes")
	assert.Equal(t, "query [rows_in=0, rows_out=0, segments=0]\n", RenderSummaryText(summary))
	assert.Equal(t, "", RenderSummaryText(nil))
}

func TestSummarizeSkipsEmptyNodes(t *testing.T) {
	summary := Summarize([]*internalpb.ExecutionExplain{
		nil,
		segmentNode("1", 10, nil, &internalpb.ExecutionExplain{Kind: KindQueryNode, Name: "1"}),
	}, nil)

	assert.Equal(t, 1, summary.Segments)
	assert.Equal(t, int64(0), summary.RowsIn)
	assert.Empty(t, summary.Conditions)
}

func TestRenderSummaryText(t *testing.T) {
	assert.Equal(t, ""+
		"query [rows_in=450000, rows_out=17853, segments=3]\n"+
		`  expr title contains "%降噪%" [idx_scan=150000, raw_scan=0, rows_in=150000, rows_out=90, segments=3] {expr_id=3, field_id=101, type=contains}`+"\n"+
		"    index 459123456789100001 ngram [idx_scan=150000, raw_scan=0, rows_in=150000, rows_out=90, segments=3]\n"+
		"  expr price in (100, +inf) [idx_scan=0, raw_scan=150000, rows_in=150000, rows_out=17526, segments=3] {expr_id=4, field_id=103, type=range}\n"+
		"    raw [idx_scan=0, raw_scan=150000, rows_in=150000, rows_out=17526, segments=3] {decline_reason=no_index_on_field}\n"+
		`  expr brand == "Acme" [idx_scan=100000, raw_scan=50000, rows_in=150000, rows_out=237, segments=3] {expr_id=6, field_id=102, type=equal}`+"\n"+
		"    index 459123456789100002 bitmap [idx_scan=100000, raw_scan=0, rows_in=100000, rows_out=200, segments=2]\n"+
		"    raw [idx_scan=0, raw_scan=50000, rows_in=50000, rows_out=37, segments=1] {decline_reason=index_unusable}\n",
		RenderSummaryText(Summarize(threeSegments(), threeLeaves())))
}

func TestRenderSummaryTextOrdersReasons(t *testing.T) {
	// Hand-built: the renderer sorts conditions, reasons and exprs without an id
	// on its own. Each reason carries its own counters, so two reasons on one
	// condition do not repeat the condition totals.
	summary := &Summary{
		Level:    "summary",
		Segments: 2,
		RowsIn:   900,
		RowsOut:  42,
		Conditions: []Condition{
			{
				ExprID:          0,
				Condition:       "field_106",
				Type:            "unknown",
				FieldID:         106,
				Segments:        1,
				IdxScan:         0,
				RawScan:         300,
				RowsIn:          300,
				RowsOut:         12,
				Used:            []IndexUse{},
				RawScanSegments: 1,
				Reasons: []ReasonCount{{
					Reason:   "no_index_on_field",
					Segments: 1,
					RawScan:  300,
					RowsIn:   300,
					RowsOut:  12,
				}},
			},
			{
				ExprID:          7,
				Condition:       "v > 1",
				Type:            "range",
				FieldID:         105,
				Segments:        2,
				IdxScan:         0,
				RawScan:         600,
				RowsIn:          600,
				RowsOut:         30,
				Used:            []IndexUse{},
				RawScanSegments: 2,
				Reasons: []ReasonCount{
					{Reason: "no_index_on_field", Segments: 1, RawScan: 300, RowsIn: 300, RowsOut: 15},
					{Reason: "index_unusable", Segments: 1, RawScan: 300, RowsIn: 300, RowsOut: 15},
				},
			},
		},
	}

	assert.Equal(t, ""+
		"query [rows_in=900, rows_out=42, segments=2]\n"+
		"  expr v > 1 [idx_scan=0, raw_scan=600, rows_in=600, rows_out=30, segments=2] {expr_id=7, field_id=105, type=range}\n"+
		"    raw [idx_scan=0, raw_scan=300, rows_in=300, rows_out=15, segments=1] {decline_reason=index_unusable}\n"+
		"    raw [idx_scan=0, raw_scan=300, rows_in=300, rows_out=15, segments=1] {decline_reason=no_index_on_field}\n"+
		"  expr field_106 [idx_scan=0, raw_scan=300, rows_in=300, rows_out=12, segments=1] {expr_id=0, field_id=106, type=unknown}\n"+
		"    raw [idx_scan=0, raw_scan=300, rows_in=300, rows_out=12, segments=1] {decline_reason=no_index_on_field}\n",
		RenderSummaryText(summary))
}

func TestRenderSummaryJSON(t *testing.T) {
	summary := Summarize(threeSegments(), threeLeaves())

	encoded, err := RenderSummaryJSON(summary)
	require.NoError(t, err)

	var decoded struct {
		Level      string `json:"level"`
		Segments   int    `json:"segments"`
		RowsIn     int64  `json:"rows_in"`
		RowsOut    int64  `json:"rows_out"`
		Conditions []struct {
			ExprID          int64         `json:"expr_id"`
			Condition       string        `json:"condition"`
			Type            string        `json:"type"`
			FieldID         int64         `json:"field_id"`
			Segments        int           `json:"segments"`
			IdxScan         int64         `json:"idx_scan"`
			RawScan         int64         `json:"raw_scan"`
			RowsIn          int64         `json:"rows_in"`
			RowsOut         int64         `json:"rows_out"`
			Used            []IndexUse    `json:"used"`
			RawScanSegments int           `json:"raw_scan_segments"`
			Reasons         []ReasonCount `json:"reasons"`
		} `json:"conditions"`
	}
	require.NoError(t, json.Unmarshal([]byte(encoded), &decoded))

	assert.Equal(t, "summary", decoded.Level)
	assert.Equal(t, 3, decoded.Segments)
	assert.Equal(t, int64(450000), decoded.RowsIn)
	assert.Equal(t, int64(17853), decoded.RowsOut)
	require.Len(t, decoded.Conditions, 3)

	brand := decoded.Conditions[2]
	assert.Equal(t, int64(6), brand.ExprID)
	assert.Equal(t, `brand == "Acme"`, brand.Condition)
	assert.Equal(t, "equal", brand.Type)
	assert.Equal(t, int64(102), brand.FieldID)
	assert.Equal(t, 3, brand.Segments)
	assert.Equal(t, int64(100000), brand.IdxScan)
	assert.Equal(t, int64(50000), brand.RawScan)
	assert.Equal(t, int64(150000), brand.RowsIn)
	assert.Equal(t, int64(237), brand.RowsOut)
	assert.Equal(t, 1, brand.RawScanSegments)
	assert.Equal(t, []IndexUse{{
		Kind:     "scalar_index",
		IndexID:  459123456789100002,
		Family:   "bitmap",
		Segments: 2,
		IdxScan:  100000,
		RawScan:  0,
		RowsIn:   100000,
		RowsOut:  200,
	}}, brand.Used)
	assert.Equal(t, []ReasonCount{{
		Reason:   "index_unusable",
		Segments: 1,
		RawScan:  50000,
		RowsIn:   50000,
		RowsOut:  37,
	}}, brand.Reasons)

	// Empty lists are [] and not null; the old counter names are gone.
	assert.Contains(t, encoded, `"used":[]`)
	assert.Contains(t, encoded, `"reasons":[]`)
	assert.NotContains(t, encoded, "null")
	assert.NotContains(t, encoded, "rows_matched")
	assert.NotContains(t, encoded, "rows_scanned")
	assert.NotContains(t, encoded, "rows_total")
	assert.NotContains(t, encoded, "element_hits")

	var raw map[string]any
	require.NoError(t, json.Unmarshal([]byte(encoded), &raw))
	assert.NotContains(t, raw, "rows_total")
	conditions, ok := raw["conditions"].([]any)
	require.True(t, ok)
	require.Len(t, conditions, 3)
	title, ok := conditions[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, []any{}, title["reasons"])
	assert.Equal(t, float64(150000), title["rows_in"])
	assert.Equal(t, float64(0), title["raw_scan"])
	assert.Equal(t, float64(150000), title["idx_scan"])
	assert.Equal(t, float64(90), title["rows_out"])
	price, ok := conditions[1].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, []any{}, price["used"])
	assert.Equal(t, float64(150000), price["raw_scan"])
	assert.Equal(t, float64(0), price["idx_scan"])
}

func withEvalOrder(node *internalpb.ExecutionExplain, order int) *internalpb.ExecutionExplain {
	node.Attributes["eval_order"] = strconv.Itoa(order)
	return node
}

func TestSummarizeAndRenderEvalOrders(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: `brand == "Acme"`, Type: "equal", FieldID: 102},
		2: {Condition: "price in (100, +inf)", Type: "range", FieldID: 103},
	}

	// Uniform eval_order across segments populates Condition.EvalOrder + eval_order=N.
	uniform := []*internalpb.ExecutionExplain{
		segmentNode("1", 100,
			withEvalOrder(indexGroup(102, 1, 11, "bitmap", indexCounters(100, 10, 0)), 2),
			withEvalOrder(rawGroup(103, 2, "no_index_on_field", rawCounters(100, 50)), 1),
		),
		segmentNode("2", 100,
			withEvalOrder(indexGroup(102, 1, 11, "bitmap", indexCounters(100, 20, 0)), 2),
			withEvalOrder(rawGroup(103, 2, "no_index_on_field", rawCounters(100, 40)), 1),
		),
	}

	summary := Summarize(uniform, leaves)
	require.Len(t, summary.Conditions, 2)
	assert.Equal(t, 2, summary.Conditions[0].EvalOrder)
	assert.Equal(t, []EvalOrderCount{{EvalOrder: 2, Segments: 2}}, summary.Conditions[0].EvalOrders)
	assert.Equal(t, 1, summary.Conditions[1].EvalOrder)
	assert.Equal(t, []EvalOrderCount{{EvalOrder: 1, Segments: 2}}, summary.Conditions[1].EvalOrders)

	text := RenderSummaryText(summary)
	assert.Contains(t, text, "{eval_order=2, expr_id=1, field_id=102, type=equal}")
	assert.Contains(t, text, "{eval_order=1, expr_id=2, field_id=103, type=range}")

	encoded, err := RenderSummaryJSON(summary)
	require.NoError(t, err)
	assert.Contains(t, encoded, `"eval_order":2`)
	assert.Contains(t, encoded, `"eval_orders":[{"eval_order":2,"segments":2}]`)

	// Divergent per-segment eval_order preserves cross-segment condition/index
	// merging, omits Condition.EvalOrder, and renders EvalOrders distribution.
	divergent := []*internalpb.ExecutionExplain{
		segmentNode("1", 100, withEvalOrder(indexGroup(102, 1, 11, "bitmap", indexCounters(100, 10, 0)), 1)),
		segmentNode("2", 100, withEvalOrder(indexGroup(102, 1, 11, "bitmap", indexCounters(100, 10, 0)), 1)),
		segmentNode("3", 100, withEvalOrder(indexGroup(102, 1, 11, "bitmap", indexCounters(100, 10, 0)), 3)),
	}

	divSummary := Summarize(divergent, leaves)
	require.Len(t, divSummary.Conditions, 1)
	cond := divSummary.Conditions[0]
	assert.Equal(t, 3, cond.Segments)
	assert.Len(t, cond.Used, 1, "index use stays merged across segments")
	assert.Equal(t, 3, cond.Used[0].Segments)
	assert.Equal(t, 0, cond.EvalOrder)
	assert.Equal(t, []EvalOrderCount{
		{EvalOrder: 1, Segments: 2},
		{EvalOrder: 3, Segments: 1},
	}, cond.EvalOrders)

	divText := RenderSummaryText(divSummary)
	assert.Contains(t, divText, "{eval_order=1(2)|3(1), expr_id=1, field_id=102, type=equal}")

	divJSON, err := RenderSummaryJSON(divSummary)
	require.NoError(t, err)
	assert.NotContains(t, divJSON, `"eval_order":0`)
	assert.Contains(t, divJSON, `"eval_orders":[{"eval_order":1,"segments":2},{"eval_order":3,"segments":1}]`)
}

func withSecondaryOperand(node *internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	node.Attributes["secondary_operand"] = "true"
	return node
}

func withSkipped(node *internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	node.Attributes["skipped"] = "true"
	return node
}

func TestSummarizeMultiOperandConditionWithoutDoubleCounting(t *testing.T) {
	leaves := map[int64]LeafInfo{
		7: {Condition: "price lessthan budget", Type: "compare", FieldID: 103},
	}

	t.Run("both operands indexed", func(t *testing.T) {
		summary := Summarize([]*internalpb.ExecutionExplain{
			segmentNode("1", 100,
				indexGroup(103, 7, 5001, "sort", indexCounters(100, 40, 0)),
				withSecondaryOperand(indexGroup(104, 7, 5002, "sort", indexCounters(100, 40, 0))),
			),
		}, leaves)

		require.Len(t, summary.Conditions, 1)
		assert.Equal(t, int64(100), summary.RowsIn)
		assert.Equal(t, int64(40), summary.RowsOut)

		cond := summary.Conditions[0]
		assert.Equal(t, int64(7), cond.ExprID)
		assert.Equal(t, int64(103), cond.FieldID)
		assert.Equal(t, 1, cond.Segments)
		assert.Equal(t, int64(100), cond.IdxScan)
		assert.Equal(t, int64(0), cond.RawScan)
		assert.Equal(t, int64(100), cond.RowsIn)
		assert.Equal(t, int64(40), cond.RowsOut)
		require.Len(t, cond.Used, 2)
		assert.Equal(t, int64(5001), cond.Used[0].IndexID)
		assert.Equal(t, int64(0), cond.Used[0].RawScan)
		assert.Equal(t, int64(5002), cond.Used[1].IndexID)
		assert.Equal(t, int64(0), cond.Used[1].RawScan)
		assert.Empty(t, cond.Reasons)
	})

	t.Run("one operand indexed and one raw", func(t *testing.T) {
		summary := Summarize([]*internalpb.ExecutionExplain{
			segmentNode("1", 100,
				indexGroup(103, 7, 5001, "sort", indexCounters(100, 40, 0)),
				withSecondaryOperand(rawGroup(104, 7, "no_index_on_field", rawCounters(100, 40))),
			),
		}, leaves)

		require.Len(t, summary.Conditions, 1)
		cond := summary.Conditions[0]
		assert.Equal(t, int64(100), cond.IdxScan)
		assert.Equal(t, int64(100), cond.RawScan)
		assert.Equal(t, int64(100), cond.RowsIn)
		assert.Equal(t, int64(40), cond.RowsOut)
		require.Len(t, cond.Used, 1)
		assert.Equal(t, int64(0), cond.Used[0].RawScan)
		assert.Equal(t, 1, cond.RawScanSegments)
		require.Len(t, cond.Reasons, 1)
		assert.Equal(t, ReasonCount{
			Reason:   "no_index_on_field",
			Segments: 1,
			RawScan:  100,
			RowsIn:   100,
			RowsOut:  40,
		}, cond.Reasons[0])
	})

	t.Run("both operands raw with same or distinct reasons", func(t *testing.T) {
		sameReason := Summarize([]*internalpb.ExecutionExplain{
			segmentNode("1", 100,
				rawGroup(103, 7, "no_index_on_field", rawCounters(100, 40)),
				withSecondaryOperand(rawGroup(104, 7, "no_index_on_field", rawCounters(100, 40))),
			),
		}, leaves)
		require.Len(t, sameReason.Conditions, 1)
		assert.Equal(t, int64(100), sameReason.Conditions[0].RawScan)
		assert.Equal(t, []ReasonCount{{
			Reason:   "no_index_on_field",
			Segments: 1,
			RawScan:  100,
			RowsIn:   100,
			RowsOut:  40,
		}}, sameReason.Conditions[0].Reasons)

		distinctReasons := Summarize([]*internalpb.ExecutionExplain{
			segmentNode("1", 100,
				rawGroup(103, 7, "index_unusable", rawCounters(100, 40)),
				withSecondaryOperand(rawGroup(104, 7, "no_index_on_field", rawCounters(100, 40))),
			),
		}, leaves)
		require.Len(t, distinctReasons.Conditions, 1)
		assert.Equal(t, int64(100), distinctReasons.Conditions[0].RawScan)
		assert.Equal(t, 1, distinctReasons.Conditions[0].RawScanSegments)
		assert.Equal(t, []ReasonCount{
			{Reason: "index_unusable", Segments: 1, RawScan: 100, RowsIn: 100, RowsOut: 40},
			{Reason: "no_index_on_field", Segments: 1, RawScan: 100, RowsIn: 100, RowsOut: 40},
		}, distinctReasons.Conditions[0].Reasons)
	})
}

func TestSummarizeSkippedAndBackfillConditions(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: `brand == "none"`, Type: "equal", FieldID: 102},
		2: {Condition: "price lessthan budget", Type: "compare", FieldID: 103},
		3: {Condition: `title contains "%pro%"`, Type: "contains", FieldID: 101},
	}

	syntheticSkipped := &internalpb.ExecutionExplain{
		Kind: KindFilterGroup,
		Attributes: map[string]string{
			"expr_id":    "2",
			"eval_order": "2",
			"skipped":    "true",
		},
		Counters: map[string]int64{"rows_in": 0, "idx_scan": 0, "raw_scan": 0, "rows_out": 0},
	}

	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 100,
			withEvalOrder(rawGroup(102, 1, "no_index_on_field", rawCounters(100, 0)), 1),
			syntheticSkipped,
		),
	}, leaves)
	BackfillConditions(summary, leaves)

	require.Len(t, summary.Conditions, 3)
	assert.False(t, summary.Conditions[0].Skipped)
	assert.True(t, summary.Conditions[1].Skipped)
	assert.Empty(t, summary.Conditions[1].Reasons, "synthetic skipped leaf must not emit unknown decline_reason")
	assert.False(t, summary.Conditions[2].Skipped, "backfilled unobserved leaf is not marked skipped")
	assert.Equal(t, 0, summary.Conditions[2].Segments)
	assert.Equal(t, []IndexUse{}, summary.Conditions[2].Used)
	assert.Equal(t, []ReasonCount{}, summary.Conditions[2].Reasons)

	text := RenderSummaryText(summary)
	assert.Contains(t, text, "{eval_order=2, expr_id=2, field_id=103, skipped=true, type=compare}")
	assert.Contains(t, text, `expr title contains "%pro%" [idx_scan=0, raw_scan=0, rows_in=0, rows_out=0, segments=0] {expr_id=3, field_id=101, type=contains}`)

	encoded, err := RenderSummaryJSON(summary)
	require.NoError(t, err)
	assert.Contains(t, encoded, `"skipped":true`)

	// When a leaf is skipped on one segment and evaluated on another, it is not
	// skipped overall.
	partial := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 100, syntheticSkipped),
		segmentNode("2", 100, withEvalOrder(rawGroup(103, 2, "no_index_on_field", rawCounters(50, 10)), 2)),
	}, leaves)
	require.Len(t, partial.Conditions, 1)
	assert.False(t, partial.Conditions[0].Skipped)
	assert.Equal(t, int64(50), partial.Conditions[0].RowsIn)
}

func TestSummarizeConstantTrueExcludesRawScanSegments(t *testing.T) {
	leaves := map[int64]LeafInfo{
		1: {Condition: "true", Type: "always_true", FieldID: 0},
	}
	alwaysTrueGroup := &internalpb.ExecutionExplain{
		Kind: KindFilterGroup,
		Attributes: map[string]string{
			"expr_id":        "1",
			"field_id":       "0",
			"nested":         "false",
			"decline_reason": "constant_true",
		},
		Counters: map[string]int64{"rows_in": 100, "idx_scan": 0, "raw_scan": 0, "rows_out": 100},
	}
	summary := Summarize([]*internalpb.ExecutionExplain{
		segmentNode("1", 100, alwaysTrueGroup),
		segmentNode("2", 100, alwaysTrueGroup),
	}, leaves)

	require.Len(t, summary.Conditions, 1)
	cond := summary.Conditions[0]
	assert.Equal(t, 2, cond.Segments)
	assert.Equal(t, int64(200), cond.RowsIn)
	assert.Equal(t, int64(200), cond.RowsOut)
	assert.Equal(t, int64(0), cond.RawScan)
	assert.Equal(t, 0, cond.RawScanSegments)
	assert.Empty(t, cond.Reasons)
	assert.NotContains(t, RenderSummaryText(summary), "decline_reason=constant_true")
}
