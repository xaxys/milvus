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

// Package explain builds and renders the execution explain tree: what one
// query did, grouped by decision. Segcore produces the segment and
// filter_group nodes; Go wraps them with the levels it owns (querynode,
// delegator, query) and sums their counters.
package explain

import (
	"fmt"
	"sort"
	"strings"

	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
)

// Node kinds. The tree is query -> delegator -> querynode -> segment ->
// filter_group.
const (
	KindSegment     = "segment"
	KindFilterGroup = "filter_group"
	KindQueryNode   = "querynode"
	KindDelegator   = "delegator"
	KindQuery       = "query"
	// KindReduced is what one reduce pass produces before the next level wraps
	// it. It carries no meaning of its own: Splice replaces it with its children
	// so that a request reduced twice does not grow a level per pass.
	KindReduced = "reduced"
)

// Wrap builds a node that owns children and sums their counters.
func Wrap(kind, name string, children []*internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	node := &internalpb.ExecutionExplain{
		Kind:     kind,
		Name:     name,
		Version:  1,
		Counters: make(map[string]int64, 4),
	}
	for _, child := range children {
		if child == nil {
			continue
		}
		node.Children = append(node.Children, child)
		for key, value := range child.GetCounters() {
			node.Counters[key] += value
		}
	}
	return node
}

// Splice appends nodes to dst, replacing a KindReduced node with its children.
// A node of a meaningful kind is kept as a child even when it has children of
// its own.
func Splice(dst []*internalpb.ExecutionExplain, nodes ...*internalpb.ExecutionExplain) []*internalpb.ExecutionExplain {
	for _, node := range nodes {
		if node == nil {
			continue
		}
		if node.GetKind() == KindReduced {
			dst = Splice(dst, node.GetChildren()...)
			continue
		}
		dst = append(dst, node)
	}
	return dst
}

// MergeReduced joins the explains of one reduce pass into a single node that
// the next level can splice.
func MergeReduced(explains ...*internalpb.ExecutionExplain) *internalpb.ExecutionExplain {
	return Wrap(KindReduced, "", Splice(nil, explains...))
}

// Render formats the tree as indented text, one line per node:
//
//	kind name [counters] {attributes}
//
// Keys are sorted so that the same execution produces the same text.
func Render(root *internalpb.ExecutionExplain) string {
	var b strings.Builder
	renderNode(&b, root, 0)
	return b.String()
}

func renderNode(b *strings.Builder, node *internalpb.ExecutionExplain, depth int) {
	if node == nil {
		return
	}
	b.WriteString(strings.Repeat("  ", depth))
	b.WriteString(node.GetKind())
	if name := node.GetName(); name != "" {
		b.WriteByte(' ')
		b.WriteString(name)
	}
	writePairs(b, " [", "]", node.GetCounters())
	writePairs(b, " {", "}", node.GetAttributes())
	b.WriteByte('\n')
	for _, child := range node.GetChildren() {
		renderNode(b, child, depth+1)
	}
}

func writePairs[T ~int64 | ~string](b *strings.Builder, open, close string, pairs map[string]T) {
	if len(pairs) == 0 {
		return
	}
	keys := make([]string, 0, len(pairs))
	for key := range pairs {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	b.WriteString(open)
	for i, key := range keys {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(b, "%s=%v", key, pairs[key])
	}
	b.WriteString(close)
}
