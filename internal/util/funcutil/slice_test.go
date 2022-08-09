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

package funcutil

import (
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_SliceContain(t *testing.T) {
	invalid := "invalid"
	assert.Panics(t, func() { SliceContain(invalid, 1) })

	strSlice := []string{"test", "for", "SliceContain"}
	intSlice := []int{1, 2, 3}

	cases := []struct {
		s    interface{}
		item interface{}
		want bool
	}{
		{strSlice, "test", true},
		{strSlice, "for", true},
		{strSlice, "SliceContain", true},
		{strSlice, "tests", false},
		{intSlice, 1, true},
		{intSlice, 2, true},
		{intSlice, 3, true},
		{intSlice, 4, false},
	}

	for _, test := range cases {
		if got := SliceContain(test.s, test.item); got != test.want {
			t.Errorf("SliceContain(%v, %v) = %v", test.s, test.item, test.want)
		}
	}
}

func Test_SliceSetEqual(t *testing.T) {
	invalid := "invalid"
	assert.Panics(t, func() { SliceSetEqual(invalid, 1) })
	temp := []int{1, 2, 3}
	assert.Panics(t, func() { SliceSetEqual(temp, invalid) })

	cases := []struct {
		s1   interface{}
		s2   interface{}
		want bool
	}{
		{[]int{}, []int{}, true},
		{[]string{}, []string{}, true},
		{[]int{1, 2, 3}, []int{3, 2, 1}, true},
		{[]int{1, 2, 3}, []int{1, 2, 3}, true},
		{[]int{1, 2, 3}, []int{}, false},
		{[]int{1, 2, 3}, []int{1, 2}, false},
		{[]int{1, 2, 3}, []int{4, 5, 6}, false},
		{[]string{"test", "for", "SliceSetEqual"}, []string{"SliceSetEqual", "test", "for"}, true},
		{[]string{"test", "for", "SliceSetEqual"}, []string{"test", "for", "SliceSetEqual"}, true},
		{[]string{"test", "for", "SliceSetEqual"}, []string{"test", "for"}, false},
		{[]string{"test", "for", "SliceSetEqual"}, []string{}, false},
		{[]string{"test", "for", "SliceSetEqual"}, []string{"test", "for", "SliceContain"}, false},
	}

	for _, test := range cases {
		if got := SliceSetEqual(test.s1, test.s2); got != test.want {
			t.Errorf("SliceSetEqual(%v, %v) = %v", test.s1, test.s2, test.want)
		}
	}
}

func Test_SortedSliceEqual(t *testing.T) {
	invalid := "invalid"
	assert.Panics(t, func() { SortedSliceEqual(invalid, 1) })
	temp := []int{1, 2, 3}
	assert.Panics(t, func() { SortedSliceEqual(temp, invalid) })

	sortSlice := func(slice interface{}, less func(i, j int) bool) {
		sort.Slice(slice, less)
	}
	intSliceAfterSort := func(slice []int) []int {
		sortSlice(slice, func(i, j int) bool {
			return slice[i] <= slice[j]
		})
		return slice
	}
	stringSliceAfterSort := func(slice []string) []string {
		sortSlice(slice, func(i, j int) bool {
			return slice[i] <= slice[j]
		})
		return slice
	}

	cases := []struct {
		s1   interface{}
		s2   interface{}
		want bool
	}{
		{intSliceAfterSort([]int{}), intSliceAfterSort([]int{}), true},
		{stringSliceAfterSort([]string{}), stringSliceAfterSort([]string{}), true},
		{intSliceAfterSort([]int{1, 2, 3}), intSliceAfterSort([]int{3, 2, 1}), true},
		{intSliceAfterSort([]int{1, 2, 3}), intSliceAfterSort([]int{1, 2, 3}), true},
		{intSliceAfterSort([]int{1, 2, 3}), intSliceAfterSort([]int{}), false},
		{intSliceAfterSort([]int{1, 2, 3}), intSliceAfterSort([]int{1, 2}), false},
		{intSliceAfterSort([]int{1, 2, 3}), intSliceAfterSort([]int{4, 5, 6}), false},
		{stringSliceAfterSort([]string{"test", "for", "SliceSetEqual"}), stringSliceAfterSort([]string{"SliceSetEqual", "test", "for"}), true},
		{stringSliceAfterSort([]string{"test", "for", "SliceSetEqual"}), stringSliceAfterSort([]string{"test", "for", "SliceSetEqual"}), true},
		{stringSliceAfterSort([]string{"test", "for", "SliceSetEqual"}), stringSliceAfterSort([]string{"test", "for"}), false},
		{stringSliceAfterSort([]string{"test", "for", "SliceSetEqual"}), stringSliceAfterSort([]string{}), false},
		{stringSliceAfterSort([]string{"test", "for", "SliceSetEqual"}), stringSliceAfterSort([]string{"test", "for", "SliceContain"}), false},
	}

	for _, test := range cases {
		if got := SortedSliceEqual(test.s1, test.s2); got != test.want {
			t.Errorf("SliceSetEqual(%v, %v) = %v", test.s1, test.s2, test.want)
		}
	}
}

func Test_SliceContainCmp(t *testing.T) {

	// ============== strSlice ==============

	strSlice := []string{"test", "for", "SliceContain"}

	strCases := []struct {
		s    []string
		item string
		want bool
	}{
		{strSlice, "test", true},
		{strSlice, "for", true},
		{strSlice, "SliceContain", true},
		{strSlice, "tests", false},
	}

	strCmp := func(s1, s2 string) bool {
		return s1 == s2
	}

	for _, test := range strCases {
		if got := SliceContainCmp(test.s, test.item, strCmp); got != test.want {
			t.Errorf("SliceContainCmp(%v, %v) = %v", test.s, test.item, test.want)
		}
	}

	// ============== intSlice ==============

	intSlice := []int{1, 2, 3}

	intCases := []struct {
		s    []int
		item int
		want bool
	}{
		{intSlice, 1, true},
		{intSlice, 2, true},
		{intSlice, 3, true},
		{intSlice, 4, false},
	}

	intCmp := func(s1, s2 int) bool {
		return s1 == s2
	}

	for _, test := range intCases {
		if got := SliceContainCmp(test.s, test.item, intCmp); got != test.want {
			t.Errorf("SliceContainCmp(%v, %v) = %v", test.s, test.item, test.want)
		}
	}

	// ============== structSlice ==============

	type testStruct struct {
		Name string
	}

	structSlice := []testStruct{
		{Name: "test"},
		{Name: "for"},
		{Name: "SliceContain"},
	}

	testCases := []struct {
		s    []testStruct
		item testStruct
		want bool
	}{
		{structSlice, testStruct{Name: "test"}, true},
		{structSlice, testStruct{Name: "for"}, true},
		{structSlice, testStruct{Name: "SliceContain"}, true},
		{structSlice, testStruct{Name: "tests"}, false},
	}

	testCmp := func(s1, s2 testStruct) bool {
		return s1.Name == s2.Name
	}

	for _, test := range testCases {
		if got := SliceContainCmp(test.s, test.item, testCmp); got != test.want {
			t.Errorf("SliceContainCmp(%v, %v) = %v", test.s, test.item, test.want)
		}
	}
}

func Test_SliceSetEqualCmp(t *testing.T) {

	// ============== intSlice ==============

	intCases := []struct {
		s1   []int
		s2   []int
		want bool
	}{
		{[]int{}, []int{}, true},
		{[]int{1, 2, 3}, []int{3, 2, 1}, true},
		{[]int{1, 2, 3}, []int{1, 2, 3}, true},
		{[]int{1, 2, 3}, []int{}, false},
		{[]int{1, 2, 3}, []int{1, 2}, false},
		{[]int{1, 2, 3}, []int{4, 5, 6}, false},
	}

	intCmp := func(s1, s2 int) bool {
		return s1 == s2
	}

	for _, test := range intCases {
		if got := SliceSetEqualCmp(test.s1, test.s2, intCmp); got != test.want {
			t.Errorf("SliceSetEqualCmp(%v, %v) = %v", test.s1, test.s2, test.want)
		}
	}

	// ============== strSlice ==============

	strCases := []struct {
		s1   []string
		s2   []string
		want bool
	}{
		{[]string{}, []string{}, true},
		{[]string{"test", "for", "SliceSetEqual"}, []string{"SliceSetEqual", "test", "for"}, true},
		{[]string{"test", "for", "SliceSetEqual"}, []string{"test", "for", "SliceSetEqual"}, true},
		{[]string{"test", "for", "SliceSetEqual"}, []string{"test", "for"}, false},
		{[]string{"test", "for", "SliceSetEqual"}, []string{}, false},
		{[]string{"test", "for", "SliceSetEqual"}, []string{"test", "for", "SliceContain"}, false},
	}

	strCmp := func(s1, s2 string) bool {
		return s1 == s2
	}

	for _, test := range strCases {
		if got := SliceSetEqualCmp(test.s1, test.s2, strCmp); got != test.want {
			t.Errorf("SliceSetEqualCmp(%v, %v) = %v", test.s1, test.s2, test.want)
		}
	}

	// ============== structSlice ==============

	type testStruct struct {
		Name string
	}

	structCases := []struct {
		s1   []testStruct
		s2   []testStruct
		want bool
	}{
		{[]testStruct{}, []testStruct{}, true},
		{[]testStruct{{Name: "test"}, {Name: "for"}, {Name: "SliceSetEqual"}}, []testStruct{{Name: "SliceSetEqual"}, {Name: "test"}, {Name: "for"}}, true},
		{[]testStruct{{Name: "test"}, {Name: "for"}, {Name: "SliceSetEqual"}}, []testStruct{{Name: "test"}, {Name: "for"}, {Name: "SliceSetEqual"}}, true},
		{[]testStruct{{Name: "test"}, {Name: "for"}, {Name: "SliceSetEqual"}}, []testStruct{{Name: "test"}, {Name: "for"}}, false},
		{[]testStruct{{Name: "test"}, {Name: "for"}, {Name: "SliceSetEqual"}}, []testStruct{}, false},
		{[]testStruct{{Name: "test"}, {Name: "for"}, {Name: "SliceSetEqual"}}, []testStruct{{Name: "test"}, {Name: "for"}, {Name: "SliceContain"}}, false},
	}

	structCmp := func(s1, s2 testStruct) bool {
		return s1.Name == s2.Name
	}

	for _, test := range structCases {
		if got := SliceSetEqualCmp(test.s1, test.s2, structCmp); got != test.want {
			t.Errorf("SliceSetEqualCmp(%v, %v) = %v", test.s1, test.s2, test.want)
		}
	}
}

func Test_SortedSliceEqualCmp(t *testing.T) {

	// ============== intSlice ==============

	sortSlice := func(slice interface{}, less func(i, j int) bool) {
		sort.Slice(slice, less)
	}
	intSliceAfterSort := func(slice []int) []int {
		sortSlice(slice, func(i, j int) bool {
			return slice[i] <= slice[j]
		})
		return slice
	}

	intCases := []struct {
		s1   []int
		s2   []int
		want bool
	}{
		{intSliceAfterSort([]int{}), intSliceAfterSort([]int{}), true},
		{intSliceAfterSort([]int{1, 2, 3}), intSliceAfterSort([]int{3, 2, 1}), true},
		{intSliceAfterSort([]int{1, 2, 3}), intSliceAfterSort([]int{1, 2, 3}), true},
		{intSliceAfterSort([]int{1, 2, 3}), intSliceAfterSort([]int{}), false},
		{intSliceAfterSort([]int{1, 2, 3}), intSliceAfterSort([]int{1, 2}), false},
		{intSliceAfterSort([]int{1, 2, 3}), intSliceAfterSort([]int{4, 5, 6}), false},
	}

	intCmp := func(s1, s2 int) bool {
		return s1 == s2
	}

	for _, test := range intCases {
		if got := SortedSliceEqualCmp(test.s1, test.s2, intCmp); got != test.want {
			t.Errorf("SortedSliceEqualCmp(%v, %v) = %v", test.s1, test.s2, test.want)
		}
	}

	// ============== stringSlice ==============

	stringSliceAfterSort := func(slice []string) []string {
		sortSlice(slice, func(i, j int) bool {
			return slice[i] <= slice[j]
		})
		return slice
	}

	strCases := []struct {
		s1   []string
		s2   []string
		want bool
	}{
		{stringSliceAfterSort([]string{}), stringSliceAfterSort([]string{}), true},
		{stringSliceAfterSort([]string{"test", "for", "SliceSetEqual"}), stringSliceAfterSort([]string{"SliceSetEqual", "test", "for"}), true},
		{stringSliceAfterSort([]string{"test", "for", "SliceSetEqual"}), stringSliceAfterSort([]string{"test", "for", "SliceSetEqual"}), true},
		{stringSliceAfterSort([]string{"test", "for", "SliceSetEqual"}), stringSliceAfterSort([]string{"test", "for"}), false},
		{stringSliceAfterSort([]string{"test", "for", "SliceSetEqual"}), stringSliceAfterSort([]string{}), false},
		{stringSliceAfterSort([]string{"test", "for", "SliceSetEqual"}), stringSliceAfterSort([]string{"test", "for", "SliceContain"}), false},
	}

	strCmp := func(s1, s2 string) bool {
		return s1 == s2
	}

	for _, test := range strCases {
		if got := SortedSliceEqualCmp(test.s1, test.s2, strCmp); got != test.want {
			t.Errorf("SortedSliceEqualCmp(%v, %v) = %v", test.s1, test.s2, test.want)
		}
	}
}

func Test_SliceFilter(t *testing.T) {

	// ============== structSlice ==============

	type testStruct struct {
		Name string
	}

	structCases := []struct {
		src []testStruct
		tgt []testStruct
		fn  func(testStruct) bool
	}{
		{[]testStruct{}, []testStruct{}, func(testStruct) bool { return true }},
		{[]testStruct{{Name: "a"}, {Name: "b"}, {Name: "c"}}, []testStruct{{Name: "a"}}, func(s testStruct) bool { return s.Name == "a" }},
		{[]testStruct{{Name: "a"}, {Name: "aa"}, {Name: "b"}}, []testStruct{{Name: "a"}, {Name: "aa"}}, func(s testStruct) bool { return strings.HasPrefix(s.Name, "a") }},
	}

	for _, test := range structCases {
		if got := SliceFilter(test.src, test.fn); !SortedSliceEqual(got, test.tgt) {
			t.Errorf("SliceFilter(%v, %T) = %v", test.src, test.fn, got)
		}
	}

	// ============== intSlice ==============

	intCases := []struct {
		src []int
		tgt []int
		fn  func(int) bool
	}{
		{[]int{}, []int{}, func(int) bool { return true }},
		{[]int{1, 2, 3}, []int{1}, func(i int) bool { return i == 1 }},
		{[]int{1, 2, 3}, []int{1, 2}, func(i int) bool { return i <= 2 }},
		{[]int{1, 2, 3}, []int{1, 2, 3}, func(i int) bool { return i <= 3 }},
		{[]int{1, 2, 3}, []int{}, func(i int) bool { return i > 3 }},
	}

	for _, test := range intCases {
		if got := SliceFilter(test.src, test.fn); !SortedSliceEqual(got, test.tgt) {
			t.Errorf("SliceFilter(%v, %T) = %v", test.src, test.fn, got)
		}
	}
}
