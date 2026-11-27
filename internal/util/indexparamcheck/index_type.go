// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package indexparamcheck

import (
	"strconv"

	"github.com/milvus-io/milvus/internal/util/vecindexmgr"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// IndexType string.
type IndexType = string

// IndexType definitions
const (
	IndexVector IndexType = "VECINDEX"

	// scalar index
	IndexSTLSORT    IndexType = "STL_SORT"
	IndexAsceneding IndexType = "Asceneding"
	IndexTRIE       IndexType = "TRIE"
	IndexTrie       IndexType = "Trie"
	IndexMarisaTrie IndexType = "marisa-trie"
	IndexBitmap     IndexType = "BITMAP"
	IndexHybrid     IndexType = "HYBRID" // BITMAP + INVERTED
	IndexINVERTED   IndexType = "INVERTED"
	IndexNGRAM      IndexType = "NGRAM"
	IndexFMINDEX    IndexType = "FMINDEX"
	IndexRTREE      IndexType = "RTREE"

	AutoIndex IndexType = "AUTOINDEX"
)

// CanonicalScalarIndexType normalizes legacy scalar index_type aliases to their
// canonical upper-case names so duplicate-definition and idempotency checks
// treat equivalent spellings identically.
func CanonicalScalarIndexType(indexType IndexType) IndexType {
	switch indexType {
	case IndexTrie, IndexMarisaTrie:
		return IndexTRIE
	case IndexAsceneding:
		return IndexSTLSORT
	default:
		return indexType
	}
}

func IsScalarIndexType(indexType IndexType) bool {
	switch CanonicalScalarIndexType(indexType) {
	case IndexSTLSORT, IndexTRIE, IndexBitmap, IndexHybrid, IndexINVERTED, IndexNGRAM, IndexFMINDEX, IndexRTREE:
		return true
	default:
		return false
	}
}

func IsGpuIndex(indexType IndexType) bool {
	return vecindexmgr.GetVecIndexMgrInstance().IsGPUVecIndex(indexType)
}

// IsVectorMmapIndex check if the vector index can be mmaped
func IsVectorMmapIndex(indexType IndexType) bool {
	return vecindexmgr.GetVecIndexMgrInstance().IsMMapSupported(indexType)
}

func IsOffsetCacheSupported(indexType IndexType) bool {
	return indexType == IndexBitmap
}

func IsDiskIndex(indexType IndexType) bool {
	return vecindexmgr.GetVecIndexMgrInstance().IsDiskANN(indexType)
}

func IsScalarMmapIndex(indexType IndexType) bool {
	switch CanonicalScalarIndexType(indexType) {
	case IndexSTLSORT, IndexINVERTED, IndexBitmap, IndexHybrid, IndexTRIE, IndexNGRAM, IndexFMINDEX:
		return true
	default:
		return false
	}
}

func ValidateMmapIndexParams(indexType IndexType, indexParams map[string]string) error {
	mmapEnable, ok := indexParams[common.MmapEnabledKey]
	if !ok {
		return nil
	}
	enable, err := strconv.ParseBool(mmapEnable)
	if err != nil {
		return merr.WrapErrParameterInvalidMsg("invalid %s value: %s, expected: true, false", common.MmapEnabledKey, mmapEnable)
	}
	mmapSupport := indexType == AutoIndex || IsVectorMmapIndex(indexType) || IsScalarMmapIndex(indexType)
	if enable && !mmapSupport {
		return merr.WrapErrParameterInvalidMsg("index type %s does not support mmap", indexType)
	}
	return nil
}

func ValidateOffsetCacheIndexParams(indexType IndexType, indexParams map[string]string) error {
	offsetCacheEnable, ok := indexParams[common.IndexOffsetCacheEnabledKey]
	if !ok {
		return nil
	}
	enable, err := strconv.ParseBool(offsetCacheEnable)
	if err != nil {
		return merr.WrapErrParameterInvalidMsg("invalid %s value: %s, expected: true, false", common.IndexOffsetCacheEnabledKey, offsetCacheEnable)
	}
	if enable && !IsOffsetCacheSupported(indexType) {
		return merr.WrapErrParameterInvalidMsg("only bitmap index support %s now", common.IndexOffsetCacheEnabledKey)
	}
	return nil
}

func ValidateWarmupIndexParams(indexParams map[string]string) error {
	warmupPolicy, ok := indexParams[common.WarmupKey]
	if !ok {
		return nil
	}
	if err := common.ValidateWarmupPolicy(warmupPolicy); err != nil {
		return err
	}
	return nil
}

func ValidateVisibleIndexParams(isVectorField bool, indexParams map[string]string) error {
	raw, ok := indexParams[common.IndexVisibleKey]
	if !ok {
		return nil
	}
	visible, err := strconv.ParseBool(raw)
	if err != nil {
		return merr.WrapErrParameterInvalidMsg("invalid %s value: %s, expected: true, false", common.IndexVisibleKey, raw)
	}
	if isVectorField && !visible {
		return merr.WrapErrParameterInvalidMsg("hiding vector index via %s=false is not supported", common.IndexVisibleKey)
	}
	return nil
}
