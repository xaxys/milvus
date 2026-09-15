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

package datacoord

import (
	"math"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// maxIDsPerAllocBatch mirrors the per-call ceiling of rootCoordAllocator.AllocN.
const maxIDsPerAllocBatch = int64(math.MaxUint32)

// assignExactPKRanges allocates contiguous per-file autoID PK ranges sized to the
// files' exact post-preimport row counts. fileRows is aligned with the job's file
// order. It returns one range per file in the same order; a zero-row file gets an
// empty range (Begin == End). A single file needing more than one allocation batch
// (MaxUint32 ids) is rejected: a contiguous range cannot cover it and the datanode's
// one-cursor-per-file consumption cannot walk a straddling range.
//
// Two-phase flow: the ImportMsg broadcast carries no ranges and does no file I/O.
// After the PreImport tasks have read every file and produced exact row counts, the
// cluster acting as primary allocates the ranges here and ships them to every
// cluster via the ImportIDRange WAL message, so each derives identical autoID
// primary keys. Reservations are exact (Σ fileRows ids, no expansion factor): the
// counts come from fully reading the files, so there is no estimate to absorb.
func assignExactPKRanges(fileRows []int64, allocN func(int64) (int64, int64, error), clusterID uint64) ([]*commonpb.IDRange, error) {
	for i, rows := range fileRows {
		if rows < 0 {
			return nil, merr.WrapErrImportSysFailedMsg("import file %d has a negative row count %d", i, rows)
		}
		if rows > maxIDsPerAllocBatch {
			return nil, merr.WrapErrParameterInvalidMsg(
				"import file %d holds %d rows, more than one allocation batch can reserve (max %d); split the file",
				i, rows, maxIDsPerAllocBatch)
		}
	}

	ranges := make([]*commonpb.IDRange, len(fileRows))
	// Pack files into allocation batches greedily so no allocN call exceeds the
	// ceiling and no file's range straddles two batches: every range stays
	// contiguous while the reservation total may exceed the ceiling, only a
	// single file may not.
	for i := 0; i < len(fileRows); {
		var batch int64
		j := i
		for ; j < len(fileRows) && batch+fileRows[j] <= maxIDsPerAllocBatch; j++ {
			batch += fileRows[j]
		}
		// A group of only zero-row files reserves nothing; allocN is never called
		// with 0 and their empty ranges keep the zero value.
		var cur int64
		if batch > 0 {
			begin, _, err := common.AllocAutoIDN(allocN, batch, clusterID)
			if err != nil {
				return nil, err
			}
			cur = begin
		}
		for k := i; k < j; k++ {
			ranges[k] = &commonpb.IDRange{Begin: cur, End: cur + fileRows[k]}
			cur = ranges[k].GetEnd()
		}
		i = j
	}
	return ranges, nil
}
