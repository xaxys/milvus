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

#pragma once

#include "index/Families.h"
#include "segcore/indexing/FieldIndexCapability.h"

// Field-level readiness of one loaded scalar index, when a field carries
// several of them.
//
// This header used to carry the per-index `ScalarIndexCandidate` record --
// index id, index type name, JSON path/cast, is_json/is_ngram and a coarse
// `ScalarIndexCapability` rank -- that the expression layer sorted and then
// probed with IndexBase::ShouldUseOp. Both halves have since moved to where the
// metadata now lives:
//   * load metadata and identity are `segcore::IndexCapabilityEntry`
//     (key, family, value_type, json_path, json_cast_type);
//   * what an entry can answer is `index::ReaderCaps` (predicate,
//     value_lookup, cheap_value_lookup, pattern_match, text_match,
//     ngram_candidates, spatial, exact, ...).
// Expression consumers therefore select one exact entry with
// exec::DetermineExecPath(ExprIndexRequirement, FieldIndexCapability) -- from
// metadata only, never pinning to discover support -- and pin exactly that key,
// so no candidate record is needed any more and none is reinstated here.
//
// What survives is the set of predicates the segment itself needs when several
// entries of one field meet the *field-level* readiness bits, which are keyed by
// FieldId and cannot describe more than one index.

namespace milvus::segcore {

// A JSON path/cast index, as opposed to a whole-field scalar index. Keyed on the
// normalized cast type, not on json_path being empty: a flat JSON index on the
// ROOT path legitimately has an empty path.
inline bool
IsJsonPathIndex(const IndexCapabilityEntry& entry) {
    return entry.json_cast_type.data_type() != JsonCastType::DataType::UNKNOWN;
}

// An NGRAM index over a JSON path. Its cast type is VARCHAR, so a plain string
// comparison would look compatible; its hit set is a superset (caps.exact is
// false) and only a pattern requirement selects it (RequiredReader::Ngram),
// never a predicate or value-lookup requirement.
inline bool
IsJsonNgramIndex(const IndexCapabilityEntry& entry) {
    return IsJsonPathIndex(entry) && entry.family == index::families::kNgram;
}

// An index folds into its field's readiness bit only if an older reader would
// have seen it there: a whole-field scalar index, or a JSON NGRAM one (whose bit
// ngram_fields/ngram_indexings used to set). A JSON path index without NGRAM
// deliberately stays out, exactly as json_indices did -- HasIndex() is sampled
// by the plan compiler and must not widen for JSON fields.
inline bool
FoldsIntoIndexReadyBit(const IndexCapabilityEntry& entry) {
    return !IsJsonPathIndex(entry) || IsJsonNgramIndex(entry);
}

// HasJsonIndex() stays what it was when it scanned json_indices: the presence of
// a JSON path index that is NOT an NGRAM one.
inline bool
CountsAsJsonIndex(const IndexCapabilityEntry& entry) {
    return IsJsonPathIndex(entry) && !IsJsonNgramIndex(entry);
}

}  // namespace milvus::segcore
