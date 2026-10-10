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

#include <cstdint>
#include <string>

#include "common/JsonCastType.h"

namespace milvus::segcore {

// What a loaded scalar index can answer, as far as its *type* tells us. HYBRID
// and AUTOINDEX choose a physical implementation from the data cardinality at
// build time, so they stay Unknown here and are judged after pinning, by
// IndexBase::ShouldUseOp -- the ordering is a preference, never a decision.
enum class ScalarIndexCapability {
    Unknown = 0,
    TermIndexed,
    PatternIndexed,
};

// Metadata about one loaded scalar index: enough for the selection layer to
// order candidates without pinning (and paying a cold fetch for) any of them.
// One field may hold several; index_id is the identity, because one JSON path
// may itself carry indexes of several cast types.
struct ScalarIndexCandidate {
    int64_t index_id{0};
    std::string index_type;
    std::string json_path;  // empty => not a JSON path index
    JsonCastType json_cast_type{JsonCastType::UNKNOWN};
    bool is_ngram{false};
    ScalarIndexCapability capability{ScalarIndexCapability::Unknown};
};

// An index folds into its field's readiness bit only if an older reader would
// have seen it there: a plain scalar index, or an NGRAM one (whose bit
// ngram_fields/ngram_indexings used to set). A JSON path index without NGRAM
// deliberately stays out, exactly as json_indices did -- HasIndex() is sampled
// by the plan compiler and must not widen for JSON fields.
inline bool
FoldsIntoIndexReadyBit(const ScalarIndexCandidate& candidate) {
    return candidate.json_path.empty() || candidate.is_ngram;
}

// HasJsonIndex() stays what it was when it scanned json_indices: the presence
// of a JSON path index that is NOT an NGRAM one.
inline bool
CountsAsJsonIndex(const ScalarIndexCandidate& candidate) {
    return !candidate.json_path.empty() && !candidate.is_ngram;
}

}  // namespace milvus::segcore
