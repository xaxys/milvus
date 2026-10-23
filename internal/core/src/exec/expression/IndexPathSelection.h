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

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>

#include "common/Types.h"
#include "common/Utils.h"
#include "index/contracts/query/IPatternMatchReader.h"
#include "index/contracts/query/IScalarPredicateReader.h"
#include "index/contracts/query/ISpatialReader.h"
#include "segcore/indexing/FieldIndexCapability.h"

// Metadata-only execution-path selection. Choose an entry whose capabilities
// satisfy the expression without pinning payloads, casting reader objects, or
// probing support by exception. Native operator enums are mapped by expression
// construction at the plan boundary.
//
// Prefer a usable index; otherwise scan the raw or shredded column. This does
// not compare cold-index fetch cost with an already-resident column scan.
// TODO: add such cost modeling separately if needed. Literal-dependent guards
// are queried on the chosen pinned interface after metadata selection.

namespace milvus::exec {

// Execution path for expression evaluation. Determines how the expression
// result bitmap is produced.
//
// MOVED HERE FROM `exec/expression/Expr.h:54` so that the enum and the function
// that decides it live together, and so that nothing on the decision path needs
// to include the expression kernels.
enum class ExprExecPath {
    RawData,      // brute-force scan of the raw column
    ScalarIndex,  // a pinned index reader interface
    PkIndex,      // segment_->pk_range / search_ids
    TextIndex,    // the text-match interface
    // Shredded JSON column scan, not an index lookup. Kept distinct while execution
    // uses JsonKeyStats directly. TODO: fold this into RawData when typed JSON
    // sub-columns use the ordinary column-access interface.
    JsonStats,
};

// Query interface required by this expression, from index/contracts/.
enum class RequiredReader {
    Predicate,     // IScalarPredicateReader<T>: In / NotIn / Range
    PatternMatch,  // IPatternMatchReader: the LIKE family
    TextMatch,     // ITextMatchReader
    Ngram,         // INgramReader     — candidate family, caps.exact == false
    Spatial,       // ISpatialReader   — candidate family, caps.exact == false
    Null,          // INullReader      — unconditional, carries no caps bit
    ValueLookup,   // IScalarValueReader<T>
    JsonPath,      // IJsonIndexReader
};

// Everything the decision needs from the EXPRESSION side. All of it is known at
// plan time; none of it requires touching an index object.
struct ExprIndexRequirement {
    FieldId field_id;
    RequiredReader reader{RequiredReader::Predicate};

    // The value type the predicate is expressed in. For an element-level ARRAY
    // expression this is the element type, not `DataType::ARRAY`.
    DataType value_type{DataType::NONE};

    // JSON Pointer encodes the document root as an empty string, so path
    // emptiness cannot distinguish a root-path JSON predicate from a
    // non-JSON field.
    bool is_json_field{false};

    // Empty unless the expression addresses a JSON path. The path selects among
    // capability metadata; inventory identity is `(field, persisted/local id)`.
    std::string json_path;

    // Projected JSON distinguishes a scalar cast from an array-element cast
    // even when both expose the same scalar value type.
    bool json_array_cast{false};

    // Whether execution can refine a candidate superset. If false, an entry with
    // caps.exact == false is unusable and selection must fall back to RawData.
    bool accepts_candidates{true};

    // Short-circuit paths bypass the inventory; do not pin unrelated scalar cells.
    bool is_pk_compare{false};
    bool json_stats_eligible{false};

    // The upstream operator already handed down candidate offsets
    // (`has_offset_input_`). Some paths are unavailable then — e.g. today's
    // `CanUseNgramIndex()` (exec/expression/UnaryExpr.cpp:2346) refuses.
    bool has_offset_input{false};
};

struct ExecPathDecision {
    ExprExecPath path{ExprExecPath::RawData};

    // Which inventory entry to pin. Present exactly when the selected path is
    // index-backed. The caller passes this straight to
    // `segcore::SegmentInterface::PinIndex` — ONCE, for the whole expression node.
    std::optional<segcore::IndexKey> key;
};

// Pure function of requirements and metadata: no pin, cast, or exception probe.
// TODO: consolidate the base and subclass path-selection logic in Expr,
// UnaryExpr, TermExpr, BinaryRangeExpr, NullExpr, ExistsExpr, JsonContainsExpr,
// GIS expressions, BloomFilterExpr, and RoaringFilterExpr. Family support is a
// caps read; expression literal/type checks populate ExprIndexRequirement.
// Runtime literal-dependent guards still belong on the selected pinned reader.
namespace index_path_detail {

inline DataType
NormalizeJsonValueType(DataType type) {
    if (type == DataType::INT8 || type == DataType::INT16 ||
        type == DataType::INT32 || type == DataType::INT64 ||
        type == DataType::FLOAT || type == DataType::DOUBLE) {
        return DataType::DOUBLE;
    }
    if (type == DataType::STRING || type == DataType::TEXT) {
        return DataType::VARCHAR;
    }
    return type;
}

inline bool
IsPathPrefix(std::string_view prefix, std::string_view path) {
    return prefix.empty() || path == prefix ||
           (path.size() > prefix.size() &&
            path.compare(0, prefix.size(), prefix) == 0 &&
            path[prefix.size()] == '/');
}

inline bool
RelativePathContainsInteger(std::string_view prefix, std::string_view path) {
    auto relative = path.substr(prefix.size());
    size_t pos = 0;
    while (pos < relative.size()) {
        if (relative[pos] == '/') {
            ++pos;
            continue;
        }
        const auto end = relative.find('/', pos);
        const auto token = relative.substr(
            pos,
            end == std::string_view::npos ? relative.size() - pos : end - pos);
        if (!token.empty() && milvus::IsInteger(std::string(token))) {
            return true;
        }
        pos = end == std::string_view::npos ? relative.size() : end;
    }
    return false;
}

inline bool
ReaderMatches(const ExprIndexRequirement& req,
              const segcore::IndexCapabilityEntry& entry) {
    const auto& caps = entry.caps;
    const bool multi_path_router = req.is_json_field && caps.json_paths &&
                                   entry.value_type == DataType::JSON;
    if (multi_path_router) {
        return req.reader == RequiredReader::Predicate ||
               req.reader == RequiredReader::PatternMatch ||
               req.reader == RequiredReader::Null ||
               req.reader == RequiredReader::JsonPath;
    }
    switch (req.reader) {
        case RequiredReader::Predicate:
            return caps.predicate;
        case RequiredReader::PatternMatch:
            return caps.pattern_match;
        case RequiredReader::TextMatch:
            return caps.text_match;
        case RequiredReader::Ngram:
            return caps.ngram_candidates;
        case RequiredReader::Spatial:
            return caps.spatial;
        case RequiredReader::Null:
            return caps.predicate || caps.ngram_candidates || caps.spatial ||
                   caps.json_paths;
        case RequiredReader::ValueLookup:
            return caps.value_lookup;
        case RequiredReader::JsonPath:
            return caps.json_paths;
    }
    return false;
}

inline int
PathMatchScore(const ExprIndexRequirement& req,
               const segcore::IndexCapabilityEntry& entry) {
    if (!req.is_json_field) {
        return entry.json_path.empty() ? 1 : -1;
    }
    if (entry.json_path == req.json_path) {
        return 2;
    }
    const bool multi_path_router =
        entry.caps.json_paths && entry.value_type == DataType::JSON;
    if (!multi_path_router || !IsPathPrefix(entry.json_path, req.json_path) ||
        RelativePathContainsInteger(entry.json_path, req.json_path)) {
        return -1;
    }
    return 1;
}

// AdaptIndexType builds a TIMESTAMPTZ index over its INT64 storage, so the
// registered value type of such an index is either type. Scalar requirements
// compare storage types.
inline DataType
ScalarIndexStorageType(DataType type) {
    return type == DataType::TIMESTAMPTZ ? DataType::INT64 : type;
}

inline bool
ValueTypeMatches(const ExprIndexRequirement& req,
                 const segcore::IndexCapabilityEntry& entry) {
    if (req.value_type == DataType::NONE) {
        return true;
    }
    const bool multi_path_router = req.is_json_field && entry.caps.json_paths &&
                                   entry.value_type == DataType::JSON;
    if (multi_path_router) {
        return true;
    }
    const bool matches =
        req.is_json_field
            ? entry.value_type == NormalizeJsonValueType(req.value_type)
            : ScalarIndexStorageType(entry.value_type) ==
                  ScalarIndexStorageType(req.value_type);
    if (!matches) {
        return false;
    }
    if (!req.is_json_field) {
        return true;
    }
    const auto cast_shape = entry.json_cast_type.data_type();
    if (cast_shape == JsonCastType::DataType::UNKNOWN) {
        // Legacy projected JSON metadata may not carry the cast-shape
        // sidecar.  Its ordinary scalar predicate remains usable, but an
        // array-element query must not guess that an unknown cast was ARRAY_*.
        return !req.json_array_cast;
    }
    return (cast_shape == JsonCastType::DataType::ARRAY) == req.json_array_cast;
}

inline bool
NeedsCandidateRefine(RequiredReader required, const index::ReaderCaps& caps) {
    return !caps.exact && (required == RequiredReader::Predicate ||
                           required == RequiredReader::Ngram ||
                           required == RequiredReader::Spatial);
}

// Static family preference within one capability class, from the design doc's
// priority table (docs/design-docs/design_docs/multi-index-scalar-v1-plan.md,
// section "三"). It only breaks ties between entries that are equally good by
// path specificity, exactness and lookup cost, so it can never make the selector
// choose a reader for an operator it cannot serve -- the capability filters above
// already ran. Indexes are compared by the registry family key carried in the
// metadata, so no payload is opened to answer this.
//
// The table distinguishes prefix matching from inner/suffix/regex matches; both
// reach here as RequiredReader::PatternMatch, so this ranking uses the coarser
// class the requirement exposes: sort/marisa first for patterns, then inverted,
// then the candidate families. A literal-dependent refinement would have to run
// after pinning.
inline int
FamilyPriority(RequiredReader reader, const std::string& family) {
    const bool is_bitmap = family == "bitmap";
    const bool is_inverted = family == "inverted";
    const bool is_sort = family == "sort";
    const bool is_marisa = family == "marisa";
    const bool is_hybrid = family == "hybrid";
    const bool is_fmindex = family == "fmindex";
    const bool is_ngram = family == "ngram";
    switch (reader) {
        case RequiredReader::Predicate:
            // equality / IN: bitmap > inverted > hybrid > sort / marisa
            if (is_bitmap) return 3;
            if (is_inverted) return 2;
            if (is_hybrid) return 1;
            return 0;
        case RequiredReader::ValueLookup:
            // ranges: sort > hybrid > inverted > bitmap
            if (is_sort) return 3;
            if (is_hybrid) return 2;
            if (is_inverted) return 1;
            return 0;
        case RequiredReader::PatternMatch:
            // Deliberately neutral. The design doc ranks prefix matches differently
            // from inner/suffix/regex ones (inverted first vs ngram first), and this
            // requirement does not say which it is: both arrive as PatternMatch. Rank
            // by family here and one of the two rows is wrong, so the pattern family
            // keeps the precedence it had before this table existed (an ngram-capable
            // candidate is what pattern leaves are matched against).
            return 0;
        case RequiredReader::Ngram:
            // the candidate family was asked for explicitly: prefer ngram, then fmindex
            if (is_ngram) return 3;
            if (is_fmindex) return 2;
            return 0;
        case RequiredReader::TextMatch:
            return family == "text" ? 3 : 0;
        case RequiredReader::Spatial:
            return family == "rtree" ? 3 : 0;
        default:
            return 0;
    }
}

}  // namespace index_path_detail

inline ExecPathDecision
DetermineExecPath(const ExprIndexRequirement& req,
                  const segcore::FieldIndexCapability& capabilities) {
    if (req.is_pk_compare) {
        return {.path = ExprExecPath::PkIndex};
    }
    if (req.json_stats_eligible) {
        return {.path = ExprExecPath::JsonStats};
    }

    const segcore::IndexCapabilityEntry* selected = nullptr;
    int selected_score = -1;
    // A flat JSON index covers a whole subtree, so several of them can prefix one
    // query path. The longest prefix is the more specific one, and that used to
    // be the rule when these indexes lived one-per-path. It only refines a tie:
    // it never outranks a typed index on the exact path, which scores higher.
    size_t selected_prefix_len = 0;
    for (const auto& entry : capabilities.entries()) {
        const bool needs_refine =
            index_path_detail::NeedsCandidateRefine(req.reader, entry.caps);
        if (entry.key.field_id != req.field_id ||
            !index_path_detail::ReaderMatches(req, entry) ||
            (needs_refine && !req.accepts_candidates) ||
            (req.has_offset_input && req.reader == RequiredReader::Ngram) ||
            !index_path_detail::ValueTypeMatches(req, entry)) {
            continue;
        }
        const auto path_score = index_path_detail::PathMatchScore(req, entry);
        if (path_score < 0) {
            continue;
        }
        // Family preference is the last tie-break: path specificity, exactness and
        // lookup cost keep their weights above it, so it only decides between two
        // indexes that can both serve this operator equally well.
        const int family_rank =
            index_path_detail::FamilyPriority(req.reader, entry.family);
        const int score = path_score * 16 + (!needs_refine ? 8 : 0) +
                          (req.reader == RequiredReader::ValueLookup &&
                                   entry.caps.cheap_value_lookup
                               ? 4
                               : 0) +
                          family_rank;
        const size_t prefix_len = entry.json_path.size();
        // capabilities.entries() is ordered by index identity, and a strictly
        // greater score is required to displace the current choice, so the
        // remaining tie-break is the lowest indexID -- deterministic across
        // segments and reload orders.
        if (selected == nullptr || score > selected_score ||
            (score == selected_score && prefix_len > selected_prefix_len)) {
            selected = &entry;
            selected_score = score;
            selected_prefix_len = prefix_len;
        }
    }
    if (selected == nullptr) {
        return {};
    }

    return {
        .path = req.reader == RequiredReader::TextMatch
                    ? ExprExecPath::TextIndex
                    : ExprExecPath::ScalarIndex,
        .key = selected->key,
    };
}

// Per-call eligibility cannot live in ReaderCaps. Ngram min-length checks and
// FM count-first cost guards depend on the concrete literal, so ask the chosen
// pinned query interface after metadata-only selection. A false answer requests
// a fallback; it is not an unsupported-operation exception.

}  // namespace milvus::exec
