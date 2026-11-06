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
#include <stdexcept>
#include <utility>

#include "common/Types.h"
#include "common/Utils.h"
#include "index/contracts/query/IPatternMatchReader.h"
#include "index/contracts/query/IScalarPredicateReader.h"
#include "index/contracts/query/ISpatialReader.h"
#include "index/scalar/ngram/NgramIndexReader.h"
#include "segcore/indexing/FieldIndexCapability.h"

// Metadata-only execution-path selection. Choose an entry whose capabilities
// satisfy the expression without pinning payloads, casting reader objects, or
// probing support by exception. Native operator enums are mapped by expression
// construction at the plan boundary.
//
// Prefer a usable index; otherwise scan the raw or shredded column. This does
// not compare cold-index fetch cost with an already-resident column scan.
// TODO: add such cost modeling separately if needed. Metadata-only literal/type
// constraints live in ExprIndexRequirement; reader-specific guards are queried
// on the chosen reader interface after metadata selection.

namespace milvus::exec {

// Execution path for expression evaluation. Determines how the expression
// result bitmap is produced.
//
// MOVED HERE FROM `exec/expression/Expr.h:54` so that the enum and the function
// that decides it live together, and so that nothing on the decision path needs
// to include the expression kernels.
enum class ExprExecPath {
    RawData,      // brute-force scan of the raw column
    ScalarIndex,  // a selected index reader interface
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
    Null,  // INullReader      — checked after family-capability selection
    ValueLookup,  // IScalarValueReader<T>
    JsonPath,     // IJsonIndexReader
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

    // Int64 JSON literals at or beyond +/-2^53 can alias in DOUBLE projections.
    // Require a field-level JSON router (flat) that keeps integer precision.
    bool requires_flat_json_index{false};

    // Whether execution can refine a candidate superset. If false, an entry with
    // caps.exact == false is unusable and selection must fall back to RawData.
    bool accepts_candidates{true};

    // Short-circuit path that bypasses the inventory; do not pin unrelated
    // scalar cells. JSON shredding is not selected here: it is the fallback
    // SegmentExpr applies when this selector returns RawData for a JSON path.
    // Ordered comparison (>, >=, <, <=), as opposed to equality / IN / NotIn. Both
    // arrive as RequiredReader::Predicate, but the family priority differs: a range
    // wants SORT (binary search) over BITMAP, while equality wants BITMAP over the
    // rest. Set by the leaf through Expr::IsRangeExpr().
    bool is_range{false};

    // Pattern literals: what the index has to be able to answer. Filled by pattern
    // leaves so selection can apply the same rule the reader would apply after a pin
    // (an NGRAM index refuses a literal shorter than min_gram), keeping the pin a
    // single attempt.
    bool has_pattern{false};
    index::PatternOp pattern_op{index::PatternOp::Match};
    std::string pattern;

    // Ordered (field_id, index_id) hints from the query. Borrowed from PlanOptions,
    // which outlives every expression of the request; nullptr when the query has none.
    const std::vector<std::pair<int64_t, int64_t>>* index_hints{nullptr};
    // strict: a hint naming this field's indexes that cannot be used fails the query
    // instead of falling back to rule-based selection.
    bool index_hint_strict{false};

    // Short-circuit paths bypass the inventory; do not pin unrelated scalar cells.
    bool is_pk_compare{false};

    // The upstream operator already handed down candidate offsets
    // (`has_offset_input_`). Some paths are unavailable then — e.g. today's
    // `CanUseNgramIndex()` (exec/expression/UnaryExpr.cpp:2346) refuses.
    bool has_offset_input{false};
};

struct ExecPathDecision {
    ExprExecPath path{ExprExecPath::RawData};

    // Which inventory entry to create a reader from. Present exactly when the selected path is
    // Why an index-backed path was not chosen, when it was not. Diagnostic only:
    // selection never reads it back. One name per gate so a single field can be
    // compared across queries (see the decision-record section of the design doc).
    const char* fallback_reason{nullptr};

    // Which inventory entry to pin. Present exactly when the selected path is
    // index-backed. The caller passes this straight to
    // `segcore::SegmentInterface::CreateIndexReader` — ONCE, for the whole expression node.
    std::optional<segcore::IndexKey> key;
};

// Pure function of requirements and metadata: no pin, cast, or exception probe.
// TODO: consolidate the base and subclass path-selection logic in Expr,
// UnaryExpr, TermExpr, BinaryRangeExpr, NullExpr, ExistsExpr, JsonContainsExpr,
// GIS expressions, BloomFilterExpr, and RoaringFilterExpr. Family support is a
// caps read; expression literal/type checks populate ExprIndexRequirement.
// Runtime literal-dependent guards still belong on the selected reader.
namespace index_path_detail {

inline bool
IsFlatJsonIndex(const segcore::IndexCapabilityEntry& entry) {
    return entry.caps.json_paths && entry.value_type == DataType::JSON;
}

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
    const bool multi_path_router = req.is_json_field && IsFlatJsonIndex(entry);
    if (multi_path_router) {
        // Field-level JSON null checks cannot use a typed path view. The
        // selector resolves views after reader creation, and Resolve(JSON) is absent.
        if (req.reader == RequiredReader::Null &&
            req.value_type == DataType::JSON) {
            return false;
        }
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
            return caps.predicate || caps.pattern_match ||
                   caps.ngram_candidates || caps.spatial || caps.json_paths;
        case RequiredReader::ValueLookup:
            return caps.value_lookup;
        case RequiredReader::JsonPath:
            return caps.json_paths;
    }
    return false;
}

inline std::pair<int, size_t>
PathMatchScore(const ExprIndexRequirement& req,
               const segcore::IndexCapabilityEntry& entry) {
    if (!req.is_json_field) {
        return {entry.json_path.empty() ? 1 : -1, 0};
    }
    if (entry.json_path == req.json_path) {
        return {2, entry.json_path.size()};
    }
    const bool multi_path_router = IsFlatJsonIndex(entry);
    if (!multi_path_router || !IsPathPrefix(entry.json_path, req.json_path) ||
        RelativePathContainsInteger(entry.json_path, req.json_path)) {
        return {-1, 0};
    }
    return {1, entry.json_path.size()};
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
    const bool multi_path_router = req.is_json_field && IsFlatJsonIndex(entry);
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

// Position of this entry in the query's hint list for its own field, or -1 when the
// query did not hint it. Hints of other fields are ignored, so a hint never changes
// (or, in strict mode, fails) selection on a field it does not name.
inline int
HintRank(const ExprIndexRequirement& req,
         const segcore::IndexCapabilityEntry& entry) {
    if (req.index_hints == nullptr) {
        return -1;
    }
    const auto id = entry.key.identity.pre_built_id();
    if (!id.has_value()) {
        return -1;
    }
    int rank = 0;
    for (const auto& [field_id, index_id] : *req.index_hints) {
        if (field_id != entry.key.field_id.get()) {
            continue;
        }
        if (index_id == *id) {
            return rank;
        }
        ++rank;
    }
    return -1;
}

// Whether the query hinted at least one index of this requirement's field.
inline bool
HasHintForField(const ExprIndexRequirement& req) {
    if (req.index_hints == nullptr) {
        return false;
    }
    return std::any_of(req.index_hints->begin(),
                       req.index_hints->end(),
                       [&req](const auto& hint) {
                           return hint.first == req.field_id.get();
                       });
}

// Whether this index can answer the specific literal of the query. Only families whose
// reader refuses literals need an entry here; the rule is shared with the reader (see
// NgramIndexReader::AcceptsLiteral) so selection and execution cannot drift apart.
inline bool
LiteralSupported(const ExprIndexRequirement& req,
                 const segcore::IndexCapabilityEntry& entry) {
    if (!req.has_pattern || entry.family != "ngram") {
        return true;
    }
    const size_t min_gram = entry.params.ngram.min_gram;
    if (min_gram == 0) {
        // Metadata did not carry min_gram: defer to the reader's own check after the
        // pin rather than refusing an index we cannot describe.
        return true;
    }
    return index::NgramIndexReader::AcceptsLiteral(
        req.pattern, req.pattern_op, min_gram);
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
FamilyPriority(RequiredReader reader,
               bool is_range,
               const std::string& family) {
    const bool is_bitmap = family == "bitmap";
    const bool is_inverted = family == "inverted";
    const bool is_sort = family == "sort";
    const bool is_marisa = family == "marisa";
    const bool is_hybrid = family == "hybrid";
    const bool is_fmindex = family == "fmindex";
    const bool is_ngram = family == "ngram";
    switch (reader) {
        case RequiredReader::Predicate:
            if (is_range) {
                // ordered comparison: sort > hybrid > inverted > bitmap
                if (is_sort) return 3;
                if (is_hybrid) return 2;
                if (is_inverted) return 1;
                return 0;
            }
            // equality / IN / NotIn: bitmap > inverted > hybrid > sort / marisa
            if (is_bitmap) return 3;
            if (is_inverted) return 2;
            if (is_hybrid) return 1;
            return 0;
        case RequiredReader::ValueLookup:
            // reverse lookup (retrieve / output_fields / group-by): the cheapest
            // reader wins; cheap_value_lookup already contributes a point, so only
            // prefer a stable order here.
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

    const segcore::IndexCapabilityEntry* selected = nullptr;
    int selected_score = -1;
    // First gate that rejected a candidate, kept for the fallback reason below.
    const char* first_reject = nullptr;
    bool saw_field_entry = false;
    size_t selected_root_length = 0;
    int selected_family_rank = -1;
    int selected_hint_rank = -1;
    for (const auto& entry : capabilities.entries()) {
        const bool needs_refine =
            index_path_detail::NeedsCandidateRefine(req.reader, entry.caps);
        const auto reject = [&](const char* reason) {
            if (first_reject == nullptr) {
                first_reject = reason;
            }
        };
        if (entry.key.field_id != req.field_id) {
            continue;
        }
        saw_field_entry = true;
        if (req.requires_flat_json_index &&
            !index_path_detail::IsFlatJsonIndex(entry)) {
            reject("flat_json_required");
            continue;
        }
        if (!index_path_detail::ReaderMatches(req, entry)) {
            reject("capability_mismatch");
            continue;
        }
        if (needs_refine && !req.accepts_candidates) {
            reject("candidate_needs_refine");
            continue;
        }
        if (req.has_offset_input && req.reader == RequiredReader::Ngram) {
            reject("offset_input_with_ngram");
            continue;
        }
        if (!index_path_detail::ValueTypeMatches(req, entry)) {
            reject("value_type_mismatch");
            continue;
        }
        if (!index_path_detail::LiteralSupported(req, entry)) {
            reject("literal_not_supported");
            continue;
        }
        const auto [path_score, root_length] =
            index_path_detail::PathMatchScore(req, entry);
        if (path_score < 0) {
            reject("json_path_not_covered");
            continue;
        }
        // Family preference is the last tie-break: path specificity, exactness and
        // lookup cost keep their weights above it, so it only decides between two
        // indexes that can both serve this operator equally well.
        const int family_rank = index_path_detail::FamilyPriority(
            req.reader, req.is_range, entry.family);
        const int score = path_score * 16 + (!needs_refine ? 8 : 0) +
                          (req.reader == RequiredReader::ValueLookup &&
                                   entry.caps.cheap_value_lookup
                               ? 1
                               : 0);
        // Preserve exact/prefix and refine/cheap scoring. Among equal scores,
        // prefer the most specific JSON root before the stable identity order.
        // A hint outranks the score: the query asked for this index by name, so it is
        // used as long as it is eligible at all. Among hinted candidates the earliest
        // wins, and the score only orders the non-hinted ones.
        const int hint_rank = index_path_detail::HintRank(req, entry);
        const bool better_tie_break =
            root_length > selected_root_length ||
            (root_length == selected_root_length &&
             family_rank > selected_family_rank);
        const bool hinted_now = hint_rank >= 0;
        const bool hinted_prev = selected_hint_rank >= 0;
        if (selected == nullptr ||
            (hinted_now && (!hinted_prev || hint_rank < selected_hint_rank)) ||
            (hinted_now == hinted_prev && score > selected_score) ||
            (hinted_now == hinted_prev && score == selected_score &&
             better_tie_break)) {
            selected = &entry;
            selected_score = score;
            selected_root_length = root_length;
            selected_family_rank = family_rank;
            selected_hint_rank = hint_rank;
        }
    }
    if (selected == nullptr) {
        const char* reason =
            saw_field_entry ? first_reject : "no_index_on_field";
        if (req.index_hint_strict && index_path_detail::HasHintForField(req)) {
            // The query named an index of this field and asked for strict hints: say
            // so instead of silently selecting something else.
            ThrowInfo(ErrorCode::InvalidParameter,
                      fmt::format(
                          "index hint cannot be used for field {}: {}",
                          req.field_id.get(),
                          reason));
        }
        // One line per expression node per segment, at DEBUG: the FeatureBit
        // accounting already counts the fallback, this only explains it.
        LOG_DEBUG(
            "scalar index fallback, fieldID:{}, reader:{}, reason:{}",
            req.field_id.get(),
            static_cast<int>(req.reader),
            reason == nullptr ? "no_candidate" : reason);
        return {.path = ExprExecPath::RawData, .fallback_reason = reason};
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
// selected query interface after metadata-only selection. A false answer requests
// a fallback; it is not an unsupported-operation exception.

}  // namespace milvus::exec
