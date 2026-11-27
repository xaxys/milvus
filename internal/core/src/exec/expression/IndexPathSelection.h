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
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "common/EasyAssert.h"
#include "common/QueryInfo.h"
#include "common/Types.h"
#include "common/Utils.h"
#include "index/Families.h"
#include "index/contracts/query/INgramReader.h"
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
    PatternMatch,  // IPatternMatchReader or INgramReader: LIKE / regex family
    TextMatch,     // ITextMatchReader
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

    // Whole-array comparison (`arr == [1, 2]` / `arr != [1, 2]`). Element-only
    // posting families (`bitmap`, `hybrid`) do not store per-row array lengths
    // needed by ExecArrayEqualForIndex and must be excluded before pinning.
    bool is_whole_array{false};

    // Keys already tried and declined by a post-pin runtime guard on this leaf.
    std::vector<segcore::IndexKey> excluded_keys;

    // Ordered query-level index hints. Null or empty means rule-based selection.
    const std::vector<IndexHintOption>* index_hints{nullptr};
    bool index_hint_strict{false};

    // Short-circuit paths bypass the inventory; do not pin unrelated scalar cells.
    bool is_pk_compare{false};

    // The upstream operator already handed down candidate offsets
    // (`has_offset_input_`). Some paths are unavailable then — e.g. today's
    // `CanUseNgramIndex()` (exec/expression/UnaryExpr.cpp:2346) refuses.
    bool has_offset_input{false};
};

// Why a candidate was not usable for one leaf. Kept deliberately coarse: each value
// names the gate that rejected it, which is what a user can act on. Strings only
// exist for logs and tests (see ToString).
enum class DeclineReason {
    kNone = 0,
    // Nothing to choose from: the field carries no index entry on this segment.
    kNoIndexOnField,
    // The index cannot serve this kind of predicate at all (family capability).
    kCapabilityMismatch,
    // The index serves predicates of another value type.
    kValueTypeMismatch,
    // The index covers a different JSON path than the leaf asks for.
    kJsonPathNotCovered,
    // The index returns a superset and the leaf cannot refine it.
    kCandidatesNeedRefine,
    // NGRAM hits are element offsets, which do not line up with the offset input.
    kOffsetInputWithNgram,
    // A flat JSON index is required for this literal (out-of-range int64).
    kFlatJsonRequired,
    // The index refused this literal (for example NGRAM shorter than min_gram).
    kLiteralNotSupported,
    // A hinted index for this field/path is not loaded on the segment.
    kHintedIndexNotSelected,
    // The field has entries, but none of them could serve this leaf (coarse
    // bucket used by the explain payload; the debug log carries the exact gate).
    kIndexUnusable,
    // Answered from an in-memory expression/geometry cache with zero storage scan.
    kCached,
};

inline const char*
ToString(DeclineReason reason) {
    switch (reason) {
        case DeclineReason::kNoIndexOnField:
            return "no_index_on_field";
        case DeclineReason::kCapabilityMismatch:
            return "capability_mismatch";
        case DeclineReason::kValueTypeMismatch:
            return "value_type_mismatch";
        case DeclineReason::kJsonPathNotCovered:
            return "json_path_not_covered";
        case DeclineReason::kCandidatesNeedRefine:
            return "candidate_needs_refine";
        case DeclineReason::kOffsetInputWithNgram:
            return "offset_input_with_ngram";
        case DeclineReason::kFlatJsonRequired:
            return "flat_json_required";
        case DeclineReason::kLiteralNotSupported:
            return "literal_not_supported";
        case DeclineReason::kHintedIndexNotSelected:
            return "hinted_index_not_selected";
        case DeclineReason::kIndexUnusable:
            return "index_unusable";
        case DeclineReason::kCached:
            return "cached";
        case DeclineReason::kNone:
        default:
            return "none";
    }
}

struct ExecPathDecision {
    ExprExecPath path{ExprExecPath::RawData};

    // Which inventory entry to create a reader from. Present exactly when the selected path is
    // Why an index-backed path was not chosen, when it was not. Diagnostic only:
    // selection never reads it back. One name per gate so a single field can be
    // compared across queries (see the decision-record section of the design doc).
    DeclineReason fallback_reason{DeclineReason::kNone};

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
InvertedSupportsPatternOp(index::PatternOp op) {
    return op == index::PatternOp::Match || op == index::PatternOp::PrefixMatch;
}

inline bool
ReaderMatches(const ExprIndexRequirement& req,
              const segcore::IndexCapabilityEntry& entry) {
    const auto& caps = entry.caps;
    if (req.is_whole_array && entry.family == index::families::kBitmap) {
        return false;
    }
    const bool multi_path_router = req.is_json_field && IsFlatJsonIndex(entry);
    if (multi_path_router) {
        // Field-level JSON null checks cannot use a typed path view. The
        // selector resolves views after reader creation, and Resolve(JSON) is absent.
        if (req.reader == RequiredReader::Null &&
            req.value_type == DataType::JSON) {
            return false;
        }
        if (req.reader == RequiredReader::PatternMatch && req.has_pattern &&
            !InvertedSupportsPatternOp(req.pattern_op)) {
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
            if (caps.ngram_candidates) {
                return true;
            }
            if (!caps.pattern_match) {
                return false;
            }
            if (req.has_pattern && entry.family == index::families::kInverted &&
                !InvertedSupportsPatternOp(req.pattern_op)) {
                return false;
            }
            return true;
        case RequiredReader::TextMatch:
            return caps.text_match;
        case RequiredReader::Spatial:
            return caps.spatial;
        case RequiredReader::Null:
            return !caps.nested &&
                   (caps.predicate || caps.pattern_match ||
                    caps.ngram_candidates || caps.spatial || caps.json_paths);
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
    if (IsFlatJsonIndex(entry)) {
        if (!IsPathPrefix(entry.json_path, req.json_path) ||
            RelativePathContainsInteger(entry.json_path, req.json_path)) {
            return {-1, 0};
        }
        return {1, entry.json_path.size()};
    }
    if (entry.json_path == req.json_path) {
        return {2, entry.json_path.size()};
    }
    return {-1, 0};
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

// Whether this index can answer the specific literal of the query. Only families whose
// reader refuses literals need an entry here; the rule is shared with the reader (see
// AcceptsNgramLiteral) so selection and execution cannot drift apart.
inline bool
LiteralSupported(const ExprIndexRequirement& req,
                 const segcore::IndexCapabilityEntry& entry) {
    if (!req.has_pattern || entry.family != index::families::kNgram) {
        return true;
    }
    const size_t min_gram = entry.params.ngram.min_gram;
    if (min_gram == 0) {
        // Metadata did not carry min_gram: defer to the reader's own check after the
        // pin rather than refusing an index we cannot describe.
        return true;
    }
    return index::AcceptsNgramLiteral(req.pattern, req.pattern_op, min_gram);
}

inline bool
NeedsCandidateRefine(RequiredReader required, const index::ReaderCaps& caps) {
    return !caps.exact && (required == RequiredReader::Predicate ||
                           required == RequiredReader::PatternMatch ||
                           required == RequiredReader::Spatial);
}

inline bool
NeedsCandidateRefine(const ExprIndexRequirement& req,
                     const segcore::IndexCapabilityEntry& entry) {
    if (NeedsCandidateRefine(req.reader, entry.caps)) {
        return true;
    }
    if (req.reader == RequiredReader::PatternMatch && req.has_pattern &&
        entry.family == index::families::kFmIndex &&
        (req.pattern_op == index::PatternOp::Match ||
         req.pattern_op == index::PatternOp::RegexMatch)) {
        return true;
    }
    return false;
}

// Static family preference within one capability class, from the design doc's
// priority table (docs/design-docs/design_docs/multi-index-scalar-v1-plan.md,
// section "三"). It only breaks ties between entries that are equally good by
// path specificity, exactness and lookup cost, so it can never make the selector
// choose a reader for an operator it cannot serve -- the capability filters above
// already ran. Indexes are compared by the registry family key carried in the
// metadata, so no payload is opened to answer this.
inline int
FamilyPriority(const ExprIndexRequirement& req, const std::string& family) {
    const bool is_bitmap = family == index::families::kBitmap;
    const bool is_inverted = family == index::families::kInverted;
    const bool is_sort = family == index::families::kSort;
    const bool is_marisa = family == index::families::kMarisa;
    const bool is_fmindex = family == index::families::kFmIndex;
    const bool is_ngram = family == index::families::kNgram;
    switch (req.reader) {
        case RequiredReader::Predicate:
            if (req.is_range) {
                // ordered comparison: sort > inverted > bitmap
                if (is_sort)
                    return 3;
                if (is_inverted)
                    return 1;
                return 0;
            }
            // equality / IN / NotIn: bitmap > inverted > sort / marisa
            if (is_bitmap)
                return 3;
            if (is_inverted)
                return 2;
            return 0;
        case RequiredReader::ValueLookup:
            // reverse lookup (retrieve / output_fields / group-by): the cheapest
            // reader wins; cheap_value_lookup already contributes a point, so only
            // prefer a stable order here.
            if (is_sort)
                return 3;
            if (is_inverted)
                return 1;
            return 0;
        case RequiredReader::PatternMatch:
            if (req.has_pattern &&
                req.pattern_op == index::PatternOp::PrefixMatch) {
                // Prefix match: sort / marisa > inverted > fmindex > bitmap > ngram
                if (is_sort || is_marisa)
                    return 5;
                if (is_inverted)
                    return 4;
                if (is_fmindex)
                    return 3;
                if (is_bitmap)
                    return 2;
                if (is_ngram)
                    return 1;
                return 0;
            }
            // Substring / suffix / LIKE / regex: ngram > fmindex > bitmap / marisa > sort / inverted
            if (is_ngram)
                return 4;
            if (is_fmindex)
                return 3;
            if (is_bitmap || is_marisa)
                return 2;
            if (is_sort || is_inverted)
                return 1;
            return 0;
        case RequiredReader::TextMatch:
            return family == index::families::kText ? 3 : 0;
        case RequiredReader::Spatial:
            return family == index::families::kRTree ? 3 : 0;
        default:
            return 0;
    }
}

inline bool
HasHintForField(const ExprIndexRequirement& req) {
    if (req.reader == RequiredReader::TextMatch || req.index_hints == nullptr ||
        req.index_hints->empty()) {
        return false;
    }
    return std::any_of(req.index_hints->begin(),
                       req.index_hints->end(),
                       [&](const IndexHintOption& hint) {
                           return hint.field_id == req.field_id.get();
                       });
}

inline bool
HintAppliesToPath(const ExprIndexRequirement& req,
                  const segcore::IndexCapabilityEntry& entry) {
    if (!req.is_json_field) {
        return entry.json_path.empty();
    }
    if (IsFlatJsonIndex(entry)) {
        return IsPathPrefix(entry.json_path, req.json_path);
    }
    return entry.json_path == req.json_path;
}

inline int
HintRank(const ExprIndexRequirement& req,
         const segcore::IndexCapabilityEntry& entry) {
    if (req.index_hints == nullptr || entry.key.field_id != req.field_id ||
        !HintAppliesToPath(req, entry)) {
        return -1;
    }
    for (size_t i = 0; i < req.index_hints->size(); ++i) {
        const auto& hint = (*req.index_hints)[i];
        if (hint.field_id == entry.key.field_id.get() &&
            entry.key.identity ==
                segcore::IndexIdentity::PreBuiltIndex(hint.index_id)) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

inline bool
HasHintedEntryOnSegment(const ExprIndexRequirement& req,
                        const segcore::FieldIndexCapability& capabilities) {
    if (req.index_hints == nullptr || req.index_hints->empty()) {
        return false;
    }
    return std::any_of(capabilities.entries().begin(),
                       capabilities.entries().end(),
                       [&](const segcore::IndexCapabilityEntry& entry) {
                           return HintRank(req, entry) >= 0;
                       });
}

}  // namespace index_path_detail

inline ExecPathDecision
DetermineExecPath(const ExprIndexRequirement& req,
                  const segcore::FieldIndexCapability& capabilities) {
    const bool has_field_hint = index_path_detail::HasHintForField(req);
    if (req.is_pk_compare && !has_field_hint) {
        return {.path = ExprExecPath::PkIndex};
    }
    const bool saw_hinted_entry =
        has_field_hint &&
        index_path_detail::HasHintedEntryOnSegment(req, capabilities);

    using CandidateRank = std::tuple<int, int, int, int, int, int, size_t, int>;
    const segcore::IndexCapabilityEntry* selected = nullptr;
    std::optional<CandidateRank> selected_rank;
    auto decline_specificity = [](DeclineReason r) -> int {
        switch (r) {
            case DeclineReason::kNone:
                return 0;
            case DeclineReason::kJsonPathNotCovered:
                return 1;
            case DeclineReason::kCapabilityMismatch:
                return 2;
            case DeclineReason::kValueTypeMismatch:
                return 3;
            case DeclineReason::kFlatJsonRequired:
            case DeclineReason::kCandidatesNeedRefine:
            case DeclineReason::kOffsetInputWithNgram:
                return 4;
            case DeclineReason::kLiteralNotSupported:
                return 5;
            case DeclineReason::kHintedIndexNotSelected:
                return 6;
            default:
                return 1;
        }
    };
    // Most specific gate reached across candidates on this field, plus the
    // most specific gate reached among hinted candidates for strict-hint errors.
    DeclineReason best_reject = capabilities.has_unmatched_entries()
                                    ? DeclineReason::kJsonPathNotCovered
                                    : DeclineReason::kNone;
    DeclineReason hint_reject = DeclineReason::kNone;
    bool saw_field_entry = capabilities.has_unmatched_entries();
    for (const auto& entry : capabilities.entries()) {
        if (entry.key.field_id != req.field_id) {
            continue;
        }
        saw_field_entry = true;
        const int hint_rank =
            has_field_hint ? index_path_detail::HintRank(req, entry) : -1;
        if (has_field_hint && req.index_hint_strict && saw_hinted_entry &&
            hint_rank < 0) {
            continue;
        }
        const bool needs_refine =
            index_path_detail::NeedsCandidateRefine(req, entry);
        const auto reject = [&](DeclineReason reason) {
            if (decline_specificity(reason) >
                decline_specificity(best_reject)) {
                best_reject = reason;
            }
            if (hint_rank >= 0 && decline_specificity(reason) >
                                      decline_specificity(hint_reject)) {
                hint_reject = reason;
            }
        };
        if (std::find(req.excluded_keys.begin(),
                      req.excluded_keys.end(),
                      entry.key) != req.excluded_keys.end()) {
            reject(req.has_pattern ? DeclineReason::kLiteralNotSupported
                                   : DeclineReason::kCapabilityMismatch);
            continue;
        }
        const auto [path_score, root_length] =
            index_path_detail::PathMatchScore(req, entry);
        if (path_score < 0) {
            reject(DeclineReason::kJsonPathNotCovered);
            continue;
        }
        if (req.requires_flat_json_index &&
            !index_path_detail::IsFlatJsonIndex(entry)) {
            reject(DeclineReason::kFlatJsonRequired);
            continue;
        }
        if (!index_path_detail::ReaderMatches(req, entry)) {
            reject(DeclineReason::kCapabilityMismatch);
            continue;
        }
        if (needs_refine && !req.accepts_candidates) {
            reject(DeclineReason::kCandidatesNeedRefine);
            continue;
        }
        if (req.has_offset_input && entry.caps.ngram_candidates) {
            reject(DeclineReason::kOffsetInputWithNgram);
            continue;
        }
        if (!index_path_detail::ValueTypeMatches(req, entry)) {
            reject(DeclineReason::kValueTypeMismatch);
            continue;
        }
        if (!index_path_detail::LiteralSupported(req, entry)) {
            reject(DeclineReason::kLiteralNotSupported);
            continue;
        }
        // Hint priority precedes rule-based preference; family preference is
        // the last tie-break after path specificity, exactness, cast-function
        // precedence, lookup cost, and JSON root prefix length.
        const int exact_rank = (req.reader != RequiredReader::PatternMatch &&
                                entry.caps.exact && !needs_refine)
                                   ? 1
                                   : 0;
        const CandidateRank rank{
            hint_rank >= 0 ? 1 : 0,
            hint_rank >= 0 ? -hint_rank : 0,
            path_score,
            exact_rank,
            !entry.json_cast_function.empty() ? 1 : 0,
            (req.reader == RequiredReader::ValueLookup &&
             entry.caps.cheap_value_lookup)
                ? 1
                : 0,
            root_length,
            index_path_detail::FamilyPriority(req, entry.family),
        };
        if (!selected_rank.has_value() || rank > *selected_rank) {
            selected = &entry;
            selected_rank = rank;
        }
    }
    if (selected == nullptr) {
        if (has_field_hint && req.index_hint_strict && saw_hinted_entry) {
            const DeclineReason reason =
                hint_reject != DeclineReason::kNone
                    ? hint_reject
                    : DeclineReason::kHintedIndexNotSelected;
            ThrowInfo(InvalidParameter,
                      "index_hint cannot be satisfied on field {} (strict "
                      "mode): {}",
                      req.field_id.get(),
                      ToString(reason));
        }
        if (req.is_pk_compare) {
            return {.path = ExprExecPath::PkIndex};
        }
        const DeclineReason reason =
            !saw_field_entry
                ? DeclineReason::kNoIndexOnField
                : (has_field_hint && !req.is_json_field && !saw_hinted_entry
                       ? DeclineReason::kHintedIndexNotSelected
                       : best_reject);
        // One line per expression node per segment, at DEBUG: the FeatureBit
        // accounting already counts the fallback, this only explains it.
        LOG_DEBUG("scalar index fallback, fieldID:{}, reader:{}, reason:{}",
                  req.field_id.get(),
                  static_cast<int>(req.reader),
                  ToString(reason));
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
