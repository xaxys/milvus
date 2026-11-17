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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "common/JsonCastType.h"
#include "common/Types.h"
#include "index/contracts/query/ReaderCaps.h"

// Segment-level collection of index capabilities, readable without pinning.
// Keep a list of per-entry descriptors rather than OR-ing all caps: exactness
// and query interfaces must describe the same index the consumer will select.

namespace milvus::segcore {

// Distinguishes indexes identified before segment construction from
// segment-local registrations without borrowing one integer namespace for
// both. The variant preserves the complete signed pre-built index id and
// unsigned segment-local registration id ranges.
class IndexIdentity {
 public:
    enum class Kind : uint8_t {
        PreBuiltIndex,
        SegmentLocal,
    };

    static IndexIdentity
    PreBuiltIndex(int64_t index_id);

    static IndexIdentity
    SegmentLocal(uint64_t registration_id);

    Kind
    kind() const;

    bool
    operator==(const IndexIdentity& other) const;

    bool
    operator<(const IndexIdentity& other) const;

    // The persisted index id when this identity names a pre-built index. Used by
    // selection to match a query hint without knowing how identities are encoded.
    std::optional<int64_t>
    pre_built_id() const;

    // Renders the identity for logs: the pre-built index id as it was persisted,
    // or "local:<n>" for a segment-local registration. The two namespaces are
    // kept apart so a logged id always names the same index.
    std::string
    ToString() const;

 private:
    explicit IndexIdentity(std::variant<int64_t, uint64_t> value);

    std::variant<int64_t, uint64_t> value_;

    friend struct IndexKeyHash;
};

// Addresses one index inside the segment's inventory. JSON path is capability
// metadata, not identity: an ordinary catalog index keeps its index id, a
// file-backed TextMatch index uses its build id, and a segment-built/interim
// index receives a segment-local registration id.
struct IndexKey {
    FieldId field_id;
    IndexIdentity identity;

    bool
    operator==(const IndexKey& other) const {
        return field_id == other.field_id && identity == other.identity;
    }
};

struct IndexKeyHash {
    size_t
    operator()(const IndexKey& key) const;
};

// Resolved metadata for one index, used to select and validate its reader.
// Runtime parameters of one scalar index, grouped by family. Every family fills only
// its own member; the rest stay default and mean "not carried by this family".
struct ScalarIndexRuntimeParams {
    struct Ngram {
        // 0 means the metadata did not carry the value; selection then defers to the
        // reader's own check after the pin rather than refusing the index.
        size_t min_gram{0};
        size_t max_gram{0};
    } ngram;

    struct FmIndex {
        // Recorded for completeness. Selection cannot judge a literal for this family:
        // the guard compares against the index's own engine statistics, which only
        // exist once the reader is built (see the design doc, section 6.1).
        double cost_ratio{0.001};
    } fmindex;
};

struct IndexCapabilityEntry {
    IndexKey key;

    // Empty for whole-field indexes. For a per-path JSON cast index this is
    // the selected path; it never substitutes for the stable inventory key.
    std::string json_path;

    // Normalized JSON_CAST_TYPE. This distinguishes, for example, DOUBLE from
    // ARRAY_DOUBLE even though both readers expose DOUBLE as their value type.
    // Non-JSON entries keep UNKNOWN.
    JsonCastType json_cast_type{JsonCastType::UNKNOWN};

    // "inverted" / "bitmap" / "sort" / "marisa" / "fmindex" / "text" /
    // "ngram" / "rtree" / "json_flat" ... — the canonical registry key.
    std::string family;

    // The value type indexed by the reader. ARRAY uses its element type; a JSON
    // per-path index uses its cast target, not the parent field type.
    DataType value_type{DataType::NONE};

    // Metadata-derived caps must equal the opened reader's Caps(); the inventory
    // checks that invariant after pinning.
    index::ReaderCaps caps;

    // Parameters selection needs to decide, BEFORE pinning, whether this index can
    // serve a specific literal. Structured on purpose: the reader of these values is
    // the selector, and adding a family adds a member here instead of changing the
    // selector interface (LiteralSupported(req, entry)).
    ScalarIndexRuntimeParams params;
};

// Index category available without initializing a lazy entry.
enum class IndexKind : uint8_t {
    Scalar,
    Vector,
    Text,
    Ngram,
};

struct IndexRegistration {
    IndexKey key;
    IndexKind kind{IndexKind::Scalar};
    std::string json_path;
    JsonCastType json_cast_type{JsonCastType::UNKNOWN};
    // Parameters the selection layer needs before pinning (e.g. NGRAM gram
    // bounds). Carried here so the inventory can hand them back unchanged.
    ScalarIndexRuntimeParams params;
};

// Copies registration metadata and classifies the resolved family.
IndexRegistration
RegistrationFromCapability(const IndexCapabilityEntry& entry);

class FieldIndexCapability {
 public:
    explicit FieldIndexCapability(
        FieldId field_id, std::vector<IndexCapabilityEntry> entries = {});

    FieldId
    field_id() const {
        return field_id_;
    }

    bool
    empty() const {
        return entries_->empty();
    }

    const std::vector<IndexCapabilityEntry>&
    entries() const {
        return *entries_;
    }

    // Returns the exact entry, or null when `key` names another field or is not
    // registered. Copies of this capability share one immutable entry snapshot,
    // so the pointer remains valid while any copy retaining that snapshot lives.
    const IndexCapabilityEntry*
    Find(const IndexKey& key) const;

 private:
    FieldId field_id_;
    std::shared_ptr<const std::vector<IndexCapabilityEntry>> entries_;
};

// Capability equality for the inventory's post-pin consistency check.
bool
SameCaps(const index::ReaderCaps& a, const index::ReaderCaps& b);

}  // namespace milvus::segcore
