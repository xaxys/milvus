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
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <fmt/core.h>

#include <map>
#include <mutex>
#include <string>

#include "common/Types.h"
#include "pb/internal.pb.h"

namespace milvus::exec {

// Collects what one segment did for one query and groups it by decision, so the
// payload size follows the number of distinct decisions rather than the number of
// filter leaves. Counters are totals over the leaves of a group.
class ExecutionExplainRecorder {
 public:
    explicit ExecutionExplainRecorder(int64_t rows_total) : rows_total_(rows_total) {
    }

    // One call per filter leaf, before it runs. Returns the group's handle, which
    // the leaf uses to report the rows it actually evaluated.
    int64_t
    RecordDecision(FieldId field_id,
                   bool used_index,
                   int64_t index_id,
                   std::string family,
                   bool nested,
                   std::string path,
                   std::string decline_reason) {
        std::lock_guard lock(mu_);
        auto& group = groups_[Key(field_id,
                                  used_index,
                                  index_id,
                                  family,
                                  nested,
                                  path,
                                  decline_reason)];
        if (group.id == 0) {
            group.id = next_group_id_++;
        }
        ++group.leaves;
        if (group.family.empty()) {
            group.family = std::move(family);
        }
        if (group.path.empty()) {
            group.path = std::move(path);
        }
        if (group.decline_reason.empty()) {
            group.decline_reason = std::move(decline_reason);
        }
        group.field_id = field_id.get();
        group.index_id = index_id;
        group.used_index = used_index;
        group.nested = nested;
        return group.id;
    }

    // Rows an index returned for this leaf group, already row-level (nested hits
    // are folded by the caller).
    void
    AddIndexHits(FieldId field_id, int64_t index_id, int64_t rows) {
        std::lock_guard lock(mu_);
        for (auto& [key, group] : groups_) {
            if (group.used_index && group.field_id == field_id.get() &&
                group.index_id == index_id) {
                group.rows_matched += rows;
            }
        }
    }

    // Hits an index returned in the element domain, when the caller folds them
    // to rows afterwards. They stay separate from rows_matched: one row may own
    // several matching elements.
    void
    AddElementHits(FieldId field_id, int64_t index_id, int64_t elements) {
        std::lock_guard lock(mu_);
        for (auto& [key, group] : groups_) {
            if (group.used_index && group.field_id == field_id.get() &&
                group.index_id == index_id) {
                group.element_hits += elements;
            }
        }
    }

    // Rows the leaf behind group_id actually evaluated. An index leaf reports
    // nothing: whatever it reads after the index lookup is not the scan this
    // counter describes.
    void
    AddScannedRows(int64_t group_id, int64_t rows) {
        if (group_id == 0 || rows <= 0) {
            return;
        }
        std::lock_guard lock(mu_);
        for (auto& [key, group] : groups_) {
            if (group.id == group_id) {
                if (!group.used_index) {
                    group.rows_scanned += rows;
                }
                return;
            }
        }
    }

    // Serialized segment node: one filter_group child per decision group.
    std::string
    Finish() const {
        proto::internal::ExecutionExplain node;
        node.set_kind("segment");
        node.set_version(1);
        (*node.mutable_counters())["rows_total"] = rows_total_;
        int64_t rows_scanned = 0;
        int64_t rows_matched = 0;
        int64_t leaves = 0;
        std::lock_guard lock(mu_);
        for (const auto& [key, group] : groups_) {
            auto* child = node.add_children();
            child->set_kind("filter_group");
            child->set_version(1);
            auto& attrs = *child->mutable_attributes();
            attrs["field_id"] = std::to_string(group.field_id);
            if (group.used_index) {
                attrs["index_id"] = std::to_string(group.index_id);
            }
            if (!group.family.empty()) {
                attrs["family"] = group.family;
            }
            if (!group.decline_reason.empty()) {
                attrs["decline_reason"] = group.decline_reason;
            }
            if (!group.path.empty()) {
                attrs["path"] = group.path;
            }
            attrs["nested"] = group.nested ? "true" : "false";
            auto& counters = *child->mutable_counters();
            counters["leaves"] = group.leaves;
            counters["rows_matched"] = group.rows_matched;
            counters["rows_scanned"] = group.rows_scanned;
            if (group.element_hits > 0) {
                counters["element_hits"] = group.element_hits;
            }
            rows_scanned += group.rows_scanned;
            rows_matched += group.rows_matched;
            leaves += group.leaves;
        }
        (*node.mutable_counters())["rows_scanned"] = rows_scanned;
        (*node.mutable_counters())["rows_matched"] = rows_matched;
        (*node.mutable_counters())["leaves_total"] = leaves;
        return node.SerializeAsString();
    }

 private:
    struct Group {
        int64_t id{0};
        int64_t field_id{0};
        int64_t index_id{0};
        bool used_index{false};
        bool nested{false};
        std::string family;
        std::string path;
        std::string decline_reason;
        int64_t leaves{0};
        int64_t rows_matched{0};
        int64_t rows_scanned{0};
        int64_t element_hits{0};
    };

    static std::string
    Key(FieldId field_id,
        bool used_index,
        int64_t index_id,
        const std::string& family,
        bool nested,
        const std::string& path,
        const std::string& decline_reason) {
        return fmt::format("{}|{}|{}|{}|{}|{}|{}",
                           field_id.get(),
                           used_index,
                           index_id,
                           family,
                           nested,
                           path,
                           decline_reason);
    }

    const int64_t rows_total_;
    mutable std::mutex mu_;
    mutable int64_t next_group_id_{1};
    mutable std::map<std::string, Group> groups_;
};

}  // namespace milvus::exec
