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

#include <chrono>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <folly/ScopeGuard.h>
#include "common/OpContext.h"
#include "common/QueryInfo.h"
#include "common/QueryResult.h"
#include "exec/ExecutionExplainRecorder.h"
#include "index/contracts/query/IVectorReader.h"
#include "knowhere/index/index_node.h"
#include "log/Log.h"
#include "segcore/SegmentInterface.h"
#include "segcore/SegmentGrowingImpl.h"
#include "segcore/ConcurrentVector.h"
#include "common/Span.h"
#include "query/Utils.h"
#include "common/EasyAssert.h"

namespace milvus {
namespace exec {
// Binary search to find insert position for sorted order
// Used by iterative filter and element filter nodes
template <bool large_is_better>
inline size_t
find_binsert_position(const std::vector<float>& distances,
                      size_t lo,
                      size_t hi,
                      float dist) {
    auto first = distances.begin() + lo;
    auto last = distances.begin() + hi;
    auto it = large_is_better
                  ? std::upper_bound(first, last, dist, std::greater<float>{})
                  : std::upper_bound(first, last, dist);
    return static_cast<size_t>(it - distances.begin());
}

// Insert one (distance, seg_offset[, elem_idx]) into the per-query result
// window [base_idx, base_idx + count) of search_result, keeping that window
// sorted best-first for the metric. Advances count. elem_idx == std::nullopt
// means non-element-level (element_indices_ is left untouched).
//
// Single owner of the top-k insertion invariant, shared by the iterative
// filter and iterative element filter nodes so the two cannot drift apart.
inline void
topk_binsert(SearchResult& search_result,
             size_t base_idx,
             int64_t& count,
             bool large_is_better,
             float distance,
             int64_t seg_offset,
             std::optional<int32_t> elem_idx) {
    size_t pos = large_is_better
                     ? find_binsert_position<true>(search_result.distances_,
                                                   base_idx,
                                                   base_idx + count,
                                                   distance)
                     : find_binsert_position<false>(search_result.distances_,
                                                    base_idx,
                                                    base_idx + count,
                                                    distance);

    // The window is [base_idx, base_idx + count); shift only when inserting
    // before its current end. Guard/size with the ABSOLUTE window end
    // (base_idx + count), not the relative count — otherwise for nq_index >= 1
    // (base_idx > 0) the shift is wrongly skipped and out-of-order insertions
    // overwrite existing top-k entries.
    if (count > 0 && pos < base_idx + count) {
        size_t n = base_idx + count - pos;
        std::memmove(&search_result.distances_[pos + 1],
                     &search_result.distances_[pos],
                     n * sizeof(float));
        std::memmove(&search_result.seg_offsets_[pos + 1],
                     &search_result.seg_offsets_[pos],
                     n * sizeof(int64_t));
        if (elem_idx.has_value()) {
            std::memmove(&search_result.element_indices_[pos + 1],
                         &search_result.element_indices_[pos],
                         n * sizeof(int32_t));
        }
    }
    search_result.seg_offsets_[pos] = seg_offset;
    if (elem_idx.has_value()) {
        search_result.element_indices_[pos] = elem_idx.value();
    }
    search_result.distances_[pos] = distance;
    ++count;
}

[[maybe_unused]] static bool
UseVectorIterator(const SearchInfo& search_info) {
    return search_info.has_group_by() || search_info.iterative_filter_execution;
}

inline index::VectorSearchParams
ProjectVectorSearchParams(const SearchInfo& search_info) {
    return index::VectorSearchParams{search_info.search_params_,
                                     search_info.metric_type_,
                                     search_info.topk_,
                                     search_info.trace_ctx_};
}

[[maybe_unused]] static bool
PrepareVectorIteratorsFromIndex(const SearchInfo& search_info,
                                int nq,
                                const DatasetPtr dataset,
                                SearchResult& search_result,
                                const BitsetView& bitset,
                                const index::IVectorReader& reader) {
    // when we use group by, we will use vector iterator to continously get results and group on them
    // when we use iterative filtered search, we will use vector iterator to continously get results and check scalar attr on them
    // until we get valid topk results
    if (UseVectorIterator(search_info)) {
        try {
            auto search_conf = reader.PrepareSearchParams(
                ProjectVectorSearchParams(search_info));
            query::ApplyStrictGroupSkipRefine(search_info, nq, search_conf);
            knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>
                iterators_val = reader.Iterators(dataset, search_conf, bitset);
            if (iterators_val.has_value()) {
                bool larger_is_closer =
                    PositivelyRelated(search_info.metric_type_);
                search_result.AssembleChunkVectorIterators(
                    nq, 1, iterators_val.value(), larger_is_closer);
            } else {
                std::string operator_type = "";
                if (search_info.has_group_by()) {
                    operator_type = "group_by";
                } else {
                    operator_type = "iterative filter";
                }
                LOG_ERROR(
                    "Returned knowhere iterator has non-ready iterators "
                    "inside, terminate {} operation:{}",
                    operator_type,
                    knowhere::Status2String(iterators_val.error()));
                ThrowInfo(ErrorCode::Unsupported,
                          fmt::format(
                              "Failed to {}, current index:{} doesn't support "
                              "the requested iterator operation: {}",
                              operator_type,
                              reader.KnowhereIndexType(),
                              knowhere::Status2String(iterators_val.error())));
            }
            search_result.total_nq_ = nq;
            search_result.unity_topK_ = search_info.topk_;
        } catch (const std::runtime_error& e) {
            // Preserve a coded transient SegcoreError (S3Error / FileReadFailed
            // / MemAllocateFailed, etc. thrown while reading the index) so the
            // driver can classify retriability. Only a genuine "this index
            // cannot produce iterators for the operation" -- Unsupported,
            // whether reported by the non-ready-iterators branch above or thrown
            // here -- or a non-SegcoreError failure becomes the unified
            // "doesn't support" error.
            if (auto* se = dynamic_cast<const milvus::SegcoreError*>(&e);
                se != nullptr &&
                se->get_error_code() != milvus::ErrorCode::Unsupported) {
                throw;
            }
            std::string operator_type = "";
            if (search_info.has_group_by()) {
                operator_type = "group_by";
            } else {
                operator_type = "iterative filter";
            }
            LOG_ERROR(
                "Caught error:{} when trying to initialize ann iterators for "
                "{}: "
                "operation will be terminated",
                e.what(),
                operator_type);
            ThrowInfo(ErrorCode::Unsupported,
                      "Failed to {}, current index:{} doesn't support",
                      operator_type,
                      reader.KnowhereIndexType());
        }
        return true;
    }
    return false;
}

inline void
sort_search_result(milvus::SearchResult& result, bool large_is_better) {
    auto nq = result.total_nq_;
    auto topk = result.unity_topK_;
    auto size = nq * topk;

    std::vector<float> new_distances = std::vector<float>();
    std::vector<int64_t> new_seg_offsets = std::vector<int64_t>();
    new_distances.reserve(size);
    new_seg_offsets.reserve(size);

    bool has_element_indices = !result.element_indices_.empty();
    std::vector<int32_t> new_element_indices;
    if (has_element_indices) {
        new_element_indices.reserve(size);
    }

    std::vector<size_t> idx(topk);

    for (size_t start = 0; start < size; start += topk) {
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = start + i;

        if (large_is_better) {
            std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j) {
                if (result.seg_offsets_[j] < 0 || result.seg_offsets_[i] < 0) {
                    return result.seg_offsets_[i] >= 0;
                }
                return result.distances_[i] > result.distances_[j];
            });
        } else {
            std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j) {
                if (result.seg_offsets_[j] < 0 || result.seg_offsets_[i] < 0) {
                    return result.seg_offsets_[i] >= 0;
                }
                return result.distances_[i] < result.distances_[j];
            });
        }
        for (auto i : idx) {
            new_distances.push_back(result.distances_[i]);
            new_seg_offsets.push_back(result.seg_offsets_[i]);
            if (has_element_indices) {
                new_element_indices.push_back(result.element_indices_[i]);
            }
        }
    }

    result.distances_ = std::move(new_distances);
    result.seg_offsets_ = std::move(new_seg_offsets);
    if (has_element_indices) {
        result.element_indices_ = std::move(new_element_indices);
    }
}

class CountingVectorIterator final : public VectorIterator {
 public:
    class ScopedBatchTimer {
     public:
        explicit ScopedBatchTimer(VectorIterator* iter) noexcept
            : target_(ResolveTarget(iter)),
              start_(target_ != nullptr
                         ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point{}) {
        }

        ScopedBatchTimer(const ScopedBatchTimer&) = delete;
        ScopedBatchTimer&
        operator=(const ScopedBatchTimer&) = delete;

        ~ScopedBatchTimer() noexcept {
            if (target_ != nullptr) {
                const auto ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start_)
                        .count();
                if (ns > 0) {
                    target_->iter_ns_ += ns;
                }
            }
        }

     private:
        static CountingVectorIterator*
        ResolveTarget(VectorIterator* iter) noexcept {
            auto* counting = dynamic_cast<CountingVectorIterator*>(iter);
            return (counting != nullptr && counting->track_time_) ? counting
                                                                  : nullptr;
        }

        CountingVectorIterator* target_{nullptr};
        std::chrono::steady_clock::time_point start_;
    };

    CountingVectorIterator(std::shared_ptr<VectorIterator> inner,
                           std::weak_ptr<ExecutionExplainRecorder> explain,
                           int32_t out_step,
                           int32_t in_step = -1)
        : inner_(std::move(inner)),
          explain_(std::move(explain)),
          out_step_(out_step),
          in_step_(in_step),
          track_time_(!explain_.expired() && out_step >= 0 && in_step >= 0) {
        AssertInfo(inner_ != nullptr,
                   "CountingVectorIterator inner iterator cannot be null");
    }

    ~CountingVectorIterator() override {
        if (yielded_ > 0 || iter_ns_ > 0) {
            if (auto rec = explain_.lock(); rec != nullptr) {
                if (yielded_ > 0) {
                    if (out_step_ >= 0) {
                        rec->AddOperatorRowsOut(out_step_, yielded_);
                    }
                    if (in_step_ >= 0) {
                        rec->AddOperatorRowsIn(in_step_, yielded_);
                    }
                }
                if (iter_ns_ > 0 && out_step_ >= 0 && in_step_ >= 0) {
                    rec->TransferOperatorTimeNs(out_step_, in_step_, iter_ns_);
                }
            }
        }
    }

    bool
    HasNext() override {
        return inner_->HasNext();
    }

    std::optional<std::pair<int64_t, float>>
    Next() override {
        auto val = inner_->Next();
        if (val.has_value()) {
            ++yielded_;
        }
        return val;
    }

 private:
    std::shared_ptr<VectorIterator> inner_;
    std::weak_ptr<ExecutionExplainRecorder> explain_;
    int32_t out_step_{-1};
    int32_t in_step_{-1};
    bool track_time_{false};
    int64_t yielded_{0};
    int64_t iter_ns_{0};
};

inline int64_t
CountValidSearchOffsets(const std::vector<int64_t>& offsets) {
    int64_t count = 0;
    for (auto off : offsets) {
        if (off >= 0) {
            ++count;
        }
    }
    return count;
}

}  // namespace exec
}  // namespace milvus
