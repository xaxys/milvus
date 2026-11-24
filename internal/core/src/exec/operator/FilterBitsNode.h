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

#include <folly/ScopeGuard.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

#include "common/Types.h"
#include "exec/Driver.h"
#include "exec/expression/Expr.h"
#include "exec/operator/Operator.h"
#include "exec/QueryContext.h"

namespace milvus {
namespace exec {

bool
ConvertPredicateToFilteredBitset(TargetBitmapView data,
                                 TargetBitmapView valid,
                                 size_t size);

class PhyFilterBitsNode : public Operator {
 public:
    PhyFilterBitsNode(
        int32_t operator_id,
        DriverContext* ctx,
        const std::shared_ptr<const plan::FilterBitsNode>& filter);

    bool
    IsFilter() const override {
        return true;
    }

    bool
    NeedInput() const override {
        return !input_;
    }

    void
    AddInput(RowVectorPtr& input) override;

    RowVectorPtr
    GetOutput() override;

    bool
    IsFinished() override;

    void
    Close() override {
        Operator::Close();
        prefetched_cached_output_.reset();
        exprs_->Clear();
    }

    BlockingReason
    IsBlocked(ContinueFuture* /* unused */) override {
        return BlockingReason::kNotBlocked;
    }

    bool
    AllInputProcessed();

    virtual std::string
    ToString() const override {
        return "PhyFilterBitsNode";
    }

    void
    PrefetchAsync(const std::shared_ptr<folly::CPUThreadPoolExecutor>
                      prefetch_pool) override {
        if (EnsureOperatorCacheLookup()) {
            return;
        }
        prefetch_started_ = true;
        exprs_->PrefetchAsync(prefetch_pool);
    }

    void
    WaitPrefetch() override {
        if (whole_filter_cache_hit_ && !prefetch_started_) {
            return;
        }
        const bool measure_wait = prefetch_started_ && !prefetch_waited_ &&
                                  query_context_ != nullptr &&
                                  query_context_->explain() != nullptr;
        prefetch_waited_ = true;
        if (!measure_wait) {
            exprs_->WaitPrefetch();
            return;
        }
        const auto start = std::chrono::steady_clock::now();
        auto guard = folly::makeGuard([&]() noexcept {
            const auto ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count();
            if (ns > 0) {
                query_context_->explain()->AddOperatorPrefetchWaitNs(
                    get_operator_id(), ns);
            }
        });
        exprs_->WaitPrefetch();
    }

 private:
    bool
    CanUseExprCache() const;

    bool
    EnsureOperatorCacheLookup();

    std::unique_ptr<ExprSet> exprs_;
    QueryContext* query_context_;
    int64_t need_process_rows_;
    int64_t num_processed_rows_;
    // Expression filter cache for two-stage search.
    // Cache backend is the process-level ExprResCacheManager.
    bool enable_expr_cache_ = false;
    bool cache_lookup_done_ = false;
    bool prefetch_started_ = false;
    bool prefetch_waited_ = false;
    bool whole_filter_cache_hit_ = false;
    int64_t operator_cache_lookup_ns_ = 0;
    RowVectorPtr prefetched_cached_output_;
    std::string expr_cache_key_;
};
}  // namespace exec
}  // namespace milvus
