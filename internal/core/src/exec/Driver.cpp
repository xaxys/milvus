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

#include "Driver.h"

#include <folly/ExceptionWrapper.h>
#include <folly/ScopeGuard.h>
#include <folly/Try.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <exception>
#include <memory>
#include <string>

#include "common/Common.h"
#include "common/EasyAssert.h"
#include "common/Exception.h"
#include "common/Tracer.h"
#include "common/protobuf_utils.h"
#include "exec/QueryContext.h"
#include "exec/Task.h"
#include "exec/ExecutionExplainRecorder.h"
#include "exec/operator/AggregationNode.h"
#include "exec/operator/CallbackSink.h"
#include "exec/operator/ElementFilterBitsNode.h"
#include "exec/operator/IterativeElementFilterNode.h"
#include "exec/operator/FilterBitsNode.h"
#include "exec/operator/IterativeFilterNode.h"
#include "exec/operator/MvccNode.h"
#include "exec/operator/Operator.h"
#include "exec/operator/ProjectNode.h"
#include "exec/operator/RandomSampleNode.h"
#include "exec/operator/RescoresNode.h"
#include "exec/operator/SearchGroupByNode.h"
#include "exec/operator/VectorSearchNode.h"
#include "exec/operator/QueryOrderByNode.h"
#include "expr/ITypeExpr.h"
#include "fmt/core.h"
#include "folly/Executor.h"
#include "folly/Unit.h"
#include "glog/logging.h"
#include "log/Log.h"
#include "plan/PlanNode.h"
#include "storage/PrefetchThreadPool.h"

namespace milvus {
namespace exec {

namespace {

bool
HasExplainableLogicalLeaf(const expr::TypedExprPtr& expr,
                          const Schema* schema) {
    if (!expr) {
        return false;
    }
    if (std::dynamic_pointer_cast<const expr::AlwaysTrueExpr>(expr)) {
        return expr->expr_id() != 0;
    }
    if (std::dynamic_pointer_cast<const expr::LogicalBinaryExpr>(expr) ||
        std::dynamic_pointer_cast<const expr::LogicalUnaryExpr>(expr) ||
        std::dynamic_pointer_cast<const expr::MatchExpr>(expr)) {
        for (const auto& child : expr->inputs()) {
            if (HasExplainableLogicalLeaf(child, schema)) {
                return true;
            }
        }
        return false;
    }
    if (expr->expr_id() == 0 && schema != nullptr) {
        if (auto unary =
                std::dynamic_pointer_cast<const expr::UnaryRangeFilterExpr>(
                    expr)) {
            if (schema->is_internal_filter_field(unary->column_.field_id_)) {
                return false;
            }
        }
        if (auto null_expr =
                std::dynamic_pointer_cast<const expr::NullExpr>(expr)) {
            if (schema->is_internal_filter_field(
                    null_expr->column_.field_id_)) {
                return false;
            }
        }
    }
    return true;
}

const char*
ExplainOperatorDomain(ExplainOperatorKind kind,
                      bool element_level_search,
                      bool has_element_filter,
                      bool has_aggregation) noexcept {
    switch (kind) {
        case ExplainOperatorKind::kElementFilter:
        case ExplainOperatorKind::kIterativeElementFilter:
            return "element";
        case ExplainOperatorKind::kVectorSearch:
        case ExplainOperatorKind::kIterativeFilter:
        case ExplainOperatorKind::kGroupBy:
        case ExplainOperatorKind::kRescore:
            return element_level_search ? "element" : "";
        case ExplainOperatorKind::kProject:
            return (element_level_search || has_element_filter) ? "element"
                                                                : "";
        case ExplainOperatorKind::kOrderBy:
            return (element_level_search ||
                    (has_element_filter && !has_aggregation))
                       ? "element"
                       : "";
        default:
            return "";
    }
}

void
RegisterOperatorsFromNodes(
    const std::vector<std::shared_ptr<const plan::PlanNode>>& plannodes,
    bool element_level_search,
    ExecutionExplainRecorder* explain,
    const Schema* schema) {
    if (explain == nullptr) {
        return;
    }
    bool has_element_filter = false;
    bool has_aggregation = false;
    bool has_mvcc = false;
    std::vector<ExplainOperatorKind> kinds(plannodes.size(),
                                           ExplainOperatorKind::kUnknown);
    for (size_t i = 0; i < plannodes.size(); ++i) {
        const auto kind = plannodes[i] ? plannodes[i]->explain_operator_kind()
                                       : ExplainOperatorKind::kUnknown;
        kinds[i] = kind;
        if (kind == ExplainOperatorKind::kElementFilter) {
            has_element_filter = true;
        } else if (kind == ExplainOperatorKind::kAggregation) {
            has_aggregation = true;
        } else if (kind == ExplainOperatorKind::kMvcc) {
            has_mvcc = true;
        }
    }
    for (size_t i = 0; i < plannodes.size(); ++i) {
        const auto kind = kinds[i];
        if (kind == ExplainOperatorKind::kUnknown) {
            continue;
        }
        bool next_is_mvcc = false;
        for (size_t j = i + 1; j < plannodes.size(); ++j) {
            if (kinds[j] == ExplainOperatorKind::kUnknown) {
                continue;
            }
            next_is_mvcc = (kinds[j] == ExplainOperatorKind::kMvcc);
            break;
        }
        bool internal_folded = false;
        if (kind == ExplainOperatorKind::kFilter && has_mvcc && next_is_mvcc) {
            const auto* fb =
                static_cast<const plan::FilterBitsNode*>(plannodes[i].get());
            internal_folded = !HasExplainableLogicalLeaf(fb->filter(), schema);
        }
        explain->RegisterOperator(static_cast<int32_t>(i),
                                  kind,
                                  ExplainOperatorDomain(kind,
                                                        element_level_search,
                                                        has_element_filter,
                                                        has_aggregation),
                                  internal_folded);
    }
}

void
LinearizePlanNodes(const std::shared_ptr<const plan::PlanNode>& node,
                   std::vector<std::shared_ptr<const plan::PlanNode>>& out) {
    if (!node) {
        return;
    }
    const auto& sources = node->sources();
    if (!sources.empty()) {
        LinearizePlanNodes(sources[0], out);
    }
    out.push_back(node);
}

}  // namespace

void
RegisterPlanOperators(const std::shared_ptr<const plan::PlanNode>& root,
                      bool element_level_search,
                      ExecutionExplainRecorder* explain,
                      const Schema* schema) {
    if (explain == nullptr || !root) {
        return;
    }
    std::vector<std::shared_ptr<const plan::PlanNode>> nodes;
    LinearizePlanNodes(root, nodes);
    RegisterOperatorsFromNodes(nodes, element_level_search, explain, schema);
}

std::atomic_uint64_t BlockingState::num_blocked_drivers_{0};

std::shared_ptr<QueryConfig>
DriverContext::GetQueryConfig() {
    return task_->query_context()->query_config();
}

std::shared_ptr<Driver>
DriverFactory::CreateDriver(
    std::unique_ptr<DriverContext> ctx,
    const std::function<int(int pipelineid)>& num_drivers) {
    auto driver = std::shared_ptr<Driver>(new Driver());
    ctx->driver_ = driver.get();
    if (auto* qc = ctx->task_ ? ctx->task_->query_context().get() : nullptr;
        qc != nullptr && qc->explain() != nullptr) {
        driver->explain_ = qc->explain();
        const auto* pg = qc->get_placeholder_group();
        const bool element_level_search =
            pg != nullptr && !pg->empty() && pg->at(0).element_level_;
        const auto schema = qc->get_optional_read_schema();
        RegisterOperatorsFromNodes(
            plannodes_, element_level_search, qc->explain(), schema.get());
    }
    std::vector<std::shared_ptr<Operator>> operators;
    operators.reserve(plannodes_.size());

    for (size_t i = 0; i < plannodes_.size(); ++i) {
        auto id = operators.size();
        const auto& plannode = plannodes_[i];
        if (auto filterbitsnode =
                std::dynamic_pointer_cast<const plan::FilterBitsNode>(
                    plannode)) {
            tracer::AddEvent("create_operator: FilterBitsNode");
            operators.push_back(std::make_shared<PhyFilterBitsNode>(
                id, ctx.get(), filterbitsnode));
        } else if (auto filternode = std::dynamic_pointer_cast<
                       const plan::IterativeFilterNode>(plannode)) {
            tracer::AddEvent("create_operator: IterativeFilterNode");
            operators.push_back(std::make_shared<PhyIterativeFilterNode>(
                id, ctx.get(), filternode));
        } else if (auto mvccnode =
                       std::dynamic_pointer_cast<const plan::MvccNode>(
                           plannode)) {
            tracer::AddEvent("create_operator: MvccNode");
            operators.push_back(
                std::make_shared<PhyMvccNode>(id, ctx.get(), mvccnode));
        } else if (auto vectorsearchnode =
                       std::dynamic_pointer_cast<const plan::VectorSearchNode>(
                           plannode)) {
            tracer::AddEvent("create_operator: VectorSearchNode");
            operators.push_back(std::make_shared<PhyVectorSearchNode>(
                id, ctx.get(), vectorsearchnode));
        } else if (auto searchGroupByNode =
                       std::dynamic_pointer_cast<const plan::SearchGroupByNode>(
                           plannode)) {
            tracer::AddEvent("create_operator: SearchGroupByNode");
            operators.push_back(std::make_shared<PhySearchGroupByNode>(
                id, ctx.get(), searchGroupByNode));
        } else if (auto queryGroupByNode =
                       std::dynamic_pointer_cast<const plan::AggregationNode>(
                           plannode)) {
            tracer::AddEvent("create_operator: AggregationNode");
            operators.push_back(std::make_shared<PhyAggregationNode>(
                id, ctx.get(), queryGroupByNode));
        } else if (auto projectNode =
                       std::dynamic_pointer_cast<const plan::ProjectNode>(
                           plannode)) {
            tracer::AddEvent("create_operator: ProjectNode");
            operators.push_back(
                std::make_shared<PhyProjectNode>(id, ctx.get(), projectNode));
        } else if (auto orderByNode =
                       std::dynamic_pointer_cast<const plan::OrderByNode>(
                           plannode)) {
            tracer::AddEvent("create_operator: QueryOrderByNode");
            operators.push_back(std::make_shared<PhyQueryOrderByNode>(
                id, ctx.get(), orderByNode));
        } else if (auto samplenode =
                       std::dynamic_pointer_cast<const plan::RandomSampleNode>(
                           plannode)) {
            tracer::AddEvent("create_operator: RandomSampleNode");
            operators.push_back(std::make_shared<PhyRandomSampleNode>(
                id, ctx.get(), samplenode));
        } else if (auto rescoresnode =
                       std::dynamic_pointer_cast<const plan::RescoresNode>(
                           plannode)) {
            tracer::AddEvent("create_operator: RescoresNode");
            operators.push_back(
                std::make_shared<PhyRescoresNode>(id, ctx.get(), rescoresnode));
        } else if (auto node = std::dynamic_pointer_cast<
                       const plan::IterativeElementFilterNode>(plannode)) {
            tracer::AddEvent("create_operator: IterativeElementFilterNode");
            operators.push_back(std::make_shared<PhyIterativeElementFilterNode>(
                id, ctx.get(), node));
        } else if (auto node = std::dynamic_pointer_cast<
                       const plan::ElementFilterBitsNode>(plannode)) {
            tracer::AddEvent("create_operator: ElementFilterBitsNode");
            operators.push_back(std::make_shared<PhyElementFilterBitsNode>(
                id, ctx.get(), node));
        } else {
            ThrowInfo(ErrorCode::UnexpectedError, "Unknown plan node type");
        }
        // TODO: add more operators
    }

    if (consumer_supplier_) {
        operators.push_back(consumer_supplier_(operators.size(), ctx.get()));
    }

    driver->Init(std::move(ctx), std::move(operators));

    return driver;
}

void
Driver::Enqueue(std::shared_ptr<Driver> driver) {
    if (driver->closed_) {
        return;
    }

    driver->get_task()->query_context()->executor()->add(
        [driver]() { Driver::Run(driver); });
}

void
Driver::Run(std::shared_ptr<Driver> self) {
    std::shared_ptr<BlockingState> blocking_state;
    RowVectorPtr result;
    auto reason = self->RunInternal(self, blocking_state, result);

    AssertInfo(result == nullptr,
               "The last operator (sink) must not produce any results.");

    if (reason == StopReason::kBlock) {
        return;
    }

    switch (reason) {
        case StopReason::kBlock:
            BlockingState::SetResume(blocking_state);
            return;
        case StopReason::kYield:
            Enqueue(self);
        case StopReason::kPause:
        case StopReason::kTerminate:
        case StopReason::kAlreadyTerminated:
        case StopReason::kAtEnd:
            return;
        default:
            AssertInfo(false, "Unhandled stop reason");
    }
}

void
Driver::initializeOperators() {
    // Atomically check and set: only the first thread to call this will
    // get false (the previous value) and proceed with initialization.
    // Other threads will get true and skip initialization.
    // Use memory barriers to ensure initialization writes are visible
    // before the flag is seen by other threads.
    if (!operatorsInitialized_.exchange(true, std::memory_order_acq_rel)) {
        // This thread won the initialization race. Perform initialization.
        for (auto& op : operators_) {
            if (explain_ == nullptr) {
                op->initialize();
                continue;
            }
            const auto start = std::chrono::steady_clock::now();
            auto guard = folly::makeGuard([&]() noexcept {
                const auto ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
                if (ns > 0) {
                    explain_->AddOperatorTimeNs(op->get_operator_id(), ns);
                }
            });
            op->initialize();
        }
        // Use release semantics to ensure all initialization writes are
        // visible to other threads before they see the flag as true.
        // The flag was already set to true by exchange above, but this
        // barrier ensures all writes from initialization are visible.
        std::atomic_thread_fence(std::memory_order_release);
    } else {
        // Another thread is initializing or has completed initialization.
        // Use acquire semantics to ensure we see all initialization writes
        // after the flag becomes true.
        operatorsInitialized_.load(std::memory_order_acquire);
    }
}

void
Driver::Init(std::unique_ptr<DriverContext> ctx,
             std::vector<std::shared_ptr<Operator>> operators) {
    assert(ctx != nullptr);
    ctx_ = std::move(ctx);
    AssertInfo(operators.size() != 0, "operators in driver must not empty");
    operators_ = std::move(operators);
    current_operator_index_ = operators_.size() - 1;
}

void
Driver::CloseByTask() noexcept {
    try {
        Close();
    } catch (const std::exception& e) {
        LOG_WARN("Ignore driver cleanup error while terminating task: {}",
                 e.what());
    } catch (...) {
        LOG_WARN("Ignore unknown driver cleanup error while terminating task");
    }
}

void
Driver::Close() {
    if (closed_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }

    auto remove_driver_guard = folly::makeGuard(
        [task = ctx_->task_, this]() { Task::RemoveDriver(task, this); });
    std::exception_ptr close_error;
    for (auto& op : operators_) {
        const auto close_start = explain_ != nullptr
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
        try {
            op->WaitPrefetch();
        } catch (...) {
            if (close_error == nullptr) {
                close_error = std::current_exception();
            }
        }
        try {
            op->Close();
        } catch (...) {
            if (close_error == nullptr) {
                close_error = std::current_exception();
            }
        }
        if (explain_ != nullptr) {
            const auto ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - close_start)
                    .count();
            if (ns > 0) {
                explain_->AddOperatorTimeNs(op->get_operator_id(), ns);
            }
        }
    }

    if (close_error != nullptr) {
        std::rethrow_exception(close_error);
    }
}

RowVectorPtr
Driver::Next(std::shared_ptr<BlockingState>& blocking_state) {
    auto self = shared_from_this();

    RowVectorPtr result;
    auto stop = RunInternal(self, blocking_state, result);

    Assert(stop == StopReason::kBlock || stop == StopReason::kAtEnd ||
           stop == StopReason::kAlreadyTerminated);
    return result;
}

// Wraps an operator call so a failure carries the operator context, mapping
// each exception class the way the async consume arm (futures/Future.h) does:
//   - folly::FutureCancellation (query cancel; a timeout is converted to a
//     cancel upstream) -> ErrorCode::FollyCancel, so a canceled query surfaces
//     as a real cancellation at the CGO boundary and is retried by the
//     scheduler, instead of collapsing to a retriable-looking UnexpectedError.
//     Expected control flow, not a crash, so no stack trace is attached.
//   - milvus::SegcoreError -> rethrown with its ORIGINAL error code preserved
//     (the classification chosen at the throw site, e.g. ExprInvalid, must
//     survive to the CGO boundary instead of collapsing to UnexpectedError).
//   - any other std::exception -> legacy UnexpectedError classification.
#define CALL_OPERATOR(call_func, operator, method_name)                    \
    try {                                                                  \
        if (explain_ == nullptr) {                                         \
            call_func;                                                     \
        } else {                                                           \
            const int32_t op_step_ = (operator)->get_operator_id();        \
            const auto op_call_start_ = std::chrono::steady_clock::now();  \
            auto op_time_guard_ = folly::makeGuard([&]() noexcept {        \
                const auto ns =                                            \
                    std::chrono::duration_cast<std::chrono::nanoseconds>(  \
                        std::chrono::steady_clock::now() - op_call_start_) \
                        .count();                                          \
                if (ns > 0 && op_step_ >= 0) {                             \
                    if (static_cast<size_t>(op_step_) <                    \
                        local_op_time_ns.size()) {                         \
                        local_op_time_ns[op_step_] += ns;                  \
                    } else {                                               \
                        explain_->AddOperatorTimeNs(op_step_, ns);         \
                    }                                                      \
                }                                                          \
            });                                                            \
            call_func;                                                     \
        }                                                                  \
    } catch (folly::FutureCancellation & e) {                              \
        auto err_msg = fmt::format(                                        \
            "Operator::{} cancelled for [Operator:{}, plan node "          \
            "id: {}] : {}",                                                \
            method_name,                                                   \
            operator->ToString(),                                          \
            operator->get_plannode_id(),                                   \
            e.what());                                                     \
        throw ExecOperatorException(ErrorCode::FollyCancel, err_msg);      \
    } catch (milvus::SegcoreError & e) {                                   \
        std::string stack_trace = milvus::impl::EasyStackTrace();          \
        auto err_msg = fmt::format(                                        \
            "Operator::{} failed for [Operator:{}, plan node id: "         \
            "{}] : {}\nStack trace: {}",                                   \
            method_name,                                                   \
            operator->ToString(),                                          \
            operator->get_plannode_id(),                                   \
            e.what(),                                                      \
            stack_trace);                                                  \
        LOG_ERROR("{}", err_msg);                                          \
        throw ExecOperatorException(e.get_error_code(), err_msg);          \
    } catch (std::exception & e) {                                         \
        std::string stack_trace = milvus::impl::EasyStackTrace();          \
        auto err_msg = fmt::format(                                        \
            "Operator::{} failed for [Operator:{}, plan node id: "         \
            "{}] : {}\nStack trace: {}",                                   \
            method_name,                                                   \
            operator->ToString(),                                          \
            operator->get_plannode_id(),                                   \
            e.what(),                                                      \
            stack_trace);                                                  \
        LOG_ERROR("{}", err_msg);                                          \
        throw ExecOperatorException(err_msg);                              \
    }

StopReason
Driver::RunInternal(std::shared_ptr<Driver>& self,
                    std::shared_ptr<BlockingState>& blocking_state,
                    RowVectorPtr& result) {
    std::array<int64_t, 16> local_op_time_ns{};
    auto flush_local_op_time = folly::makeGuard([&]() noexcept {
        if (explain_ == nullptr) {
            return;
        }
        for (size_t step = 0; step < local_op_time_ns.size(); ++step) {
            if (local_op_time_ns[step] > 0) {
                explain_->AddOperatorTimeNs(static_cast<int32_t>(step),
                                            local_op_time_ns[step]);
                local_op_time_ns[step] = 0;
            }
        }
    });
    try {
        initializeOperators();
        if (ENABLE_DRIVER_PREFETCH.load(std::memory_order_relaxed)) {
            std::call_once(self->once_, [self]() { self->PrefetchAsync(); });
        }
        int num_operators = operators_.size();
        ContinueFuture future;

        for (;;) {
            for (int32_t i = num_operators - 1; i >= 0; --i) {
                auto op = operators_[i].get();
                current_operator_index_ = i;
                CALL_OPERATOR(
                    blocking_reason_ = op->IsBlocked(&future), op, "IsBlocked");
                if (blocking_reason_ != BlockingReason::kNotBlocked) {
                    blocking_state = std::make_shared<BlockingState>(
                        self, std::move(future), op, blocking_reason_);
                    return StopReason::kBlock;
                }
                Operator* next_op = nullptr;

                if (i < operators_.size() - 1) {
                    next_op = operators_[i + 1].get();
                    CALL_OPERATOR(
                        blocking_reason_ = next_op->IsBlocked(&future),
                        next_op,
                        "IsBlocked");
                    if (blocking_reason_ != BlockingReason::kNotBlocked) {
                        blocking_state = std::make_shared<BlockingState>(
                            self, std::move(future), next_op, blocking_reason_);
                        return StopReason::kBlock;
                    }

                    bool needs_input;
                    CALL_OPERATOR(needs_input = next_op->NeedInput(),
                                  next_op,
                                  "NeedInput");
                    if (needs_input) {
                        RowVectorPtr result;
                        {
                            CALL_OPERATOR(
                                result = op->GetOutput(), op, "GetOutput");
                            if (result) {
                                AssertInfo(
                                    result->size() > 0,
                                    fmt::format(
                                        "GetOutput must return nullptr or "
                                        "a non-empty vector: {}",
                                        op->get_operator_type()));
                            }
                        }
                        if (result) {
                            CALL_OPERATOR(
                                next_op->AddInput(result), next_op, "AddInput");
                            i += 2;
                            continue;
                        } else {
                            CALL_OPERATOR(
                                blocking_reason_ = op->IsBlocked(&future),
                                op,
                                "IsBlocked");
                            if (blocking_reason_ !=
                                BlockingReason::kNotBlocked) {
                                blocking_state =
                                    std::make_shared<BlockingState>(
                                        self,
                                        std::move(future),
                                        next_op,
                                        blocking_reason_);
                                return StopReason::kBlock;
                            }
                            if (op->IsFinished()) {
                                CALL_OPERATOR(next_op->NoMoreInput(),
                                              next_op,
                                              "NoMoreInput");
                                break;
                            }
                        }
                    }
                } else {
                    {
                        CALL_OPERATOR(
                            result = op->GetOutput(), op, "GetOutput");
                        if (result) {
                            AssertInfo(
                                result->size() > 0,
                                fmt::format("GetOutput must return nullptr or "
                                            "a non-empty vector: {}",
                                            op->get_operator_type()));
                            blocking_reason_ = BlockingReason::kWaitForConsumer;
                            return StopReason::kBlock;
                        }
                    }
                    if (op->IsFinished()) {
                        Close();
                        return StopReason::kAtEnd;
                    }
                    continue;
                }
            }
        }
    } catch (std::exception& e) {
        get_task()->SetError(std::current_exception());
        return StopReason::kAlreadyTerminated;
    }
}

void
Driver::PrefetchAsync() {
    auto prefetch_pool = GetPrefetchThreadPool();
    if (explain_ == nullptr) {
        for (auto& op : operators_) {
            op->PrefetchAsync(prefetch_pool);
        }
        return;
    }
    for (auto& op : operators_) {
        const auto start = std::chrono::steady_clock::now();
        auto guard = folly::makeGuard([this, op = op.get(), start]() noexcept {
            const auto ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count();
            if (ns > 0) {
                explain_->AddOperatorTimeNs(op->get_operator_id(), ns);
            }
        });
        op->PrefetchAsync(prefetch_pool);
    }
}

static bool
MustStartNewPipeline(std::shared_ptr<const plan::PlanNode> plannode,
                     int source_id) {
    //TODO: support LocalMerge and other shuffle
    return source_id != 0;
}

OperatorSupplier
MakeConsumerSupplier(ConsumerSupplier supplier) {
    if (supplier) {
        return [supplier](int32_t operator_id, DriverContext* ctx) {
            return std::make_unique<CallbackSink>(operator_id, ctx, supplier());
        };
    }
    return nullptr;
}

uint32_t
MaxDrivers(const DriverFactory* factory, const QueryConfig& config) {
    return 1;
}

static void
SplitPlan(const std::shared_ptr<const plan::PlanNode>& plannode,
          std::vector<std::shared_ptr<const plan::PlanNode>>* current_plannodes,
          const std::shared_ptr<const plan::PlanNode>& consumer_node,
          OperatorSupplier operator_supplier,
          std::vector<std::unique_ptr<DriverFactory>>* driver_factories) {
    if (!current_plannodes) {
        driver_factories->push_back(std::make_unique<DriverFactory>());
        current_plannodes = &driver_factories->back()->plannodes_;
        driver_factories->back()->consumer_supplier_ = operator_supplier;
        driver_factories->back()->consumer_node_ = consumer_node;
    }

    const auto& sources = plannode->sources();
    if (sources.empty()) {
        driver_factories->back()->is_input_driver_ = true;
    } else {
        for (int i = 0; i < sources.size(); ++i) {
            SplitPlan(
                sources[i],
                MustStartNewPipeline(plannode, i) ? nullptr : current_plannodes,
                plannode,
                nullptr,
                driver_factories);
        }
    }
    current_plannodes->push_back(plannode);
}

void
LocalPlanner::Plan(
    const plan::PlanFragment& fragment,
    ConsumerSupplier consumer_supplier,
    std::vector<std::unique_ptr<DriverFactory>>* driver_factories,
    const QueryConfig& config,
    uint32_t max_drivers) {
    SplitPlan(fragment.plan_node_,
              nullptr,
              nullptr,
              MakeConsumerSupplier(consumer_supplier),
              driver_factories);

    (*driver_factories)[0]->is_output_driver_ = true;

    for (auto& factory : *driver_factories) {
        factory->max_drivers_ = MaxDrivers(factory.get(), config);
        factory->num_drivers_ = std::min(factory->max_drivers_, max_drivers);

        if (factory->is_group_execution_) {
            factory->num_total_drivers_ =
                factory->num_drivers_ * fragment.num_splitgroups_;
        } else {
            factory->num_total_drivers_ = factory->num_drivers_;
        }
    }
}

}  // namespace exec
}  // namespace milvus

#undef CALL_OPERATOR
