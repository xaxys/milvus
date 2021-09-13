// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once
// Generated File
// DO NOT EDIT
#include <optional>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <boost/dynamic_bitset.hpp>
#include <utility>
#include <deque>
#include <boost_ext/dynamic_bitset_ext.hpp>
#include "segcore/SegmentGrowingImpl.h"
#include "query/ExprImpl.h"
#include "ExprVisitor.h"

namespace milvus::query {
class ExecExprVisitor : public ExprVisitor {
 public:
    void
    visit(ColumnExpr& expr) override;

    void
    visit(ValueExpr& expr) override;

    void
    visit(UnaryLogicalExpr& expr) override;

    void
    visit(BinaryLogicalExpr& expr) override;

    void
    visit(TermExpr& expr) override;

    void
    visit(UnaryRangeExpr& expr) override;

    void
    visit(BinaryRangeExpr& expr) override;

    void
    visit(CompareExpr& expr) override;

    void
    visit(UnaryArithExpr& expr) override;

    void
    visit(BinaryArithExpr& expr) override;

    void
    visit(CastExpr& expr) override;

 public:
    using RetType = arrow::Datum;
    using Bitmasks = std::deque<std::vector<bool>>;
    using ArrayPtr = std::shared_ptr<arrow::Array>;
    ExecExprVisitor(const segcore::SegmentInternalInterface& segment, int64_t row_count, Timestamp timestamp)
        : segment_(segment), row_count_(row_count), timestamp_(timestamp) {
    }
    RetType
    call_child(Expr& expr) {
        Assert(!ret_.has_value());
        expr.accept(*this);
        Assert(ret_.has_value());
        auto res = std::move(ret_);
        ret_ = std::nullopt;
        return std::move(res.value());
    }

 public:
    template <typename T, typename IndexFunc>
    auto
    GetBitmaskFromIndex(FieldOffset field_offset, IndexFunc func) -> Bitmasks;

    template <typename T>
    auto
    ExecUnaryRangeVisitorDispatcher(const FieldOffset& field_offset, const UnaryRangeExpr& expr)
        -> std::tuple<RetType, ArrayPtr>;

    template <typename T>
    auto
    ExecBinaryRangeVisitorDispatcher(const FieldOffset& field_offset, const BinaryRangeExpr& expr)
        -> std::tuple<RetType, ArrayPtr>;

    auto
    BuildFieldArray(const FieldOffset& offset, int64_t chunk_offset = 0) -> RetType;

    template <typename T, typename Builder>
    void
    ExtractFieldData(const FieldOffset& offset, Builder& builder, int64_t chunk_offset);

 private:
    const segcore::SegmentInternalInterface& segment_;
    int64_t row_count_;
    std::optional<RetType> ret_;
    Timestamp timestamp_;
};
}  // namespace milvus::query
