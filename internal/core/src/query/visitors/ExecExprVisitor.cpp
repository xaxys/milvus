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

#include <optional>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <boost/dynamic_bitset.hpp>
#include <utility>
#include <deque>
#include <boost_ext/dynamic_bitset_ext.hpp>
#include "segcore/SegmentGrowingImpl.h"
#include "query/ExprImpl.h"
#include "query/generated/ExecExprVisitor.h"

namespace milvus::query {
#if 1
// THIS CONTAINS EXTRA BODY FOR VISITOR
// WILL BE USED BY GENERATOR
namespace impl {
class ExecExprVisitor : ExprVisitor {
 public:
    using RetType = arrow::Datum;
    using Bitmask = std::deque<std::vector<bool>>;
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
    GetBitmaskFromIndex(FieldOffset field_offset, IndexFunc func) -> Bitmask;

    template <typename T>
    auto
    ExecUnaryRangeVisitorDispatcher(UnaryRangeExpr& expr_raw) -> RetType;

    template <typename T>
    auto
    ExecBinaryRangeVisitorDispatcher(BinaryRangeExpr& expr_raw) -> RetType;

    template <typename T>
    auto
    ExecTermVisitorImpl(TermExpr& expr_raw) -> RetType;

    auto
    BuildFieldArray(const FieldOffset& offset, std::optional<Bitmask> bitmask = std::nullopt) -> RetType;

    template <typename T, typename Builder>
    void
    ExtractFieldData(const FieldOffset& offset, Builder& builder, std::optional<Bitmask>& bitmask);

 private:
    const segcore::SegmentInternalInterface& segment_;
    int64_t row_count_;
    std::optional<RetType> ret_;
    Timestamp timestamp_;
};
}  // namespace impl
#endif

namespace cp = ::arrow::compute;

void
ExecExprVisitor::visit(UnaryLogicalExpr& expr) {
    auto child_res = call_child(*expr.child_);
    RetType res;
    switch (expr.op_type_) {
        case UnaryLogicalOp::LogicalNot: {
            res = std::move(cp::Invert(child_res).ValueOrDie());
            break;
        }
        default: {
            PanicInfo("Invalid Unary Op");
        }
    }
    Assert(res.length() == row_count_);
    ret_ = std::move(res);
}

void
ExecExprVisitor::visit(BinaryLogicalExpr& expr) {
    auto left_res = call_child(*expr.left_);
    auto right_res = call_child(*expr.right_);
    RetType res;
    switch (expr.op_type_) {
        case BinaryLogicalOp::LogicalAnd: {
            res = cp::And(left_res, right_res).ValueOrDie();
            break;
        }
        case BinaryLogicalOp::LogicalOr: {
            res = cp::Or(left_res, right_res).ValueOrDie();
            break;
        }
        case BinaryLogicalOp::LogicalXor: {
            res = cp::Xor(left_res, right_res).ValueOrDie();
            break;
        }
        default: {
            PanicInfo("Invalid Binary Op");
        }
    }
    Assert(res.length() == row_count_);
    ret_ = std::move(res);
}

template <typename T, typename IndexFunc>
auto
ExecExprVisitor::GetBitmaskFromIndex(FieldOffset field_offset, IndexFunc index_func) -> Bitmask {
    auto indexing_barrier = segment_.num_chunk_index(field_offset);
    auto size_per_chunk = segment_.size_per_chunk();
    auto num_chunk = upper_div(row_count_, size_per_chunk);
    Bitmask bitmask;
    using Index = knowhere::scalar::StructuredIndex<T>;
    for (auto chunk_id = 0; chunk_id < indexing_barrier; ++chunk_id) {
        const Index& indexing = segment_.chunk_scalar_index<T>(field_offset, chunk_id);
        // NOTE: knowhere is not const-ready
        // This is a dirty workaround
        auto data = index_func(const_cast<Index*>(&indexing));
        Assert(data->size() == size_per_chunk);
        bitmask.emplace_back(std::move(*data));
    }
    for (auto chunk_id = indexing_barrier; chunk_id < num_chunk; ++chunk_id) {
        auto size = chunk_id == num_chunk - 1 ? row_count_ - chunk_id * size_per_chunk : size_per_chunk;
        bitmask.emplace_back(std::vector<bool>(size, true));
    }
    return bitmask;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "Simplify"
template <typename T>
auto
ExecExprVisitor::ExecUnaryRangeVisitorDispatcher(UnaryRangeExpr& expr_raw) -> RetType {
    auto& expr = static_cast<UnaryRangeExprImpl<T>&>(expr_raw);
    auto op = expr.op_type_;
    auto val = expr.value_;
    static const std::map<CompareOp, std::string> op_name = {
        {CompareOp::Equal, "equal"},
        {CompareOp::NotEqual, "not_equal"},
        {CompareOp::GreaterEqual, "greater_equal"},
        {CompareOp::GreaterThan, "greater"},
        {CompareOp::LessEqual, "less_equal"},
        {CompareOp::LessThan, "less"},
    };
    RetType child_res;
    if (const auto child = dynamic_cast<ColumnExpr*>(expr.child_.get()); child) {
        // get bitmask from index
        using Index = knowhere::scalar::StructuredIndex<T>;
        using Operator = knowhere::scalar::OperatorType;
        auto field_offset = child->field_offset_;
        Bitmask bitmask;
        switch (op) {
            case CompareOp::Equal: {
                auto index_func = [val](Index* index) { return index->In(1, &val); };
                bitmask = GetBitmaskFromIndex<T>(field_offset, index_func);
                break;
            }
            case CompareOp::NotEqual: {
                auto index_func = [val](Index* index) { return index->NotIn(1, &val); };
                bitmask = GetBitmaskFromIndex<T>(field_offset, index_func);
                break;
            }
            case CompareOp::GreaterEqual: {
                auto index_func = [val](Index* index) { return index->Range(val, Operator::GE); };
                bitmask = GetBitmaskFromIndex<T>(field_offset, index_func);
                break;
            }
            case CompareOp::GreaterThan: {
                auto index_func = [val](Index* index) { return index->Range(val, Operator::GT); };
                bitmask = GetBitmaskFromIndex<T>(field_offset, index_func);
                break;
            }
            case CompareOp::LessEqual: {
                auto index_func = [val](Index* index) { return index->Range(val, Operator::LE); };
                bitmask = GetBitmaskFromIndex<T>(field_offset, index_func);
                break;
            }
            case CompareOp::LessThan: {
                auto index_func = [val](Index* index) { return index->Range(val, Operator::LT); };
                bitmask = GetBitmaskFromIndex<T>(field_offset, index_func);
                break;
            }
            default: {
                PanicInfo("unsupported range node");
            }
        }
        child_res = BuildFieldArray(field_offset, std::move(bitmask));
    } else {
        child_res = call_child(*expr.child_);
    }
    auto scalar = arrow::Datum(val);
    auto final_res = cp::CallFunction(op_name.at(op), {child_res, scalar}).ValueOrDie();
    return cp::FillNull(final_res, arrow::Datum(false)).ValueOrDie();
}
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma ide diagnostic ignored "Simplify"
template <typename T>
auto
ExecExprVisitor::ExecBinaryRangeVisitorDispatcher(BinaryRangeExpr& expr_raw) -> RetType {
    auto& expr = static_cast<BinaryRangeExprImpl<T>&>(expr_raw);
    bool lower_inclusive = expr.lower_inclusive_;
    bool upper_inclusive = expr.upper_inclusive_;
    T val1 = expr.lower_value_;
    T val2 = expr.upper_value_;
    RetType child_res;
    if (val1 > val2 || (val1 == val2 && !(lower_inclusive && upper_inclusive))) {
        return arrow::Datum(false);
    }
    if (const auto child = dynamic_cast<ColumnExpr*>(expr.child_.get()); child) {
        // get bitmask from index
        using Index = knowhere::scalar::StructuredIndex<T>;
        using Operator = knowhere::scalar::OperatorType;
        auto field_offset = child->field_offset_;
        auto index_func = [=](Index* index) { return index->Range(val1, lower_inclusive, val2, upper_inclusive); };
        auto bitmask = GetBitmaskFromIndex<T>(field_offset, index_func);
        child_res = BuildFieldArray(field_offset, std::move(bitmask));
    } else {
        child_res = call_child(*expr.child_);
    }
    auto scalar1 = arrow::Datum(val1);
    auto scalar2 = arrow::Datum(val2);
    std::string op_name1 = lower_inclusive ? "greater_equal" : "greater";
    std::string op_name2 = upper_inclusive ? "less_equal" : "less";
    auto res1 = cp::CallFunction(op_name1, {child_res, scalar1}).ValueOrDie();
    auto res2 = cp::CallFunction(op_name2, {child_res, scalar2}).ValueOrDie();
    auto final_res = cp::And(res1, res2).ValueOrDie();
    return cp::FillNull(final_res, arrow::Datum(false)).ValueOrDie();
}
#pragma clang diagnostic pop

void
ExecExprVisitor::visit(UnaryRangeExpr& expr) {
    RetType res;
    switch (expr.child_->data_type_) {
        case DataType::BOOL: {
            res = ExecUnaryRangeVisitorDispatcher<bool>(expr);
            break;
        }
        case DataType::INT8: {
            res = ExecUnaryRangeVisitorDispatcher<int8_t>(expr);
            break;
        }
        case DataType::INT16: {
            res = ExecUnaryRangeVisitorDispatcher<int16_t>(expr);
            break;
        }
        case DataType::INT32: {
            res = ExecUnaryRangeVisitorDispatcher<int32_t>(expr);
            break;
        }
        case DataType::INT64: {
            res = ExecUnaryRangeVisitorDispatcher<int64_t>(expr);
            break;
        }
        case DataType::FLOAT: {
            res = ExecUnaryRangeVisitorDispatcher<float>(expr);
            break;
        }
        case DataType::DOUBLE: {
            res = ExecUnaryRangeVisitorDispatcher<double>(expr);
            break;
        }
        default:
            PanicInfo("unsupported datatype");
    }
    Assert(res.length() == row_count_);
    ret_ = std::move(res);
}

void
ExecExprVisitor::visit(BinaryRangeExpr& expr) {
    RetType res;
    switch (expr.child_->data_type_) {
        case DataType::BOOL: {
            res = ExecBinaryRangeVisitorDispatcher<bool>(expr);
            break;
        }
        case DataType::INT8: {
            res = ExecBinaryRangeVisitorDispatcher<int8_t>(expr);
            break;
        }
        case DataType::INT16: {
            res = ExecBinaryRangeVisitorDispatcher<int16_t>(expr);
            break;
        }
        case DataType::INT32: {
            res = ExecBinaryRangeVisitorDispatcher<int32_t>(expr);
            break;
        }
        case DataType::INT64: {
            res = ExecBinaryRangeVisitorDispatcher<int64_t>(expr);
            break;
        }
        case DataType::FLOAT: {
            res = ExecBinaryRangeVisitorDispatcher<float>(expr);
            break;
        }
        case DataType::DOUBLE: {
            res = ExecBinaryRangeVisitorDispatcher<double>(expr);
            break;
        }
        default:
            PanicInfo("unsupported datatype");
    }
    Assert(res.is_scalar() || res.length() == row_count_);
    ret_ = std::move(res);
}

void
ExecExprVisitor::visit(CompareExpr& expr) {
    auto op = expr.op_type_;
    auto left_res = call_child(*expr.left_);
    auto right_res = call_child(*expr.right_);
    static const std::map<CompareOp, std::string> op_name = {
        {CompareOp::Equal, "equal"},
        {CompareOp::NotEqual, "not_equal"},
        {CompareOp::GreaterEqual, "greater_equal"},
        {CompareOp::GreaterThan, "greater"},
        {CompareOp::LessEqual, "less_equal"},
        {CompareOp::LessThan, "less"},
    };
    RetType res = cp::CallFunction(op_name.at(op), {left_res, right_res}).ValueOrDie();
    Assert(res.is_scalar() || res.length() == row_count_);
    ret_ = std::move(res);
}

template <typename T>
auto
ExecExprVisitor::ExecTermVisitorImpl(TermExpr& expr_raw) -> RetType {
    auto& expr = static_cast<TermExprImpl<T>&>(expr_raw);
    auto child_res = call_child(*expr.child_);
    arrow::Datum terms;
    if constexpr (std::is_same_v<T, bool>) {
        arrow::BooleanBuilder builder;
        builder.AppendValues(expr.terms_);
        terms = builder.Finish().ValueOrDie();
    } else if constexpr (std::is_same_v<T, int8_t>) {
        arrow::Int8Builder builder;
        builder.AppendValues(expr.terms_);
        terms = builder.Finish().ValueOrDie();
    } else if constexpr (std::is_same_v<T, int16_t>) {
        arrow::Int16Builder builder;
        builder.AppendValues(expr.terms_);
        terms = builder.Finish().ValueOrDie();
    } else if constexpr (std::is_same_v<T, int32_t>) {
        arrow::Int32Builder builder;
        builder.AppendValues(expr.terms_);
        terms = builder.Finish().ValueOrDie();
    } else if constexpr (std::is_same_v<T, int64_t>) {
        arrow::Int64Builder builder;
        builder.AppendValues(expr.terms_);
        terms = builder.Finish().ValueOrDie();
    } else if constexpr (std::is_same_v<T, float>) {
        arrow::FloatBuilder builder;
        builder.AppendValues(expr.terms_);
        terms = builder.Finish().ValueOrDie();
    } else if constexpr (std::is_same_v<T, double>) {
        arrow::DoubleBuilder builder;
        builder.AppendValues(expr.terms_);
        terms = builder.Finish().ValueOrDie();
    } else {
        PanicInfo("unsupported datatype");
    }
    return cp::IsIn(child_res, cp::SetLookupOptions(terms, true)).ValueOrDie();
}

void
ExecExprVisitor::visit(TermExpr& expr) {
    RetType res;
    switch (expr.child_->data_type_) {
        case DataType::BOOL: {
            res = ExecTermVisitorImpl<bool>(expr);
            break;
        }
        case DataType::INT8: {
            res = ExecTermVisitorImpl<int8_t>(expr);
            break;
        }
        case DataType::INT16: {
            res = ExecTermVisitorImpl<int16_t>(expr);
            break;
        }
        case DataType::INT32: {
            res = ExecTermVisitorImpl<int32_t>(expr);
            break;
        }
        case DataType::INT64: {
            res = ExecTermVisitorImpl<int64_t>(expr);
            break;
        }
        case DataType::FLOAT: {
            res = ExecTermVisitorImpl<float>(expr);
            break;
        }
        case DataType::DOUBLE: {
            res = ExecTermVisitorImpl<double>(expr);
            break;
        }
        default:
            PanicInfo("unsupported datatype");
    }
    Assert(res.length() == row_count_);
    ret_ = std::move(res);
}

template <typename T, typename Builder>
void
ExecExprVisitor::ExtractFieldData(const FieldOffset& offset, Builder& builder, std::optional<Bitmask>& bitmask) {
    auto size_per_chunk = segment_.size_per_chunk();
    auto num_chunk = upper_div(row_count_, size_per_chunk);
    auto num_bitmasks = bitmask ? bitmask->size() : 0;
    for (int64_t chunk_id = 0; chunk_id < num_bitmasks; ++chunk_id) {
        auto size = chunk_id == num_chunk - 1 ? row_count_ - chunk_id * size_per_chunk : size_per_chunk;
        auto data = segment_.chunk_data<T>(offset, chunk_id).data();
        auto validity = bitmask->at(chunk_id);
        Assert(validity.size() == size);
        builder.AppendValues(data, size, validity);
    }
    for (int64_t chunk_id = num_bitmasks; chunk_id < num_chunk; ++chunk_id) {
        auto size = chunk_id == num_chunk - 1 ? row_count_ - chunk_id * size_per_chunk : size_per_chunk;
        auto chunk_offset = chunk_id * size_per_chunk;
        auto data = segment_.chunk_data<T>(offset, chunk_id).data();
        builder.AppendValues(data, size);
    }
}

auto
ExecExprVisitor::BuildFieldArray(const FieldOffset& offset, std::optional<Bitmask> bitmask) -> RetType {
    auto data_type = segment_.get_schema()[offset].get_data_type();
    RetType res;
    switch (data_type) {
        case DataType::BOOL: {
            auto builder = arrow::BooleanBuilder();
            ExtractFieldData<uint8_t>(offset, builder, bitmask);
            res = builder.Finish().ValueOrDie();
            break;
        }
        case DataType::INT8: {
            auto builder = arrow::Int8Builder();
            ExtractFieldData<int8_t>(offset, builder, bitmask);
            res = builder.Finish().ValueOrDie();
            break;
        }
        case DataType::INT16: {
            auto builder = arrow::Int16Builder();
            ExtractFieldData<int16_t>(offset, builder, bitmask);
            res = builder.Finish().ValueOrDie();
            break;
        }
        case DataType::INT32: {
            auto builder = arrow::Int32Builder();
            ExtractFieldData<int32_t>(offset, builder, bitmask);
            res = builder.Finish().ValueOrDie();
            break;
        }
        case DataType::INT64: {
            auto builder = arrow::Int64Builder();
            ExtractFieldData<int64_t>(offset, builder, bitmask);
            res = builder.Finish().ValueOrDie();
            break;
        }
        case DataType::FLOAT: {
            auto builder = arrow::FloatBuilder();
            ExtractFieldData<float>(offset, builder, bitmask);
            res = builder.Finish().ValueOrDie();
            break;
        }
        case DataType::DOUBLE: {
            auto builder = arrow::DoubleBuilder();
            ExtractFieldData<double>(offset, builder, bitmask);
            res = builder.Finish().ValueOrDie();
            break;
        }
        default:
            PanicInfo("unsupported datatype");
    }
    return res;
}

void
ExecExprVisitor::visit(ColumnExpr& expr) {
    auto& field_meta = segment_.get_schema()[expr.field_offset_];
    Assert(expr.data_type_ == field_meta.get_data_type());
    RetType res = BuildFieldArray(expr.field_offset_);
    Assert(res.length() == row_count_);
    ret_ = std::move(res);
}

void
ExecExprVisitor::visit(ArithExpr& expr) {
    auto op = expr.op_type_;
    auto left_res = call_child(*expr.left_);
    auto right_res = call_child(*expr.right_);
    static const std::map<ArithOp, std::string> op_name = {
        {ArithOp::Add, "add"},
        {ArithOp::Subtract, "subtract"},
        {ArithOp::Multiply, "multiply"},
        {ArithOp::Divide, "divide"},
        {ArithOp::Modulo, "modulo"},
        {ArithOp::Power, "power"},
        {ArithOp::BitAnd, "bit_wise_and"},
        {ArithOp::BitOr, "bit_wise_or"},
        {ArithOp::BitXor, "bit_wise_xor"},
    };
    RetType res = cp::CallFunction(op_name.at(op), {left_res, right_res}).ValueOrDie();
    Assert(res.is_scalar() || res.length() == row_count_);
    ret_ = std::move(res);
}

void
ExecExprVisitor::visit(ValueExpr& expr) {
    RetType res;
    switch (expr.data_type_) {
        case DataType::BOOL: {
            auto& expr_impl = static_cast<ValueExprImpl<bool>&>(expr);
            res = arrow::Datum(expr_impl.value_);
            break;
        }
        case DataType::INT8: {
            auto& expr_impl = static_cast<ValueExprImpl<int8_t>&>(expr);
            res = arrow::Datum(expr_impl.value_);
            break;
        }
        case DataType::INT16: {
            auto& expr_impl = static_cast<ValueExprImpl<int16_t>&>(expr);
            res = arrow::Datum(expr_impl.value_);
            break;
        }
        case DataType::INT32: {
            auto& expr_impl = static_cast<ValueExprImpl<int32_t>&>(expr);
            res = arrow::Datum(expr_impl.value_);
            break;
        }
        case DataType::INT64: {
            auto& expr_impl = static_cast<ValueExprImpl<int64_t>&>(expr);
            res = arrow::Datum(expr_impl.value_);
            break;
        }
        case DataType::FLOAT: {
            auto& expr_impl = static_cast<ValueExprImpl<float>&>(expr);
            res = arrow::Datum(expr_impl.value_);
            break;
        }
        case DataType::DOUBLE: {
            auto& expr_impl = static_cast<ValueExprImpl<double>&>(expr);
            res = arrow::Datum(expr_impl.value_);
            break;
        }
        default:
            PanicInfo("unsupported datatype");
    }
    ret_ = std::move(res);
}

void
ExecExprVisitor::visit(CastExpr& expr) {
    auto child_res = call_child(*expr.child_);
    static const std::map<DataType, std::shared_ptr<arrow::DataType>> type_name = {
        {DataType::BOOL, arrow::boolean()},   {DataType::INT8, arrow::int8()},   {DataType::INT16, arrow::int16()},
        {DataType::INT32, arrow::int32()},    {DataType::INT64, arrow::int64()}, {DataType::FLOAT, arrow::float32()},
        {DataType::DOUBLE, arrow::float64()},
    };
    RetType res = cp::Cast(child_res, cp::CastOptions::Unsafe(type_name.at(expr.data_type_))).ValueOrDie();
    Assert(res.is_scalar() || res.length() == row_count_);
    ret_ = std::move(res);
}
}  // namespace milvus::query
