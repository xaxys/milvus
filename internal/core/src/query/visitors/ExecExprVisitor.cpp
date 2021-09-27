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
#include <vector>
#include <memory>
#include <tuple>
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
    using Bitmasks = std::deque<std::vector<bool>>;
    using ArrayPtr = std::shared_ptr<arrow::Array>;
    ExecExprVisitor(const segcore::SegmentInternalInterface& segment, int64_t row_count, Timestamp timestamp)
        : segment_(segment), row_count_(row_count), timestamp_(timestamp) {
    }
    RetType
    call_child(Expr& expr) {
        AssertInfo(!ret_.has_value(), "[ExecExprVisitor]Bitset already has value before accept");
        expr.accept(*this);
        AssertInfo(ret_.has_value(), "[ExecExprVisitor]Bitset doesn't have value after accept");
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
}  // namespace impl
#endif

namespace cp = ::arrow::compute;

static arrow::Datum
ExecGenericValueVisitor(const GenericValue& gv) {
    switch (gv.data_type_) {
        case DataType::BOOL: {
            auto& gv_impl = static_cast<const GenericValueImpl<bool>&>(gv);
            return arrow::Datum(gv_impl.value_);
        }
        case DataType::INT8: {
            auto& gv_impl = static_cast<const GenericValueImpl<int8_t>&>(gv);
            return arrow::Datum(gv_impl.value_);
        }
        case DataType::INT16: {
            auto& gv_impl = static_cast<const GenericValueImpl<int16_t>&>(gv);
            return arrow::Datum(gv_impl.value_);
        }
        case DataType::INT32: {
            auto& gv_impl = static_cast<const GenericValueImpl<int32_t>&>(gv);
            return arrow::Datum(gv_impl.value_);
        }
        case DataType::INT64: {
            auto& gv_impl = static_cast<const GenericValueImpl<int64_t>&>(gv);
            return arrow::Datum(gv_impl.value_);
        }
        case DataType::FLOAT: {
            auto& gv_impl = static_cast<const GenericValueImpl<float>&>(gv);
            return arrow::Datum(gv_impl.value_);
        }
        case DataType::DOUBLE: {
            auto& gv_impl = static_cast<const GenericValueImpl<double>&>(gv);
            return arrow::Datum(gv_impl.value_);
        }
        default:
            PanicInfo("unsupported datatype");
    }
}

template <typename T>
static T
ExtractGenericValue(const GenericValue& gv) {
    switch (gv.data_type_) {
        case DataType::BOOL: {
            auto& gv_impl = static_cast<const GenericValueImpl<bool>&>(gv);
            return static_cast<T>(gv_impl.value_);
        }
        case DataType::INT8: {
            auto& gv_impl = static_cast<const GenericValueImpl<int8_t>&>(gv);
            return static_cast<T>(gv_impl.value_);
        }
        case DataType::INT16: {
            auto& gv_impl = static_cast<const GenericValueImpl<int16_t>&>(gv);
            return static_cast<T>(gv_impl.value_);
        }
        case DataType::INT32: {
            auto& gv_impl = static_cast<const GenericValueImpl<int32_t>&>(gv);
            return static_cast<T>(gv_impl.value_);
        }
        case DataType::INT64: {
            auto& gv_impl = static_cast<const GenericValueImpl<int64_t>&>(gv);
            return static_cast<T>(gv_impl.value_);
        }
        case DataType::FLOAT: {
            auto& gv_impl = static_cast<const GenericValueImpl<float>&>(gv);
            return static_cast<T>(gv_impl.value_);
        }
        case DataType::DOUBLE: {
            auto& gv_impl = static_cast<const GenericValueImpl<double>&>(gv);
            return static_cast<T>(gv_impl.value_);
        }
        default:
            PanicInfo("unsupported datatype");
    }
}

template <typename T, typename Builder>
static void
ExtractGenericValueList(Builder& builder, const std::vector<GenericValuePtr>& gvl) {
    for (const auto& gv : gvl) {
        auto gv_impl = dynamic_cast<const GenericValueImpl<T>*>(gv.get());
        Assert(gv_impl);
        builder.Append(gv_impl->value_);
    }
}

static arrow::Datum
ExecGenericValueListVisitor(const std::vector<GenericValuePtr>& gvl) {
    if (gvl.size() <= 0) {
        return arrow::Datum(false);
    }
    switch (gvl[0]->data_type_) {
        case DataType::BOOL: {
            arrow::BooleanBuilder builder;
            ExtractGenericValueList<bool>(builder, gvl);
            return builder.Finish().ValueOrDie();
        }
        case DataType::INT8: {
            arrow::Int8Builder builder;
            ExtractGenericValueList<int8_t>(builder, gvl);
            return builder.Finish().ValueOrDie();
        }
        case DataType::INT16: {
            arrow::Int16Builder builder;
            ExtractGenericValueList<int16_t>(builder, gvl);
            return builder.Finish().ValueOrDie();
        }
        case DataType::INT32: {
            arrow::Int32Builder builder;
            ExtractGenericValueList<int32_t>(builder, gvl);
            return builder.Finish().ValueOrDie();
        }
        case DataType::INT64: {
            arrow::Int64Builder builder;
            ExtractGenericValueList<int64_t>(builder, gvl);
            return builder.Finish().ValueOrDie();
        }
        case DataType::FLOAT: {
            arrow::FloatBuilder builder;
            ExtractGenericValueList<float>(builder, gvl);
            return builder.Finish().ValueOrDie();
        }
        case DataType::DOUBLE: {
            arrow::DoubleBuilder builder;
            ExtractGenericValueList<double>(builder, gvl);
            return builder.Finish().ValueOrDie();
        }
        default:
            PanicInfo("unsupported datatype");
    }
}

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
    AssertInfo(res.is_scalar() || res.length() == row_count_, "[ExecExprVisitor]Size of results not equal row count");
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
    AssertInfo(res.is_scalar() || res.length() == row_count_, "[ExecExprVisitor]Size of results not equal row count");
    ret_ = std::move(res);
}

template <typename T, typename IndexFunc>
auto
ExecExprVisitor::GetBitmaskFromIndex(FieldOffset field_offset, IndexFunc index_func) -> Bitmasks {
    auto indexing_barrier = segment_.num_chunk_index(field_offset);
    auto size_per_chunk = segment_.size_per_chunk();
    auto num_chunk = upper_div(row_count_, size_per_chunk);
    Bitmasks bitmasks;
    using Index = knowhere::scalar::StructuredIndex<T>;
    for (auto chunk_id = 0; chunk_id < indexing_barrier; ++chunk_id) {
        const Index& indexing = segment_.chunk_scalar_index<T>(field_offset, chunk_id);
        // NOTE: knowhere is not const-ready
        // This is a dirty workaround
        auto data = index_func(const_cast<Index*>(&indexing));
        AssertInfo(data->size() == size_per_chunk, "[ExecExprVisitor]Data size not equal to size_per_chunk");
        bitmasks.emplace_back(std::move(*data));
    }
    return bitmasks;
}

static ExecExprVisitor::ArrayPtr
BuildBitmasksArray(const ExecExprVisitor::Bitmasks& bitmasks) {
    arrow::BooleanBuilder builder;
    for (const auto& bitmask : bitmasks) {
        builder.AppendValues(bitmask);
    }
    return builder.Finish().ValueOrDie();
}

template <typename T>
auto
ExecExprVisitor::ExecBinaryRangeVisitorDispatcher(const FieldOffset& field_offset, const BinaryRangeExpr& expr)
    -> std::tuple<RetType, ArrayPtr> {
    using Index = knowhere::scalar::StructuredIndex<T>;
    auto lower_inclusive = expr.lower_inclusive_;
    auto upper_inclusive = expr.upper_inclusive_;
    auto lower_value = ExtractGenericValue<T>(*expr.lower_value_);
    auto upper_value = ExtractGenericValue<T>(*expr.upper_value_);
    if (lower_inclusive && upper_inclusive) {
        if (lower_value > upper_value)
            return {arrow::Datum(false), nullptr};
    } else {
        if (lower_value >= upper_value)
            return {arrow::Datum(false), nullptr};
    }
    auto index_func = [=](Index* index) {
        return index->Range(lower_value, lower_inclusive, upper_value, upper_inclusive);
    };
    auto bitmasks = GetBitmaskFromIndex<T>(field_offset, index_func);
    auto child_res = BuildFieldArray(field_offset, bitmasks.size());
    auto index_res = BuildBitmasksArray(bitmasks);
    return {child_res, index_res};
}

template <typename T>
auto
ExecExprVisitor::ExecUnaryRangeVisitorDispatcher(const FieldOffset& field_offset, const UnaryRangeExpr& expr)
    -> std::tuple<RetType, ArrayPtr> {
    using Index = knowhere::scalar::StructuredIndex<T>;
    using Operator = knowhere::scalar::OperatorType;
    auto value = ExtractGenericValue<T>(*expr.value_);
    Bitmasks bitmasks;
    switch (expr.op_type_) {
        case CompareOp::Equal: {
            auto index_func = [=](Index* index) { return index->In(1, &value); };
            bitmasks = GetBitmaskFromIndex<T>(field_offset, index_func);
            break;
        }
        case CompareOp::NotEqual: {
            auto index_func = [=](Index* index) { return index->NotIn(1, &value); };
            bitmasks = GetBitmaskFromIndex<T>(field_offset, index_func);
            break;
        }
        case CompareOp::GreaterEqual: {
            auto index_func = [=](Index* index) { return index->Range(value, Operator::GE); };
            bitmasks = GetBitmaskFromIndex<T>(field_offset, index_func);
            break;
        }
        case CompareOp::GreaterThan: {
            auto index_func = [=](Index* index) { return index->Range(value, Operator::GT); };
            bitmasks = GetBitmaskFromIndex<T>(field_offset, index_func);
            break;
        }
        case CompareOp::LessEqual: {
            auto index_func = [=](Index* index) { return index->Range(value, Operator::LE); };
            bitmasks = GetBitmaskFromIndex<T>(field_offset, index_func);
            break;
        }
        case CompareOp::LessThan: {
            auto index_func = [=](Index* index) { return index->Range(value, Operator::LT); };
            bitmasks = GetBitmaskFromIndex<T>(field_offset, index_func);
            break;
        }
        default: {
            PanicInfo("unsupported range node");
        }
    }
    auto child_res = BuildFieldArray(field_offset, bitmasks.size());
    auto index_res = BuildBitmasksArray(bitmasks);
    return {child_res, index_res};
}

void
ExecExprVisitor::visit(UnaryRangeExpr& expr) {
    auto op = expr.op_type_;
    static const std::map<CompareOp, std::string> op_name = {
        {CompareOp::Equal, "equal"},
        {CompareOp::NotEqual, "not_equal"},
        {CompareOp::GreaterEqual, "greater_equal"},
        {CompareOp::GreaterThan, "greater"},
        {CompareOp::LessEqual, "less_equal"},
        {CompareOp::LessThan, "less"},
    };
    RetType child_res;
    ArrayPtr index_res;
    UnaryExprBase* column_expr = &expr;
    for (auto child = dynamic_cast<CastExpr*>(column_expr->child_.get()); child;
         child = dynamic_cast<CastExpr*>(child->child_.get())) {
        column_expr = child;
    }
    if (const auto child = dynamic_cast<ColumnExpr*>(column_expr->child_.get()); child) {
        // get bitmask from index
        switch (child->data_type_) {
            case DataType::BOOL: {
                std::tie(child_res, index_res) = ExecUnaryRangeVisitorDispatcher<bool>(child->field_offset_, expr);
                break;
            }
            case DataType::INT8: {
                std::tie(child_res, index_res) = ExecUnaryRangeVisitorDispatcher<int8_t>(child->field_offset_, expr);
                break;
            }
            case DataType::INT16: {
                std::tie(child_res, index_res) = ExecUnaryRangeVisitorDispatcher<int16_t>(child->field_offset_, expr);
                break;
            }
            case DataType::INT32: {
                std::tie(child_res, index_res) = ExecUnaryRangeVisitorDispatcher<int32_t>(child->field_offset_, expr);
                break;
            }
            case DataType::INT64: {
                std::tie(child_res, index_res) = ExecUnaryRangeVisitorDispatcher<int64_t>(child->field_offset_, expr);
                break;
            }
            case DataType::FLOAT: {
                std::tie(child_res, index_res) = ExecUnaryRangeVisitorDispatcher<float>(child->field_offset_, expr);
                break;
            }
            case DataType::DOUBLE: {
                std::tie(child_res, index_res) = ExecUnaryRangeVisitorDispatcher<double>(child->field_offset_, expr);
                break;
            }
            default:
                PanicInfo("unsupported datatype");
        }
    } else {
        child_res = call_child(*expr.child_);
    }
    auto scalar = ExecGenericValueVisitor(*expr.value_);
    auto res = cp::CallFunction(op_name.at(op), {child_res, scalar}).ValueOrDie();
    if (index_res) {
        res = arrow::Concatenate({index_res, res.make_array()}).ValueOrDie();
    }
    AssertInfo(res.length() == row_count_, "[ExecExprVisitor]Size of results not equal row count");
    ret_ = std::move(res);
}

void
ExecExprVisitor::visit(BinaryRangeExpr& expr) {
    bool lower_inclusive = expr.lower_inclusive_;
    bool upper_inclusive = expr.upper_inclusive_;
    RetType child_res;
    ArrayPtr index_res;
    UnaryExprBase* column_expr = &expr;
    for (auto child = dynamic_cast<CastExpr*>(column_expr->child_.get()); child;
         child = dynamic_cast<CastExpr*>(child->child_.get())) {
        column_expr = child;
    }
    if (const auto child = dynamic_cast<ColumnExpr*>(column_expr->child_.get()); child) {
        // get bitmask from index
        switch (child->data_type_) {
            case DataType::BOOL: {
                std::tie(child_res, index_res) = ExecBinaryRangeVisitorDispatcher<bool>(child->field_offset_, expr);
                break;
            }
            case DataType::INT8: {
                std::tie(child_res, index_res) = ExecBinaryRangeVisitorDispatcher<int8_t>(child->field_offset_, expr);
                break;
            }
            case DataType::INT16: {
                std::tie(child_res, index_res) = ExecBinaryRangeVisitorDispatcher<int16_t>(child->field_offset_, expr);
                break;
            }
            case DataType::INT32: {
                std::tie(child_res, index_res) = ExecBinaryRangeVisitorDispatcher<int32_t>(child->field_offset_, expr);
                break;
            }
            case DataType::INT64: {
                std::tie(child_res, index_res) = ExecBinaryRangeVisitorDispatcher<int64_t>(child->field_offset_, expr);
                break;
            }
            case DataType::FLOAT: {
                std::tie(child_res, index_res) = ExecBinaryRangeVisitorDispatcher<float>(child->field_offset_, expr);
                break;
            }
            case DataType::DOUBLE: {
                std::tie(child_res, index_res) = ExecBinaryRangeVisitorDispatcher<double>(child->field_offset_, expr);
                break;
            }
            default:
                PanicInfo("unsupported datatype");
        }
        // invalid case: lowerbound > upperbound
        if (child_res.is_scalar()) {
            ret_ = std::move(child_res);
            return;
        }
    } else {
        child_res = call_child(*expr.child_);
    }
    auto scalar1 = ExecGenericValueVisitor(*expr.lower_value_);
    auto scalar2 = ExecGenericValueVisitor(*expr.upper_value_);
    std::string op_name1 = lower_inclusive ? "greater_equal" : "greater";
    std::string op_name2 = upper_inclusive ? "less_equal" : "less";
    auto res1 = cp::CallFunction(op_name1, {child_res, scalar1}).ValueOrDie();
    auto res2 = cp::CallFunction(op_name2, {child_res, scalar2}).ValueOrDie();
    auto res = cp::And(res1, res2).ValueOrDie();
    if (index_res) {
        res = arrow::Concatenate({index_res, res.make_array()}).ValueOrDie();
    }
    AssertInfo(res.length() == row_count_, "[ExecExprVisitor]Size of results not equal row count");
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

void
ExecExprVisitor::visit(TermExpr& expr) {
    auto child_res = call_child(*expr.child_);
    auto terms = ExecGenericValueListVisitor(expr.terms_);
    if (terms.is_scalar()) {  // terms is empty
        ret_ = std::move(terms);
        return;
    }
    auto res = cp::IsIn(child_res, cp::SetLookupOptions(terms, true)).ValueOrDie();
    AssertInfo(res.length() == row_count_, "[ExecExprVisitor]Size of results not equal row count");
    ret_ = std::move(res);
}

template <typename T, typename Builder>
void
ExecExprVisitor::ExtractFieldData(const FieldOffset& offset, Builder& builder, int64_t chunk_offset) {
    auto size_per_chunk = segment_.size_per_chunk();
    auto num_chunk = upper_div(row_count_, size_per_chunk);
    for (int64_t chunk_id = chunk_offset; chunk_id < num_chunk; ++chunk_id) {
        auto size = chunk_id == num_chunk - 1 ? row_count_ - chunk_id * size_per_chunk : size_per_chunk;
        auto data = segment_.chunk_data<T>(offset, chunk_id).data();
        builder.AppendValues(data, size);
    }
}

auto
ExecExprVisitor::BuildFieldArray(const FieldOffset& offset, int64_t chunk_offset) -> RetType {
    auto data_type = segment_.get_schema()[offset].get_data_type();
    RetType res;
    switch (data_type) {
        case DataType::BOOL: {
            auto builder = arrow::BooleanBuilder();
            ExtractFieldData<uint8_t>(offset, builder, chunk_offset);
            res = builder.Finish().ValueOrDie();
            break;
        }
        case DataType::INT8: {
            auto builder = arrow::Int8Builder();
            ExtractFieldData<int8_t>(offset, builder, chunk_offset);
            res = builder.Finish().ValueOrDie();
            break;
        }
        case DataType::INT16: {
            auto builder = arrow::Int16Builder();
            ExtractFieldData<int16_t>(offset, builder, chunk_offset);
            res = builder.Finish().ValueOrDie();
            break;
        }
        case DataType::INT32: {
            auto builder = arrow::Int32Builder();
            ExtractFieldData<int32_t>(offset, builder, chunk_offset);
            res = builder.Finish().ValueOrDie();
            break;
        }
        case DataType::INT64: {
            auto builder = arrow::Int64Builder();
            ExtractFieldData<int64_t>(offset, builder, chunk_offset);
            res = builder.Finish().ValueOrDie();
            break;
        }
        case DataType::FLOAT: {
            auto builder = arrow::FloatBuilder();
            ExtractFieldData<float>(offset, builder, chunk_offset);
            res = builder.Finish().ValueOrDie();
            break;
        }
        case DataType::DOUBLE: {
            auto builder = arrow::DoubleBuilder();
            ExtractFieldData<double>(offset, builder, chunk_offset);
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
    AssertInfo(res.is_scalar() || res.length() == row_count_, "[ExecExprVisitor]Size of results not equal row count");
    ret_ = std::move(res);
}

void
ExecExprVisitor::visit(UnaryArithExpr& expr) {
    auto op = expr.op_type_;
    auto child_res = call_child(*expr.child_);
    static const std::map<UnaryArithOp, std::string> op_name = {
        {UnaryArithOp::Minus, "negate"},
        {UnaryArithOp::BitNot, "bit_wise_not"},
    };
    RetType res = cp::CallFunction(op_name.at(op), {child_res}).ValueOrDie();
    AssertInfo(res.is_scalar() || res.length() == row_count_, "[ExecExprVisitor]Size of results not equal row count");
    ret_ = std::move(res);
}

void
ExecExprVisitor::visit(BinaryArithExpr& expr) {
    auto op = expr.op_type_;
    auto left_res = call_child(*expr.left_);
    auto right_res = call_child(*expr.right_);
    // TODO: implement modulo
    static const std::map<BinaryArithOp, std::string> op_name = {
        {BinaryArithOp::Add, "add"},
        {BinaryArithOp::Subtract, "subtract"},
        {BinaryArithOp::Multiply, "multiply"},
        {BinaryArithOp::Divide, "divide"},
        {BinaryArithOp::Modulo, "modulo"},
        {BinaryArithOp::Power, "power"},
        {BinaryArithOp::BitAnd, "bit_wise_and"},
        {BinaryArithOp::BitOr, "bit_wise_or"},
        {BinaryArithOp::BitXor, "bit_wise_xor"},
        {BinaryArithOp::ShiftLeft, "shift_left"},
        {BinaryArithOp::BitXor, "shift_right"},
    };
    RetType res = cp::CallFunction(op_name.at(op), {left_res, right_res}).ValueOrDie();
    AssertInfo(res.is_scalar() || res.length() == row_count_, "[ExecExprVisitor]Size of results not equal row count");
    ret_ = std::move(res);
}

void
ExecExprVisitor::visit(ValueExpr& expr) {
    ret_ = ExecGenericValueVisitor(*expr.value_);
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
    AssertInfo(res.is_scalar() || res.length() == row_count_, "[ExecExprVisitor]Size of results not equal row count");
    ret_ = std::move(res);
}
}  // namespace milvus::query
