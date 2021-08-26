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
#include <memory>
#include <vector>
#include <any>
#include <string>
#include <optional>
#include <map>
#include "common/Schema.h"

namespace milvus::query {
class ExprVisitor;

enum class CompareOp {
    InvalidCompareOp = 0,
    GreaterThan = 1,
    GreaterEqual = 2,
    LessThan = 3,
    LessEqual = 4,
    Equal = 5,
    NotEqual = 6,
};

enum class ArithOp {
    InvalidArithOp = 0,
    Add = 1,
    Subtract = 2,
    Multiply = 3,
    Divide = 4,
    Modulo = 5,
    Power = 6,
    BitAnd = 7,
    BitOr = 8,
    BitXor = 9,
};

enum class UnaryLogicalOp {
    InvalidUnaryOp = 0,
    LogicalNot = 1,
};

enum class BinaryLogicalOp {
    InvalidBinaryOp = 0,
    LogicalAnd = 1,
    LogicalOr = 2,
    LogicalXor = 3,
};

// Base of all Exprs
struct Expr {
    DataType data_type_ = DataType::NONE;

 public:
    virtual ~Expr() = default;
    virtual void
    accept(ExprVisitor&) = 0;
};

using ExprPtr = std::unique_ptr<Expr>;

struct BinaryExprBase : Expr {
    ExprPtr left_;
    ExprPtr right_;
};

struct UnaryExprBase : Expr {
    ExprPtr child_;
};

struct ColumnExpr : Expr {
    FieldOffset field_offset_;

 public:
    void
    accept(ExprVisitor&) override;
};

struct ValueExpr : Expr {
 protected:
    // prevent accidential instantiation
    ValueExpr() = default;

 public:
    void
    accept(ExprVisitor&) override;
};

struct UnaryLogicalExpr : UnaryExprBase {
    UnaryLogicalOp op_type_;

 public:
    void
    accept(ExprVisitor&) override;
};

struct BinaryLogicalExpr : BinaryExprBase {
    BinaryLogicalOp op_type_;

 public:
    void
    accept(ExprVisitor&) override;
};

struct TermExpr : UnaryExprBase {
 protected:
    // prevent accidential instantiation
    TermExpr() = default;

 public:
    void
    accept(ExprVisitor&) override;
};

// deprecated
static const std::map<std::string, CompareOp> mapping_ = {
    // op_name -> op
    {"lt", CompareOp::LessThan},    {"le", CompareOp::LessEqual},    {"lte", CompareOp::LessEqual},
    {"gt", CompareOp::GreaterThan}, {"ge", CompareOp::GreaterEqual}, {"gte", CompareOp::GreaterEqual},
    {"eq", CompareOp::Equal},       {"ne", CompareOp::NotEqual},
};

struct UnaryRangeExpr : UnaryExprBase {
    CompareOp op_type_;

 protected:
    // prevent accidential instantiation
    UnaryRangeExpr() = default;

 public:
    void
    accept(ExprVisitor&) override;
};

struct BinaryRangeExpr : UnaryExprBase {
    bool lower_inclusive_;
    bool upper_inclusive_;

 protected:
    // prevent accidential instantiation
    BinaryRangeExpr() = default;

 public:
    void
    accept(ExprVisitor&) override;
};

struct CompareExpr : BinaryExprBase {
    CompareOp op_type_;

 public:
    void
    accept(ExprVisitor&) override;
};

struct ArithExpr : BinaryExprBase {
    ArithOp op_type_;

 public:
    void
    accept(ExprVisitor&) override;
};
}  // namespace milvus::query
