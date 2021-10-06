// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package proxy

import (
	"fmt"

	"strconv"

	"github.com/antlr/antlr4/runtime/Go/antlr"
	"github.com/milvus-io/milvus/internal/proto/planpb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	parser "github.com/milvus-io/milvus/internal/proxy/plan_parser"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

func parseExpr(schema *typeutil.SchemaHelper, exprStr string) (*planpb.Expr, error) {
	if exprStr == "" {
		return nil, nil
	}

	inputStream := antlr.NewInputStream(exprStr)
	lexer := parser.NewPlanLexer(inputStream)
	stream := antlr.NewCommonTokenStream(lexer, antlr.TokenDefaultChannel)
	parse := parser.NewPlanParser(stream)
	parse.BuildParseTrees = true

	errorListener := &errorListener{}
	parse.AddErrorListener(errorListener)
	ast := parse.Expr()
	if errorListener.err != nil {
		return nil, errorListener.err
	}

	visitor := &Visitor{schema: schema}
	ret := ast.Accept(visitor)

	if err := getError(ret); err != nil {
		return nil, err
	}
	if n := getNumber(ret); n != nil {
		return nil, fmt.Errorf("predicate is a constant expression")
	}
	predicate := ret.(*ExprWithType)
	if !typeutil.IsBooleanType(predicate.dataType) {
		return nil, fmt.Errorf("predicate is not a boolean expression")
	}
	return predicate.expr, nil
}

func CreateExprPlan(schemaPb *schemapb.CollectionSchema, exprStr string) (*planpb.PlanNode, error) {
	schema, err := typeutil.CreateSchemaHelper(schemaPb)
	if err != nil {
		return nil, err
	}

	expr, err := parseExpr(schema, exprStr)
	if err != nil {
		return nil, err
	}

	planNode := &planpb.PlanNode{
		Node: &planpb.PlanNode_Predicates{
			Predicates: expr,
		},
	}
	return planNode, nil
}

func CreateQueryPlan(schemaPb *schemapb.CollectionSchema, exprStr string, vectorFieldName string, queryInfo *planpb.QueryInfo) (*planpb.PlanNode, error) {
	schema, err := typeutil.CreateSchemaHelper(schemaPb)
	if err != nil {
		return nil, err
	}

	expr, err := parseExpr(schema, exprStr)
	if err != nil {
		return nil, err
	}
	vectorField, err := schema.GetFieldFromName(vectorFieldName)
	if err != nil {
		return nil, err
	}
	fieldID := vectorField.FieldID
	dataType := vectorField.DataType

	if !typeutil.IsVectorType(dataType) {
		return nil, fmt.Errorf("field (%s) to search is not of vector data type", vectorFieldName)
	}

	planNode := &planpb.PlanNode{
		Node: &planpb.PlanNode_VectorAnns{
			VectorAnns: &planpb.VectorANNS{
				IsBinary:       dataType == schemapb.DataType_BinaryVector,
				Predicates:     expr,
				QueryInfo:      queryInfo,
				PlaceholderTag: "$0",
				FieldId:        fieldID,
			},
		},
	}
	return planNode, nil
}

// utils

func getSameType(a, b schemapb.DataType) schemapb.DataType {
	if a == schemapb.DataType_Double || b == schemapb.DataType_Double {
		return schemapb.DataType_Double
	}
	if a == schemapb.DataType_Int64 || b == schemapb.DataType_Int64 {
		return schemapb.DataType_Int64
	}
	if a == schemapb.DataType_Bool || b == schemapb.DataType_Bool {
		return schemapb.DataType_Bool
	}
	panic("can not get same type")
}

func calcDataType(left, right *ExprWithType) schemapb.DataType {
	dataType := getSameType(left.dataType, right.dataType)
	return dataType
}

func isSameOrder(op1, op2 planpb.CompareOp) bool {
	isLess1 := op1 == planpb.CompareOp_LessThan || op2 == planpb.CompareOp_LessEqual
	isLess2 := op2 == planpb.CompareOp_LessThan || op2 == planpb.CompareOp_LessEqual
	return isLess1 == isLess2
}

func reverseOrder(op planpb.CompareOp) planpb.CompareOp {
	switch op {
	case planpb.CompareOp_LessThan:
		return planpb.CompareOp_GreaterThan
	case planpb.CompareOp_LessEqual:
		return planpb.CompareOp_GreaterEqual
	case planpb.CompareOp_GreaterThan:
		return planpb.CompareOp_LessThan
	case planpb.CompareOp_GreaterEqual:
		return planpb.CompareOp_LessEqual
	case planpb.CompareOp_Equal:
		return planpb.CompareOp_Equal
	case planpb.CompareOp_NotEqual:
		return planpb.CompareOp_NotEqual
	default:
		panic("cannot reverse order")
	}
}

func toGenericValue(n *number) (value *planpb.GenericValue) {
	switch n.numType {
	case Bool:
		value = &planpb.GenericValue{
			Val: &planpb.GenericValue_BoolVal{
				BoolVal: n.Bool(),
			},
		}
	case Int:
		value = &planpb.GenericValue{
			Val: &planpb.GenericValue_Int64Val{
				Int64Val: n.Int(),
			},
		}
	case Float:
		value = &planpb.GenericValue{
			Val: &planpb.GenericValue_FloatVal{
				FloatVal: n.Float(),
			},
		}
	default:
		panic("invalid constant")
	}
	return
}

func toValueExpr(n *number) *ExprWithType {
	value := toGenericValue(n)
	expr := &planpb.Expr{
		Expr: &planpb.Expr_ValueExpr{
			ValueExpr: &planpb.ValueExpr{
				Value: value,
			},
		},
	}
	switch n.numType {
	case Bool:
		return &ExprWithType{
			expr:     expr,
			dataType: schemapb.DataType_Bool,
		}
	case Int:
		return &ExprWithType{
			expr:     expr,
			dataType: schemapb.DataType_Int64,
		}
	case Float:
		return &ExprWithType{
			expr:     expr,
			dataType: schemapb.DataType_Double,
		}
	default:
		panic("invalid constant")
	}
}

// errorListener

type errorListener struct {
	*antlr.DefaultErrorListener
	err error
}

func (l *errorListener) SyntaxError(recognizer antlr.Recognizer, offendingSymbol interface{}, line, column int, msg string, e antlr.RecognitionException) {
	l.err = fmt.Errorf("line " + strconv.Itoa(line) + ":" + strconv.Itoa(column) + " " + msg)
}

// ExprWithType

type ExprWithType struct {
	expr     *planpb.Expr
	dataType schemapb.DataType
}

// Visitor

type Visitor struct {
	parser.BasePlanVisitor
	schema *typeutil.SchemaHelper
}

func getError(obj interface{}) error {
	err, _ := obj.(error)
	return err
}

func getNumber(obj interface{}) *number {
	n, _ := obj.(*number)
	return n
}

func getExpr(obj interface{}) *ExprWithType {
	n, _ := obj.(*ExprWithType)
	return n
}

func (v *Visitor) VisitBoolean(ctx *parser.BooleanContext) interface{} {
	literal := ctx.BooleanConstant().GetText()
	b, err := strconv.ParseBool(literal)
	if err != nil {
		return err
	}
	return NewBool(b)
}

func (v *Visitor) VisitFloating(ctx *parser.FloatingContext) interface{} {
	literal := ctx.FloatingConstant().GetText()
	f, err := strconv.ParseFloat(literal, 64)
	if err != nil {
		return err
	}
	return NewFloat(f)
}

func (v *Visitor) VisitInteger(ctx *parser.IntegerContext) interface{} {
	literal := ctx.IntegerConstant().GetText()
	i, err := strconv.ParseInt(literal, 0, 64)
	if err != nil {
		return err
	}
	return NewInt(i)
}

func (v *Visitor) VisitString(ctx *parser.StringContext) interface{} {
	literal := ctx.StringLiteral().GetText()
	return fmt.Errorf("string is not supported yet: %s", literal)
}

func (v *Visitor) VisitIdentifier(ctx *parser.IdentifierContext) interface{} {
	fieldName := ctx.Identifier().GetText()
	field, err := v.schema.GetFieldFromName(fieldName)
	if err != nil {
		return err
	}
	expr := &planpb.Expr{
		Expr: &planpb.Expr_ColumnExpr{
			ColumnExpr: &planpb.ColumnExpr{
				ColumnInfo: &planpb.ColumnInfo{
					FieldId:      field.FieldID,
					DataType:     field.DataType,
					IsPrimaryKey: field.IsPrimaryKey,
				},
			},
		},
	}

	// TODO: optimize the space usage of the columns
	switch field.DataType {
	case schemapb.DataType_Bool:
		return &ExprWithType{
			expr:     expr,
			dataType: field.DataType,
		}
	case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32:
		expr = &planpb.Expr{
			Expr: &planpb.Expr_CastExpr{
				CastExpr: &planpb.CastExpr{
					Child:    expr,
					DataType: schemapb.DataType_Int64,
				},
			},
		}
		return &ExprWithType{
			expr:     expr,
			dataType: schemapb.DataType_Int64,
		}
	case schemapb.DataType_Int64:
		return &ExprWithType{
			expr:     expr,
			dataType: schemapb.DataType_Int64,
		}
	case schemapb.DataType_Float:
		expr = &planpb.Expr{
			Expr: &planpb.Expr_CastExpr{
				CastExpr: &planpb.CastExpr{
					Child:    expr,
					DataType: schemapb.DataType_Double,
				},
			},
		}
		return &ExprWithType{
			expr:     expr,
			dataType: schemapb.DataType_Double,
		}
	case schemapb.DataType_Double:
		return &ExprWithType{
			expr:     expr,
			dataType: schemapb.DataType_Double,
		}
	default:
		return fmt.Errorf("data type of field '%s' must be scalar", fieldName)
	}
}

func (v *Visitor) VisitParens(ctx *parser.ParensContext) interface{} {
	return ctx.Expr().Accept(v)
}

func (v *Visitor) VisitPower(ctx *parser.PowerContext) interface{} {
	left := ctx.Expr(0).Accept(v)
	if getError(left) != nil {
		return left
	}
	right := ctx.Expr(1).Accept(v)
	if getError(right) != nil {
		return right
	}

	leftNumber, rightNumber := getNumber(left), getNumber(right)
	if leftNumber != nil && rightNumber != nil {
		return Power(leftNumber, rightNumber)
	}

	var leftExpr *ExprWithType
	var rightExpr *ExprWithType
	if leftNumber != nil {
		leftExpr = toValueExpr(leftNumber)
	} else {
		leftExpr = getExpr(left)
	}
	if rightNumber != nil {
		rightExpr = toValueExpr(rightNumber)
	} else {
		rightExpr = getExpr(right)
	}
	if !typeutil.IsNumberType(leftExpr.dataType) || !typeutil.IsNumberType(rightExpr.dataType) {
		return fmt.Errorf("'power' can only be used between integer or floating expressions")
	}
	expr := &planpb.Expr{
		Expr: &planpb.Expr_BinaryArithExpr{
			BinaryArithExpr: &planpb.BinaryArithExpr{
				Left:  leftExpr.expr,
				Right: rightExpr.expr,
				Op:    planpb.BinaryArithOp_Power,
			},
		},
	}
	dataType := calcDataType(leftExpr, rightExpr)
	return &ExprWithType{
		expr:     expr,
		dataType: dataType,
	}
}

func (v *Visitor) VisitUnary(ctx *parser.UnaryContext) interface{} {
	child := ctx.Expr().Accept(v)
	if getError(child) != nil {
		return child
	}

	childNumber := getNumber(child)
	if childNumber != nil {
		switch ctx.GetOp().GetTokenType() {
		case parser.PlanParserADD:
			return childNumber
		case parser.PlanParserSUB:
			return Negative(childNumber)
		case parser.PlanParserBNOT:
			n, err := BitNot(childNumber)
			if err != nil {
				return err
			}
			return n
		case parser.PlanParserNOT:
			return Not(childNumber)
		default:
			return fmt.Errorf("unexpected op: %s", ctx.GetOp().GetText())
		}
	}

	childExpr := getExpr(child)
	switch ctx.GetOp().GetTokenType() {
	case parser.PlanParserADD:
		return childExpr
	case parser.PlanParserSUB:
		expr := &planpb.Expr{
			Expr: &planpb.Expr_UnaryArithExpr{
				UnaryArithExpr: &planpb.UnaryArithExpr{
					Child: childExpr.expr,
					Op:    planpb.UnaryArithOp_Minus,
				},
			},
		}
		return &ExprWithType{
			expr:     expr,
			dataType: childExpr.dataType,
		}
	case parser.PlanParserBNOT:
		if !typeutil.IsIntegerType(childExpr.dataType) {
			return fmt.Errorf("'bitnot' can only be used on integer expression")
		}
		expr := &planpb.Expr{
			Expr: &planpb.Expr_UnaryArithExpr{
				UnaryArithExpr: &planpb.UnaryArithExpr{
					Child: childExpr.expr,
					Op:    planpb.UnaryArithOp_BitNot,
				},
			},
		}
		return &ExprWithType{
			expr:     expr,
			dataType: childExpr.dataType,
		}
	case parser.PlanParserNOT:
		if !typeutil.IsBooleanType(childExpr.dataType) {
			return fmt.Errorf("'not' can only be used on boolean expression")
		}
		expr := &planpb.Expr{
			Expr: &planpb.Expr_UnaryLogicalExpr{
				UnaryLogicalExpr: &planpb.UnaryLogicalExpr{
					Child: childExpr.expr,
					Op:    planpb.UnaryLogicalOp_Not,
				},
			},
		}
		return &ExprWithType{
			expr:     expr,
			dataType: schemapb.DataType_Bool,
		}
	default:
		return fmt.Errorf("unexpected op: %s", ctx.GetOp().GetText())
	}
}

func (v *Visitor) VisitMulDivMod(ctx *parser.MulDivModContext) interface{} {
	left := ctx.Expr(0).Accept(v)
	if getError(left) != nil {
		return left
	}
	right := ctx.Expr(1).Accept(v)
	if getError(right) != nil {
		return right
	}

	leftNumber, rightNumber := getNumber(left), getNumber(right)
	if leftNumber != nil && rightNumber != nil {
		switch ctx.GetOp().GetTokenType() {
		case parser.PlanParserMUL:
			return Multiply(leftNumber, rightNumber)
		case parser.PlanParserDIV:
			n, err := Divide(leftNumber, rightNumber)
			if err != nil {
				return err
			}
			return n
		case parser.PlanParserMOD:
			n, err := Modulo(leftNumber, rightNumber)
			if err != nil {
				return err
			}
			return n
		default:
			return fmt.Errorf("unexpected op: %s", ctx.GetOp().GetText())
		}
	}

	var leftExpr *ExprWithType
	var rightExpr *ExprWithType
	if leftNumber != nil {
		leftExpr = toValueExpr(leftNumber)
	} else {
		leftExpr = getExpr(left)
	}
	if rightNumber != nil {
		rightExpr = toValueExpr(rightNumber)
	} else {
		rightExpr = getExpr(right)
	}
	switch ctx.GetOp().GetTokenType() {
	case parser.PlanParserMUL:
		if !typeutil.IsNumberType(leftExpr.dataType) || !typeutil.IsNumberType(rightExpr.dataType) {
			return fmt.Errorf("'multiply' can only be used between integer or floating expressions")
		}
		expr := &planpb.Expr{
			Expr: &planpb.Expr_BinaryArithExpr{
				BinaryArithExpr: &planpb.BinaryArithExpr{
					Left:  leftExpr.expr,
					Right: rightExpr.expr,
					Op:    planpb.BinaryArithOp_Multiply,
				},
			},
		}
		dataType := calcDataType(leftExpr, rightExpr)
		return &ExprWithType{
			expr:     expr,
			dataType: dataType,
		}
	case parser.PlanParserDIV:
		if !typeutil.IsNumberType(leftExpr.dataType) || !typeutil.IsNumberType(rightExpr.dataType) {
			return fmt.Errorf("'divide' can only be used between integer or floating expressions")
		}
		if !typeutil.IsFloatingType(leftExpr.dataType) && rightNumber != nil && Equal(rightNumber, NewInt(0)).Bool() {
			return fmt.Errorf("divisor cannot be 0")
		}
		expr := &planpb.Expr{
			Expr: &planpb.Expr_BinaryArithExpr{
				BinaryArithExpr: &planpb.BinaryArithExpr{
					Left:  leftExpr.expr,
					Right: rightExpr.expr,
					Op:    planpb.BinaryArithOp_Divide,
				},
			},
		}
		dataType := calcDataType(leftExpr, rightExpr)
		return &ExprWithType{
			expr:     expr,
			dataType: dataType,
		}
	case parser.PlanParserMOD:
		if !typeutil.IsIntegerType(leftExpr.dataType) || !typeutil.IsIntegerType(rightExpr.dataType) {
			return fmt.Errorf("'modulo' can only be used between integer expressions")
		}
		if !typeutil.IsFloatingType(leftExpr.dataType) && rightNumber != nil && Equal(rightNumber, NewInt(0)).Bool() {
			return fmt.Errorf("modulo cannot be 0")
		}
		return fmt.Errorf("modulo between non-const expressions not supported yet")
		// expr := &planpb.Expr{
		// 	Expr: &planpb.Expr_BinaryArithExpr{
		// 		BinaryArithExpr: &planpb.BinaryArithExpr{
		// 			Left:  leftExpr.expr,
		// 			Right: rightExpr.expr,
		// 			Op:    planpb.BinaryArithOp_Divide,
		// 		},
		// 	},
		// }
		// dataType := calcDataType(leftExpr, rightExpr)
		// return &ExprWithType{
		// 	expr:     expr,
		// 	dataType: dataType,
		// }
	default:
		return fmt.Errorf("unexpected op: %s", ctx.GetOp().GetText())
	}
}

var addSubExprMap = map[int]planpb.BinaryArithOp{
	parser.PlanParserADD: planpb.BinaryArithOp_Add,
	parser.PlanParserSUB: planpb.BinaryArithOp_Subtract,
}

var addSubNameMap = map[int]string{
	parser.PlanParserADD: "add",
	parser.PlanParserSUB: "subtract",
}

func (v *Visitor) VisitAddSub(ctx *parser.AddSubContext) interface{} {
	left := ctx.Expr(0).Accept(v)
	if getError(left) != nil {
		return left
	}
	right := ctx.Expr(1).Accept(v)
	if getError(right) != nil {
		return right
	}

	leftNumber, rightNumber := getNumber(left), getNumber(right)
	if leftNumber != nil && rightNumber != nil {
		switch ctx.GetOp().GetTokenType() {
		case parser.PlanParserADD:
			return Add(leftNumber, rightNumber)
		case parser.PlanParserSUB:
			return Subtract(leftNumber, rightNumber)
		default:
			return fmt.Errorf("unexpected op: %s", ctx.GetOp().GetText())
		}
	}

	var leftExpr *ExprWithType
	var rightExpr *ExprWithType
	if leftNumber != nil {
		leftExpr = toValueExpr(leftNumber)
	} else {
		leftExpr = getExpr(left)
	}
	if rightNumber != nil {
		rightExpr = toValueExpr(rightNumber)
	} else {
		rightExpr = getExpr(right)
	}
	if !typeutil.IsNumberType(leftExpr.dataType) || !typeutil.IsNumberType(rightExpr.dataType) {
		return fmt.Errorf("'%s' can only be used between integer or floating expressions", addSubNameMap[ctx.GetOp().GetTokenType()])
	}
	expr := &planpb.Expr{
		Expr: &planpb.Expr_BinaryArithExpr{
			BinaryArithExpr: &planpb.BinaryArithExpr{
				Left:  leftExpr.expr,
				Right: rightExpr.expr,
				Op:    addSubExprMap[ctx.GetOp().GetTokenType()],
			},
		},
	}
	dataType := calcDataType(leftExpr, rightExpr)
	return &ExprWithType{
		expr:     expr,
		dataType: dataType,
	}
}

var shiftExprMap = map[int]planpb.BinaryArithOp{
	parser.PlanParserSHL: planpb.BinaryArithOp_ShiftLeft,
	parser.PlanParserSHR: planpb.BinaryArithOp_ShiftRight,
}

var shiftNameMap = map[int]string{
	parser.PlanParserSHL: "shiftleft",
	parser.PlanParserSHR: "shiftright",
}

func (v *Visitor) VisitShift(ctx *parser.ShiftContext) interface{} {
	left := ctx.Expr(0).Accept(v)
	if getError(left) != nil {
		return left
	}
	right := ctx.Expr(1).Accept(v)
	if getError(right) != nil {
		return right
	}

	leftNumber, rightNumber := getNumber(left), getNumber(right)
	if leftNumber != nil && rightNumber != nil {
		switch ctx.GetOp().GetTokenType() {
		case parser.PlanParserSHL:
			n, err := ShiftLeft(leftNumber, rightNumber)
			if err != nil {
				return err
			}
			return n
		case parser.PlanParserSHR:
			n, err := ShiftRight(leftNumber, rightNumber)
			if err != nil {
				return err
			}
			return n
		}

	}

	var leftExpr *ExprWithType
	var rightExpr *ExprWithType
	if leftNumber != nil {
		leftExpr = toValueExpr(leftNumber)
	} else {
		leftExpr = getExpr(left)
	}
	if rightNumber != nil {
		rightExpr = toValueExpr(rightNumber)
	} else {
		rightExpr = getExpr(right)
	}
	if !typeutil.IsIntegerType(leftExpr.dataType) || !typeutil.IsIntegerType(rightExpr.dataType) {
		return fmt.Errorf("'%s' can only be used between integer expressions", shiftNameMap[ctx.GetOp().GetTokenType()])
	}
	expr := &planpb.Expr{
		Expr: &planpb.Expr_BinaryArithExpr{
			BinaryArithExpr: &planpb.BinaryArithExpr{
				Left:  leftExpr.expr,
				Right: rightExpr.expr,
				Op:    shiftExprMap[ctx.GetOp().GetTokenType()],
			},
		},
	}
	return &ExprWithType{
		expr:     expr,
		dataType: leftExpr.dataType,
	}
}

func (v *Visitor) VisitTerm(ctx *parser.TermContext) interface{} {
	child := ctx.Expr(0).Accept(v)
	if getError(child) != nil {
		return child
	}

	childNumber := getNumber(child)
	if childNumber != nil {
		return fmt.Errorf("'term' can only be used on non-const expression")
	}

	childExpr := getExpr(child)
	if !typeutil.IsNumberType(childExpr.dataType) {
		return fmt.Errorf("'term' can only be used on integer or floating expression")
	}
	var values []*planpb.GenericValue
	for i := 1; i < len(ctx.AllExpr()); i++ {
		term := ctx.Expr(i).Accept(v)
		if getError(term) != nil {
			return term
		}
		n := getNumber(term)
		if n == nil {
			return fmt.Errorf("value '%s' in list cannot be a non-const expression", ctx.Expr(i).GetText())
		}
		if n.IsBool() {
			return fmt.Errorf("value '%s' in list cannot be a boolean expression", ctx.Expr(i).GetText())
		}
		if n.IsFloat() && typeutil.IsIntegerType(childExpr.dataType) {
			n = NewInt(n.Int())
		}
		if n.IsInt() && typeutil.IsFloatingType(childExpr.dataType) {
			n = NewFloat(n.Float())
		}
		value := toGenericValue(n)
		values = append(values, value)
	}
	if len(values) <= 0 {
		return fmt.Errorf("'term' has empty value list")
	}
	expr := &planpb.Expr{
		Expr: &planpb.Expr_TermExpr{
			TermExpr: &planpb.TermExpr{
				Child:  childExpr.expr,
				Values: values,
			},
		},
	}
	if ctx.GetOp().GetTokenType() == parser.PlanParserNIN {
		expr = &planpb.Expr{
			Expr: &planpb.Expr_UnaryLogicalExpr{
				UnaryLogicalExpr: &planpb.UnaryLogicalExpr{
					Child: expr,
					Op:    planpb.UnaryLogicalOp_Not,
				},
			},
		}
	}
	return &ExprWithType{
		expr:     expr,
		dataType: schemapb.DataType_Bool,
	}
}

var cmpOpMap = map[int]planpb.CompareOp{
	parser.PlanParserLT: planpb.CompareOp_LessThan,
	parser.PlanParserLE: planpb.CompareOp_LessEqual,
	parser.PlanParserGT: planpb.CompareOp_GreaterThan,
	parser.PlanParserGE: planpb.CompareOp_GreaterEqual,
	parser.PlanParserEQ: planpb.CompareOp_Equal,
	parser.PlanParserNE: planpb.CompareOp_NotEqual,
}

var cmpNameMap = map[int]string{
	parser.PlanParserLT: "less",
	parser.PlanParserLE: "lessequal",
	parser.PlanParserGT: "greater",
	parser.PlanParserGE: "greatequal",
	parser.PlanParserEQ: "equal",
	parser.PlanParserNE: "notequal",
}

func (v *Visitor) VisitRange(ctx *parser.RangeContext) interface{} {
	child := ctx.Expr(1).Accept(v)
	if getError(child) != nil {
		return child
	}

	childNumber := getNumber(child)
	if childNumber != nil {
		return fmt.Errorf("'range' can only be used on non-const expression")
	}

	childExpr := getExpr(child)
	if !typeutil.IsNumberType(childExpr.dataType) {
		return fmt.Errorf("'range' can only be used on integer or floating expression")
	}

	lower := ctx.Expr(0).Accept(v)
	upper := ctx.Expr(2).Accept(v)
	if getError(lower) != nil {
		return lower
	}
	if getError(upper) != nil {
		return upper
	}
	lowerNumber := getNumber(lower)
	upperNumber := getNumber(upper)
	if lowerNumber == nil {
		return fmt.Errorf("lowerbound '%s' cannot be a non-const expression", ctx.Expr(0).GetText())
	}
	if lowerNumber.IsBool() {
		return fmt.Errorf("lowerbound '%s' cannot be a boolean expression", ctx.Expr(0).GetText())
	}
	if lowerNumber.IsFloat() && typeutil.IsIntegerType(childExpr.dataType) {
		lowerNumber = NewInt(lowerNumber.Int())
	}
	if lowerNumber.IsInt() && typeutil.IsFloatingType(childExpr.dataType) {
		lowerNumber = NewFloat(lowerNumber.Float())
	}
	if upperNumber == nil {
		return fmt.Errorf("upperbound '%s' cannot be a non-const expression", ctx.Expr(1).GetText())
	}
	if upperNumber.IsBool() {
		return fmt.Errorf("upperbound '%s' cannot be a boolean expression", ctx.Expr(1).GetText())
	}
	if upperNumber.IsFloat() && typeutil.IsIntegerType(childExpr.dataType) {
		upperNumber = NewInt(upperNumber.Int())
	}
	if upperNumber.IsInt() && typeutil.IsFloatingType(childExpr.dataType) {
		upperNumber = NewFloat(upperNumber.Float())
	}
	lowerInclusive := (ctx.GetOp1().GetTokenType() == parser.PlanParserLE)
	upperInclusive := (ctx.GetOp2().GetTokenType() == parser.PlanParserLE)
	if !(lowerInclusive && upperInclusive) {
		if GreaterEqual(lowerNumber, upperNumber).Bool() {
			return fmt.Errorf("invalid range: lowerbound is greater than upperbound")
		}
	} else {
		if Greater(lowerNumber, upperNumber).Bool() {
			return fmt.Errorf("invalid range: lowerbound is greater than upperbound")
		}
	}
	lowerValue := toGenericValue(lowerNumber)
	upperValue := toGenericValue(upperNumber)

	expr := &planpb.Expr{
		Expr: &planpb.Expr_BinaryRangeExpr{
			BinaryRangeExpr: &planpb.BinaryRangeExpr{
				Child:          childExpr.expr,
				LowerInclusive: lowerInclusive,
				UpperInclusive: upperInclusive,
				LowerValue:     lowerValue,
				UpperValue:     upperValue,
			},
		},
	}
	return &ExprWithType{
		expr:     expr,
		dataType: schemapb.DataType_Bool,
	}
}

func HandleCompare(op int, left, right *ExprWithType) (*planpb.Expr, error) {
	// handle CompareExpr and UnaryRangeExpr
	cmpOp := cmpOpMap[op]
	if valueExpr := left.expr.GetValueExpr(); valueExpr != nil {
		expr := &planpb.Expr{
			Expr: &planpb.Expr_UnaryRangeExpr{
				UnaryRangeExpr: &planpb.UnaryRangeExpr{
					Child: right.expr,
					Op:    reverseOrder(cmpOp),
					Value: valueExpr.Value,
				},
			},
		}
		return expr, nil
	} else if valueExpr := right.expr.GetValueExpr(); valueExpr != nil {
		expr := &planpb.Expr{
			Expr: &planpb.Expr_UnaryRangeExpr{
				UnaryRangeExpr: &planpb.UnaryRangeExpr{
					Child: left.expr,
					Op:    cmpOp,
					Value: valueExpr.Value,
				},
			},
		}
		return expr, nil
	} else {
		calcDataType(left, right)
		expr := &planpb.Expr{
			Expr: &planpb.Expr_CompareExpr{
				CompareExpr: &planpb.CompareExpr{
					Left:  left.expr,
					Right: right.expr,
					Op:    cmpOp,
				},
			},
		}
		return expr, nil
	}
}

func (v *Visitor) VisitRelational(ctx *parser.RelationalContext) interface{} {
	left := ctx.Expr(0).Accept(v)
	if getError(left) != nil {
		return left
	}
	right := ctx.Expr(1).Accept(v)
	if getError(right) != nil {
		return right
	}

	leftNumber, rightNumber := getNumber(left), getNumber(right)
	if leftNumber != nil && rightNumber != nil {
		switch ctx.GetOp().GetTokenType() {
		case parser.PlanParserLT:
			return Less(leftNumber, rightNumber)
		case parser.PlanParserLE:
			return LessEqual(leftNumber, rightNumber)
		case parser.PlanParserGT:
			return Greater(leftNumber, rightNumber)
		case parser.PlanParserGE:
			return GreaterEqual(leftNumber, rightNumber)
		default:
			return fmt.Errorf("unexpected op: %s", ctx.GetOp().GetText())
		}
	}

	var leftExpr *ExprWithType
	var rightExpr *ExprWithType
	if leftNumber != nil {
		leftExpr = toValueExpr(leftNumber)
	} else {
		leftExpr = getExpr(left)
	}
	if rightNumber != nil {
		rightExpr = toValueExpr(rightNumber)
	} else {
		rightExpr = getExpr(right)
	}
	if !typeutil.IsNumberType(leftExpr.dataType) || !typeutil.IsNumberType(rightExpr.dataType) {
		return fmt.Errorf("'%s' can only be used between integer or floating expressions", cmpNameMap[ctx.GetOp().GetTokenType()])
	}
	expr, err := HandleCompare(ctx.GetOp().GetTokenType(), leftExpr, rightExpr)
	if err != nil {
		return err
	}
	return &ExprWithType{
		expr:     expr,
		dataType: schemapb.DataType_Bool,
	}
}

func (v *Visitor) VisitEquality(ctx *parser.EqualityContext) interface{} {
	left := ctx.Expr(0).Accept(v)
	if getError(left) != nil {
		return left
	}
	right := ctx.Expr(1).Accept(v)
	if getError(right) != nil {
		return right
	}

	leftNumber, rightNumber := getNumber(left), getNumber(right)
	if leftNumber != nil && rightNumber != nil {
		switch ctx.GetOp().GetTokenType() {
		case parser.PlanParserEQ:
			return Equal(leftNumber, rightNumber)
		case parser.PlanParserNE:
			return NotEqual(leftNumber, rightNumber)
		default:
			return fmt.Errorf("unexpected op: %s", ctx.GetOp().GetText())
		}
	}

	var leftExpr *ExprWithType
	var rightExpr *ExprWithType
	if leftNumber != nil {
		rightExpr = getExpr(right)
		if typeutil.IsBooleanType(rightExpr.dataType) && leftNumber.IsInt() {
			if Equal(leftNumber, NewInt(0)).Bool() || Equal(leftNumber, NewInt(1)).Bool() {
				leftNumber = NewBool(leftNumber.Bool())
			}
		}
		leftExpr = toValueExpr(leftNumber)
	} else if rightNumber != nil {
		leftExpr = getExpr(left)
		if typeutil.IsBooleanType(leftExpr.dataType) && rightNumber.IsInt() {
			if Equal(rightNumber, NewInt(0)).Bool() || Equal(rightNumber, NewInt(1)).Bool() {
				rightNumber = NewBool(rightNumber.Bool())
			}
		}
		rightExpr = toValueExpr(rightNumber)
	} else {
		leftExpr = getExpr(left)
		rightExpr = getExpr(right)
	}
	if typeutil.IsBooleanType(leftExpr.dataType) != typeutil.IsBooleanType(rightExpr.dataType) {
		return fmt.Errorf("operands of '%s' must be boolean expressions or not at same time", cmpNameMap[ctx.GetOp().GetTokenType()])
	}
	expr, err := HandleCompare(ctx.GetOp().GetTokenType(), leftExpr, rightExpr)
	if err != nil {
		return err
	}
	return &ExprWithType{
		expr:     expr,
		dataType: schemapb.DataType_Bool,
	}
}

func (v *Visitor) VisitBitAnd(ctx *parser.BitAndContext) interface{} {
	left := ctx.Expr(0).Accept(v)
	if getError(left) != nil {
		return left
	}
	right := ctx.Expr(1).Accept(v)
	if getError(right) != nil {
		return right
	}

	leftNumber, rightNumber := getNumber(left), getNumber(right)
	if leftNumber != nil && rightNumber != nil {
		n, err := BitAnd(leftNumber, rightNumber)
		if err != nil {
			return err
		}
		return n
	}

	var leftExpr *ExprWithType
	var rightExpr *ExprWithType
	if leftNumber != nil {
		leftExpr = toValueExpr(leftNumber)
	} else {
		leftExpr = getExpr(left)
	}
	if rightNumber != nil {
		rightExpr = toValueExpr(rightNumber)
	} else {
		rightExpr = getExpr(right)
	}
	if !typeutil.IsIntegerType(leftExpr.dataType) || !typeutil.IsIntegerType(rightExpr.dataType) {
		return fmt.Errorf("'bitand' can only be used between integer expressions")
	}
	expr := &planpb.Expr{
		Expr: &planpb.Expr_BinaryArithExpr{
			BinaryArithExpr: &planpb.BinaryArithExpr{
				Left:  leftExpr.expr,
				Right: rightExpr.expr,
				Op:    planpb.BinaryArithOp_BitAnd,
			},
		},
	}
	dataType := calcDataType(leftExpr, rightExpr)
	return &ExprWithType{
		expr:     expr,
		dataType: dataType,
	}
}

func (v *Visitor) VisitBitXor(ctx *parser.BitXorContext) interface{} {
	left := ctx.Expr(0).Accept(v)
	if getError(left) != nil {
		return left
	}
	right := ctx.Expr(1).Accept(v)
	if getError(right) != nil {
		return right
	}

	leftNumber, rightNumber := getNumber(left), getNumber(right)
	if leftNumber != nil && rightNumber != nil {
		n, err := BitXor(leftNumber, rightNumber)
		if err != nil {
			return err
		}
		return n
	}

	var leftExpr *ExprWithType
	var rightExpr *ExprWithType
	if leftNumber != nil {
		leftExpr = toValueExpr(leftNumber)
	} else {
		leftExpr = getExpr(left)
	}
	if rightNumber != nil {
		rightExpr = toValueExpr(rightNumber)
	} else {
		rightExpr = getExpr(right)
	}
	if !typeutil.IsIntegerType(leftExpr.dataType) || !typeutil.IsIntegerType(rightExpr.dataType) {
		return fmt.Errorf("'bitxor' can only be used between integer expressions")
	}
	expr := &planpb.Expr{
		Expr: &planpb.Expr_BinaryArithExpr{
			BinaryArithExpr: &planpb.BinaryArithExpr{
				Left:  leftExpr.expr,
				Right: rightExpr.expr,
				Op:    planpb.BinaryArithOp_BitXor,
			},
		},
	}
	dataType := calcDataType(leftExpr, rightExpr)
	return &ExprWithType{
		expr:     expr,
		dataType: dataType,
	}
}

func (v *Visitor) VisitBitOr(ctx *parser.BitOrContext) interface{} {
	left := ctx.Expr(0).Accept(v)
	if getError(left) != nil {
		return left
	}
	right := ctx.Expr(1).Accept(v)
	if getError(right) != nil {
		return right
	}

	leftNumber, rightNumber := getNumber(left), getNumber(right)
	if leftNumber != nil && rightNumber != nil {
		n, err := BitOr(leftNumber, rightNumber)
		if err != nil {
			return err
		}
		return n
	}

	var leftExpr *ExprWithType
	var rightExpr *ExprWithType
	if leftNumber != nil {
		leftExpr = toValueExpr(leftNumber)
	} else {
		leftExpr = getExpr(left)
	}
	if rightNumber != nil {
		rightExpr = toValueExpr(rightNumber)
	} else {
		rightExpr = getExpr(right)
	}
	if !typeutil.IsIntegerType(leftExpr.dataType) || !typeutil.IsIntegerType(rightExpr.dataType) {
		return fmt.Errorf("'bitor' can only be used between integer expressions")
	}
	expr := &planpb.Expr{
		Expr: &planpb.Expr_BinaryArithExpr{
			BinaryArithExpr: &planpb.BinaryArithExpr{
				Left:  leftExpr.expr,
				Right: rightExpr.expr,
				Op:    planpb.BinaryArithOp_BitOr,
			},
		},
	}
	dataType := calcDataType(leftExpr, rightExpr)
	return &ExprWithType{
		expr:     expr,
		dataType: dataType,
	}
}

func (v *Visitor) VisitLogicalAnd(ctx *parser.LogicalAndContext) interface{} {
	left := ctx.Expr(0).Accept(v)
	if getError(left) != nil {
		return left
	}
	right := ctx.Expr(1).Accept(v)
	if getError(right) != nil {
		return right
	}

	leftNumber, rightNumber := getNumber(left), getNumber(right)
	if leftNumber != nil && rightNumber != nil {
		n, err := And(leftNumber, rightNumber)
		if err != nil {
			return err
		}
		return n
	}

	var leftExpr *ExprWithType
	var rightExpr *ExprWithType
	if leftNumber != nil {
		leftExpr = toValueExpr(leftNumber)
	} else {
		leftExpr = getExpr(left)
	}
	if rightNumber != nil {
		rightExpr = toValueExpr(rightNumber)
	} else {
		rightExpr = getExpr(right)
	}
	if !typeutil.IsBooleanType(leftExpr.dataType) || !typeutil.IsBooleanType(rightExpr.dataType) {
		return fmt.Errorf("'and' can only be used between boolean expressions")
	}
	expr := &planpb.Expr{
		Expr: &planpb.Expr_BinaryLogicalExpr{
			BinaryLogicalExpr: &planpb.BinaryLogicalExpr{
				Left:  leftExpr.expr,
				Right: rightExpr.expr,
				Op:    planpb.BinaryLogicalOp_LogicalAnd,
			},
		},
	}
	dataType := calcDataType(leftExpr, rightExpr)
	return &ExprWithType{
		expr:     expr,
		dataType: dataType,
	}
}

func (v *Visitor) VisitLogicalOr(ctx *parser.LogicalOrContext) interface{} {
	left := ctx.Expr(0).Accept(v)
	if getError(left) != nil {
		return left
	}
	right := ctx.Expr(1).Accept(v)
	if getError(right) != nil {
		return right
	}

	leftNumber, rightNumber := getNumber(left), getNumber(right)
	if leftNumber != nil && rightNumber != nil {
		n, err := Or(leftNumber, rightNumber)
		if err != nil {
			return err
		}
		return n
	}

	var leftExpr *ExprWithType
	var rightExpr *ExprWithType
	if leftNumber != nil {
		leftExpr = toValueExpr(leftNumber)
	} else {
		leftExpr = getExpr(left)
	}
	if rightNumber != nil {
		rightExpr = toValueExpr(rightNumber)
	} else {
		rightExpr = getExpr(right)
	}
	if !typeutil.IsBooleanType(leftExpr.dataType) || !typeutil.IsBooleanType(rightExpr.dataType) {
		return fmt.Errorf("'or' can only be used between boolean expressions")
	}
	expr := &planpb.Expr{
		Expr: &planpb.Expr_BinaryLogicalExpr{
			BinaryLogicalExpr: &planpb.BinaryLogicalExpr{
				Left:  leftExpr.expr,
				Right: rightExpr.expr,
				Op:    planpb.BinaryLogicalOp_LogicalOr,
			},
		},
	}
	dataType := calcDataType(leftExpr, rightExpr)
	return &ExprWithType{
		expr:     expr,
		dataType: dataType,
	}
}
