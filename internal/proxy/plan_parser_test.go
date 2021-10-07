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
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/proto/planpb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

func newTestSchema() *schemapb.CollectionSchema {
	fields := []*schemapb.FieldSchema{
		{FieldID: 0, Name: "FieldID", IsPrimaryKey: false, Description: "field no.1", DataType: schemapb.DataType_Int64},
	}

	for name, value := range schemapb.DataType_value {
		dataType := schemapb.DataType(value)
		if !typeutil.IsIntegerType(dataType) && !typeutil.IsFloatingType(dataType) && !typeutil.IsVectorType(dataType) {
			continue
		}
		newField := &schemapb.FieldSchema{
			FieldID: int64(100 + value), Name: name + "Field", IsPrimaryKey: false, Description: "", DataType: dataType,
		}
		fields = append(fields, newField)
	}

	return &schemapb.CollectionSchema{
		Name:        "test",
		Description: "schema for test used",
		AutoID:      true,
		Fields:      fields,
	}
}

func TestParseExpr_Naive(t *testing.T) {
	schemaPb := newTestSchema()
	schema, err := typeutil.CreateSchemaHelper(schemaPb)
	assert.Nil(t, err)

	t.Run("test UnaryNode", func(t *testing.T) {
		exprStrs := []string{
			"Int64Field > +1",
			"Int64Field > -1",
			"FloatField > +1.0",
			"FloatField > -1.0",
		}
		for _, exprStr := range exprStrs {
			exprProto, err := parseExpr(schema, exprStr)
			assert.Nil(t, err)
			str := proto.MarshalTextString(exprProto)
			println(str)
		}
	})

	t.Run("test UnaryNode invalid", func(t *testing.T) {
		exprStrs := []string{
			"Int64Field > +aa",
			"FloatField > -aa",
		}
		for _, exprStr := range exprStrs {
			exprProto, err := parseExpr(schema, exprStr)
			assert.Error(t, err)
			assert.Nil(t, exprProto)
		}
	})

	t.Run("test BinaryNode", func(t *testing.T) {
		exprStrs := []string{
			// "+"
			"FloatField > 1 + 2",
			"FloatField > 1 + 2.0",
			"FloatField > 1.0 + 2",
			"FloatField > 1.0 + 2.0",
			// "-"
			"FloatField > 1 - 2",
			"FloatField > 1 - 2.0",
			"FloatField > 1.0 - 2",
			"FloatField > 1.0 - 2.0",
			// "*"
			"FloatField > 1 * 2",
			"FloatField > 1 * 2.0",
			"FloatField > 1.0 * 2",
			"FloatField > 1.0 * 2.0",
			// "/"
			"FloatField > 1 / 2",
			"FloatField > 1 / 2.0",
			"FloatField > 1.0 / 2",
			"FloatField > 1.0 / 2.0",
			// "%"
			"FloatField > 1 % 2",
			// "**"
			"FloatField > 1 ** 2",
			"FloatField > 1 ** 2.0",
			"FloatField > 1.0 ** 2",
			"FloatField > 1.0 ** 2.0",
		}
		for _, exprStr := range exprStrs {
			exprProto, err := parseExpr(schema, exprStr)
			assert.Nil(t, err)
			str := proto.MarshalTextString(exprProto)
			println(str)
		}
	})

	t.Run("test BinaryNode invalid", func(t *testing.T) {
		exprStrs := []string{
			// "/"
			"FloatField > 1 / 0",
			"FloatField > aa / 0",
			"FloatField > aa / 0.0",
			// "%"
			"FloatField > 1 % 0",
			"FloatField > 1 % 0.0",
			"FloatField > aa % 0",
		}
		for _, exprStr := range exprStrs {
			exprProto, err := parseExpr(schema, exprStr)
			assert.Error(t, err)
			assert.Nil(t, exprProto)
		}
	})
}

func TestParsePlanNode_Naive(t *testing.T) {
	exprStrs := []string{
		"not (Int64Field > 3)",
		"not (3 > Int64Field)",
		"Int64Field in [1, 2, 3]",
		"Int64Field < 3 and (Int64Field > 2 || Int64Field == 1)",
		"DoubleField in [1.0, 2, 3]",
		"DoubleField in [1.0, 2, 3] && Int64Field < 3 or Int64Field > 2",
	}

	schema := newTestSchema()
	queryInfo := &planpb.QueryInfo{
		Topk:         10,
		MetricType:   "L2",
		SearchParams: "{\"nprobe\": 10}",
	}

	// Note: use pointer to string to represent nullable string
	// TODO: change it to better solution
	for offset, exprStr := range exprStrs {
		fmt.Printf("case %d: %s\n", offset, exprStr)
		planProto, err := CreateQueryPlan(schema, exprStr, "FloatVectorField", queryInfo)
		assert.Nil(t, err)
		dbgStr := proto.MarshalTextString(planProto)
		println(dbgStr)
	}

}

func TestExprPlan_Str(t *testing.T) {
	fields := []*schemapb.FieldSchema{
		{FieldID: 100, Name: "fakevec", DataType: schemapb.DataType_FloatVector},
		{FieldID: 101, Name: "age", DataType: schemapb.DataType_Int64},
	}

	schema := &schemapb.CollectionSchema{
		Name:        "default-collection",
		Description: "",
		AutoID:      true,
		Fields:      fields,
	}

	queryInfo := &planpb.QueryInfo{
		Topk:         10,
		MetricType:   "L2",
		SearchParams: "{\"nprobe\": 10}",
	}

	// without filter
	planProto, err := CreateQueryPlan(schema, "", "fakevec", queryInfo)
	assert.Nil(t, err)
	dbgStr := proto.MarshalTextString(planProto)
	println(dbgStr)

	exprStrs := []string{
		"age >= 420000 && age < 420010", // range
		"age == 420000 || age == 420001 || age == 420002 || age == 420003 || age == 420004", // term
		"age not in [1, 2, 3]",
	}

	for offset, exprStr := range exprStrs {
		fmt.Printf("case %d: %s\n", offset, exprStr)
		planProto, err := CreateQueryPlan(schema, exprStr, "fakevec", queryInfo)
		assert.Nil(t, err)
		dbgStr := proto.MarshalTextString(planProto)
		println(dbgStr)
	}
}

func TestExprRange_Str(t *testing.T) {
	exprStrs := []string{
		"1 == Int8Field",
		"1.0 != Int8Field",
		"1 < Int8Field",
		"1.0 < Int8Field",
		"1.0 >= Int8Field",
		"Int8Field == 2",
		"Int8Field != 2.0",
		"Int8Field < 2",
		"Int8Field < 2.0",
		"Int8Field >= 2.0",
		"1 == Int64Field",
		"1.0 != Int64Field",
		"1 < Int64Field",
		"1.0 < Int64Field",
		"1.0 >= Int64Field",
		"Int64Field == 2",
		"Int64Field != 2.0",
		"Int64Field < 2",
		"Int64Field < 2.0",
		"Int64Field >= 2.0",
		"1 == FloatField",
		"1.0 != FloatField",
		"1 < FloatField",
		"1.0 < FloatField",
		"1.0 >= FloatField",
		"FloatField == 2",
		"FloatField != 2.0",
		"FloatField < 2",
		"FloatField < 2.0",
		"FloatField >= 2.0",
		"1 == DoubleField",
		"1.0 != DoubleField",
		"1 < DoubleField",
		"1.0 < DoubleField",
		"1.0 >= DoubleField",
		"DoubleField == 2",
		"DoubleField != 2.0",
		"DoubleField < 2",
		"DoubleField < 2.0",
		"DoubleField >= 2.0",
		"1 < Int8Field < 2",
		"1 < Int8Field < 2.0",
		"1.0 < Int8Field < 2.0",
		"1.0 <= Int8Field <= 2.0",
		"1 < Int64Field < 2",
		"1 < Int64Field < 2.0",
		"1.0 < Int64Field < 2.0",
		"1.0 <= Int64Field <= 2.0",
		"1 < FloatField < 2",
		"1 < FloatField < 2.0",
		"1.0 < FloatField < 2.0",
		"1.0 <= FloatField <= 2.0",
		"1 < DoubleField < 2",
		"1 < DoubleField < 2.0",
		"1.0 < DoubleField < 2.0",
		"1.0 <= DoubleField <= 2.0",
		"BoolField == True",
		"BoolField == true",
		"BoolField == 1",
		"True == BoolField",
		"BoolField == False",
		"BoolField == false",
		"BoolField == 0",
	}

	invalidExprStrs := []string{
		"1 < BoolField",
		"1.0 < BoolField",
		"1.0 >= BoolField",
		"BoolField < 2",
		"BoolField < 2.0",
		"BoolField >= 2.0",
		"1 < BoolField < 2",
		"1 < BoolField < 2.0",
		"1.0 < BoolField < 2.0",
		"1.0 <= BoolField <= 2.0",
		"1 < BoolField < 2",
		"1 < BoolField < 2.0",
		"1.0 < BoolField < 2.0",
		"1.0 <= BoolField <= 2.0",
		"1 > Int64Field > 2",
		"1 > Int64Field > 2.0",
		"1.0 > Int64Field > 2.0",
		"1.0 >= Int64Field >= 2.0",
		"1.0 < Int64Field > 2.0",
		"0 < 1 < Int64Field < 2 < 3",
		"Int8Field < 1 < Int64Field < 2.0 < FloatField",
		"1.0 == Int64Field == 2.0",
		"1.0 != Int64Field != 2.0",
	}

	fields := []*schemapb.FieldSchema{
		{FieldID: 100, Name: "fakevec", DataType: schemapb.DataType_FloatVector},
		{FieldID: 101, Name: "Int64Field", DataType: schemapb.DataType_Int64},
		{FieldID: 102, Name: "Int8Field", DataType: schemapb.DataType_Int8},
		{FieldID: 103, Name: "FloatField", DataType: schemapb.DataType_Float},
		{FieldID: 104, Name: "DoubleField", DataType: schemapb.DataType_Double},
		{FieldID: 105, Name: "BoolField", DataType: schemapb.DataType_Bool},
	}

	schema := &schemapb.CollectionSchema{
		Name:        "default-collection",
		Description: "",
		AutoID:      true,
		Fields:      fields,
	}

	queryInfo := &planpb.QueryInfo{
		Topk:         10,
		MetricType:   "L2",
		SearchParams: "{\"nprobe\": 10}",
	}

	for offset, exprStr := range exprStrs {
		fmt.Printf("case %d: %s\n", offset, exprStr)
		planProto, err := CreateQueryPlan(schema, exprStr, "fakevec", queryInfo)
		assert.Nil(t, err)
		dbgStr := proto.MarshalTextString(planProto)
		println(dbgStr)
	}

	for offset, invalidExprStr := range invalidExprStrs {
		fmt.Printf("case %d: %s\n", offset, invalidExprStr)
		planProto, err := CreateQueryPlan(schema, invalidExprStr, "fakevec", queryInfo)
		assert.Error(t, err)
		dbgStr := proto.MarshalTextString(planProto)
		println(dbgStr)
	}
}

func TestExprCompare_Str(t *testing.T) {
	exprStrs := []string{
		"age1 < age2",
		"age1 <= age2",
		"age1 > age2",
		"age1 >= age2",
		"age1 != age2",
		"age1 == age2",
	}

	invalidExprStrs := []string{
		"age1 == age2 == age1",
		"age1 < age2 < age1",
		"age1 >= age2 >= age1",
	}

	fields := []*schemapb.FieldSchema{
		{FieldID: 100, Name: "fakevec", DataType: schemapb.DataType_FloatVector},
		{FieldID: 101, Name: "age1", DataType: schemapb.DataType_Int64},
		{FieldID: 102, Name: "age2", DataType: schemapb.DataType_Int32},
	}

	schema := &schemapb.CollectionSchema{
		Name:        "default-collection",
		Description: "",
		AutoID:      true,
		Fields:      fields,
	}

	queryInfo := &planpb.QueryInfo{
		Topk:         10,
		MetricType:   "L2",
		SearchParams: "{\"nprobe\": 10}",
	}

	for offset, exprStr := range exprStrs {
		fmt.Printf("case %d: %s\n", offset, exprStr)
		planProto, err := CreateQueryPlan(schema, exprStr, "fakevec", queryInfo)
		assert.Nil(t, err)
		dbgStr := proto.MarshalTextString(planProto)
		println(dbgStr)
	}

	for offset, invalidExprStr := range invalidExprStrs {
		fmt.Printf("case %d: %s\n", offset, invalidExprStr)
		planProto, err := CreateQueryPlan(schema, invalidExprStr, "fakevec", queryInfo)
		assert.Error(t, err)
		dbgStr := proto.MarshalTextString(planProto)
		println(dbgStr)
	}
}

func TestExprArith_Str(t *testing.T) {
	exprStrs := []string{
		"Int64Field ** Int8Field == 0",
		"Int64Field ** DoubleField == 0",
		"DoubleField ** Int8Field == 0",
		"DoubleField ** FloatField == 0",
		"Int64Field + Int8Field == 0",
		"Int64Field - FloatField == 0",
		"FloatField * DoubleField == 0",
		"FloatField / Int8Field == 0",
		"Int8Field / DoubleField == 0",
		"DoubleField / 0 == 0",
		// "Int64Field % Int8Field == 0", // modulo unimplemented yet.
		// "Int64Field % 3 == 0",
		"Int64Field << Int8Field == 0",
		"Int8Field >> Int8Field == 0",
		"(Int64Field & Int8Field) == 0",
		"(Int64Field ^ Int8Field) == 0",
		"(Int64Field | Int8Field) == 0",
		"BoolField and true",
		"BoolField or false",
	}

	invalidExprStrs := []string{
		"Int8Field",
		"Int64Field",
		"FloatField",
		"DoubleField",
		"Int64Field + true == 0",
		"Int64Field + BoolField == 0",
		"BoolField - FloatField == 0",
		"FloatField * BoolField == 0",
		"Int8Field / 0 == 0",
		"Int8Field / 0 * 0 == 0",
		"BoolField % Int8Field == 0",
		"Int64Field % 0 == 0",
		"DoubleField % Int64Field == 0",
		"FloatField << Int8Field == 0",
		"Int8Field >> BoolField == 0",
		"FloatField << 0.0 == 0",
		"(Int64Field & FloatField) == 0",
		"(Int64Field & BoolField) == 0",
		"(Int64Field & false) == 0",
		"(Int64Field ^ FloatField) == 0",
		"(Int64Field ^ BoolField) == 0",
		"(Int64Field ^ false) == 0",
		"(Int64Field | FloatField) == 0",
		"(Int64Field | BoolField) == 0",
		"(Int64Field | false) == 0",
		"Int64Field and FloatField",
		"Int64Field or false",
		"FloatField or false",
	}

	fields := []*schemapb.FieldSchema{
		{FieldID: 100, Name: "fakevec", DataType: schemapb.DataType_FloatVector},
		{FieldID: 101, Name: "Int64Field", DataType: schemapb.DataType_Int64},
		{FieldID: 102, Name: "Int8Field", DataType: schemapb.DataType_Int8},
		{FieldID: 103, Name: "FloatField", DataType: schemapb.DataType_Float},
		{FieldID: 104, Name: "DoubleField", DataType: schemapb.DataType_Double},
		{FieldID: 105, Name: "BoolField", DataType: schemapb.DataType_Bool},
	}

	schema := &schemapb.CollectionSchema{
		Name:        "default-collection",
		Description: "",
		AutoID:      true,
		Fields:      fields,
	}

	queryInfo := &planpb.QueryInfo{
		Topk:         10,
		MetricType:   "L2",
		SearchParams: "{\"nprobe\": 10}",
	}

	for offset, exprStr := range exprStrs {
		fmt.Printf("case %d: %s\n", offset, exprStr)
		planProto, err := CreateQueryPlan(schema, exprStr, "fakevec", queryInfo)
		assert.Nil(t, err)
		dbgStr := proto.MarshalTextString(planProto)
		println(dbgStr)
	}

	for offset, invalidExprStr := range invalidExprStrs {
		fmt.Printf("case %d: %s\n", offset, invalidExprStr)
		planProto, err := CreateQueryPlan(schema, invalidExprStr, "fakevec", queryInfo)
		assert.Error(t, err)
		dbgStr := proto.MarshalTextString(planProto)
		println(dbgStr)
	}
}

func TestPlanParseAPIs(t *testing.T) {
	t.Run("get compare op type", func(t *testing.T) {
		var op planpb.OpType
		var reverse bool

		reverse = false
		op = getCompareOpType(">", reverse)
		assert.Equal(t, planpb.OpType_GreaterThan, op)
		op = getCompareOpType(">=", reverse)
		assert.Equal(t, planpb.OpType_GreaterEqual, op)
		op = getCompareOpType("<", reverse)
		assert.Equal(t, planpb.OpType_LessThan, op)
		op = getCompareOpType("<=", reverse)
		assert.Equal(t, planpb.OpType_LessEqual, op)
		op = getCompareOpType("==", reverse)
		assert.Equal(t, planpb.OpType_Equal, op)
		op = getCompareOpType("!=", reverse)
		assert.Equal(t, planpb.OpType_NotEqual, op)
		op = getCompareOpType("*", reverse)
		assert.Equal(t, planpb.OpType_Invalid, op)

		reverse = true
		op = getCompareOpType(">", reverse)
		assert.Equal(t, planpb.OpType_LessThan, op)
		op = getCompareOpType(">=", reverse)
		assert.Equal(t, planpb.OpType_LessEqual, op)
		op = getCompareOpType("<", reverse)
		assert.Equal(t, planpb.OpType_GreaterThan, op)
		op = getCompareOpType("<=", reverse)
		assert.Equal(t, planpb.OpType_GreaterEqual, op)
		op = getCompareOpType("==", reverse)
		assert.Equal(t, planpb.OpType_Equal, op)
		op = getCompareOpType("!=", reverse)
		assert.Equal(t, planpb.OpType_NotEqual, op)
		op = getCompareOpType("*", reverse)
		assert.Equal(t, planpb.OpType_Invalid, op)
	})

	t.Run("parse bool node", func(t *testing.T) {
		var nodeRaw1, nodeRaw2, nodeRaw3, nodeRaw4 ant_ast.Node
		nodeRaw1 = &ant_ast.IdentifierNode{
			Value: "True",
		}
		boolNode1 := parseBoolNode(&nodeRaw1)
		assert.Equal(t, boolNode1.Value, true)

		nodeRaw2 = &ant_ast.IdentifierNode{
			Value: "False",
		}
		boolNode2 := parseBoolNode(&nodeRaw2)
		assert.Equal(t, boolNode2.Value, false)

		nodeRaw3 = &ant_ast.IdentifierNode{
			Value: "abcd",
		}
		assert.Nil(t, parseBoolNode(&nodeRaw3))

		nodeRaw4 = &ant_ast.BoolNode{
			Value: true,
		}
		assert.Nil(t, parseBoolNode(&nodeRaw4))
	})
}
