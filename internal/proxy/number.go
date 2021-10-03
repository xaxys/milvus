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
	"math"
)

type numberType int

const (
	None numberType = iota
	Bool
	Int
	Float
)

type number struct {
	boolValue  bool
	intValue   int64
	floatValue float64
	numType    numberType
}

func NewBool(value bool) *number {
	return &number{boolValue: value, numType: Bool}
}

func NewInt(value int64) *number {
	return &number{intValue: value, numType: Int}
}

func NewFloat(value float64) *number {
	return &number{floatValue: value, numType: Float}
}

func (n *number) IsBool() bool {
	return n.numType == Bool
}

func (n *number) IsInt() bool {
	return n.numType == Int
}

func (n *number) IsFloat() bool {
	return n.numType == Float
}

func (n *number) Bool() bool {
	switch n.numType {
	case Bool:
		return n.boolValue
	case Int:
		return n.intValue != 0
	case Float:
		return n.floatValue != 0
	default:
		panic("unreachable")
	}
}

func (n *number) Int() int64 {
	switch n.numType {
	case Bool:
		if n.boolValue {
			return 1
		}
		return 0
	case Int:
		return n.intValue
	case Float:
		return int64(n.floatValue)
	default:
		panic("unreachable")
	}
}

func (n *number) Float() float64 {
	switch n.numType {
	case Bool:
		if n.boolValue {
			return 1
		}
		return 0
	case Int:
		return float64(n.intValue)
	case Float:
		return n.floatValue
	default:
		panic("unreachable")
	}
}

// arithmatic functions

func Add(a, b *number) *number {
	aFloat, bFloat := a.IsFloat(), b.IsFloat()
	if aFloat && bFloat {
		return NewFloat(a.Float() + b.Float())
	} else if aFloat && !bFloat {
		return NewFloat(a.Float() + float64(b.Int()))
	} else if !aFloat && bFloat {
		return NewFloat(float64(a.Int()) + b.Float())
	}
	return NewInt(a.Int() + b.Int())
}

func Subtract(a, b *number) *number {
	aFloat, bFloat := a.IsFloat(), b.IsFloat()
	if aFloat && bFloat {
		return NewFloat(a.Float() - b.Float())
	} else if aFloat && !bFloat {
		return NewFloat(a.Float() - float64(b.Int()))
	} else if !aFloat && bFloat {
		return NewFloat(float64(a.Int()) - b.Float())
	}
	return NewInt(a.Int() - b.Int())
}

func Multiply(a, b *number) *number {
	aFloat, bFloat := a.IsFloat(), b.IsFloat()
	if aFloat && bFloat {
		return NewFloat(a.Float() * b.Float())
	} else if aFloat && !bFloat {
		return NewFloat(a.Float() * float64(b.Int()))
	} else if !aFloat && bFloat {
		return NewFloat(float64(a.Int()) * b.Float())
	}
	return NewInt(a.Int() * b.Int())
}

func Divide(a, b *number) (*number, error) {
	aFloat, bFloat := a.IsFloat(), b.IsFloat()
	if aFloat && bFloat {
		if a.Float() == 0 && b.Float() == 0 {
			return nil, fmt.Errorf("zero can not divide by zero")
		}
		return NewFloat(a.Float() / b.Float()), nil
	} else if aFloat && !bFloat {
		if a.Float() == 0 && b.Int() == 0 {
			return nil, fmt.Errorf("zero can not divide by zero")
		}
		return NewFloat(a.Float() / float64(b.Int())), nil
	} else if !aFloat && bFloat {
		if a.Int() == 0 && b.Float() == 0 {
			return nil, fmt.Errorf("zero can not divide by zero")
		}
		return NewFloat(float64(a.Int()) / b.Float()), nil
	}
	if b.Int() == 0 {
		return nil, fmt.Errorf("integer can not divide by zero")
	}
	return NewInt(a.Int() / b.Int()), nil
}

func Modulo(a, b *number) (*number, error) {
	if a.IsInt() && b.IsInt() {
		if b.Int() == 0 {
			return nil, fmt.Errorf("number can not modulo by zero")
		}
		return NewInt(a.Int() / b.Int()), nil
	}
	return nil, fmt.Errorf("'modulo' can only be used between integer expressions")
}

func Power(a, b *number) *number {
	aFloat, bFloat := a.IsFloat(), b.IsFloat()
	if aFloat && bFloat {
		return NewFloat(math.Pow(a.Float(), b.Float()))
	} else if aFloat && !bFloat {
		return NewFloat(math.Pow(a.Float(), float64(b.Int())))
	} else if !aFloat && bFloat {
		return NewFloat(math.Pow(float64(a.Int()), b.Float()))
	}
	return NewInt(int64(math.Pow(float64(a.Int()), float64(b.Int()))))
}

func BitAnd(a, b *number) (*number, error) {
	if a.IsBool() && b.IsBool() {
		return NewBool(a.Bool() && b.Bool()), nil
	} else if !a.IsFloat() && !b.IsFloat() {
		return NewInt(a.Int() & b.Int()), nil
	}
	return nil, fmt.Errorf("'bitand' can only be used between integer expressions")
}

func BitOr(a, b *number) (*number, error) {
	if a.IsBool() && b.IsBool() {
		return NewBool(a.Bool() || b.Bool()), nil
	} else if !a.IsFloat() && !b.IsFloat() {
		return NewInt(a.Int() | b.Int()), nil
	}
	return nil, fmt.Errorf("'bitor' can only be used between integer expressions")
}

func BitXor(a, b *number) (*number, error) {
	if a.IsBool() && b.IsBool() {
		return NewBool(a.Bool() != b.Bool()), nil
	} else if !a.IsFloat() && !b.IsFloat() {
		return NewInt(a.Int() ^ b.Int()), nil
	}
	return nil, fmt.Errorf("'bitxor' can only be used between integer expressions")
}

func ShiftLeft(a, b *number) (*number, error) {
	if !a.IsFloat() && !b.IsFloat() {
		return NewInt(a.Int() << b.Int()), nil
	}
	return nil, fmt.Errorf("'shiftleft' can only be used between integer expressions")
}

func ShiftRight(a, b *number) (*number, error) {
	if !a.IsFloat() && !b.IsFloat() {
		return NewInt(a.Int() >> b.Int()), nil
	}
	return nil, fmt.Errorf("'shiftright' can only be used between integer expressions")
}

func And(a, b *number) (*number, error) {
	if !a.IsFloat() && !b.IsFloat() {
		return NewBool(a.Bool() && b.Bool()), nil
	}
	return nil, fmt.Errorf("'and' can only be used between integer expressions")
}

func Or(a, b *number) (*number, error) {
	if !a.IsFloat() && !b.IsFloat() {
		return NewBool(a.Bool() || b.Bool()), nil
	}
	return nil, fmt.Errorf("'or' can only be used between integer expressions")
}

func BitNot(a *number) (*number, error) {
	if !a.IsFloat() {
		return NewInt(^a.Int()), nil
	}
	return nil, fmt.Errorf("bitnot' can only be used on integer expression")
}

func Negative(a *number) *number {
	if !a.IsFloat() {
		return NewInt(-a.Int())
	}
	return NewFloat(-a.Float())
}

func Not(a *number) *number {
	return NewBool(!a.Bool())
}

func Less(a, b *number) *number {
	aFloat, bFloat := a.IsFloat(), b.IsFloat()
	if aFloat && bFloat {
		return NewBool(a.Float() < b.Float())
	} else if aFloat && !bFloat {
		return NewBool(a.Float() < float64(b.Int()))
	} else if !aFloat && bFloat {
		return NewBool(float64(a.Int()) < b.Float())
	}
	return NewBool(a.Int() < b.Int())
}

func LessEqual(a, b *number) *number {
	aFloat, bFloat := a.IsFloat(), b.IsFloat()
	if aFloat && bFloat {
		return NewBool(a.Float() <= b.Float())
	} else if aFloat && !bFloat {
		return NewBool(a.Float() <= float64(b.Int()))
	} else if !aFloat && bFloat {
		return NewBool(float64(a.Int()) <= b.Float())
	}
	return NewBool(a.Int() <= b.Int())
}

func Greater(a, b *number) *number {
	aFloat, bFloat := a.IsFloat(), b.IsFloat()
	if aFloat && bFloat {
		return NewBool(a.Float() > b.Float())
	} else if aFloat && !bFloat {
		return NewBool(a.Float() > float64(b.Int()))
	} else if !aFloat && bFloat {
		return NewBool(float64(a.Int()) > b.Float())
	}
	return NewBool(a.Int() > b.Int())
}

func GreaterEqual(a, b *number) *number {
	aFloat, bFloat := a.IsFloat(), b.IsFloat()
	if aFloat && bFloat {
		return NewBool(a.Float() > b.Float())
	} else if aFloat && !bFloat {
		return NewBool(a.Float() > float64(b.Int()))
	} else if !aFloat && bFloat {
		return NewBool(float64(a.Int()) > b.Float())
	}
	return NewBool(a.Int() > b.Int())
}

func Equal(a, b *number) *number {
	aFloat, bFloat := a.IsFloat(), b.IsFloat()
	if aFloat && bFloat {
		return NewBool(a.Float() == b.Float())
	} else if aFloat && !bFloat {
		return NewBool(a.Float() == float64(b.Int()))
	} else if !aFloat && bFloat {
		return NewBool(float64(a.Int()) == b.Float())
	}
	return NewBool(a.Int() == b.Int())
}

func NotEqual(a, b *number) *number {
	aFloat, bFloat := a.IsFloat(), b.IsFloat()
	if aFloat && bFloat {
		return NewBool(a.Float() != b.Float())
	} else if aFloat && !bFloat {
		return NewBool(a.Float() != float64(b.Int()))
	} else if !aFloat && bFloat {
		return NewBool(float64(a.Int()) != b.Float())
	}
	return NewBool(a.Int() != b.Int())
}
