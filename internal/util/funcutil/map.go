package funcutil

import (
	"fmt"

	"google.golang.org/protobuf/types/known/structpb"
)

var aggMethod = map[string]func(total, partial *structpb.Value) error{
	"sum": func(total, partial *structpb.Value) error {
		count, ok := partial.GetKind().(*structpb.Value_NumberValue)
		if !ok {
			return fmt.Errorf("sum value is not number")
		}
		totalValue, _ := total.GetKind().(*structpb.Value_NumberValue)
		totalValue.NumberValue += count.NumberValue
		return nil
	},
	"min": func(total, partial *structpb.Value) error {
		partialValue, ok := partial.GetKind().(*structpb.Value_NumberValue)
		if !ok {
			return fmt.Errorf("min value is not number")
		}
		if partialValue.NumberValue < total.GetNumberValue() {
			total.Kind = partial.Kind
		}
		return nil
	},
	"max": func(total, partial *structpb.Value) error {
		partialValue, ok := partial.GetKind().(*structpb.Value_NumberValue)
		if !ok {
			return fmt.Errorf("max value is not number")
		}
		if partialValue.NumberValue > total.GetNumberValue() {
			total.Kind = partial.Kind
		}
		return nil
	},
}

func NewMapReducer(schema map[string]string) MapReducer {
	m := &mapReducer{
		method: make(map[string]func(total, partial *structpb.Value) error),
	}
	for k, v := range schema {
		fn, ok := aggMethod[v]
		if !ok {
			continue
		}
		m.method[k] = fn
	}
	return m
}

type MapReducer interface {
	Reduce(results []*structpb.Struct) (*structpb.Struct, error)
}

type mapReducer struct {
	method map[string]func(total, partial *structpb.Value) error
}

func (m *mapReducer) Reduce(results []*structpb.Struct) (*structpb.Struct, error) {
	mergedResult, _ := structpb.NewStruct(nil)
	for _, result := range results {
		for k, v := range result.GetFields() {
			fn, ok := m.method[k]
			// if not exist, just copy
			if _, exist := mergedResult.GetFields()[k]; !exist {
				mergedResult.Fields[k] = v
				continue
			}
			// if exist and collide with other result, return error
			if !ok {
				return nil, fmt.Errorf("field %s without aggregation method has collision", k)
			}
			if err := fn(mergedResult.Fields[k], v); err != nil {
				return nil, err
			}
		}
	}
	return mergedResult, nil
}
