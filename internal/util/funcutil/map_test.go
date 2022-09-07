package funcutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestMapReducer_Reduce_Sum(t *testing.T) {
	schema := map[string]string{
		"row_count": "sum",
	}
	results := []*structpb.Struct{
		{
			Fields: map[string]*structpb.Value{
				"row_count": {
					Kind: &structpb.Value_NumberValue{
						NumberValue: 1472,
					},
				},
			},
		},
		{
			Fields: map[string]*structpb.Value{
				"row_count": {
					Kind: &structpb.Value_NumberValue{
						NumberValue: 1528,
					},
				},
			},
		},
	}
	m := NewMapReducer(schema)
	result, err := m.Reduce(results)
	assert.Nil(t, err)
	assert.Equal(t, float64(3000), result.GetFields()["row_count"].GetNumberValue())
}

func TestMapReducer_Reduce_Sum_Invalid(t *testing.T) {
	schema := map[string]string{
		"row_count": "sum",
	}
	results := []*structpb.Struct{
		{
			Fields: map[string]*structpb.Value{
				"row_count": {
					Kind: &structpb.Value_NumberValue{
						NumberValue: 1472,
					},
				},
			},
		},
		{
			Fields: map[string]*structpb.Value{
				"row_count": {
					Kind: &structpb.Value_StringValue{
						StringValue: "not a number",
					},
				},
			},
		},
	}
	m := NewMapReducer(schema)
	_, err := m.Reduce(results)
	assert.Error(t, err)
}

func TestMapReducer_Reduce_Max(t *testing.T) {
	schema := map[string]string{
		"max(id)": "max",
	}
	results := []*structpb.Struct{
		{
			Fields: map[string]*structpb.Value{
				"max(id)": {
					Kind: &structpb.Value_NumberValue{
						NumberValue: 99,
					},
				},
			},
		},
		{
			Fields: map[string]*structpb.Value{
				"max(id)": {
					Kind: &structpb.Value_NumberValue{
						NumberValue: 100,
					},
				},
			},
		},
	}
	m := NewMapReducer(schema)
	result, err := m.Reduce(results)
	assert.Nil(t, err)
	assert.Equal(t, float64(100), result.GetFields()["max(id)"].GetNumberValue())
}

func TestMapReducer_Reduce_Max_Invalid(t *testing.T) {
	schema := map[string]string{
		"max(id)": "max",
	}
	results := []*structpb.Struct{
		{
			Fields: map[string]*structpb.Value{
				"max(id)": {
					Kind: &structpb.Value_NumberValue{
						NumberValue: 99,
					},
				},
			},
		},
		{
			Fields: map[string]*structpb.Value{
				"max(id)": {
					Kind: &structpb.Value_StringValue{
						StringValue: "not a number",
					},
				},
			},
		},
	}
	m := NewMapReducer(schema)
	_, err := m.Reduce(results)
	assert.Error(t, err)
}

func TestMapReducer_Reduce_Min(t *testing.T) {
	schema := map[string]string{
		"min(id)": "min",
	}
	results := []*structpb.Struct{
		{
			Fields: map[string]*structpb.Value{
				"min(id)": {
					Kind: &structpb.Value_NumberValue{
						NumberValue: 1,
					},
				},
			},
		},
		{
			Fields: map[string]*structpb.Value{
				"min(id)": {
					Kind: &structpb.Value_NumberValue{
						NumberValue: 5,
					},
				},
			},
		},
	}
	m := NewMapReducer(schema)
	result, err := m.Reduce(results)
	assert.Nil(t, err)
	assert.Equal(t, float64(1), result.GetFields()["min(id)"].GetNumberValue())
}

func TestMapReducer_Reduce_Min_Invalid(t *testing.T) {
	schema := map[string]string{
		"min(id)": "min",
	}
	results := []*structpb.Struct{
		{
			Fields: map[string]*structpb.Value{
				"min(id)": {
					Kind: &structpb.Value_NumberValue{
						NumberValue: 1,
					},
				},
			},
		},
		{
			Fields: map[string]*structpb.Value{
				"min(id)": {
					Kind: &structpb.Value_StringValue{
						StringValue: "not a number",
					},
				},
			},
		},
	}
	m := NewMapReducer(schema)
	_, err := m.Reduce(results)
	assert.Error(t, err)
}

func TestMapReducer_Reduce_Collision(t *testing.T) {
	results := []*structpb.Struct{
		{
			Fields: map[string]*structpb.Value{
				"a": {
					Kind: &structpb.Value_StringValue{
						StringValue: "some string from a",
					},
				},
			},
		},
		{
			Fields: map[string]*structpb.Value{
				"a": {
					Kind: &structpb.Value_StringValue{
						StringValue: "some string from another a",
					},
				},
			},
		},
	}
	m := NewMapReducer(nil)
	_, err := m.Reduce(results)
	assert.Error(t, err)
}
