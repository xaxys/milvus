package model

import (
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
)

type Field struct {
	FieldID      int64
	Name         string
	IsPrimaryKey bool
	Description  string
	DataType     schemapb.DataType
	TypeParams   []*commonpb.KeyValuePair
	IndexParams  []*commonpb.KeyValuePair
	AutoID       bool
	State        schemapb.FieldState
}

func (f Field) Available() bool {
	return f.State == schemapb.FieldState_FieldCreated
}

func (f Field) Clone() *Field {
	tps := make([]*commonpb.KeyValuePair, len(f.TypeParams))
	ips := make([]*commonpb.KeyValuePair, len(f.IndexParams))
	for i, tp := range f.TypeParams {
		tps[i] = &commonpb.KeyValuePair{Key: tp.GetKey(), Value: tp.GetValue()}
	}
	for i, ip := range f.IndexParams {
		ips[i] = &commonpb.KeyValuePair{Key: ip.GetKey(), Value: ip.GetValue()}
	}
	return &Field{
		FieldID:      f.FieldID,
		Name:         f.Name,
		IsPrimaryKey: f.IsPrimaryKey,
		Description:  f.Description,
		DataType:     f.DataType,
		TypeParams:   tps,
		IndexParams:  ips,
		AutoID:       f.AutoID,
		State:        f.State,
	}
}

func MarshalFieldModel(field *Field) *schemapb.FieldSchema {
	if field == nil {
		return nil
	}

	return &schemapb.FieldSchema{
		FieldID:      field.FieldID,
		Name:         field.Name,
		IsPrimaryKey: field.IsPrimaryKey,
		Description:  field.Description,
		DataType:     field.DataType,
		TypeParams:   field.TypeParams,
		IndexParams:  field.IndexParams,
		AutoID:       field.AutoID,
	}
}

func MarshalFieldModels(fields []*Field) []*schemapb.FieldSchema {
	if fields == nil {
		return nil
	}

	fieldSchemas := make([]*schemapb.FieldSchema, len(fields))
	for idx, field := range fields {
		fieldSchemas[idx] = MarshalFieldModel(field)
	}
	return fieldSchemas
}

func UnmarshalFieldModel(fieldSchema *schemapb.FieldSchema) *Field {
	if fieldSchema == nil {
		return nil
	}

	return &Field{
		FieldID:      fieldSchema.FieldID,
		Name:         fieldSchema.Name,
		IsPrimaryKey: fieldSchema.IsPrimaryKey,
		Description:  fieldSchema.Description,
		DataType:     fieldSchema.DataType,
		TypeParams:   fieldSchema.TypeParams,
		IndexParams:  fieldSchema.IndexParams,
		AutoID:       fieldSchema.AutoID,
	}
}

func UnmarshalFieldModels(fieldSchemas []*schemapb.FieldSchema) []*Field {
	if fieldSchemas == nil {
		return nil
	}

	fields := make([]*Field, len(fieldSchemas))
	for idx, fieldSchema := range fieldSchemas {
		fields[idx] = UnmarshalFieldModel(fieldSchema)
	}
	return fields
}
