package rootcoord

import (
	"context"
	"errors"
	"testing"

	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"
	"github.com/stretchr/testify/assert"
)

type mockStep struct {
	rootcoordpb.NullStep
	id      int64
	execErr error
}

func (m *mockStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_NullStep{
			NullStep: &m.NullStep,
		},
	}
}

func (m *mockStep) Execute(ctx context.Context) error {
	return m.execErr
}

func Test_StepLogger(t *testing.T) {
	logs := stepLogger{
		writeFunc: func(data []byte) error {
			return errors.New("mock write error")
		},
	}
	logs.AddStep(&mockStep{id: 1})
	logs.AddStep(&mockStep{id: 2})
	logs.AddStep(&mockStep{id: 3})
	logs.AddStep(&mockStep{id: 4, execErr: errors.New("mock")})

	logs.Rollback(context.Background())
	assert.Equal(t, 4, len(logs.steps))
	assert.Equal(t, 4, len(logs.info.Steps))
	logs.PopStep()
	logs.Rollback(context.Background())
	assert.Empty(t, logs.steps)
}
