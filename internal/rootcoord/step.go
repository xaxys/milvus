package rootcoord

import (
	"context"
	"errors"
	"fmt"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"
	"go.uber.org/zap"
)

// stepLogger is a log of steps
type stepLogger struct {
	steps     []Step
	info      rootcoordpb.TaskInfo
	writeFunc func(data []byte) error
}

func UnmarshalTaskInfo(info *rootcoordpb.TaskInfo, c *RootCoord) (*stepLogger, error) {
	logs := &stepLogger{
		info: *info,
	}
	for _, rawStep := range info.Steps {
		base := baseStep{
			core: c,
		}
		step, err := UnmarshalStep(base, rawStep)
		if err != nil {
			return nil, fmt.Errorf("extract step recover info failed: %+v", err)
		}
		logs.steps = append(logs.steps, step)
	}
	return logs, nil
}

func (s *stepLogger) save() {
	data, err := proto.Marshal(&s.info)
	if err != nil {
		log.Error("serialize task info failed", zap.Error(err))
		return
	}
	err = s.writeFunc(data)
	if err != nil {
		log.Error("write task info failed", zap.Error(err))
	}
}

func (s *stepLogger) AddStep(step Step) {
	s.steps = append(s.steps, step)
	s.info.Steps = append(s.info.Steps, step.Serialize())
	s.save()
}

func (s *stepLogger) PopStep() {
	n := len(s.steps)
	if n == 0 {
		return
	}
	s.steps = s.steps[:n-1]
	s.info.Steps = s.info.Steps[:n-1]
	s.save()
}

func (s *stepLogger) LastStep() Step {
	n := len(s.steps)
	if n == 0 {
		return nil
	}
	return s.steps[n-1]
}

func (s *stepLogger) Clear() {
	s.steps = s.steps[:0]
	s.info.Steps = s.info.Steps[:0]
	s.writeFunc(nil)
}

func (s *stepLogger) Rollback(ctx context.Context) {
	step := s.LastStep()
	for step != nil {
		err := step.Execute(ctx)
		if err != nil {
			log.Error("rollback step failed", zap.Error(err))
			return
		}
		s.PopStep()
		step = s.LastStep()
	}
	s.Clear()
}

// Step is a step of a task
type Step interface {
	Execute(ctx context.Context) error
	Serialize() *rootcoordpb.Step
}

type baseStep struct {
	core *RootCoord
}

func UnmarshalStep(base baseStep, step *rootcoordpb.Step) (Step, error) {
	switch v := step.GetStep().(type) {
	case *rootcoordpb.Step_DeleteCollMetaStep:
		return &DeleteCollectionMetaStep{
			baseStep:                 base,
			DeleteCollectionMetaStep: *v.DeleteCollMetaStep,
		}, nil
	case *rootcoordpb.Step_DisableCollMetaStep:
		return &DisableCollectionMetaStep{
			baseStep:                  base,
			DisableCollectionMetaStep: *v.DisableCollMetaStep,
		}, nil
	case *rootcoordpb.Step_RemoveChannelStep:
		return &RemoveChannelStep{
			baseStep:          base,
			RemoveChannelStep: *v.RemoveChannelStep,
		}, nil
	case *rootcoordpb.Step_UnwatchChannelStep:
		return &UnwatchChannelStep{
			baseStep:           base,
			UnwatchChannelStep: *v.UnwatchChannelStep,
		}, nil
	case *rootcoordpb.Step_RemoveIndexStep:
		return &RemoveIndexStep{
			baseStep:        base,
			RemoveIndexStep: *v.RemoveIndexStep,
		}, nil
	case *rootcoordpb.Step_ReleaseCollStep:
		return &ReleaseCollectionStep{
			baseStep:              base,
			ReleaseCollectionStep: *v.ReleaseCollStep,
		}, nil
	case *rootcoordpb.Step_DeleteCollDataStep:
		return &DeleteCollectionDataStep{
			baseStep:                 base,
			DeleteCollectionDataStep: *v.DeleteCollDataStep,
		}, nil
	case *rootcoordpb.Step_AddCollMetaStep:
		return &AddCollectionMetaStep{
			baseStep:              base,
			AddCollectionMetaStep: *v.AddCollMetaStep,
		}, nil
	case *rootcoordpb.Step_EnableCollMetaStep:
		return &EnableCollectionMetaStep{
			baseStep:                 base,
			EnableCollectionMetaStep: *v.EnableCollMetaStep,
		}, nil
	case *rootcoordpb.Step_CreateChannelStep:
		return &CreateChannelStep{
			baseStep:          base,
			CreateChannelStep: *v.CreateChannelStep,
		}, nil
	case *rootcoordpb.Step_WatchChannelStep:
		return &WatchChannelStep{
			baseStep:         base,
			WatchChannelStep: *v.WatchChannelStep,
		}, nil
	case *rootcoordpb.Step_DeletePartMetaStep:
		return &DeletePartitionMetaStep{
			baseStep:                base,
			DeletePartitionMetaStep: *v.DeletePartMetaStep,
		}, nil
	case *rootcoordpb.Step_DisablePartMetaStep:
		return &DisablePartitionMetaStep{
			baseStep:                 base,
			DisablePartitionMetaStep: *v.DisablePartMetaStep,
		}, nil
	case *rootcoordpb.Step_ReleasePartStep:
		return &ReleasePartitionStep{
			baseStep:             base,
			ReleasePartitionStep: *v.ReleasePartStep,
		}, nil
	case *rootcoordpb.Step_DeletePartDataStep:
		return &DeletePartitionDataStep{
			baseStep:                base,
			DeletePartitionDataStep: *v.DeletePartDataStep,
		}, nil
	case *rootcoordpb.Step_AddPartMetaStep:
		return &AddPartitionMetaStep{
			baseStep:             base,
			AddPartitionMetaStep: *v.AddPartMetaStep,
		}, nil
	case *rootcoordpb.Step_EnablePartMetaStep:
		return &EnablePartitionMetaStep{
			baseStep:                base,
			EnablePartitionMetaStep: *v.EnablePartMetaStep,
		}, nil
	case *rootcoordpb.Step_ExpireCollCacheStep:
		return &ExpireCollectionCacheStep{
			baseStep:                  base,
			ExpireCollectionCacheStep: *v.ExpireCollCacheStep,
		}, nil
	case *rootcoordpb.Step_NullStep:
		return &NullStep{}, nil
	default:
		return nil, fmt.Errorf("unknown step type: %v", step.GetStep())
	}
}

type DeleteCollectionMetaStep struct {
	baseStep
	rootcoordpb.DeleteCollectionMetaStep
}

func (s *DeleteCollectionMetaStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_DeleteCollMetaStep{
			DeleteCollMetaStep: &s.DeleteCollectionMetaStep,
		},
	}
}

func (s *DeleteCollectionMetaStep) Execute(ctx context.Context) error {
	return nil
}

type DisableCollectionMetaStep struct {
	baseStep
	rootcoordpb.DisableCollectionMetaStep
}

func (s *DisableCollectionMetaStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_DisableCollMetaStep{
			DisableCollMetaStep: &s.DisableCollectionMetaStep,
		},
	}
}

func (s *DisableCollectionMetaStep) Execute(ctx context.Context) error {
	return nil
}

type RemoveChannelStep struct {
	baseStep
	rootcoordpb.RemoveChannelStep
}

func (s *RemoveChannelStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_RemoveChannelStep{
			RemoveChannelStep: &s.RemoveChannelStep,
		},
	}
}

func (s *RemoveChannelStep) Execute(ctx context.Context) (err error) {
	return nil
}

type UnwatchChannelStep struct {
	baseStep
	rootcoordpb.UnwatchChannelStep
}

func (s *UnwatchChannelStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_UnwatchChannelStep{
			UnwatchChannelStep: &s.UnwatchChannelStep,
		},
	}
}

func (s *UnwatchChannelStep) Execute(ctx context.Context) error {
	return errors.New("not implemented")
}

type RemoveIndexStep struct {
	baseStep
	rootcoordpb.RemoveIndexStep
}

func (s *RemoveIndexStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_RemoveIndexStep{
			RemoveIndexStep: &s.RemoveIndexStep,
		},
	}
}

func (s *RemoveIndexStep) Execute(ctx context.Context) error {
	return nil
}

type ReleaseCollectionStep struct {
	baseStep
	rootcoordpb.ReleaseCollectionStep
}

func (s *ReleaseCollectionStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_ReleaseCollStep{
			ReleaseCollStep: &s.ReleaseCollectionStep,
		},
	}
}

func (s *ReleaseCollectionStep) Execute(ctx context.Context) error {
	return nil
}

type DeleteCollectionDataStep struct {
	baseStep
	rootcoordpb.DeleteCollectionDataStep
}

func (s *DeleteCollectionDataStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_DeleteCollDataStep{
			DeleteCollDataStep: &s.DeleteCollectionDataStep,
		},
	}
}

func (s *DeleteCollectionDataStep) Execute(ctx context.Context) error {
	return nil
}

type AddCollectionMetaStep struct {
	baseStep
	coll *model.Collection
	rootcoordpb.AddCollectionMetaStep
}

func (s *AddCollectionMetaStep) Serialize() *rootcoordpb.Step {
	s.CollInfo = model.MarshalCollectionModel(s.coll)
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_AddCollMetaStep{
			AddCollMetaStep: &s.AddCollectionMetaStep,
		},
	}
}

func (s *AddCollectionMetaStep) Execute(ctx context.Context) error {
	return nil
}

type EnableCollectionMetaStep struct {
	baseStep
	rootcoordpb.EnableCollectionMetaStep
}

func (s *EnableCollectionMetaStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_EnableCollMetaStep{
			EnableCollMetaStep: &s.EnableCollectionMetaStep,
		},
	}
}

func (s *EnableCollectionMetaStep) Execute(ctx context.Context) error {
	return nil
}

type CreateChannelStep struct {
	baseStep
	rootcoordpb.CreateChannelStep
}

func (s *CreateChannelStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_CreateChannelStep{
			CreateChannelStep: &s.CreateChannelStep,
		},
	}
}

func (s *CreateChannelStep) Execute(ctx context.Context) (err error) {
	return nil
}

type WatchChannelStep struct {
	baseStep
	rootcoordpb.WatchChannelStep
}

func (s *WatchChannelStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_WatchChannelStep{
			WatchChannelStep: &s.WatchChannelStep,
		},
	}
}

func (s *WatchChannelStep) Execute(ctx context.Context) error {
	return nil
}

type DeletePartitionMetaStep struct {
	baseStep
	rootcoordpb.DeletePartitionMetaStep
}

func (s *DeletePartitionMetaStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_DeletePartMetaStep{
			DeletePartMetaStep: &s.DeletePartitionMetaStep,
		},
	}
}

func (s *DeletePartitionMetaStep) Execute(ctx context.Context) error {
	return nil
}

type DisablePartitionMetaStep struct {
	baseStep
	rootcoordpb.DisablePartitionMetaStep
}

func (s *DisablePartitionMetaStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_DisablePartMetaStep{
			DisablePartMetaStep: &s.DisablePartitionMetaStep,
		},
	}
}

func (s *DisablePartitionMetaStep) Execute(ctx context.Context) error {
	return nil
}

type ReleasePartitionStep struct {
	baseStep
	rootcoordpb.ReleasePartitionStep
}

func (s *ReleasePartitionStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_ReleasePartStep{
			ReleasePartStep: &s.ReleasePartitionStep,
		},
	}
}

func (s *ReleasePartitionStep) Execute(ctx context.Context) error {
	return nil
}

type DeletePartitionDataStep struct {
	baseStep
	rootcoordpb.DeletePartitionDataStep
}

func (s *DeletePartitionDataStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_DeletePartDataStep{
			DeletePartDataStep: &s.DeletePartitionDataStep,
		},
	}
}

func (s *DeletePartitionDataStep) Execute(ctx context.Context) error {
	return nil
}

type AddPartitionMetaStep struct {
	baseStep
	rootcoordpb.AddPartitionMetaStep
}

func (s *AddPartitionMetaStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_AddPartMetaStep{
			AddPartMetaStep: &s.AddPartitionMetaStep,
		},
	}
}

func (s *AddPartitionMetaStep) Execute(ctx context.Context) error {
	return nil
}

type EnablePartitionMetaStep struct {
	baseStep
	rootcoordpb.EnablePartitionMetaStep
}

func (s *EnablePartitionMetaStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_EnablePartMetaStep{
			EnablePartMetaStep: &s.EnablePartitionMetaStep,
		},
	}
}

func (s *EnablePartitionMetaStep) Execute(ctx context.Context) error {
	return nil
}

type ExpireCollectionCacheStep struct {
	baseStep
	rootcoordpb.ExpireCollectionCacheStep
}

func (s *ExpireCollectionCacheStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_ExpireCollCacheStep{
			ExpireCollCacheStep: &s.ExpireCollectionCacheStep,
		},
	}
}

func (s *ExpireCollectionCacheStep) Execute(ctx context.Context) error {
	return nil
}

type NullStep struct {
	rootcoordpb.NullStep
}

func (s *NullStep) Serialize() *rootcoordpb.Step {
	return &rootcoordpb.Step{
		Step: &rootcoordpb.Step_NullStep{
			NullStep: &s.NullStep,
		},
	}
}

func (s *NullStep) Execute(ctx context.Context) error {
	return nil
}
