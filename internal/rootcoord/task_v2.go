package rootcoord

import (
	"context"
)

type taskV2 interface {
	SetTs(ts Timestamp)
	GetTs() Timestamp
	SetID(id UniqueID)
	GetID() UniqueID
	SetSuccess(success bool)
	IsSuccess() bool
	Prepare(ctx context.Context) error
	Execute(ctx context.Context) error
	PostExecute(ctx context.Context) error
	WaitToFinish() error
	NotifyDone(err error)
}

type baseTaskV2 struct {
	core    *RootCoord
	done    chan error
	ts      Timestamp
	id      UniqueID
	success bool
}

func (b *baseTaskV2) SetTs(ts Timestamp) {
	b.ts = ts
}

func (b *baseTaskV2) GetTs() Timestamp {
	return b.ts
}

func (b *baseTaskV2) SetID(id UniqueID) {
	b.id = id
}

func (b *baseTaskV2) GetID() UniqueID {
	return b.id
}

func (b *baseTaskV2) SetSuccess(success bool) {
	b.success = success
}

func (b *baseTaskV2) IsSuccess() bool {
	return b.success
}

func (b *baseTaskV2) Prepare(ctx context.Context) error {
	return nil
}

func (b *baseTaskV2) Execute(ctx context.Context) error {
	return nil
}

func (b *baseTaskV2) PostExecute(ctx context.Context) error {
	return nil
}

func (b *baseTaskV2) WaitToFinish() error {
	return <-b.done
}

func (b *baseTaskV2) NotifyDone(err error) {
	b.done <- err
}
