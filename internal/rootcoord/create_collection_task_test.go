package rootcoord

import (
	"context"
	"errors"
	"os"
	"testing"

	"github.com/milvus-io/milvus/internal/proto/datapb"

	"github.com/milvus-io/milvus/internal/log"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/util/dependency"

	"github.com/milvus-io/milvus/internal/proto/commonpb"

	"github.com/milvus-io/milvus/internal/proto/milvuspb"

	"github.com/golang/protobuf/proto"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/proto/schemapb"
)

const (
	testCollectionName = "test"
	testFieldName1     = "test_1"
	testFieldName2     = "test_2"
	testFieldType1     = schemapb.DataType_Int64
	testFieldType2     = schemapb.DataType_VarChar
	testServerID       = 100
)

func genTestSchema() *schemapb.CollectionSchema {
	return &schemapb.CollectionSchema{
		Name:        testCollectionName,
		Description: "",
		AutoID:      false,
		Fields: []*schemapb.FieldSchema{
			{Name: testFieldName1, DataType: testFieldType1},
			{Name: testFieldName2, DataType: testFieldType2},
		},
	}
}

func genTestCreateCollectionReq() *milvuspb.CreateCollectionRequest {
	schema := genTestSchema()
	marshaledSchema, err := proto.Marshal(schema)
	if err != nil {
		panic(err)
	}

	req := &milvuspb.CreateCollectionRequest{
		Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_CreateCollection},
		CollectionName: testCollectionName,
		Schema:         marshaledSchema,
	}
	return req
}

func genTestCore() *RootCoord {
	Params.InitOnce()
	Params.RootCoordCfg.DmlChannelNum = 4

	ctx := context.Background()
	factory := dependency.NewDefaultFactory(true)
	chans := map[UniqueID][]string{}
	core := &RootCoord{
		txn:          newTxnKV(),
		idAllocator:  newIDAllocator(),
		tsoAllocator: newTsoAllocator(),
		chanTimeTick: newTimeTickSync(ctx, testServerID, factory, chans),
		meta:         newMeta(),
		dataCoord:    newDC(),
	}
	return core
}

// cleanTestEnv clean test environment, for example, files generated by rocksmq.
func cleanTestEnv() {
	path := "/tmp/milvus"
	if err := os.RemoveAll(path); err != nil {
		log.Warn("failed to clean test directories", zap.Error(err), zap.String("path", path))
	}
}

func Test_createCollectionTask_succeed(t *testing.T) {
	defer cleanTestEnv()

	req := genTestCreateCollectionReq()
	task := &createCollectionTask{
		baseUndoTask: baseUndoTask{
			baseTaskV2: baseTaskV2{
				core: genTestCore(),
				done: make(chan error, 1),
			},
		},
		Req: req,
	}

	ts := Timestamp(100)
	id := UniqueID(101)
	task.SetTs(ts)
	task.SetID(id)

	ctx := context.Background()
	err := task.Prepare(ctx)
	assert.NoError(t, err)
	err = task.Execute(ctx)
	assert.NoError(t, err)
}

func Test_createCollectionTask_partial_fail(t *testing.T) {
	defer cleanTestEnv()

	unwatchChannelCalled := false
	deleteCollectionCalled := false
	core := genTestCore()
	core.meta.(*mockMetaTable).DeleteCollectionF = func(ctx context.Context, collectionID UniqueID, ts Timestamp) error {
		deleteCollectionCalled = true
		return nil
	}
	core.dataCoord.(*mockDataCoord).WatchChannelsF = func(ctx context.Context, req *datapb.WatchChannelsRequest) (*datapb.WatchChannelsResponse, error) {
		return nil, errors.New("mock")
	}
	core.CallUnwatchChannels = func(ctx context.Context, collectionID UniqueID, vChannels []string) error {
		unwatchChannelCalled = true
		return nil
	}

	req := genTestCreateCollectionReq()
	task := &createCollectionTask{
		baseUndoTask: baseUndoTask{
			baseTaskV2: baseTaskV2{
				core: core,
				done: make(chan error, 1),
			},
		},
		Req: req,
	}

	ts := Timestamp(100)
	id := UniqueID(101)
	task.SetTs(ts)
	task.SetID(id)

	ctx := context.Background()
	err := task.Prepare(ctx)
	assert.NoError(t, err)
	err = task.Execute(ctx)
	assert.Error(t, err)
	// check if undo worked.
	assert.True(t, unwatchChannelCalled)
	assert.True(t, deleteCollectionCalled)
}
