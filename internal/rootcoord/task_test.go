package rootcoord

import (
	"context"
	"errors"
	"fmt"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/etcd"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/util/dependency"
)

//func TestDescribeSegmentReqTask_Type(t *testing.T) {
//	tsk := &DescribeSegmentsReqTask{
//		Req: &rootcoordpb.DescribeSegmentsRequest{
//			Base: &commonpb.MsgBase{
//				MsgType: commonpb.MsgType_DescribeSegments,
//			},
//		},
//	}
//	assert.Equal(t, commonpb.MsgType_DescribeSegments, tsk.Type())
//}

//func TestDescribeSegmentsReqTask_Execute(t *testing.T) {
//	collID := typeutil.UniqueID(1)
//	partID := typeutil.UniqueID(2)
//	segID := typeutil.UniqueID(100)
//	fieldID := typeutil.UniqueID(3)
//	buildID := typeutil.UniqueID(4)
//	indexID := typeutil.UniqueID(1000)
//	indexName := "test_describe_segments_index"
//
//	c := &Core{}
//
//	// failed to get flushed segments.
//	c.CallGetFlushedSegmentsService = func(ctx context.Context, collID, partID typeutil.UniqueID) ([]typeutil.UniqueID, error) {
//		return nil, errors.New("mock")
//	}
//	tsk := &DescribeSegmentsReqTask{
//		baseReqTask: baseReqTask{
//			core: c,
//		},
//		Req: &rootcoordpb.DescribeSegmentsRequest{
//			Base: &commonpb.MsgBase{
//				MsgType: commonpb.MsgType_DescribeSegments,
//			},
//			CollectionID: collID,
//			SegmentIDs:   []typeutil.UniqueID{segID},
//		},
//		Rsp: &rootcoordpb.DescribeSegmentsResponse{},
//	}
//	assert.Error(t, tsk.Execute(context.Background()))
//
//	// requested segment not found in flushed segments.
//	c.CallGetFlushedSegmentsService = func(ctx context.Context, collID, partID typeutil.UniqueID) ([]typeutil.UniqueID, error) {
//		return []typeutil.UniqueID{}, nil
//	}
//	assert.Error(t, tsk.Execute(context.Background()))
//
//	// segment not found in meta.
//	c.CallGetFlushedSegmentsService = func(ctx context.Context, collID, partID typeutil.UniqueID) ([]typeutil.UniqueID, error) {
//		return []typeutil.UniqueID{segID}, nil
//	}
//	c.MetaTable = &MetaTable{
//		segID2IndexID: make(map[typeutil.UniqueID]typeutil.UniqueID, 1),
//	}
//	assert.NoError(t, tsk.Execute(context.Background()))
//
//	// index not found in meta
//	c.MetaTable = &MetaTable{
//		segID2IndexID: map[typeutil.UniqueID]typeutil.UniqueID{segID: indexID},
//		indexID2Meta: map[typeutil.UniqueID]*model.Index{
//			indexID: {
//				CollectionID: collID,
//				FieldID:      fieldID,
//				IndexID:      indexID,
//				SegmentIndexes: map[int64]model.SegmentIndex{
//					segID + 1: {
//						Segment: model.Segment{
//							SegmentID:   segID,
//							PartitionID: partID,
//						},
//						BuildID:     buildID,
//						EnableIndex: true,
//					},
//				},
//			},
//		},
//	}
//	assert.Error(t, tsk.Execute(context.Background()))
//
//	// success.
//	c.MetaTable = &MetaTable{
//		segID2IndexID: map[typeutil.UniqueID]typeutil.UniqueID{segID: indexID},
//		indexID2Meta: map[typeutil.UniqueID]*model.Index{
//			indexID: {
//				CollectionID: collID,
//				FieldID:      fieldID,
//				IndexID:      indexID,
//				IndexName:    indexName,
//				IndexParams:  nil,
//				SegmentIndexes: map[int64]model.SegmentIndex{
//					segID: {
//						Segment: model.Segment{
//							SegmentID:   segID,
//							PartitionID: partID,
//						},
//						BuildID:     buildID,
//						EnableIndex: true,
//					},
//				},
//			},
//		},
//	}
//	assert.NoError(t, tsk.Execute(context.Background()))
//}

func Test_hasSystemFields(t *testing.T) {
	t.Run("no system fields", func(t *testing.T) {
		schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{Name: "not_system_field"}}}
		assert.False(t, hasSystemFields(schema, []string{RowIDFieldName, TimeStampFieldName}))
	})

	t.Run("has row id field", func(t *testing.T) {
		schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{Name: RowIDFieldName}}}
		assert.True(t, hasSystemFields(schema, []string{RowIDFieldName, TimeStampFieldName}))
	})

	t.Run("has timestamp field", func(t *testing.T) {
		schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{{Name: TimeStampFieldName}}}
		assert.True(t, hasSystemFields(schema, []string{RowIDFieldName, TimeStampFieldName}))
	})
}

func TestCreateCollectionReqTask_Execute_hasSystemFields(t *testing.T) {
	schema := &schemapb.CollectionSchema{Name: "test", Fields: []*schemapb.FieldSchema{{Name: TimeStampFieldName}}}
	marshaledSchema, err := proto.Marshal(schema)
	assert.NoError(t, err)
	ctx := context.Background()
	task := &CreateCollectionReqTask{
		baseUndoTask: baseUndoTask{
			baseReqTask: baseReqTask{
				ctx:  ctx,
				core: nil,
				done: make(chan error, 1),
			},
			logs: stepLogger{
				writeFunc: func(data []byte) error {
					return nil
				},
			},
		},
		Req: &milvuspb.CreateCollectionRequest{
			Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_CreateCollection},
			CollectionName: "test",
			Schema:         marshaledSchema,
		},
	}
	scheduler := newTaskScheduler(ctx)
	scheduler.Start()
	scheduler.AddTask(task)
	err = task.WaitToFinish()
	assert.Error(t, err)
}

func TestCreateCollectionReqTask_ChannelMismatch(t *testing.T) {
	schema := &schemapb.CollectionSchema{Name: "test", Fields: []*schemapb.FieldSchema{{Name: "f1"}}}
	marshaledSchema, err := proto.Marshal(schema)
	assert.NoError(t, err)
	msFactory := dependency.NewDefaultFactory(true)

	Params.Init()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	core, err := NewCore(ctx, msFactory)
	assert.NoError(t, err)
	core.IDAllocator = func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error) {
		return 0, 0, nil
	}
	core.chanTimeTick = newTimeTickSync(core.ctx, 1, core.factory, nil)
	core.TSOAllocator = func(count uint32) (typeutil.Timestamp, error) {
		return 0, nil
	}
	core.SendDdCreateCollectionReq = func(context.Context, *internalpb.CreateCollectionRequest, []string) (map[string][]byte, error) {
		return map[string][]byte{}, nil
	}

	// set RootCoordDml="" to trigger a error for code coverage
	Params.CommonCfg.RootCoordDml = ""
	task := &CreateCollectionReqTask{
		baseUndoTask: baseUndoTask{
			baseReqTask: baseReqTask{
				ctx:  ctx,
				core: core,
				done: make(chan error, 1),
			},
			logs: stepLogger{
				writeFunc: func(data []byte) error {
					return nil
				},
			},
		},
		Req: &milvuspb.CreateCollectionRequest{
			Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_CreateCollection},
			CollectionName: "test",
			Schema:         marshaledSchema,
		},
	}
	scheduler := newTaskScheduler(ctx)
	scheduler.Start()
	scheduler.AddTask(task)
	err = task.WaitToFinish()
	assert.Error(t, err)
}

func newMockCore(ctx context.Context) *Core {
	Params.Init()
	core, _ := NewCore(ctx, dependency.NewDefaultFactory(true))

	rand.Seed(time.Now().UnixNano())
	randVal := rand.Int()
	rootPath := fmt.Sprintf("/test/meta/%d", randVal)

	etcdCli, _ := etcd.GetEtcdClient(&Params.EtcdCfg)
	defer etcdCli.Close()
	core.etcdCli = etcdCli

	txnKV := etcdkv.NewEtcdKV(etcdCli, rootPath)

	mockTxnKV := &mockTestTxnKV{
		TxnKV: txnKV,
		loadWithPrefix: func(key string) ([]string, []string, error) {
			return []string{}, []string{}, nil
		},
		save: func(key, value string) error {
			return nil
		},
		multiSave: func(kvs map[string]string) error {
			return nil
		},
		multiSaveAndRemoveWithPrefix: func(kvs map[string]string, removal []string) error {
			return nil
		},
		remove: func(key string) error {
			return nil
		},
	}
	mockKV := &mockTestKV{
		TxnKV: txnKV,
		loadWithPrefix: func(key string, ts typeutil.Timestamp) ([]string, []string, error) {
			return nil, nil, nil
		},
		save: func(key, value string, ts typeutil.Timestamp) error {
			return nil
		},
		multiSave: func(kvs map[string]string, ts typeutil.Timestamp) error {
			return nil
		},
		multiSaveAndRemoveWithPrefix: func(kvs map[string]string, removal []string, ts typeutil.Timestamp) error {
			return nil
		},
	}
	mockMt, _ := NewMetaTable(context.TODO(), mockTxnKV, mockKV)

	core.MetaTable = mockMt

	mockID := typeutil.UniqueID(0)
	core.IDAllocator = func(count uint32) (typeutil.UniqueID, typeutil.UniqueID, error) {
		oldID := mockID
		mockID += typeutil.UniqueID(count)
		return oldID + 1, mockID, nil
	}

	core.chanTimeTick = newTimeTickSync(core.ctx, 1, core.factory, nil)

	mockTSO := typeutil.Timestamp(0)
	core.TSOAllocator = func(count uint32) (typeutil.Timestamp, error) {
		mockTSO++
		return mockTSO, nil
	}

	core.SendDdDropPartitionReq = func(ctx context.Context, req *internalpb.DropPartitionRequest, channelNames []string) error {
		return nil
	}

	core.SendDdDropCollectionReq = func(ctx context.Context, req *internalpb.DropCollectionRequest, channelNames []string) error {
		return nil
	}

	core.CallReleasePartitionService = func(ctx context.Context, ts typeutil.Timestamp, dbID, collectionID typeutil.UniqueID, partitionIDs []typeutil.UniqueID) error {
		return nil
	}

	core.CallReleaseCollectionService = func(ctx context.Context, ts typeutil.Timestamp, dbID, collectionID typeutil.UniqueID) error {
		return nil
	}

	core.CallWatchChannels = func(ctx context.Context, collectionID typeutil.UniqueID, channelNames []string) error {
		return nil
	}

	pnm := &proxyMock{
		collArray: make([]string, 0, 16),
		mutex:     sync.Mutex{},
	}

	core.NewProxyClient = func(*sessionutil.Session) (types.Proxy, error) {
		return pnm, nil
	}

	core.initSession()
	chanMap := core.MetaTable.ListCollectionPhysicalChannels()
	core.chanTimeTick = newTimeTickSync(core.ctx, core.session.ServerID, core.factory, chanMap)
	core.chanTimeTick.addSession(core.session)
	core.proxyClientManager = newProxyClientManager(core)
	core.proxyManager = newProxyManager(
		core.ctx,
		core.etcdCli,
		core.chanTimeTick.initSessions,
		core.proxyClientManager.GetProxyClients,
	)
	core.proxyManager.AddSessionFunc(core.chanTimeTick.addSession, core.proxyClientManager.AddProxyClient)
	core.proxyManager.DelSessionFunc(core.chanTimeTick.delSession, core.proxyClientManager.DelProxyClient)

	return core
}

func Test_Task(t *testing.T) {
	schema := &schemapb.CollectionSchema{Name: "test coll", Fields: []*schemapb.FieldSchema{{Name: "f1"}}}
	marshaledSchema, err := proto.Marshal(schema)
	assert.NoError(t, err)
	ctx := context.Background()
	core := newMockCore(ctx)

	scheduler := newTaskScheduler(ctx)
	scheduler.Start()

	// ============================ create collection ============================

	t.Run("CreateCollectionReqTask fail", func(t *testing.T) {
		task := &CreateCollectionReqTask{
			baseUndoTask: baseUndoTask{
				baseReqTask: newBaseReqTask(ctx, core),
				logs: stepLogger{
					writeFunc: func(data []byte) error {
						return nil
					},
				},
			},
			Req: &milvuspb.CreateCollectionRequest{
				Base:             &commonpb.MsgBase{MsgType: commonpb.MsgType_CreateCollection},
				CollectionName:   "test coll",
				Schema:           marshaledSchema,
				ShardsNum:        8,
				ConsistencyLevel: commonpb.ConsistencyLevel_Eventually,
			},
		}

		watchFunc := core.CallWatchChannels
		core.CallWatchChannels = func(ctx context.Context, collectionID typeutil.UniqueID, channelNames []string) error {
			return errors.New("mock watch channels error")
		}
		scheduler.AddTask(task)
		err = task.WaitToFinish()
		assert.Error(t, err)
		// uncomment this check after unwatch channel is implemented
		//assert.Empty(t, task.logs.steps)
		core.CallWatchChannels = watchFunc
	})

	t.Run("CreateCollectionReqTask", func(t *testing.T) {
		task := &CreateCollectionReqTask{
			baseUndoTask: baseUndoTask{
				baseReqTask: newBaseReqTask(ctx, core),
				logs: stepLogger{
					writeFunc: func(data []byte) error {
						return nil
					},
				},
			},
			Req: &milvuspb.CreateCollectionRequest{
				Base:             &commonpb.MsgBase{MsgType: commonpb.MsgType_CreateCollection},
				CollectionName:   "test coll",
				Schema:           marshaledSchema,
				ShardsNum:        8,
				ConsistencyLevel: commonpb.ConsistencyLevel_Eventually,
			},
		}

		scheduler.AddTask(task)
		err = task.WaitToFinish()
		assert.NoError(t, err)
		assert.Empty(t, task.logs.steps)
	})

	t.Run("CreateCollectionReqTask duplicate", func(t *testing.T) {
		task := &CreateCollectionReqTask{
			baseUndoTask: baseUndoTask{
				baseReqTask: newBaseReqTask(ctx, core),
				logs: stepLogger{
					writeFunc: func(data []byte) error {
						return nil
					},
				},
			},
			Req: &milvuspb.CreateCollectionRequest{
				Base:             &commonpb.MsgBase{MsgType: commonpb.MsgType_CreateCollection},
				CollectionName:   "test coll",
				Schema:           marshaledSchema,
				ShardsNum:        8,
				ConsistencyLevel: commonpb.ConsistencyLevel_Eventually,
			},
		}

		scheduler.AddTask(task)
		err = task.WaitToFinish()
		assert.NoError(t, err)
		assert.Empty(t, task.logs.steps)
	})

	t.Run("CreateCollectionReqTask duplicate with different params", func(t *testing.T) {
		task := &CreateCollectionReqTask{
			baseUndoTask: baseUndoTask{
				baseReqTask: newBaseReqTask(ctx, core),
				logs: stepLogger{
					writeFunc: func(data []byte) error {
						return nil
					},
				},
			},
			Req: &milvuspb.CreateCollectionRequest{
				Base:             &commonpb.MsgBase{MsgType: commonpb.MsgType_CreateCollection},
				CollectionName:   "test coll",
				Schema:           marshaledSchema,
				ShardsNum:        2,
				ConsistencyLevel: commonpb.ConsistencyLevel_Eventually,
			},
		}

		scheduler.AddTask(task)
		err = task.WaitToFinish()
		assert.Error(t, err)
		assert.Empty(t, task.logs.steps)
	})

	t.Run("CreatePartitionReqTask", func(t *testing.T) {
		task := &CreatePartitionReqTask{
			baseUndoTask: baseUndoTask{
				baseReqTask: newBaseReqTask(ctx, core),
				logs: stepLogger{
					writeFunc: func(data []byte) error {
						return nil
					},
				},
			},
			Req: &milvuspb.CreatePartitionRequest{
				Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_CreatePartition},
				CollectionName: "test coll",
				PartitionName:  "test part",
			},
		}

		scheduler.AddTask(task)
		err = task.WaitToFinish()
		assert.NoError(t, err)
		assert.Empty(t, task.logs.steps)
	})

	t.Run("CreatePartitionReqTask duplicate", func(t *testing.T) {
		task := &CreatePartitionReqTask{
			baseUndoTask: baseUndoTask{
				baseReqTask: newBaseReqTask(ctx, core),
				logs: stepLogger{
					writeFunc: func(data []byte) error {
						return nil
					},
				},
			},
			Req: &milvuspb.CreatePartitionRequest{
				Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_CreatePartition},
				CollectionName: "test coll",
				PartitionName:  "test part",
			},
		}

		scheduler.AddTask(task)
		err = task.WaitToFinish()
		assert.NoError(t, err)
		assert.Empty(t, task.logs.steps)
	})

	t.Run("DropPartitionReqTask failed async", func(t *testing.T) {
		releaseFunc := core.CallReleasePartitionService
		mockCount := 0
		core.CallReleasePartitionService = func(ctx context.Context, ts typeutil.Timestamp, dbID, collectionID typeutil.UniqueID, partitionIDs []typeutil.UniqueID) error {
			mockCount++
			return errors.New("mock err")
		}
		task := &DropPartitionReqTask{
			baseRedoTask: baseRedoTask{
				baseReqTask: newBaseReqTask(ctx, core),
				logs: stepLogger{
					writeFunc: func(data []byte) error {
						return nil
					},
				},
			},
			Req: &milvuspb.DropPartitionRequest{
				Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_DropPartition},
				CollectionName: "test coll",
				PartitionName:  "test part",
			},
		}

		scheduler.AddTask(task)
		err = task.WaitToFinish()
		assert.NoError(t, err)
		time.Sleep(time.Second)
		assert.Equal(t, 1, mockCount)
		assert.NotEmpty(t, task.logs.steps)

		core.CallReleasePartitionService = releaseFunc
	})

	t.Run("DropPartitionReqTask duplicate", func(t *testing.T) {
		task := &DropPartitionReqTask{
			baseRedoTask: baseRedoTask{
				baseReqTask: newBaseReqTask(ctx, core),
				logs: stepLogger{
					writeFunc: func(data []byte) error {
						return nil
					},
				},
			},
			Req: &milvuspb.DropPartitionRequest{
				Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_DropPartition},
				CollectionName: "test coll",
				PartitionName:  "test part",
			},
		}

		scheduler.AddTask(task)
		err = task.WaitToFinish()
		assert.NoError(t, err)
		assert.Empty(t, task.logs.steps)
	})

	t.Run("DropCollectionReqTask", func(t *testing.T) {
		task := &DropCollectionReqTask{
			baseRedoTask: baseRedoTask{
				baseReqTask: newBaseReqTask(ctx, core),
				logs: stepLogger{
					writeFunc: func(data []byte) error {
						return nil
					},
				},
			},
			Req: &milvuspb.DropCollectionRequest{
				Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_DropCollection},
				CollectionName: "test coll",
			},
		}

		scheduler.AddTask(task)
		err = task.WaitToFinish()
		assert.NoError(t, err)
		assert.NotEmpty(t, task.logs.steps)
	})

	t.Run("DropCollectionReqTask duplicate", func(t *testing.T) {
		task := &DropCollectionReqTask{
			baseRedoTask: baseRedoTask{
				baseReqTask: newBaseReqTask(ctx, core),
				logs: stepLogger{
					writeFunc: func(data []byte) error {
						return nil
					},
				},
			},
			Req: &milvuspb.DropCollectionRequest{
				Base:           &commonpb.MsgBase{MsgType: commonpb.MsgType_DropCollection},
				CollectionName: "test coll",
			},
		}

		scheduler.AddTask(task)
		err = task.WaitToFinish()
		assert.NoError(t, err)
		assert.Empty(t, task.logs.steps)
	})
}
