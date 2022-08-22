package rootcoord

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"
	"github.com/milvus-io/milvus/internal/util/retry"
	"go.uber.org/zap"
)

// GetImportState returns the current state of an import task.
func (c *RootCoord) GetImportState(ctx context.Context, req *milvuspb.GetImportStateRequest) (*milvuspb.GetImportStateResponse, error) {
	if code, ok := c.checkHealthy(); !ok {
		return &milvuspb.GetImportStateResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode="+internalpb.StateCode_name[int32(code)]),
		}, nil
	}
	return c.importManager.getTaskState(req.GetTask()), nil
}

// ListImportTasks returns id array of all import tasks.
func (c *RootCoord) ListImportTasks(ctx context.Context, req *milvuspb.ListImportTasksRequest) (*milvuspb.ListImportTasksResponse, error) {
	if code, ok := c.checkHealthy(); !ok {
		return &milvuspb.ListImportTasksResponse{
			Status: failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode="+internalpb.StateCode_name[int32(code)]),
		}, nil
	}

	resp := &milvuspb.ListImportTasksResponse{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		},
		Tasks: c.importManager.listAllTasks(),
	}
	return resp, nil
}

// ReportImport reports import task state to RootCoord.
func (c *RootCoord) ReportImport(ctx context.Context, ir *rootcoordpb.ImportResult) (*commonpb.Status, error) {
	log.Info("RootCoord receive import state report",
		zap.Int64("task ID", ir.GetTaskId()),
		zap.Any("import state", ir.GetState()))
	if code, ok := c.checkHealthy(); !ok {
		return failStatus(commonpb.ErrorCode_UnexpectedError, "StateCode="+internalpb.StateCode_name[int32(code)]), nil
	}
	// Special case for ImportState_ImportAllocSegment state, where we shall only add segment ref lock and do no other
	// operations.
	// TODO: This is inelegant and must get re-structured.
	if ir.GetState() == commonpb.ImportState_ImportAllocSegment {
		// Lock the segments, so we don't lose track of them when compaction happens.
		// Note that these locks will be unlocked in c.postImportPersistLoop() -> checkSegmentLoadedLoop().
		if err := c.broker.AddSegRefLock(ctx, ir.GetTaskId(), ir.GetSegments()); err != nil {
			log.Error("failed to acquire segment ref lock", zap.Error(err))
			return &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UnexpectedError,
				Reason:    fmt.Sprintf("failed to acquire segment ref lock %s", err.Error()),
			}, nil
		}
		// Update task store with new segments.
		c.importManager.appendTaskSegments(ir.GetTaskId(), ir.GetSegments())
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}
	// Upon receiving ReportImport request, update the related task's state in task store.
	ti, err := c.importManager.updateTaskState(ir)
	if err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UpdateImportTaskFailure,
			Reason:    err.Error(),
		}, nil
	}

	// This method update a busy node to idle node, and send import task to idle node
	resendTaskFunc := func() {
		func() {
			c.importManager.busyNodesLock.Lock()
			defer c.importManager.busyNodesLock.Unlock()
			delete(c.importManager.busyNodes, ir.GetDatanodeId())
			log.Info("DataNode is no longer busy",
				zap.Int64("dataNode ID", ir.GetDatanodeId()),
				zap.Int64("task ID", ir.GetTaskId()))

		}()
		c.importManager.sendOutTasks(c.importManager.ctx)
	}

	// If task failed, send task to idle datanode
	if ir.GetState() == commonpb.ImportState_ImportFailed {
		// Release segments when task fails.
		log.Info("task failed, release segment ref locks")
		err := retry.Do(ctx, func() error {
			return c.broker.ReleaseSegRefLock(ctx, ir.GetTaskId(), ir.GetSegments())
		}, retry.Attempts(100))
		if err != nil {
			log.Error("failed to release lock, about to panic!")
			panic(err)
		}
		resendTaskFunc()
	}

	// So much for reporting, unless the task just reached `ImportPersisted` state.
	if ir.GetState() != commonpb.ImportState_ImportPersisted {
		log.Debug("non import-persisted state received, return immediately",
			zap.Any("task ID", ir.GetTaskId()),
			zap.Any("import state", ir.GetState()))
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}, nil
	}

	// Look up collection name on collection ID.
	var colName string
	var colMeta *model.Collection
	if colMeta, err = c.meta.GetCollectionByID(ctx, ti.GetCollectionId(), 0); err != nil {
		log.Error("failed to get collection name",
			zap.Int64("collection ID", ti.GetCollectionId()),
			zap.Error(err))
		// In some unexpected cases, user drop collection when bulkload task still in pending list, the datanode become idle.
		// If we directly return, the pending tasks will remain in pending list. So we call resendTaskFunc() to push next pending task to idle datanode.
		resendTaskFunc()
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_CollectionNameNotFound,
			Reason:    "failed to get collection name for collection ID" + strconv.FormatInt(ti.GetCollectionId(), 10),
		}, nil
	}
	colName = colMeta.Name

	// When DataNode has done its thing, remove it from the busy node list. And send import task again
	resendTaskFunc()

	// Flush all import data segments.
	c.broker.Flush(ctx, ti.GetCollectionId(), ir.GetSegments())
	// Check if data are "queryable" and if indices are built on all segments.
	go c.postImportPersistLoop(c.ctx, ir.GetTaskId(), ti.GetCollectionId(), colName, ir.GetSegments())

	return &commonpb.Status{
		ErrorCode: commonpb.ErrorCode_Success,
	}, nil
}

// CountCompleteIndex checks indexing status of the given segments.
// It returns an error if error occurs. It also returns a boolean indicating whether indexing is done (or if no index
// is needed).
func (c *RootCoord) CountCompleteIndex(ctx context.Context, collectionName string, collectionID UniqueID,
	allSegmentIDs []UniqueID) (bool, error) {
	// Note: Index name is always Params.CommonCfg.DefaultIndexName in current Milvus designs as of today.
	indexName := Params.CommonCfg.DefaultIndexName

	// Retrieve index status and detailed index information.
	describeIndexReq := &milvuspb.DescribeIndexRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_DescribeIndex,
		},
		CollectionName: collectionName,
		IndexName:      indexName,
	}
	indexDescriptionResp, err := c.DescribeIndex(ctx, describeIndexReq)
	if err != nil {
		return false, err
	}
	if len(indexDescriptionResp.GetIndexDescriptions()) == 0 {
		log.Info("no index needed for collection, consider indexing done",
			zap.Int64("collection ID", collectionID))
		return true, nil
	}
	log.Debug("got index description",
		zap.Any("index description", indexDescriptionResp))

	// Check if the target index name exists.
	matchIndexID := int64(-1)
	foundIndexID := false
	for _, desc := range indexDescriptionResp.GetIndexDescriptions() {
		if desc.GetIndexName() == indexName {
			matchIndexID = desc.GetIndexID()
			foundIndexID = true
			break
		}
	}
	if !foundIndexID {
		return false, fmt.Errorf("no index is created")
	}
	log.Debug("found match index ID",
		zap.Int64("match index ID", matchIndexID))

	getIndexStatesRequest := &indexpb.GetIndexStatesRequest{
		IndexBuildIDs: make([]UniqueID, 0),
	}

	// Fetch index build IDs from segments.
	var seg2Check []UniqueID
	for _, segmentID := range allSegmentIDs {
		describeSegmentRequest := &milvuspb.DescribeSegmentRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_DescribeSegment,
			},
			CollectionID: collectionID,
			SegmentID:    segmentID,
		}
		segmentDesc, _ := c.DescribeSegment(ctx, describeSegmentRequest)
		if segmentDesc.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
			// Describe failed, since the segment could get compacted, simply log and ignore the error.
			log.Error("failed to describe segment",
				zap.Int64("collection ID", collectionID),
				zap.Int64("segment ID", segmentID),
				zap.String("error", segmentDesc.GetStatus().GetReason()))
		}
		if segmentDesc.GetIndexID() == matchIndexID {
			if segmentDesc.GetEnableIndex() {
				seg2Check = append(seg2Check, segmentID)
				getIndexStatesRequest.IndexBuildIDs = append(getIndexStatesRequest.GetIndexBuildIDs(), segmentDesc.GetBuildID())
			}
		}
	}
	if len(getIndexStatesRequest.GetIndexBuildIDs()) == 0 {
		log.Info("none index build IDs returned, perhaps no index is needed",
			zap.String("collection name", collectionName),
			zap.Int64("collection ID", collectionID))
		return true, nil
	}

	log.Debug("working on GetIndexState",
		zap.Int("# of IndexBuildIDs", len(getIndexStatesRequest.GetIndexBuildIDs())))

	states, err := c.broker.GetIndexStates(ctx, getIndexStatesRequest.GetIndexBuildIDs())
	if err != nil {
		log.Error("failed to get index state in checkSegmentIndexStates", zap.Error(err))
		return false, err
	}

	// Count the # of segments with finished index.
	ct := 0
	for _, s := range states {
		if s.State == commonpb.IndexState_Finished {
			ct++
		}
	}
	log.Info("segment indexing state checked",
		zap.Int64s("segments checked", seg2Check),
		zap.Int("# of checked segment", len(seg2Check)),
		zap.Int("# of segments with complete index", ct),
		zap.String("collection name", collectionName),
		zap.Int64("collection ID", collectionID),
	)
	return len(seg2Check) == ct, nil
}

func (c *RootCoord) postImportPersistLoop(ctx context.Context, taskID int64, colID int64, colName string, segIDs []UniqueID) {
	// Loop and check if segments are loaded in queryNodes.
	c.wg.Add(1)
	go c.checkSegmentLoadedLoop(ctx, taskID, colID, segIDs)
	// Check if collection has any indexed fields. If so, start a loop to check segments' index states.
	if colMeta, err := c.meta.GetCollectionByID(ctx, colID, 0); err != nil {
		log.Error("failed to find meta for collection",
			zap.Int64("collection ID", colID),
			zap.Error(err))
	} else if len(colMeta.FieldIDToIndexID) == 0 {
		log.Info("no index field found for collection", zap.Int64("collection ID", colID))
	} else {
		log.Info("start checking index state", zap.Int64("collection ID", colID))
		c.wg.Add(1)
		go c.checkCompleteIndexLoop(ctx, taskID, colID, colName, segIDs)
	}
}

// checkSegmentLoadedLoop loops and checks if all segments in `segIDs` are loaded in queryNodes.
func (c *RootCoord) checkSegmentLoadedLoop(ctx context.Context, taskID int64, colID int64, segIDs []UniqueID) {
	defer c.wg.Done()
	ticker := time.NewTicker(time.Duration(Params.RootCoordCfg.ImportSegmentStateCheckInterval*1000) * time.Millisecond)
	defer ticker.Stop()
	expireTicker := time.NewTicker(time.Duration(Params.RootCoordCfg.ImportSegmentStateWaitLimit*1000) * time.Millisecond)
	defer expireTicker.Stop()
	defer func() {
		log.Info("we are done checking segment loading state, release segment ref locks")
		err := retry.Do(ctx, func() error {
			return c.broker.ReleaseSegRefLock(ctx, taskID, segIDs)
		}, retry.Attempts(100))
		if err != nil {
			log.Error("failed to release lock, about to panic!")
			panic(err)
		}
	}()
	for {
		select {
		case <-c.ctx.Done():
			log.Info("(in check segment loaded loop) context done, exiting checkSegmentLoadedLoop")
			return
		case <-ticker.C:
			resp, err := c.broker.GetQuerySegmentInfo(ctx, colID, segIDs)
			log.Debug("(in check segment loaded loop)",
				zap.Int64("task ID", taskID),
				zap.Int64("collection ID", colID),
				zap.Int64s("segment IDs expected", segIDs),
				zap.Int("# of segments found", len(resp.GetInfos())))
			if err != nil {
				log.Warn("(in check segment loaded loop) failed to call get segment info on queryCoord",
					zap.Int64("task ID", taskID),
					zap.Int64("collection ID", colID),
					zap.Int64s("segment IDs", segIDs))
			} else if len(resp.GetInfos()) == len(segIDs) {
				// Check if all segment info are loaded in queryNodes.
				log.Info("(in check segment loaded loop) all import data segments loaded in queryNodes",
					zap.Int64("task ID", taskID),
					zap.Int64("collection ID", colID),
					zap.Int64s("segment IDs", segIDs))
				c.importManager.setTaskDataQueryable(taskID)
				return
			}
		case <-expireTicker.C:
			log.Warn("(in check segment loaded loop) segments still not loaded after max wait time",
				zap.Int64("task ID", taskID),
				zap.Int64("collection ID", colID),
				zap.Int64s("segment IDs", segIDs))
			return
		}
	}
}

// checkCompleteIndexLoop loops and checks if all indices are built for an import task's segments.
func (c *RootCoord) checkCompleteIndexLoop(ctx context.Context, taskID int64, colID int64, colName string, segIDs []UniqueID) {
	defer c.wg.Done()
	ticker := time.NewTicker(time.Duration(Params.RootCoordCfg.ImportIndexCheckInterval*1000) * time.Millisecond)
	defer ticker.Stop()
	expireTicker := time.NewTicker(time.Duration(Params.RootCoordCfg.ImportIndexWaitLimit*1000) * time.Millisecond)
	defer expireTicker.Stop()
	for {
		select {
		case <-c.ctx.Done():
			log.Info("(in check complete index loop) context done, exiting checkCompleteIndexLoop")
			return
		case <-ticker.C:
			if done, err := c.CountCompleteIndex(ctx, colName, colID, segIDs); err == nil && done {
				log.Info("(in check complete index loop) indices are built or no index needed",
					zap.Int64("task ID", taskID))
				c.importManager.setTaskDataIndexed(taskID)
				return
			} else if err != nil {
				log.Error("(in check complete index loop) an error occurs",
					zap.Error(err))
			}
		case <-expireTicker.C:
			log.Warn("(in check complete index loop) indexing is taken too long",
				zap.Int64("task ID", taskID),
				zap.Int64("collection ID", colID),
				zap.Int64s("segment IDs", segIDs))
			return
		}
	}
}
