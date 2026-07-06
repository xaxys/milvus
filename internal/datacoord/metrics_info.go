// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package datacoord

import (
	"context"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/tidwall/gjson"
	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus/internal/json"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/hardware"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metricsinfo"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
	"github.com/milvus-io/milvus/pkg/v3/util/uniquegenerator"
)

// getQuotaMetrics returns DataCoordQuotaMetrics.
func (s *Server) getQuotaMetrics() *metricsinfo.DataCoordQuotaMetrics {
	info := s.meta.GetQuotaInfo()
	return info
}

func (s *Server) getCollectionMetrics(ctx context.Context) *metricsinfo.DataCoordCollectionMetrics {
	totalNumRows := s.meta.GetAllCollectionNumRows()
	ret := &metricsinfo.DataCoordCollectionMetrics{
		Collections: make(map[int64]*metricsinfo.DataCoordCollectionInfo, len(totalNumRows)),
	}
	for collectionID, total := range totalNumRows {
		if _, ok := ret.Collections[collectionID]; !ok {
			ret.Collections[collectionID] = &metricsinfo.DataCoordCollectionInfo{
				NumEntitiesTotal: 0,
				IndexInfo:        make([]*metricsinfo.DataCoordIndexInfo, 0),
			}
		}
		ret.Collections[collectionID].NumEntitiesTotal = total
	}
	return ret
}

func (s *Server) getChannelsJSON(ctx context.Context, req *milvuspb.GetMetricsRequest) (string, error) {
	channels, err := getMetrics[*metricsinfo.Channel](ctx, s, req)
	// fill checkpoint timestamp
	channel2Checkpoints := s.meta.GetChannelCheckpoints()
	for _, channel := range channels {
		if cp, ok := channel2Checkpoints[channel.Name]; ok {
			channel.CheckpointTS = tsoutil.PhysicalTimeFormat(cp.GetTimestamp())
		} else {
			mlog.Warn(ctx, "channel not found in meta cache", mlog.String("channel", channel.Name))
		}
	}
	return metricsinfo.MarshalGetMetricsValues(channels, err)
}

// mergeChannels merges the channel metrics from data nodes and channel watch infos from channel manager
// dnChannels: a slice of Channel metrics from data nodes
// dcChannels: a map of channel watch infos from the channel manager, keyed by node ID and channel name
func mergeChannels(dnChannels []*metricsinfo.Channel, dcChannels map[int64]map[string]*datapb.ChannelWatchInfo) []*metricsinfo.Channel {
	mergedChannels := make([]*metricsinfo.Channel, 0)

	// Add or update channels from data nodes
	for _, dnChannel := range dnChannels {
		if dcChannelMap, ok := dcChannels[dnChannel.NodeID]; ok {
			if dcChannel, ok := dcChannelMap[dnChannel.Name]; ok {
				dnChannel.WatchState = dcChannel.State.String()
				delete(dcChannelMap, dnChannel.Name)
			}
		}
		mergedChannels = append(mergedChannels, dnChannel)
	}

	// Add remaining channels from channel manager
	for nodeID, dcChannelMap := range dcChannels {
		for _, dcChannel := range dcChannelMap {
			mergedChannels = append(mergedChannels, &metricsinfo.Channel{
				Name:         dcChannel.Vchan.ChannelName,
				CollectionID: dcChannel.Vchan.CollectionID,
				WatchState:   dcChannel.State.String(),
				NodeID:       nodeID,
			})
		}
	}

	return mergedChannels
}

func (s *Server) getSegmentsJSON(ctx context.Context, req *milvuspb.GetMetricsRequest, jsonReq gjson.Result) (string, error) {
	v := jsonReq.Get(metricsinfo.MetricRequestParamINKey)
	if !v.Exists() {
		// default to get all segments from datanode
		return s.getDataNodeSegmentsJSON(ctx, req)
	}

	in := v.String()
	if in == metricsinfo.MetricsRequestParamsInDN {
		return s.getDataNodeSegmentsJSON(ctx, req)
	}

	if in == metricsinfo.MetricsRequestParamsInDC {
		collectionID := metricsinfo.GetCollectionIDFromRequest(jsonReq)
		segments := s.meta.getSegmentsMetrics(collectionID)
		for _, seg := range segments {
			isIndexed, indexedFields := s.meta.indexMeta.GetSegmentIndexedFields(seg.CollectionID, seg.SegmentID)
			seg.IndexedFields = indexedFields
			seg.IsIndexed = isIndexed
		}

		bs, err := json.Marshal(segments)
		if err != nil {
			mlog.Warn(ctx, "marshal segment value failed", mlog.FieldCollectionID(collectionID), mlog.String("err", err.Error()))
			return "", nil
		}
		return string(bs), nil
	}
	return "", merr.WrapErrParameterInvalidMsg("invalid param value in=[%s], it should be dc or dn", in)
}

func (s *Server) getDistJSON(ctx context.Context, req *milvuspb.GetMetricsRequest) string {
	segments := s.meta.getSegmentsMetrics(-1)
	dist := &metricsinfo.DataCoordDist{
		Segments: segments,
	}

	bs, err := json.Marshal(dist)
	if err != nil {
		mlog.Warn(ctx, "marshal dist value failed", mlog.String("err", err.Error()))
		return ""
	}
	return string(bs)
}

func (s *Server) getDataNodeSegmentsJSON(ctx context.Context, req *milvuspb.GetMetricsRequest) (string, error) {
	ret, err := getMetrics[*metricsinfo.Segment](ctx, s, req)
	return metricsinfo.MarshalGetMetricsValues(ret, err)
}

func (s *Server) getSyncTaskJSON(ctx context.Context, req *milvuspb.GetMetricsRequest) (string, error) {
	ret, err := getMetrics[*metricsinfo.SyncTask](ctx, s, req)
	return metricsinfo.MarshalGetMetricsValues(ret, err)
}

// getSystemInfoMetrics composes data cluster metrics
func (s *Server) getSystemInfoMetrics(
	ctx context.Context,
	req *milvuspb.GetMetricsRequest,
) (string, error) {
	coordTopology := s.getDataCoordTopology(ctx, req)
	ret, err := metricsinfo.MarshalTopology(coordTopology)
	if err != nil {
		return "", err
	}
	return ret, nil
}

// getDataCoordTopology returns DataCoord topology directly without JSON serialization
// This is optimized for in-process calls in MixCoord mode to avoid marshal/unmarshal overhead
func (s *Server) getDataCoordTopology(
	ctx context.Context,
	req *milvuspb.GetMetricsRequest,
) metricsinfo.DataCoordTopology {
	// TODO(dragondriver): add more detail metrics

	// get datacoord info
	nodes := s.nodeManager.GetClientIDs()
	clusterTopology := metricsinfo.DataClusterTopology{
		Self:               s.getDataCoordMetrics(ctx),
		ConnectedDataNodes: make([]metricsinfo.DataNodeInfos, 0, len(nodes)),
	}

	// for each data node, fetch metrics info
	for _, node := range nodes {
		infos, err := s.getDataNodeMetrics(ctx, req, node)
		if err != nil {
			mlog.Warn(ctx, "fails to get DataNode metrics", mlog.Err(err))
			continue
		}
		infos.ObjectStorageDerivedMetrics = s.updateDataNodeObjectStorageMetrics(node, infos.ObjectStorageMetrics)
		clusterTopology.ConnectedDataNodes = append(clusterTopology.ConnectedDataNodes, infos)
	}
	clusterTopology.Self.ObjectStorageMetrics = aggregateObjectStorageDerivedMetrics(clusterTopology.ConnectedDataNodes)
	s.cleanupObjectStorageMetricSamples(nodes)

	// compose topology struct
	return metricsinfo.DataCoordTopology{
		Cluster: clusterTopology,
		Connections: metricsinfo.ConnTopology{
			Name: metricsinfo.ConstructComponentName(typeutil.DataCoordRole, paramtable.GetNodeID()),
			// TODO(dragondriver): fill ConnectedComponents if necessary
			ConnectedComponents: []metricsinfo.ConnectionInfo{},
		},
	}
}

// getDataCoordMetrics composes datacoord infos
func (s *Server) getDataCoordMetrics(ctx context.Context) metricsinfo.DataCoordInfos {
	used, total, err := hardware.GetDiskUsage(paramtable.Get().LocalStorageCfg.Path.GetValue())
	if err != nil {
		mlog.Warn(ctx, "get disk usage failed", mlog.Err(err))
	}

	ioWait, err := hardware.GetIOWait()
	if err != nil {
		mlog.Warn(ctx, "get iowait failed", mlog.Err(err))
	}

	ret := metricsinfo.DataCoordInfos{
		BaseComponentInfos: metricsinfo.BaseComponentInfos{
			Name: metricsinfo.ConstructComponentName(typeutil.DataCoordRole, paramtable.GetNodeID()),
			HardwareInfos: metricsinfo.HardwareMetrics{
				IP:               s.session.GetAddress(),
				CPUCoreCount:     hardware.GetCPUNum(),
				CPUCoreUsage:     hardware.GetCPUUsage(),
				Memory:           hardware.GetMemoryCount(),
				MemoryUsage:      hardware.GetUsedMemoryCount(),
				Disk:             total,
				DiskUsage:        used,
				IOWaitPercentage: ioWait,
			},
			SystemInfo:  metricsinfo.DeployMetrics{},
			CreatedTime: paramtable.GetCreateTime().String(),
			UpdatedTime: paramtable.GetUpdateTime().String(),
			Type:        typeutil.DataCoordRole,
			ID:          paramtable.GetNodeID(),
		},
		SystemConfigurations: metricsinfo.DataCoordConfiguration{
			SegmentMaxSize: Params.DataCoordCfg.SegmentMaxSize.GetAsFloat(),
		},
		QuotaMetrics:      s.getQuotaMetrics(),
		CollectionMetrics: s.getCollectionMetrics(ctx),
	}

	metricsinfo.FillDeployMetricsWithEnv(&ret.SystemInfo)

	return ret
}

type objectStorageMetricsSample struct {
	ts      time.Time
	metrics *metricsinfo.ObjectStorageMetrics
}

type objectStorageMetricKey struct {
	storageType   string
	cloudProvider string
	operation     string
}

func (s *Server) updateDataNodeObjectStorageMetrics(nodeID int64, cur *metricsinfo.ObjectStorageMetrics) *metricsinfo.ObjectStorageDerivedMetrics {
	if cur == nil {
		return nil
	}

	now := time.Now()
	s.objectStorageMetricsMu.Lock()
	defer s.objectStorageMetricsMu.Unlock()
	if s.objectStorageMetricsSamples == nil {
		s.objectStorageMetricsSamples = make(map[int64]objectStorageMetricsSample)
	}

	prev, ok := s.objectStorageMetricsSamples[nodeID]
	s.objectStorageMetricsSamples[nodeID] = objectStorageMetricsSample{
		ts:      now,
		metrics: cloneObjectStorageMetrics(cur),
	}
	if !ok {
		return &metricsinfo.ObjectStorageDerivedMetrics{
			SampleReady:           false,
			LatencyBucketBoundsMs: append([]float64(nil), cur.LatencyBucketBoundsMs...),
		}
	}

	windowSeconds := now.Sub(prev.ts).Seconds()
	if windowSeconds <= 0 {
		return &metricsinfo.ObjectStorageDerivedMetrics{
			SampleReady:           false,
			ResetDetected:         true,
			LatencyBucketBoundsMs: append([]float64(nil), cur.LatencyBucketBoundsMs...),
		}
	}

	derived, reset := calculateObjectStorageDerivedMetrics(prev.metrics, cur, windowSeconds)
	if reset {
		derived.SampleReady = false
		derived.ResetDetected = true
	}
	return derived
}

func (s *Server) cleanupObjectStorageMetricSamples(nodes []int64) {
	nodeSet := make(map[int64]struct{}, len(nodes))
	for _, node := range nodes {
		nodeSet[node] = struct{}{}
	}

	s.objectStorageMetricsMu.Lock()
	defer s.objectStorageMetricsMu.Unlock()
	for node := range s.objectStorageMetricsSamples {
		if _, ok := nodeSet[node]; !ok {
			delete(s.objectStorageMetricsSamples, node)
		}
	}
}

func calculateObjectStorageDerivedMetrics(prev, cur *metricsinfo.ObjectStorageMetrics, windowSeconds float64) (*metricsinfo.ObjectStorageDerivedMetrics, bool) {
	if prev == nil || cur == nil || windowSeconds <= 0 {
		return &metricsinfo.ObjectStorageDerivedMetrics{SampleReady: false}, true
	}
	ret := &metricsinfo.ObjectStorageDerivedMetrics{
		SampleReady:           false,
		LatencyBucketBoundsMs: append([]float64(nil), cur.LatencyBucketBoundsMs...),
	}

	prevOps := objectStorageRawOpMap(prev.Ops)
	derivedOps := make([]metricsinfo.ObjectStorageDerivedOpMetrics, 0, len(cur.Ops))
	for _, curOp := range cur.Ops {
		key := objectStorageOpKey(curOp.StorageType, curOp.CloudProvider, curOp.Operation)
		prevOp, ok := prevOps[key]
		if !ok {
			continue
		}
		derivedOp, reset := calculateObjectStorageDerivedOp(prevOp, curOp, cur.LatencyBucketBoundsMs, windowSeconds)
		if reset {
			return ret, true
		}
		derivedOps = append(derivedOps, derivedOp)
	}

	sort.Slice(derivedOps, func(i, j int) bool {
		if derivedOps[i].CloudProvider != derivedOps[j].CloudProvider {
			return derivedOps[i].CloudProvider < derivedOps[j].CloudProvider
		}
		return derivedOps[i].Operation < derivedOps[j].Operation
	})
	ret.Ops = derivedOps
	ret.SampleReady = len(derivedOps) > 0
	return ret, false
}

func calculateObjectStorageDerivedOp(
	prev, cur metricsinfo.ObjectStorageOpMetrics,
	bounds []float64,
	windowSeconds float64,
) (metricsinfo.ObjectStorageDerivedOpMetrics, bool) {
	op := metricsinfo.ObjectStorageDerivedOpMetrics{
		StorageType:   cur.StorageType,
		CloudProvider: cur.CloudProvider,
		Operation:     cur.Operation,
		WindowSeconds: windowSeconds,
	}

	op.TotalRequests = cur.TotalRequests - prev.TotalRequests
	op.SuccessRequests = cur.SuccessRequests - prev.SuccessRequests
	op.FailedRequests = cur.FailedRequests - prev.FailedRequests
	op.CanceledRequests = cur.CanceledRequests - prev.CanceledRequests
	op.BytesRead = cur.BytesRead - prev.BytesRead
	op.BytesWritten = cur.BytesWritten - prev.BytesWritten
	totalLatencyMs := cur.TotalLatencyMs - prev.TotalLatencyMs
	op.LatencyBucketCounts = deltaObjectStorageBuckets(prev.LatencyBucketCounts, cur.LatencyBucketCounts)

	if op.TotalRequests < 0 || op.SuccessRequests < 0 || op.FailedRequests < 0 ||
		op.CanceledRequests < 0 || op.BytesRead < 0 || op.BytesWritten < 0 ||
		totalLatencyMs < 0 || hasNegativeObjectStorageBucket(op.LatencyBucketCounts) {
		return op, true
	}

	op.QPS = float64(op.TotalRequests) / windowSeconds
	if op.TotalRequests > 0 {
		op.SuccessRate = float64(op.SuccessRequests) / float64(op.TotalRequests)
		op.FailureRate = float64(op.FailedRequests) / float64(op.TotalRequests)
		op.CancelRate = float64(op.CanceledRequests) / float64(op.TotalRequests)
		op.AvgLatencyMs = float64(totalLatencyMs) / float64(op.TotalRequests)
		op.P95LatencyMs = objectStorageQuantile(bounds, op.LatencyBucketCounts, 0.95)
		op.P99LatencyMs = objectStorageQuantile(bounds, op.LatencyBucketCounts, 0.99)
	}
	return op, false
}

func aggregateObjectStorageDerivedMetrics(nodes []metricsinfo.DataNodeInfos) *metricsinfo.ObjectStorageDerivedMetrics {
	var bounds []float64
	ops := make(map[objectStorageMetricKey]metricsinfo.ObjectStorageDerivedOpMetrics)
	sampleReady := false
	resetDetected := false

	for _, node := range nodes {
		metrics := node.ObjectStorageDerivedMetrics
		if metrics == nil {
			continue
		}
		if len(bounds) == 0 {
			bounds = append([]float64(nil), metrics.LatencyBucketBoundsMs...)
		}
		resetDetected = resetDetected || metrics.ResetDetected
		if !metrics.SampleReady {
			continue
		}
		sampleReady = true
		for _, op := range metrics.Ops {
			key := objectStorageOpKey(op.StorageType, op.CloudProvider, op.Operation)
			agg := ops[key]
			if agg.StorageType == "" {
				agg.StorageType = op.StorageType
				agg.CloudProvider = op.CloudProvider
				agg.Operation = op.Operation
				agg.LatencyBucketCounts = make([]int64, len(op.LatencyBucketCounts))
			}
			agg.WindowSeconds = math.Max(agg.WindowSeconds, op.WindowSeconds)
			agg.QPS += op.QPS
			agg.TotalRequests += op.TotalRequests
			agg.SuccessRequests += op.SuccessRequests
			agg.FailedRequests += op.FailedRequests
			agg.CanceledRequests += op.CanceledRequests
			agg.BytesRead += op.BytesRead
			agg.BytesWritten += op.BytesWritten
			agg.AvgLatencyMs += op.AvgLatencyMs * float64(op.TotalRequests)
			for i := range op.LatencyBucketCounts {
				if i < len(agg.LatencyBucketCounts) {
					agg.LatencyBucketCounts[i] += op.LatencyBucketCounts[i]
				}
			}
			ops[key] = agg
		}
	}

	ret := &metricsinfo.ObjectStorageDerivedMetrics{
		SampleReady:           sampleReady,
		ResetDetected:         resetDetected,
		LatencyBucketBoundsMs: bounds,
		Ops:                   make([]metricsinfo.ObjectStorageDerivedOpMetrics, 0, len(ops)),
	}
	for _, op := range ops {
		if op.TotalRequests > 0 {
			op.SuccessRate = float64(op.SuccessRequests) / float64(op.TotalRequests)
			op.FailureRate = float64(op.FailedRequests) / float64(op.TotalRequests)
			op.CancelRate = float64(op.CanceledRequests) / float64(op.TotalRequests)
			op.AvgLatencyMs = op.AvgLatencyMs / float64(op.TotalRequests)
			op.P95LatencyMs = objectStorageQuantile(bounds, op.LatencyBucketCounts, 0.95)
			op.P99LatencyMs = objectStorageQuantile(bounds, op.LatencyBucketCounts, 0.99)
		}
		ret.Ops = append(ret.Ops, op)
	}
	sort.Slice(ret.Ops, func(i, j int) bool {
		if ret.Ops[i].CloudProvider != ret.Ops[j].CloudProvider {
			return ret.Ops[i].CloudProvider < ret.Ops[j].CloudProvider
		}
		return ret.Ops[i].Operation < ret.Ops[j].Operation
	})
	return ret
}

func objectStorageRawOpMap(ops []metricsinfo.ObjectStorageOpMetrics) map[objectStorageMetricKey]metricsinfo.ObjectStorageOpMetrics {
	ret := make(map[objectStorageMetricKey]metricsinfo.ObjectStorageOpMetrics, len(ops))
	for _, op := range ops {
		ret[objectStorageOpKey(op.StorageType, op.CloudProvider, op.Operation)] = op
	}
	return ret
}

func objectStorageOpKey(storageType, cloudProvider, operation string) objectStorageMetricKey {
	return objectStorageMetricKey{
		storageType:   storageType,
		cloudProvider: cloudProvider,
		operation:     operation,
	}
}

func deltaObjectStorageBuckets(prev, cur []int64) []int64 {
	buckets := make([]int64, len(cur))
	for i := range cur {
		prevValue := int64(0)
		if i < len(prev) {
			prevValue = prev[i]
		}
		buckets[i] = cur[i] - prevValue
	}
	return buckets
}

func hasNegativeObjectStorageBucket(buckets []int64) bool {
	for _, bucket := range buckets {
		if bucket < 0 {
			return true
		}
	}
	return false
}

func objectStorageQuantile(bounds []float64, cumulativeBuckets []int64, quantile float64) float64 {
	if len(bounds) == 0 || len(cumulativeBuckets) == 0 || quantile <= 0 {
		return 0
	}
	total := cumulativeBuckets[len(cumulativeBuckets)-1]
	if total <= 0 {
		return 0
	}
	rank := int64(math.Ceil(float64(total) * quantile))
	for i, count := range cumulativeBuckets {
		if count >= rank && i < len(bounds) {
			return bounds[i]
		}
	}
	return bounds[len(bounds)-1]
}

func cloneObjectStorageMetrics(in *metricsinfo.ObjectStorageMetrics) *metricsinfo.ObjectStorageMetrics {
	if in == nil {
		return nil
	}
	out := &metricsinfo.ObjectStorageMetrics{
		LatencyBucketBoundsMs: append([]float64(nil), in.LatencyBucketBoundsMs...),
		Ops:                   make([]metricsinfo.ObjectStorageOpMetrics, 0, len(in.Ops)),
	}
	for _, op := range in.Ops {
		copied := op
		copied.LatencyBucketCounts = append([]int64(nil), op.LatencyBucketCounts...)
		out.Ops = append(out.Ops, copied)
	}
	return out
}

// getDataNodeMetrics composes DataNode infos
// this function will invoke GetMetrics with DataNode specified in NodeInfo
func (s *Server) getDataNodeMetrics(ctx context.Context, req *milvuspb.GetMetricsRequest, node int64) (metricsinfo.DataNodeInfos, error) {
	infos := metricsinfo.DataNodeInfos{
		BaseComponentInfos: metricsinfo.BaseComponentInfos{
			HasError: true,
			ID:       int64(uniquegenerator.GetUniqueIntGeneratorIns().GetInt()),
		},
	}
	cli, err := s.nodeManager.GetClient(node)
	if err != nil {
		return infos, err
	}

	metrics, err := cli.GetMetrics(ctx, req)
	if err != nil {
		mlog.Warn(ctx, "invalid metrics of DataNode was found",
			mlog.Err(err))
		infos.ErrorReason = err.Error()
		// err handled, returns nil
		return infos, nil
	}
	infos.Name = metrics.GetComponentName()

	if metrics.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
		mlog.Warn(ctx, "invalid metrics of DataNode was found",
			mlog.Any("error_code", metrics.GetStatus().GetErrorCode()),
			mlog.Any("error_reason", metrics.GetStatus().GetReason()))
		infos.ErrorReason = metrics.GetStatus().GetReason()
		return infos, nil
	}

	err = metricsinfo.UnmarshalComponentInfos(metrics.GetResponse(), &infos)
	if err != nil {
		mlog.Warn(ctx, "invalid metrics of DataNode found",
			mlog.Err(err))
		infos.ErrorReason = err.Error()
		return infos, nil
	}
	infos.HasError = false
	return infos, nil
}

func (s *Server) getIndexNodeMetrics(ctx context.Context, req *milvuspb.GetMetricsRequest, node types.DataNodeClient) (metricsinfo.DataNodeInfos, error) {
	infos := metricsinfo.DataNodeInfos{
		BaseComponentInfos: metricsinfo.BaseComponentInfos{
			HasError: true,
			ID:       int64(uniquegenerator.GetUniqueIntGeneratorIns().GetInt()),
		},
	}
	if node == nil {
		return infos, merr.WrapErrServiceInternalMsg("index node is nil")
	}

	metrics, err := node.GetMetrics(ctx, req)
	if err != nil {
		mlog.Warn(ctx, "invalid metrics of IndexNode was found",
			mlog.Err(err))
		infos.ErrorReason = err.Error()
		// err handled, returns nil
		return infos, nil
	}
	infos.Name = metrics.GetComponentName()

	if metrics.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
		mlog.Warn(ctx, "invalid metrics of DataNode was found",
			mlog.Any("error_code", metrics.GetStatus().GetErrorCode()),
			mlog.Any("error_reason", metrics.GetStatus().GetReason()))
		infos.ErrorReason = metrics.GetStatus().GetReason()
		return infos, nil
	}

	err = metricsinfo.UnmarshalComponentInfos(metrics.GetResponse(), &infos)
	if err != nil {
		mlog.Warn(ctx, "invalid metrics of DataNode found",
			mlog.Err(err))
		infos.ErrorReason = err.Error()
		return infos, nil
	}
	infos.HasError = false
	return infos, nil
}

// getMetrics retrieves and aggregates the metrics of the datanode to a slice
func getMetrics[T any](ctx context.Context, s *Server, req *milvuspb.GetMetricsRequest) ([]T, error) {
	var metrics []T
	var mu sync.Mutex
	errorGroup, ctx := errgroup.WithContext(ctx)

	nodes := s.nodeManager.GetClientIDs()
	for _, node := range nodes {
		errorGroup.Go(func() error {
			cli, err := s.nodeManager.GetClient(node)
			if err != nil {
				return err
			}
			resp, err := cli.GetMetrics(ctx, req)
			if err != nil {
				mlog.Warn(ctx, "failed to get metric from DataNode", mlog.FieldNodeID(node))
				return err
			}

			if resp.Response == "" {
				return nil
			}

			var infos []T
			err = json.Unmarshal([]byte(resp.Response), &infos)
			if err != nil {
				mlog.Warn(ctx, "invalid metrics of data node was found", mlog.Err(err))
				return err
			}

			mu.Lock()
			metrics = append(metrics, infos...)
			mu.Unlock()
			return nil
		})
	}

	err := errorGroup.Wait()
	return metrics, err
}

// GetDataCoordTopology returns DataCoord topology directly without JSON serialization
// This is optimized for in-process calls in MixCoord mode to avoid marshal/unmarshal overhead
func (s *Server) GetDataCoordTopology(ctx context.Context, req *milvuspb.GetMetricsRequest) (*metricsinfo.DataCoordTopology, error) {
	if err := merr.CheckHealthy(s.GetStateCode()); err != nil {
		return nil, err
	}
	topology := s.getDataCoordTopology(ctx, req)
	return &topology, nil
}
