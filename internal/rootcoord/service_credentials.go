package rootcoord

import (
	"context"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metrics"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/proxypb"
	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"
	"github.com/milvus-io/milvus/internal/util/timerecord"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"go.uber.org/zap"
)

// ExpireCredCache will call invalidate credential cache
func (c *RootCoord) ExpireCredCache(ctx context.Context, username string) error {
	req := proxypb.InvalidateCredCacheRequest{
		Base: &commonpb.MsgBase{
			MsgType:  0, //TODO, msg type
			MsgID:    0, //TODO, msg id
			SourceID: c.session.ServerID,
		},
		Username: username,
	}
	return c.proxyClientManager.InvalidateCredentialCache(ctx, &req)
}

// UpdateCredCache will call update credential cache
func (c *RootCoord) UpdateCredCache(ctx context.Context, credInfo *internalpb.CredentialInfo) error {
	req := proxypb.UpdateCredCacheRequest{
		Base: &commonpb.MsgBase{
			MsgType:  0, //TODO, msg type
			MsgID:    0, //TODO, msg id
			SourceID: c.session.ServerID,
		},
		Username: credInfo.Username,
		Password: credInfo.Sha256Password,
	}
	return c.proxyClientManager.UpdateCredentialCache(ctx, &req)
}

// CreateCredential create new user and password
// 	1. decode ciphertext password to raw password
// 	2. encrypt raw password
// 	3. save in to etcd
func (c *RootCoord) CreateCredential(ctx context.Context, credInfo *internalpb.CredentialInfo) (*commonpb.Status, error) {
	method := "CreateCredential"
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.TotalLabel).Inc()
	tr := timerecord.NewTimeRecorder(method)
	log.Debug("CreateCredential", zap.String("role", typeutil.RootCoordRole),
		zap.String("username", credInfo.Username))

	usersInfo, err := c.meta.ListCredentialUsernames()
	if err != nil {
		log.Error("CreateCredential get credential username list failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("username", credInfo.Username), zap.Error(err))
		return failStatus(commonpb.ErrorCode_CreateCredentialFailure, "CreateCredential failed, fail to get credential username list to check the user number, error: "+err.Error()), nil
	}
	if len(usersInfo.Usernames) >= Params.ProxyCfg.MaxUserNum {
		return failStatus(commonpb.ErrorCode_CreateCredentialFailure, "unable to add user because the number of users has reached the limit"), nil
	}

	if cred, _ := c.meta.GetCredential(credInfo.Username); cred != nil {
		return failStatus(commonpb.ErrorCode_CreateCredentialFailure, "user already exists:"+credInfo.Username), nil
	}
	// insert to db
	err = c.meta.AddCredential(credInfo)
	if err != nil {
		log.Error("CreateCredential save credential failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("username", credInfo.Username), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.FailLabel).Inc()
		return failStatus(commonpb.ErrorCode_CreateCredentialFailure, "CreateCredential failed: "+err.Error()), nil
	}
	// update proxy's local cache
	err = c.UpdateCredCache(ctx, credInfo)
	if err != nil {
		log.Warn("CreateCredential add cache failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("username", credInfo.Username), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.FailLabel).Inc()
	}
	log.Debug("CreateCredential success", zap.String("role", typeutil.RootCoordRole),
		zap.String("username", credInfo.Username))

	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues(method).Observe(float64(tr.ElapseSpan().Milliseconds()))
	metrics.RootCoordNumOfCredentials.Inc()
	return succStatus(), nil
}

// GetCredential get credential by username
func (c *RootCoord) GetCredential(ctx context.Context, in *rootcoordpb.GetCredentialRequest) (*rootcoordpb.GetCredentialResponse, error) {
	method := "GetCredential"
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.TotalLabel).Inc()
	tr := timerecord.NewTimeRecorder(method)
	log.Debug("GetCredential", zap.String("role", typeutil.RootCoordRole),
		zap.String("username", in.Username))

	credInfo, err := c.meta.GetCredential(in.Username)
	if err != nil {
		log.Error("GetCredential query credential failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("username", in.Username), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.FailLabel).Inc()
		return &rootcoordpb.GetCredentialResponse{
			Status: failStatus(commonpb.ErrorCode_GetCredentialFailure, "GetCredential failed: "+err.Error()),
		}, err
	}
	log.Debug("GetCredential success", zap.String("role", typeutil.RootCoordRole),
		zap.String("username", in.Username))

	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues(method).Observe(float64(tr.ElapseSpan().Milliseconds()))
	return &rootcoordpb.GetCredentialResponse{
		Status:   succStatus(),
		Username: credInfo.Username,
		Password: credInfo.EncryptedPassword,
	}, nil
}

// UpdateCredential update password for a user
func (c *RootCoord) UpdateCredential(ctx context.Context, credInfo *internalpb.CredentialInfo) (*commonpb.Status, error) {
	method := "UpdateCredential"
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.TotalLabel).Inc()
	tr := timerecord.NewTimeRecorder(method)
	log.Debug("UpdateCredential", zap.String("role", typeutil.RootCoordRole),
		zap.String("username", credInfo.Username))
	// update data on storage
	err := c.meta.AddCredential(credInfo)
	if err != nil {
		log.Error("UpdateCredential save credential failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("username", credInfo.Username), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.FailLabel).Inc()
		return failStatus(commonpb.ErrorCode_UpdateCredentialFailure, "UpdateCredential failed: "+err.Error()), nil
	}
	// update proxy's local cache
	err = c.UpdateCredCache(ctx, credInfo)
	if err != nil {
		log.Error("UpdateCredential update cache failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("username", credInfo.Username), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.FailLabel).Inc()
		return failStatus(commonpb.ErrorCode_UpdateCredentialFailure, "UpdateCredential failed: "+err.Error()), nil
	}
	log.Debug("UpdateCredential success", zap.String("role", typeutil.RootCoordRole),
		zap.String("username", credInfo.Username))

	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues(method).Observe(float64(tr.ElapseSpan().Milliseconds()))
	return succStatus(), nil
}

// DeleteCredential delete a user
func (c *RootCoord) DeleteCredential(ctx context.Context, in *milvuspb.DeleteCredentialRequest) (*commonpb.Status, error) {
	method := "DeleteCredential"
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.TotalLabel).Inc()
	tr := timerecord.NewTimeRecorder(method)

	// delete data on storage
	err := c.meta.DeleteCredential(in.Username)
	if err != nil {
		log.Error("DeleteCredential remove credential failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("username", in.Username), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.FailLabel).Inc()
		return failStatus(commonpb.ErrorCode_DeleteCredentialFailure, "DeleteCredential failed: "+err.Error()), err
	}
	// invalidate proxy's local cache
	err = c.ExpireCredCache(ctx, in.Username)
	if err != nil {
		log.Error("DeleteCredential expire credential cache failed", zap.String("role", typeutil.RootCoordRole),
			zap.String("username", in.Username), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.FailLabel).Inc()
		return failStatus(commonpb.ErrorCode_DeleteCredentialFailure, "DeleteCredential failed: "+err.Error()), nil
	}
	log.Debug("DeleteCredential success", zap.String("role", typeutil.RootCoordRole),
		zap.String("username", in.Username))

	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues(method).Observe(float64(tr.ElapseSpan().Milliseconds()))
	metrics.RootCoordNumOfCredentials.Dec()
	return succStatus(), nil
}

// ListCredUsers list all usernames
func (c *RootCoord) ListCredUsers(ctx context.Context, in *milvuspb.ListCredUsersRequest) (*milvuspb.ListCredUsersResponse, error) {
	method := "ListCredUsers"
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.TotalLabel).Inc()
	tr := timerecord.NewTimeRecorder(method)

	credInfo, err := c.meta.ListCredentialUsernames()
	if err != nil {
		log.Error("ListCredUsers query usernames failed", zap.String("role", typeutil.RootCoordRole),
			zap.Int64("msgID", in.Base.MsgID), zap.Error(err))
		metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.FailLabel).Inc()
		return &milvuspb.ListCredUsersResponse{
			Status: failStatus(commonpb.ErrorCode_ListCredUsersFailure, "ListCredUsers failed: "+err.Error()),
		}, err
	}
	log.Debug("ListCredUsers success", zap.String("role", typeutil.RootCoordRole))

	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues(method).Observe(float64(tr.ElapseSpan().Milliseconds()))
	return &milvuspb.ListCredUsersResponse{
		Status:    succStatus(),
		Usernames: credInfo.Usernames,
	}, nil
}
