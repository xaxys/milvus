package rootcoord

import (
	"context"
	"errors"
	"fmt"

	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metrics"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/proxypb"
	"github.com/milvus-io/milvus/internal/util"
	"github.com/milvus-io/milvus/internal/util/errorutil"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/timerecord"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"go.uber.org/zap"
)

// CreateRole create role
// - check the node health
// - check if the role is existed
// - check if the role num has reached the limit
// - create the role by the meta api
func (c *RootCoord) CreateRole(ctx context.Context, in *milvuspb.CreateRoleRequest) (*commonpb.Status, error) {
	method := "CreateRole"
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.TotalLabel).Inc()
	tr := timerecord.NewTimeRecorder(method)
	logger.Debug(method, zap.Any("in", in))

	if code, ok := c.checkHealthy(); !ok {
		return errorutil.UnhealthyStatus(code), errorutil.UnhealthyError()
	}
	entity := in.Entity
	_, err := c.meta.SelectRole(util.DefaultTenant, &milvuspb.RoleEntity{Name: entity.Name}, false)
	if err == nil {
		errMsg := "role already exists:" + entity.Name
		return failStatus(commonpb.ErrorCode_CreateRoleFailure, errMsg), errors.New(errMsg)
	}
	if !common.IsKeyNotExistError(err) {
		return failStatus(commonpb.ErrorCode_CreateRoleFailure, err.Error()), err
	}

	results, err := c.meta.SelectRole(util.DefaultTenant, nil, false)
	if err != nil {
		logger.Error("fail to select roles", zap.Error(err))
		return failStatus(commonpb.ErrorCode_CreateRoleFailure, "fail to select roles to check the role number, error: "+err.Error()), err
	}
	if len(results) >= Params.ProxyCfg.MaxRoleNum {
		errMsg := "unable to add role because the number of roles has reached the limit"
		return failStatus(commonpb.ErrorCode_CreateRoleFailure, errMsg), errors.New(errMsg)
	}

	err = c.meta.CreateRole(util.DefaultTenant, &milvuspb.RoleEntity{Name: entity.Name})
	if err != nil {
		logger.Error("fail to create role", zap.String("role_name", entity.Name), zap.Error(err))
		return failStatus(commonpb.ErrorCode_CreateRoleFailure, "CreateCollection role failed: "+err.Error()), err
	}

	logger.Debug(method+" success", zap.String("role_name", entity.Name))
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues(method).Observe(float64(tr.ElapseSpan().Milliseconds()))
	metrics.RootCoordNumOfRoles.Inc()

	return succStatus(), nil
}

// DropRole drop role
// - check the node health
// - check if the role name is existed
// - check if the role has some grant info
// - get all role mapping of this role
// - drop these role mappings
// - drop the role by the meta api
func (c *RootCoord) DropRole(ctx context.Context, in *milvuspb.DropRoleRequest) (*commonpb.Status, error) {
	method := "DropRole"
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.TotalLabel).Inc()
	tr := timerecord.NewTimeRecorder(method)
	logger.Debug(method, zap.Any("in", in))

	if code, ok := c.checkHealthy(); !ok {
		return errorutil.UnhealthyStatus(code), errorutil.UnhealthyError()
	}
	if _, err := c.meta.SelectRole(util.DefaultTenant, &milvuspb.RoleEntity{Name: in.RoleName}, false); err != nil {
		logger.Error("the role isn't existed", zap.String("role_name", in.RoleName), zap.Error(err))
		return failStatus(commonpb.ErrorCode_DropRoleFailure, fmt.Sprintf("the role isn't existed, role name: %s", in.RoleName)), err
	}

	grantEntities, err := c.meta.SelectGrant(util.DefaultTenant, &milvuspb.GrantEntity{
		Role: &milvuspb.RoleEntity{Name: in.RoleName},
	})
	if len(grantEntities) != 0 {
		errMsg := "fail to drop the role that it has privileges. Use REVOKE API to revoke privileges"
		logger.Error(errMsg, zap.String("role_name", in.RoleName), zap.Error(err))
		return failStatus(commonpb.ErrorCode_DropRoleFailure, errMsg), errors.New(errMsg)
	}
	roleResults, err := c.meta.SelectRole(util.DefaultTenant, &milvuspb.RoleEntity{Name: in.RoleName}, true)
	if err != nil {
		errMsg := "fail to select a role by role name"
		logger.Error("fail to select a role by role name", zap.String("role_name", in.RoleName), zap.Error(err))
		return failStatus(commonpb.ErrorCode_DropRoleFailure, errMsg), err
	}
	logger.Debug("role to user info", zap.Int("counter", len(roleResults)))
	for _, roleResult := range roleResults {
		for index, userEntity := range roleResult.Users {
			if err = c.meta.OperateUserRole(util.DefaultTenant, &milvuspb.UserEntity{Name: userEntity.Name}, &milvuspb.RoleEntity{Name: roleResult.Role.Name}, milvuspb.OperateUserRoleType_RemoveUserFromRole); err != nil {
				errMsg := "fail to remove user from role"
				logger.Error(errMsg, zap.String("role_name", roleResult.Role.Name), zap.String("username", userEntity.Name), zap.Int("current_index", index), zap.Error(err))
				return failStatus(commonpb.ErrorCode_OperateUserRoleFailure, errMsg), err
			}
		}
	}
	if err = c.meta.DropRole(util.DefaultTenant, in.RoleName); err != nil {
		errMsg := "fail to drop the role"
		logger.Error(errMsg, zap.String("role_name", in.RoleName), zap.Error(err))
		return failStatus(commonpb.ErrorCode_DropRoleFailure, errMsg), err
	}

	logger.Debug(method+" success", zap.String("role_name", in.RoleName))
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues(method).Observe(float64(tr.ElapseSpan().Milliseconds()))
	metrics.RootCoordNumOfRoles.Dec()
	return succStatus(), nil
}

// OperateUserRole operate the relationship between a user and a role
// - check the node health
// - check if the role is valid
// - check if the user is valid
// - operate the user-role by the meta api
// - update the policy cache
func (c *RootCoord) OperateUserRole(ctx context.Context, in *milvuspb.OperateUserRoleRequest) (*commonpb.Status, error) {
	method := "OperateUserRole-" + in.Type.String()
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.TotalLabel).Inc()
	tr := timerecord.NewTimeRecorder(method)
	logger.Debug(method, zap.Any("in", in))

	if code, ok := c.checkHealthy(); !ok {
		return errorutil.UnhealthyStatus(code), errorutil.UnhealthyError()
	}

	if _, err := c.meta.SelectRole(util.DefaultTenant, &milvuspb.RoleEntity{Name: in.RoleName}, false); err != nil {
		errMsg := "fail to check the role name"
		logger.Error(errMsg, zap.String("role_name", in.RoleName), zap.Error(err))
		return failStatus(commonpb.ErrorCode_OperateUserRoleFailure, errMsg), err
	}
	if _, err := c.meta.SelectUser(util.DefaultTenant, &milvuspb.UserEntity{Name: in.Username}, false); err != nil {
		errMsg := "fail to check the username"
		logger.Error(errMsg, zap.String("username", in.Username), zap.Error(err))
		return failStatus(commonpb.ErrorCode_OperateUserRoleFailure, errMsg), err
	}
	if err := c.meta.OperateUserRole(util.DefaultTenant, &milvuspb.UserEntity{Name: in.Username}, &milvuspb.RoleEntity{Name: in.RoleName}, in.Type); err != nil {
		errMsg := "fail to operate user to role"
		logger.Error(errMsg, zap.String("role_name", in.RoleName), zap.String("username", in.Username), zap.Error(err))
		return failStatus(commonpb.ErrorCode_OperateUserRoleFailure, errMsg), err
	}

	var opType int32
	if in.Type == milvuspb.OperateUserRoleType_AddUserToRole {
		opType = int32(typeutil.CacheAddUserToRole)
	} else if in.Type == milvuspb.OperateUserRoleType_RemoveUserFromRole {
		opType = int32(typeutil.CacheRemoveUserFromRole)
	}
	if err := c.proxyClientManager.RefreshPolicyInfoCache(ctx, &proxypb.RefreshPolicyInfoCacheRequest{
		OpType: opType,
		OpKey:  funcutil.EncodeUserRoleCache(in.Username, in.RoleName),
	}); err != nil {
		return failStatus(commonpb.ErrorCode_OperateUserRoleFailure, err.Error()), err
	}

	logger.Debug(method + " success")
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues(method).Observe(float64(tr.ElapseSpan().Milliseconds()))
	return succStatus(), nil
}

// SelectRole select role
// - check the node health
// - check if the role is valid when this param is provided
// - select role by the meta api
func (c *RootCoord) SelectRole(ctx context.Context, in *milvuspb.SelectRoleRequest) (*milvuspb.SelectRoleResponse, error) {
	method := "SelectRole"
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.TotalLabel).Inc()
	tr := timerecord.NewTimeRecorder(method)
	logger.Debug(method, zap.Any("in", in))

	if code, ok := c.checkHealthy(); !ok {
		return &milvuspb.SelectRoleResponse{Status: errorutil.UnhealthyStatus(code)}, errorutil.UnhealthyError()
	}

	if in.Role != nil {
		if _, err := c.meta.SelectRole(util.DefaultTenant, &milvuspb.RoleEntity{Name: in.Role.Name}, false); err != nil {
			errMsg := "fail to select the role to check the role name"
			logger.Error(errMsg, zap.String("role_name", in.Role.Name), zap.Error(err))
			if common.IsKeyNotExistError(err) {
				return &milvuspb.SelectRoleResponse{
					Status: succStatus(),
				}, nil
			}
			return &milvuspb.SelectRoleResponse{
				Status: failStatus(commonpb.ErrorCode_SelectRoleFailure, errMsg),
			}, err
		}
	}
	roleResults, err := c.meta.SelectRole(util.DefaultTenant, in.Role, in.IncludeUserInfo)
	if err != nil {
		errMsg := "fail to select the role"
		logger.Error(errMsg, zap.Error(err))
		return &milvuspb.SelectRoleResponse{
			Status: failStatus(commonpb.ErrorCode_SelectRoleFailure, errMsg),
		}, err
	}

	logger.Debug(method + " success")
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues(method).Observe(float64(tr.ElapseSpan().Milliseconds()))
	return &milvuspb.SelectRoleResponse{
		Status:  succStatus(),
		Results: roleResults,
	}, nil
}

// SelectUser select user
// - check the node health
// - check if the user is valid when this param is provided
// - select user by the meta api
func (c *RootCoord) SelectUser(ctx context.Context, in *milvuspb.SelectUserRequest) (*milvuspb.SelectUserResponse, error) {
	method := "SelectUser"
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.TotalLabel).Inc()
	tr := timerecord.NewTimeRecorder(method)
	logger.Debug(method, zap.Any("in", in))

	if code, ok := c.checkHealthy(); !ok {
		return &milvuspb.SelectUserResponse{Status: errorutil.UnhealthyStatus(code)}, errorutil.UnhealthyError()
	}

	if in.User != nil {
		if _, err := c.meta.SelectUser(util.DefaultTenant, &milvuspb.UserEntity{Name: in.User.Name}, false); err != nil {
			errMsg := "fail to select the user to check the username"
			logger.Error(errMsg, zap.String("username", in.User.Name), zap.Error(err))
			if common.IsKeyNotExistError(err) {
				return &milvuspb.SelectUserResponse{
					Status: succStatus(),
				}, nil
			}
			return &milvuspb.SelectUserResponse{
				Status: failStatus(commonpb.ErrorCode_SelectUserFailure, errMsg),
			}, err
		}
	}
	userResults, err := c.meta.SelectUser(util.DefaultTenant, in.User, in.IncludeRoleInfo)
	if err != nil {
		errMsg := "fail to select the user"
		log.Error(errMsg, zap.Error(err))
		return &milvuspb.SelectUserResponse{
			Status: failStatus(commonpb.ErrorCode_SelectUserFailure, errMsg),
		}, err
	}

	logger.Debug(method + " success")
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues(method).Observe(float64(tr.ElapseSpan().Milliseconds()))
	return &milvuspb.SelectUserResponse{
		Status:  succStatus(),
		Results: userResults,
	}, nil
}

func (c *RootCoord) isValidRole(entity *milvuspb.RoleEntity) error {
	if entity == nil {
		return fmt.Errorf("the role entity is nil")
	}
	if entity.Name == "" {
		return fmt.Errorf("the name in the role entity is empty")
	}
	if _, err := c.meta.SelectRole(util.DefaultTenant, &milvuspb.RoleEntity{Name: entity.Name}, false); err != nil {
		return err
	}
	return nil
}

func (c *RootCoord) isValidObject(entity *milvuspb.ObjectEntity) error {
	if entity == nil {
		return fmt.Errorf("the object entity is nil")
	}
	if _, ok := commonpb.ObjectType_value[entity.Name]; !ok {
		return fmt.Errorf("the object type in the object entity is invalid, current value: %s", entity.Name)
	}
	return nil
}

func (c *RootCoord) isValidGrantor(entity *milvuspb.GrantorEntity, object string) error {
	if entity == nil {
		return fmt.Errorf("the grantor entity is nil")
	}
	if entity.User == nil {
		return fmt.Errorf("the user entity in the grantor entity is nil")
	}
	if entity.User.Name == "" {
		return fmt.Errorf("the name in the user entity of the grantor entity is empty")
	}
	if _, err := c.meta.SelectUser(util.DefaultTenant, &milvuspb.UserEntity{Name: entity.GetUser().Name}, false); err != nil {
		return err
	}
	if entity.Privilege == nil {
		return fmt.Errorf("the privilege entity in the grantor entity is nil")
	}
	if util.IsAnyWord(entity.Privilege.Name) {
		return nil
	}
	if privilegeName := util.PrivilegeNameForMetastore(entity.Privilege.Name); privilegeName == "" {
		return fmt.Errorf("the privilege name in the privilege entity is invalid, current value: %s", entity.Privilege.Name)
	}
	privileges, ok := util.ObjectPrivileges[object]
	if !ok {
		return fmt.Errorf("the object type is invalid, current value: %s", object)
	}
	for _, privilege := range privileges {
		if privilege == entity.Privilege.Name {
			return nil
		}
	}
	return fmt.Errorf("the privilege name is invalid, current value: %s", entity.Privilege.Name)
}

// OperatePrivilege operate the privilege, including grant and revoke
// - check the node health
// - check if the operating type is valid
// - check if the entity is nil
// - check if the params, including the resource entity, the principal entity, the grantor entity, is valid
// - operate the privilege by the meta api
// - update the policy cache
func (c *RootCoord) OperatePrivilege(ctx context.Context, in *milvuspb.OperatePrivilegeRequest) (*commonpb.Status, error) {
	method := "OperatePrivilege"
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.TotalLabel).Inc()
	tr := timerecord.NewTimeRecorder(method)
	logger.Debug(method, zap.Any("in", in))

	if code, ok := c.checkHealthy(); !ok {
		return errorutil.UnhealthyStatus(code), errorutil.UnhealthyError()
	}
	if in.Type != milvuspb.OperatePrivilegeType_Grant && in.Type != milvuspb.OperatePrivilegeType_Revoke {
		errMsg := fmt.Sprintf("invalid operate privilege type, current type: %s, valid value: [%s, %s]", in.Type, milvuspb.OperatePrivilegeType_Grant, milvuspb.OperatePrivilegeType_Revoke)
		return failStatus(commonpb.ErrorCode_OperatePrivilegeFailure, errMsg), errors.New(errMsg)
	}
	if in.Entity == nil {
		errMsg := "the grant entity in the request is nil"
		return failStatus(commonpb.ErrorCode_OperatePrivilegeFailure, errMsg), errors.New(errMsg)
	}
	if err := c.isValidObject(in.Entity.Object); err != nil {
		return failStatus(commonpb.ErrorCode_OperatePrivilegeFailure, err.Error()), err
	}
	if err := c.isValidRole(in.Entity.Role); err != nil {
		return failStatus(commonpb.ErrorCode_OperatePrivilegeFailure, err.Error()), err
	}
	if err := c.isValidGrantor(in.Entity.Grantor, in.Entity.Object.Name); err != nil {
		return failStatus(commonpb.ErrorCode_OperatePrivilegeFailure, err.Error()), err
	}

	logger.Debug("before PrivilegeNameForMetastore", zap.String("privilege", in.Entity.Grantor.Privilege.Name))
	if !util.IsAnyWord(in.Entity.Grantor.Privilege.Name) {
		in.Entity.Grantor.Privilege.Name = util.PrivilegeNameForMetastore(in.Entity.Grantor.Privilege.Name)
	}
	logger.Debug("after PrivilegeNameForMetastore", zap.String("privilege", in.Entity.Grantor.Privilege.Name))
	if in.Entity.Object.Name == commonpb.ObjectType_Global.String() {
		in.Entity.ObjectName = util.AnyWord
	}
	if err := c.meta.OperatePrivilege(util.DefaultTenant, in.Entity, in.Type); err != nil {
		errMsg := "fail to operate the privilege"
		logger.Error(errMsg, zap.Error(err))
		return failStatus(commonpb.ErrorCode_OperatePrivilegeFailure, errMsg), err
	}

	var opType int32
	if in.Type == milvuspb.OperatePrivilegeType_Grant {
		opType = int32(typeutil.CacheGrantPrivilege)
	} else if in.Type == milvuspb.OperatePrivilegeType_Revoke {
		opType = int32(typeutil.CacheRevokePrivilege)
	}
	if err := c.proxyClientManager.RefreshPolicyInfoCache(ctx, &proxypb.RefreshPolicyInfoCacheRequest{
		OpType: opType,
		OpKey:  funcutil.PolicyForPrivilege(in.Entity.Role.Name, in.Entity.Object.Name, in.Entity.ObjectName, in.Entity.Grantor.Privilege.Name),
	}); err != nil {
		return failStatus(commonpb.ErrorCode_OperatePrivilegeFailure, err.Error()), err
	}

	logger.Debug(method + " success")
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues(method).Observe(float64(tr.ElapseSpan().Milliseconds()))
	return succStatus(), nil
}

// SelectGrant select grant
// - check the node health
// - check if the principal entity is valid
// - check if the resource entity which is provided by the user is valid
// - select grant by the meta api
func (c *RootCoord) SelectGrant(ctx context.Context, in *milvuspb.SelectGrantRequest) (*milvuspb.SelectGrantResponse, error) {
	method := "SelectGrant"
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.TotalLabel).Inc()
	tr := timerecord.NewTimeRecorder(method)
	logger.Debug(method, zap.Any("in", in))

	if code, ok := c.checkHealthy(); !ok {
		return &milvuspb.SelectGrantResponse{
			Status: errorutil.UnhealthyStatus(code),
		}, errorutil.UnhealthyError()
	}
	if in.Entity == nil {
		errMsg := "the grant entity in the request is nil"
		return &milvuspb.SelectGrantResponse{
			Status: failStatus(commonpb.ErrorCode_SelectGrantFailure, errMsg),
		}, errors.New(errMsg)
	}
	if err := c.isValidRole(in.Entity.Role); err != nil {
		return &milvuspb.SelectGrantResponse{
			Status: failStatus(commonpb.ErrorCode_SelectGrantFailure, err.Error()),
		}, err
	}
	if in.Entity.Object != nil {
		if err := c.isValidObject(in.Entity.Object); err != nil {
			return &milvuspb.SelectGrantResponse{
				Status: failStatus(commonpb.ErrorCode_SelectGrantFailure, err.Error()),
			}, err
		}
	}

	grantEntities, err := c.meta.SelectGrant(util.DefaultTenant, in.Entity)
	if err != nil {
		errMsg := "fail to select the grant"
		logger.Error(errMsg, zap.Error(err))
		return &milvuspb.SelectGrantResponse{
			Status: failStatus(commonpb.ErrorCode_SelectGrantFailure, errMsg),
		}, err
	}

	logger.Debug(method + " success")
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues(method).Observe(float64(tr.ElapseSpan().Milliseconds()))
	return &milvuspb.SelectGrantResponse{
		Status:   succStatus(),
		Entities: grantEntities,
	}, nil
}

func (c *RootCoord) ListPolicy(ctx context.Context, in *internalpb.ListPolicyRequest) (*internalpb.ListPolicyResponse, error) {
	method := "PolicyList"
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.TotalLabel).Inc()
	tr := timerecord.NewTimeRecorder(method)
	logger.Debug(method, zap.Any("in", in))

	if code, ok := c.checkHealthy(); !ok {
		return &internalpb.ListPolicyResponse{
			Status: errorutil.UnhealthyStatus(code),
		}, errorutil.UnhealthyError()
	}

	policies, err := c.meta.ListPolicy(util.DefaultTenant)
	if err != nil {
		return &internalpb.ListPolicyResponse{
			Status: failStatus(commonpb.ErrorCode_ListPolicyFailure, "fail to list policy"),
		}, err
	}
	userRoles, err := c.meta.ListUserRole(util.DefaultTenant)
	if err != nil {
		return &internalpb.ListPolicyResponse{
			Status: failStatus(commonpb.ErrorCode_ListPolicyFailure, "fail to list user-role"),
		}, err
	}

	logger.Debug(method + " success")
	metrics.RootCoordDDLReqCounter.WithLabelValues(method, metrics.SuccessLabel).Inc()
	metrics.RootCoordDDLReqLatency.WithLabelValues(method).Observe(float64(tr.ElapseSpan().Milliseconds()))
	return &internalpb.ListPolicyResponse{
		Status:      succStatus(),
		PolicyInfos: policies,
		UserRoles:   userRoles,
	}, nil
}
