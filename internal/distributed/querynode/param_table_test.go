// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package grpcquerynode

import (
	"testing"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/distributed/grpcconfigs"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/stretchr/testify/assert"
)

func TestParamTable(t *testing.T) {
	Params.Init()

	assert.NotEqual(t, Params.IndexCoordAddress, "")
	t.Logf("IndexCoordAddress:%s", Params.IndexCoordAddress)

	assert.NotEqual(t, Params.DataCoordAddress, "")
	t.Logf("DataCoordAddress:%s", Params.DataCoordAddress)

	assert.NotEqual(t, Params.RootCoordAddress, "")
	t.Logf("RootCoordAddress:%s", Params.RootCoordAddress)

	assert.NotEqual(t, Params.QueryCoordAddress, "")
	t.Logf("QueryCoordAddress:%s", Params.QueryCoordAddress)

	log.Info("TestParamTable", zap.Int("ServerMaxSendSize", Params.ServerMaxSendSize))
	log.Info("TestParamTable", zap.Int("ServerMaxRecvSize", Params.ServerMaxRecvSize))

	Params.Remove("queryNode.grpc.ServerMaxSendSize")
	Params.initServerMaxSendSize()
	assert.Equal(t, Params.ServerMaxSendSize, grpcconfigs.DefaultServerMaxSendSize)

	Params.Remove("queryNode.grpc.ServerMaxRecvSize")
	Params.initServerMaxRecvSize()
	assert.Equal(t, Params.ServerMaxRecvSize, grpcconfigs.DefaultServerMaxRecvSize)

	Params.LoadFromEnv()
	assert.Equal(t, Params.QueryNodeIP, funcutil.GetLocalIP())
}
