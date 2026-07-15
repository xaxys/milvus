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

package importv2

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

func TestImportTasksInheritParentContext(t *testing.T) {
	parentCtx, cancel := context.WithCancel(context.Background())
	manager := NewTaskManager()

	tasks := []Task{
		NewPreImportTask(parentCtx, &datapb.PreImportRequest{}, manager, nil),
		NewImportTask(parentCtx, &datapb.ImportRequest{}, manager, nil, nil),
		NewL0PreImportTask(parentCtx, &datapb.PreImportRequest{}, manager, nil),
		NewL0ImportTask(parentCtx, &datapb.ImportRequest{}, manager, nil, nil),
		NewCopySegmentTask(parentCtx, &datapb.CopySegmentRequest{}, manager, nil),
	}

	cancel()

	for _, task := range tasks {
		var taskCtx context.Context
		switch typedTask := task.(type) {
		case *PreImportTask:
			taskCtx = typedTask.ctx
		case *ImportTask:
			taskCtx = typedTask.ctx
		case *L0PreImportTask:
			taskCtx = typedTask.ctx
		case *L0ImportTask:
			taskCtx = typedTask.ctx
		case *CopySegmentTask:
			taskCtx = typedTask.ctx
		default:
			t.Fatalf("unexpected task type %T", task)
		}
		require.ErrorIs(t, taskCtx.Err(), context.Canceled)
	}
}
