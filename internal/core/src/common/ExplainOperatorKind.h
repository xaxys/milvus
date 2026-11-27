// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <cstdint>

namespace milvus {

// Global topological order of physical operators on a segment. Numeric values
// match milvus::proto::internal::ExplainOperatorKind (verified via
// static_assert in ExecutionExplainRecorder.cpp) so headers included by
// expression kernels stay free of generated protobuf headers.
enum class ExplainOperatorKind : int32_t {
    kUnknown = 0,
    kFilter = 1,
    kRandomSample = 2,
    kMvcc = 3,
    kElementFilter = 4,
    kVectorSearch = 5,
    kIterativeElementFilter = 6,
    kIterativeFilter = 7,
    kGroupBy = 8,
    kRescore = 9,
    kProject = 10,
    kAggregation = 11,
    kOrderBy = 12,
};

}  // namespace milvus
