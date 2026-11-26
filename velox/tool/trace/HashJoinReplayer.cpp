/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/tool/trace/HashJoinReplayer.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/trace/TraceUtil.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace facebook::velox::tool::trace {
core::PlanNodePtr HashJoinReplayer::createPlanNode(
    const core::PlanNode* node,
    const core::PlanNodeId& nodeId,
    const core::PlanNodePtr& source) const {
  const auto* hashJoinNode = dynamic_cast<const core::HashJoinNode*>(node);
  VELOX_CHECK_NOT_NULL(hashJoinNode);
  // SQL semantics follow the captured node. TraceScan is a different input
  // producer: its batches do not inherit the original peek bound, and replay
  // must perform the build rather than reuse an unrelated runtime cache.
  return core::HashJoinNode::Builder(*hashJoinNode)
      .id(nodeId)
      .left(source)
      .right(PlanBuilder(planNodeIdGenerator_)
          .traceScan(
              nodeTraceDir_,
              pipelineIds_.at(1),
              driverIds_,
              exec::trace::getDataType(planFragment_, nodeId_, 1))
          .planNode())
      .useHashTableCache(false)
      .cacheKey(std::nullopt)
      .probeFirst(0)
      .build();
}
} // namespace facebook::velox::tool::trace
