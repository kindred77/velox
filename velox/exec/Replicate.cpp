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
#include "velox/exec/Replicate.h"

#include "velox/exec/OperatorType.h"
#include "velox/exec/ReplicateSource.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

ReplicateConsumer::ReplicateConsumer(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::string& planNodeId,
    const std::string& replicateId,
    int32_t channel,
    int32_t numConsumers,
    const RowTypePtr& outputType)
    : SourceOperator(
          ctx,
          outputType,
          operatorId,
          planNodeId,
          OperatorType::kReplicateConsumer),
      channel_(channel),
      source_(ctx->task->getOrCreateReplicateSource(
          ctx->splitGroupId,
          replicateId,
          numConsumers,
          outputType)) {}

BlockingReason ReplicateConsumer::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    auto reason = blockingReason_;
    blockingReason_ = BlockingReason::kNotBlocked;
    return reason;
  }

  return BlockingReason::kNotBlocked;
}

RowVectorPtr ReplicateConsumer::getOutput() {
  if (hasDrained()) {
    return nullptr;
  }

  RowVectorPtr data;
  bool drained{false};
  blockingReason_ = source_->next(channel_, &data, drained, &future_);
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    VELOX_CHECK(future_.valid());
    VELOX_CHECK(!drained);
    return nullptr;
  }

  if (data != nullptr) {
    VELOX_CHECK(!drained);
    auto lockedStats = stats_.wlock();
    lockedStats->addInputVector(data->estimateFlatSize(), data->size());
    return data;
  }

  if (drained) {
    VELOX_CHECK(!isDraining());
    operatorCtx_->driver()->drainOutput();
  } else {
    // Producer signalled EOF (or the task closed the hub).
    atEnd_ = true;
  }
  return nullptr;
}

void ReplicateConsumer::close() {
  Operator::close();
  source_->closeChannel(channel_);
  atEnd_ = true;
}

} // namespace facebook::velox::exec
