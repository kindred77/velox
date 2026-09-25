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
#pragma once

#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

class ReplicateSource;

/// Reads one channel of a task-level ReplicateSource hub, i.e. the consumer
/// side of the fan-out primitive. It is a leaf operator without splits: it
/// blocks until the producer delivers a batch (kWaitForProducer) and reports
/// end of data once the channel saw EOF. The channel is per consumer pipeline
/// (not per driver), so the drivers of the pipeline share it.
class ReplicateConsumer : public SourceOperator {
 public:
  ReplicateConsumer(
      int32_t operatorId,
      DriverCtx* ctx,
      const std::string& planNodeId,
      const std::string& replicateId,
      int32_t channel,
      int32_t numConsumers,
      const RowTypePtr& outputType);

  std::string toString() const override {
    return fmt::format(
        "ReplicateConsumer({}, channel {})", planNodeId(), channel_);
  }

  /// Nothing is produced while draining; the drain signal itself is delivered
  /// through the hub (see getOutput).
  bool startDrain() override {
    return false;
  }

  BlockingReason isBlocked(ContinueFuture* future) override;

  RowVectorPtr getOutput() override;

  bool isFinished() override {
    return atEnd_;
  }

  /// Signals the producer that this channel needs no more data (e.g. a Limit
  /// downstream stopped early), then releases the channel.
  void close() override;

 private:
  const int32_t channel_;
  const std::shared_ptr<ReplicateSource> source_;
  ContinueFuture future_;
  BlockingReason blockingReason_{BlockingReason::kNotBlocked};
  bool atEnd_{false};
};

} // namespace facebook::velox::exec
