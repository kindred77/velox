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
#include "velox/exec/ReplicateSource.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

namespace {

using ContinuePromises = std::vector<ContinuePromise>;

/// Diagnostic registry (env-gated, default off): while GPORCA_FANOUT_STATS is
/// set, every live hub prints its per-channel state once every 5 seconds. That
/// is what identifies the waiter that never wakes when a fan-out pipeline
/// stalls; the probe is removed with the prototype.
std::mutex& hubsMutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<ReplicateSource*>& liveHubs() {
  static std::vector<ReplicateSource*> hubs;
  return hubs;
}

bool statsEnabled() {
  static const bool enabled = ::getenv("GPORCA_FANOUT_STATS") != nullptr;
  return enabled;
}

void registerHub(ReplicateSource* hub) {
  if (!statsEnabled()) {
    return;
  }
  static std::once_flag once;
  std::call_once(once, []() {
    std::thread([]() {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::vector<ReplicateSource*> hubs;
        {
          std::lock_guard<std::mutex> lock(hubsMutex());
          hubs = liveHubs();
        }
        for (auto* hub : hubs) {
          hub->dumpState();
        }
      }
    }).detach();
  });
  std::lock_guard<std::mutex> lock(hubsMutex());
  liveHubs().push_back(hub);
}

void unregisterHub(ReplicateSource* hub) {
  if (!statsEnabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(hubsMutex());
  auto& hubs = liveHubs();
  hubs.erase(std::remove(hubs.begin(), hubs.end(), hub), hubs.end());
}

/// Releases promises collected under the hub lock. Setting values outside the
/// lock keeps the wake-up path from running under it (same discipline as
/// MergeJoinSource's ScopedPromiseNotification).
void notifyAll(ContinuePromises& promises) {
  for (auto& promise : promises) {
    promise.setValue();
  }
  promises.clear();
}

} // namespace

ReplicateSource::ReplicateSource(
    const std::string& planNodeId,
    int32_t numConsumers,
    const RowTypePtr& rowType,
    int64_t maxBufferBytes)
    : planNodeId_(planNodeId),
      numConsumers_(numConsumers),
      rowType_(rowType),
      maxBufferBytes_(maxBufferBytes) {
  VELOX_CHECK_GE(numConsumers_, 2, "Replicate hub needs >= 2 consumers");
  VELOX_CHECK_GT(maxBufferBytes_, 0, "Replicate hub needs a byte budget");
  {
    // Channel holds move-only promises, so grow by emplace_back instead of
    // resize (which needs a copy-constructible element type here).
    auto state = state_.wlock();
    state->channels.reserve(numConsumers_);
    for (int32_t i = 0; i < numConsumers_; ++i) {
      state->channels.emplace_back();
    }
  }
  registerHub(this);
}

ReplicateSource::~ReplicateSource() {
  unregisterHub(this);
}

void ReplicateSource::dumpState() const {
  auto state = state_.rlock();
  std::string line = fmt::format(
      "[fanout] hub={} consumers={} enqueue={} noMoreInput={} drain={}"
      " buffered={} waitingProducers={} pendingProducers={} producersDone={}",
      planNodeId_,
      numConsumers_,
      enqueueCalls_.load(),
      noMoreInputCalls_.load(),
      drainCalls_.load(),
      state->bufferedBytes,
      state->bufferPromises.size(),
      state->pendingProducers,
      state->producersDone ? 1 : 0);
  for (size_t i = 0; i < state->channels.size(); ++i) {
    const auto& channel = state->channels[i];
    line += fmt::format(
        " | ch{} stored={} taken={} queued={} queuedBytes={} atEnd={}"
        " closed={} drained={} waitCons={}",
        i,
        channel.storedBatches,
        channel.takenBatches,
        channel.queue.size(),
        channel.queuedBytes,
        channel.atEnd ? 1 : 0,
        channel.closed ? 1 : 0,
        channel.drained ? 1 : 0,
        channel.consumerPromises.size());
  }
  line += "\n";
  ::fprintf(stderr, "%s", line.c_str());
  ::fflush(stderr);
}

BlockingReason ReplicateSource::enqueue(
    RowVectorPtr input,
    ContinueFuture* future,
    bool drained) {
  ++enqueueCalls_;
  ContinuePromises consumerPromises;

  if (drained) {
    {
      auto state = state_.wlock();
      for (auto& channel : state->channels) {
        if (!isOpen(channel) || channel.drained) {
          continue;
        }
        channel.drained = true;
        for (auto& promise : channel.consumerPromises) {
          consumerPromises.push_back(std::move(promise));
        }
        channel.consumerPromises.clear();
      }
    }
    notifyAll(consumerPromises);
    return BlockingReason::kNotBlocked;
  }

  if (input == nullptr) {
    // This producer is done. EOF is only signalled once every producer of the
    // pipeline is done, so the consumers never observe a premature end.
    producerFinished();
    return BlockingReason::kNotBlocked;
  }

  const int64_t bytes = input->estimateFlatSize();
  bool blocked{false};
  {
    auto state = state_.wlock();
    // The batch is shared (copy-on-write) with every channel, so it is charged
    // to the budget once; it stays charged as long as any channel holds it.
    int32_t openChannels{0};
    for (auto& channel : state->channels) {
      if (!isOpen(channel)) {
        continue;
      }
      ++openChannels;
      channel.queue.push_back(Channel::QueuedBatch{input, bytes});
      ++channel.storedBatches;
      channel.queuedBytes += bytes;
      for (auto& promise : channel.consumerPromises) {
        consumerPromises.push_back(std::move(promise));
      }
      channel.consumerPromises.clear();
    }
    state->bufferedBytes += bytes * openChannels;
    if (state->bufferedBytes > maxBufferBytes_) {
      blocked = true;
      state->bufferPromises.emplace_back("ReplicateSource::enqueue");
      *future = state->bufferPromises.back().getSemiFuture();
    }
  }

  notifyAll(consumerPromises);
  return blocked ? BlockingReason::kWaitForConsumer
                 : BlockingReason::kNotBlocked;
}

void ReplicateSource::addProducer() {
  ++state_.wlock()->pendingProducers;
}

void ReplicateSource::noMoreProducers() {
  ContinuePromises consumerPromises;
  {
    auto state = state_.wlock();
    state->producersDone = true;
    if (state->pendingProducers == 0) {
      signalEndOfInputLocked(*state, consumerPromises);
    }
  }
  notifyAll(consumerPromises);
}

void ReplicateSource::producerFinished() {
  ++noMoreInputCalls_;
  ContinuePromises consumerPromises;
  {
    auto state = state_.wlock();
    VELOX_CHECK_GT(
        state->pendingProducers, 0, "No producer registered for {}", toString());
    if (--state->pendingProducers == 0 && state->producersDone) {
      signalEndOfInputLocked(*state, consumerPromises);
    }
  }
  notifyAll(consumerPromises);
}

void ReplicateSource::signalEndOfInputLocked(
    State& state,
    ContinuePromises& toNotify) {
  for (auto& channel : state.channels) {
    if (channel.atEnd || channel.closed) {
      continue;
    }
    channel.atEnd = true;
    for (auto& promise : channel.consumerPromises) {
      toNotify.push_back(std::move(promise));
    }
    channel.consumerPromises.clear();
  }
}

void ReplicateSource::drain() {
  ++drainCalls_;
  ContinueFuture ignored;
  enqueue(nullptr, &ignored, /*drained=*/true);
}

BlockingReason ReplicateSource::next(
    int32_t channel,
    RowVectorPtr* data,
    bool& drained,
    ContinueFuture* future) {
  VELOX_CHECK_LT(channel, numConsumers_, "{}", toString());
  drained = false;
  ContinuePromises bufferPromises;
  BlockingReason blockingReason{BlockingReason::kNotBlocked};
  {
    auto state = state_.wlock();
    auto& channelState = state->channels[channel];
    if (!channelState.queue.empty()) {
      auto batch = std::move(channelState.queue.front());
      channelState.queue.pop_front();
      ++channelState.takenBatches;
      channelState.queuedBytes -= batch.bytes;
      state->bufferedBytes -= batch.bytes;
      *data = std::move(batch.data);
      if (state->bufferedBytes <= maxBufferBytes_) {
        bufferPromises.swap(state->bufferPromises);
      }
    } else if (channelState.atEnd || channelState.closed) {
      *data = nullptr;
    } else if (channelState.drained) {
      channelState.drained = false;
      drained = true;
      *data = nullptr;
    } else {
      channelState.consumerPromises.emplace_back("ReplicateSource::next");
      *future = channelState.consumerPromises.back().getSemiFuture();
      blockingReason = BlockingReason::kWaitForProducer;
    }
  }
  notifyAll(bufferPromises);
  return blockingReason;
}

void ReplicateSource::closeChannel(int32_t channel) {
  VELOX_CHECK_LT(channel, numConsumers_, "{}", toString());
  ContinuePromises toNotify;
  {
    auto state = state_.wlock();
    auto& channelState = state->channels[channel];
    for (auto& batch : channelState.queue) {
      state->bufferedBytes -= batch.bytes;
    }
    channelState.queue.clear();
    channelState.queuedBytes = 0;
    channelState.closed = true;
    channelState.drained = false;
    for (auto& promise : channelState.consumerPromises) {
      toNotify.push_back(std::move(promise));
    }
    channelState.consumerPromises.clear();
    if (state->bufferedBytes <= maxBufferBytes_) {
      for (auto& promise : state->bufferPromises) {
        toNotify.push_back(std::move(promise));
      }
      state->bufferPromises.clear();
    }
  }
  notifyAll(toNotify);
}

void ReplicateSource::close() {
  ContinuePromises toNotify;
  {
    auto state = state_.wlock();
    for (auto& channel : state->channels) {
      channel.queue.clear();
      channel.queuedBytes = 0;
      channel.closed = true;
      channel.atEnd = true;
      channel.drained = false;
      for (auto& promise : channel.consumerPromises) {
        toNotify.push_back(std::move(promise));
      }
      channel.consumerPromises.clear();
    }
    state->bufferedBytes = 0;
    state->pendingProducers = 0;
    state->producersDone = true;
    for (auto& promise : state->bufferPromises) {
      toNotify.push_back(std::move(promise));
    }
    state->bufferPromises.clear();
  }
  notifyAll(toNotify);
}

bool ReplicateSource::allConsumersClosed() const {
  auto state = state_.rlock();
  for (const auto& channel : state->channels) {
    if (!channel.closed) {
      return false;
    }
  }
  return true;
}

int64_t ReplicateSource::bufferedBytes() const {
  return state_.rlock()->bufferedBytes;
}

std::string ReplicateSource::toString() const {
  return fmt::format(
      "ReplicateSource({}, consumers {})", planNodeId_, numConsumers_);
}

} // namespace facebook::velox::exec
