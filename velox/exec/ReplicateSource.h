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

#include "velox/core/PlanNode.h"
#include "velox/exec/Driver.h"

#include <deque>

namespace facebook::velox::exec {

/// Fan-out hub that delivers one producer stream to N independent consumer
/// channels: every batch enqueued by the producer is handed to every channel
/// that is still open, exactly once per channel.
///
/// Modeled on MergeJoinSource (single producer/consumer, promise-based waits)
/// combined with LocalExchangeQueue's backpressure idiom (bounded buffered
/// bytes instead of a fixed batch count per channel):
///  * start   - the hub itself is eager: it exists before any driver runs and
///              never waits for a start signal. Consumers are normal drivers
///              of the task, created together with the producer drivers.
///  * sharing - a channel is per consumer pipeline, not per driver, so every
///              driver of that pipeline competes for the batches of the same
///              channel (like a LocalExchangeQueue). Waiters are therefore
///              tracked as lists: N drivers may block on one channel.
///  * wait    - consumers block on 'next' (kWaitForProducer) while their
///              channel is empty; the producer blocks on 'enqueue'
///              (kWaitForConsumer) once the total buffered bytes exceed the
///              budget. All waits are promise-based, so no driver holds a stack
///              while waiting and no future depends on the driver that waits on
///              it.
///  * backpressure - every channel holds a queue of batches and the producer
///              blocks once the total buffered bytes across channels exceed the
///              budget. 'enqueue' always takes ownership of the batch (the
///              caller's contract: a callback may block but must not be asked
///              to hand the same batch over twice), so memory stays bounded
///              without dropping data and without a per-channel one-batch
///              handshake that turns a slow branch into a stall of the other.
///  * EOF     - the producer sends EOF ('noMoreInput') to every channel that is
///              still open; consumers drain what is already queued before they
///              observe it. A consumer that stops early calls 'closeChannel'
///              and the producer stops accounting for it; 'close' (task failure
///              or cancellation) releases all buffered data and wakes every
///              waiter.
///
/// The hub is process-local and split-group scoped: it never crosses a
/// fragment/stage boundary.
class ReplicateSource {
 public:
  /// Default total buffer budget across all channels (bytes).
  static constexpr int64_t kDefaultMaxBufferBytes = 32LL << 20;

  ReplicateSource(
      const std::string& planNodeId,
      int32_t numConsumers,
      const RowTypePtr& rowType,
      int64_t maxBufferBytes = kDefaultMaxBufferBytes);

  ~ReplicateSource();

  int32_t numConsumers() const {
    return numConsumers_;
  }

  /// Producer side: registers one producer driver. Every driver of the
  /// producer pipeline calls this before it starts feeding the hub, so that EOF
  /// is signalled only after the last of them is done (a pipeline has one sink
  /// per driver, not one global sink).
  void addProducer();

  /// Producer side: called once by the task after all driver pipelines were
  /// created; together with 'producerFinished' this sequences EOF.
  void noMoreProducers();

  /// Producer side: hands 'input' to every open channel. Returns
  /// kWaitForConsumer (and sets 'future') when the buffered bytes exceed the
  /// budget; the batch is taken over in every case, so the caller must not
  /// retry it.
  BlockingReason
  enqueue(RowVectorPtr input, ContinueFuture* future, bool drained = false);

  /// Producer side: no more data; every open channel receives EOF. Idempotent.
  /// Only the last producer may call it through 'producerFinished'.
  void producerFinished();

  /// Producer side: barrier drain signal forwarded to every open channel.
  void drain();

  /// Consumer side: fetches the next batch of 'channel'. Returns kNotBlocked
  /// with data == nullptr once the channel saw EOF (or was closed). Sets
  /// 'drained' when the producer drained its pipeline under barrier
  /// processing.
  BlockingReason next(
      int32_t channel,
      RowVectorPtr* data,
      bool& drained,
      ContinueFuture* future);

  /// Consumer side: 'channel' needs no more data (e.g. a Limit downstream).
  /// The producer releases the buffered batch and stops waiting for it.
  void closeChannel(int32_t channel);

  /// Drops buffered data and wakes all waiters; used on task failure/cancel.
  void close();

  /// True when every channel has been closed by its consumer.
  bool allConsumersClosed() const;

  /// Bytes currently buffered across all channels.
  int64_t bufferedBytes() const;

  std::string toString() const;

  /// Prints one line describing the current per-channel state. Called every few
  /// seconds by the diagnostic watchdog when GPORCA_FANOUT_STATS is set; the
  /// users of this probe read it to find the waiter that never wakes when a
  /// fan-out pipeline stalls.
  void dumpState() const;

 private:
  struct Channel {
    Channel() = default;
    // Holds move-only promises: movable, never copyable (a copy would also
    // duplicate a waiter, which is meaningless).
    Channel(Channel&&) noexcept = default;
    Channel& operator=(Channel&&) noexcept = default;
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    bool atEnd{false};
    bool drained{false};
    bool closed{false};
    struct QueuedBatch {
      RowVectorPtr data;
      int64_t bytes;
    };
    std::deque<QueuedBatch> queue;
    uint64_t storedBatches{0};
    uint64_t takenBatches{0};
    int64_t queuedBytes{0};
    // Satisfied when this channel receives data, EOF or a drain signal.
    std::vector<ContinuePromise> consumerPromises;
  };

  struct State {
    std::vector<Channel> channels;
    /// Total bytes buffered across channels (bounded by the budget).
    int64_t bufferedBytes{0};
    /// Producers waiting for 'bufferedBytes' to fall below the budget.
    std::vector<ContinuePromise> bufferPromises;
    /// Producers registered / still running.
    int32_t pendingProducers{0};
    bool producersDone{false};
  };

  /// Signals EOF on every open channel. Called once the last producer is done.
  void signalEndOfInputLocked(
      State& state,
      std::vector<ContinuePromise>& toNotify);

  bool isOpen(const Channel& channel) const {
    return !channel.closed && !channel.atEnd;
  }

  const std::string planNodeId_;
  const int32_t numConsumers_;
  const RowTypePtr rowType_;
  const int64_t maxBufferBytes_;

  std::atomic<uint64_t> enqueueCalls_{0};
  std::atomic<uint64_t> noMoreInputCalls_{0};
  std::atomic<uint64_t> drainCalls_{0};

  folly::Synchronized<State> state_;
};

} // namespace facebook::velox::exec
