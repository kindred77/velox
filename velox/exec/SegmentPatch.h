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

#include <deque>

#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

/// Task + split-group scoped state of the patch-free segmented window shape
/// (dev_tasks/performance_tuning/20260923_win/tasks/S1_patchfree_seg_window_design.md).
///
/// Every (partition, segment) bucket is processed wholly by one driver and its
/// rows are already sorted, so the bucket tail -- the value the stitched hash
/// join used to recompute with a second pass over the input -- is available for
/// free: 'SegmentPatch' reports one tail per bucket. The only cross-driver
/// information left is the small table
///
///   patch(key, seg) = tail of the previous existing bucket of 'key'
///
/// which the last producer driver computes once every producer of the pipeline
/// is done. Nothing here feeds back into the data path: producers only publish a
/// tiny side stream (one row per bucket), so the shape has no barrier on the
/// shared input.
class SegmentPatchState {
 public:
  explicit SegmentPatchState(std::string planNodeId);

  ~SegmentPatchState();

  const std::string& planNodeId() const {
    return planNodeId_;
  }

  /// Registers one producer driver (one per driver of the patching pipeline).
  void addProducer();

  /// Marks this producer as done; the last one computes the patch table.
  void producerDone();

  /// Marks a producer that stopped before finishing its input (a downstream
  /// limit stopped the pipeline, or the driver failed) as gone: its remaining
  /// bucket tails will never arrive, so waiting consumers must stop instead of
  /// blocking forever. The table is never computed from a partial set of tails
  /// (that would silently patch boundary rows with the wrong value); consumers
  /// observe abandoned() and drop their deferred rows.
  void producerAbandoned();

  /// Called by the task once all driver pipelines were created.
  void noMoreProducers();

  /// Publishes one bucket tail: 'key' identifies the partition, 'seg' the
  /// segment and 'tail' is a single value of the tail column.
  void publishTail(std::string key, int64_t seg, VectorPtr tail);

  /// Blocks until the patch table is ready (or the state was closed).
  BlockingReason waitReady(ContinueFuture* future);

  bool ready() const;

  /// True when some producer stopped before its input was exhausted, i.e. the
  /// patch table cannot be completed. Consumers must stop emitting deferred
  /// rows rather than emit unpatched ones.
  bool abandoned() const;

  /// Value that patches the first row of the bucket identified by
  /// 'patchKey' (see patchKey()), or nullptr when the bucket has no
  /// predecessor (the first existing bucket of that key).
  VectorPtr lookup(const std::string& patchKey) const;

  /// Releases waiters (task failure/cancel). The table itself is kept.
  void close();

  std::string toString() const;

  /// Number of published bucket tails (diagnostics).
  size_t numTails() const;

  /// Key of a lookup entry: partition key + segment.
  static std::string patchKey(const std::string& key, int64_t seg);

 private:
  struct Tail {
    std::string key;
    int64_t seg;
    VectorPtr value;
  };

  /// Computes (key, seg) -> previous existing bucket's tail. Caller must hold
  /// 'mutex_'.
  void computeTableLocked();

  const std::string planNodeId_;
  mutable std::mutex mutex_;
  std::vector<Tail> tails_;
  std::unordered_map<std::string, VectorPtr> table_;
  int32_t producers_{0};
  bool producersDone_{false};
  bool ready_{false};
  bool abandoned_{false};
  bool closed_{false};
  std::vector<ContinuePromise> readyPromises_;
};

/// Patching operator of the segmented window shape.
///
/// Input = the local lag output of the window: rows are sorted by (partition
/// keys..., seg, order key) and 'patchChannel' (the local lag column) is NULL
/// exactly on the first row of every (partition, seg) bucket, which is the
/// boundary marker -- no key comparison is needed per row. The operator
///  - forwards non-boundary rows (the whole batch untouched when it has no
///    boundary row, so the common path stays allocation-free),
///  - publishes the tail of every bucket it finishes,
///  - defers boundary rows until the patch table is ready, writes the patch
///    value into 'patchChannel' and then emits them.
class SegmentPatch : public Operator {
 public:
  SegmentPatch(
      int32_t operatorId,
      DriverCtx* ctx,
      const std::string& planNodeId,
      RowTypePtr outputType,
      std::shared_ptr<SegmentPatchState> state,
      std::vector<column_index_t> keyChannels,
      column_index_t segChannel,
      column_index_t tailValueChannel,
      column_index_t patchChannel);

  std::string toString() const override {
    return fmt::format("SegmentPatch({})", planNodeId());
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    return true;
  }

  BlockingReason isBlocked(ContinueFuture* future) override;

  void noMoreInput() override;

  bool isFinished() override;

  void close() override;

 private:
  static constexpr vector_size_t kHeldChunkRows = 1024;
  static constexpr vector_size_t kEmitRows = 1024;

  /// Partition key of 'row' (joined key columns).
  std::string keyOf(const RowVectorPtr& batch, vector_size_t row) const;

  /// Segment of 'row' as an integer.
  int64_t segOf(const RowVectorPtr& batch, vector_size_t row) const;

  /// Copies the single cell 'row' of 'channel' into a fresh one-row vector.
  VectorPtr valueOf(
      const RowVectorPtr& batch,
      vector_size_t row,
      column_index_t channel) const;

  /// Fresh row vector of the operator's output type.
  RowVectorPtr newOutput(vector_size_t rows) const;

  /// Publishes the bucket that ended at 'batch[row]'.
  void publishTail(const RowVectorPtr& batch, vector_size_t row);

  /// Publishes the bucket whose identity was captured in 'pendingKey_' and
  /// 'pendingTail_' (it ended in a previous batch).
  void publishPendingTail();

  /// Defers 'row': the row is copied so the batch is not pinned and downstream
  /// in-place reuse is preserved.
  void holdRow(const RowVectorPtr& batch, vector_size_t row);

  /// Emits the next chunk of deferred rows (patching them first).
  RowVectorPtr emitHeldRows();

  const std::shared_ptr<SegmentPatchState> state_;
  const std::vector<column_index_t> keyChannels_;
  const column_index_t segChannel_;
  const column_index_t tailValueChannel_;
  const column_index_t patchChannel_;

  std::deque<RowVectorPtr> pendingOutput_;

  std::vector<RowVectorPtr> heldChunks_;
  std::vector<std::string> heldKeys_;
  vector_size_t heldRows_{0};
  vector_size_t emittedHeld_{0};

  /// Identity and tail value of the bucket that is still open at the end of the
  /// last processed batch.
  std::string pendingKey_;
  int64_t pendingSeg_{0};
  VectorPtr pendingTail_;
  bool hasPendingBucket_{false};

  ContinueFuture future_;
  bool noMoreInput_{false};
  bool producerFinished_{false};
  bool done_{false};
};

} // namespace facebook::velox::exec
