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
#include "velox/exec/SegmentPatch.h"

#include <algorithm>
#include <numeric>

#include "velox/exec/OperatorType.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::exec {

// ---------------------------------------------------------------------------
// SegmentPatchState
// ---------------------------------------------------------------------------

SegmentPatchState::SegmentPatchState(std::string planNodeId)
    : planNodeId_(std::move(planNodeId)) {}

SegmentPatchState::~SegmentPatchState() {
  close();
}

std::string SegmentPatchState::patchKey(
    const std::string& key,
    int64_t seg) {
  return key + '\x1f' + std::to_string(seg);
}

void SegmentPatchState::addProducer() {
  std::lock_guard<std::mutex> lock(mutex_);
  ++producers_;
}

void SegmentPatchState::noMoreProducers() {
  std::vector<ContinuePromise> waiters;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    producersDone_ = true;
    if (producers_ == 0) {
      computeTableLocked();
      ready_ = true;
      waiters.swap(readyPromises_);
    }
  }
  for (auto& waiter : waiters) {
    waiter.setValue();
  }
}

void SegmentPatchState::producerDone() {
  std::vector<ContinuePromise> waiters;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK_GT(
        producers_, 0, "{}: no producer registered", toString());
    if (--producers_ == 0 && producersDone_ && !ready_) {
      computeTableLocked();
      ready_ = true;
      waiters.swap(readyPromises_);
    }
  }
  for (auto& waiter : waiters) {
    waiter.setValue();
  }
}

void SegmentPatchState::producerAbandoned() {
  std::vector<ContinuePromise> waiters;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || ready_) {
      return;
    }
    // The abandoned producer will never publish the rest of its tails. Do not
    // compute the table: a partial table would patch boundary rows with the
    // wrong value. Release the waiters so the peers stop instead of blocking.
    abandoned_ = true;
    closed_ = true;
    ready_ = true;
    waiters.swap(readyPromises_);
  }
  for (auto& waiter : waiters) {
    waiter.setValue();
  }
}

void SegmentPatchState::publishTail(
    std::string key,
    int64_t seg,
    VectorPtr tail) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_ || ready_) {
    // The table was already computed (or the state was closed): the bucket
    // belongs to a producer that finished after everybody else, which cannot
    // happen for a well-formed pipeline.
    return;
  }
  tails_.push_back(Tail{std::move(key), seg, std::move(tail)});
}

BlockingReason SegmentPatchState::waitReady(ContinueFuture* future) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (ready_ || closed_) {
    return BlockingReason::kNotBlocked;
  }
  readyPromises_.emplace_back("SegmentPatchState::waitReady");
  *future = readyPromises_.back().getSemiFuture();
  return BlockingReason::kWaitForJoinBuild;
}

bool SegmentPatchState::ready() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ready_;
}

bool SegmentPatchState::abandoned() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return abandoned_;
}

VectorPtr SegmentPatchState::lookup(const std::string& patchKey) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = table_.find(patchKey);
  return it == table_.end() ? nullptr : it->second;
}

void SegmentPatchState::close() {
  std::vector<ContinuePromise> waiters;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    ready_ = true;
    waiters.swap(readyPromises_);
  }
  for (auto& waiter : waiters) {
    waiter.setValue();
  }
}

size_t SegmentPatchState::numTails() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return tails_.size();
}

std::string SegmentPatchState::toString() const {
  return fmt::format("SegmentPatchState({})", planNodeId_);
}

void SegmentPatchState::computeTableLocked() {
  std::vector<size_t> order(tails_.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
    if (tails_[left].key != tails_[right].key) {
      return tails_[left].key < tails_[right].key;
    }
    return tails_[left].seg < tails_[right].seg;
  });
  for (size_t i = 1; i < order.size(); ++i) {
    const auto& previous = tails_[order[i - 1]];
    const auto& current = tails_[order[i]];
    if (current.key == previous.key) {
      // Each (key, seg) bucket is processed wholly by exactly one driver (the
      // pipeline redistributes by key + seg), so two tails of the same segment
      // mean the partitioning invariant broke: patching from a peer bucket of
      // the same segment would be silently wrong, so fail loudly instead.
      VELOX_CHECK_NE(
          current.seg,
          previous.seg,
          "{}: two producers reported a tail for bucket (key, seg) = ({}, {}); "
          "the segmented-window partition invariant is broken",
          toString(),
          current.key,
          current.seg);
      table_[patchKey(current.key, current.seg)] = previous.value;
    }
  }
}

// ---------------------------------------------------------------------------
// SegmentPatch
// ---------------------------------------------------------------------------

SegmentPatch::SegmentPatch(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::string& planNodeId,
    RowTypePtr outputType,
    std::shared_ptr<SegmentPatchState> state,
    std::vector<column_index_t> keyChannels,
    column_index_t segChannel,
    column_index_t tailValueChannel,
    column_index_t patchChannel)
    : Operator(
          ctx,
          std::move(outputType),
          operatorId,
          planNodeId,
          OperatorType::kSegmentPatch),
      state_(std::move(state)),
      keyChannels_(std::move(keyChannels)),
      segChannel_(segChannel),
      tailValueChannel_(tailValueChannel),
      patchChannel_(patchChannel) {
  VELOX_CHECK(!keyChannels_.empty(), "SegmentPatch needs partition keys");
  // One producer per driver of this pipeline: the state signals completion
  // only after the last of them finished (see SegmentPatchState).
  state_->addProducer();
}

std::string SegmentPatch::keyOf(
    const RowVectorPtr& batch,
    vector_size_t row) const {
  std::string key;
  for (size_t i = 0; i < keyChannels_.size(); ++i) {
    if (i > 0) {
      key += '\x1e';
    }
    // Length-prefix every component: a partition value may itself contain the
    // separators (or the textual form of a NULL), and two different key tuples
    // must never collapse into the same patch-table key.
    const auto component = batch->childAt(keyChannels_[i])->toString(row);
    key += std::to_string(component.size());
    key += ':';
    key += component;
  }
  return key;
}

int64_t SegmentPatch::segOf(
    const RowVectorPtr& batch,
    vector_size_t row) const {
  const auto& segVector = batch->childAt(segChannel_);
  switch (segVector->typeKind()) {
    case TypeKind::SMALLINT:
      return segVector->as<SimpleVector<int16_t>>()->valueAt(row);
    case TypeKind::INTEGER:
      return segVector->as<SimpleVector<int32_t>>()->valueAt(row);
    case TypeKind::BIGINT:
      return segVector->as<SimpleVector<int64_t>>()->valueAt(row);
    default:
      VELOX_FAIL(
          "SegmentPatch expects an integer segment column, got {}",
          segVector->type()->toString());
  }
}

VectorPtr SegmentPatch::valueOf(
    const RowVectorPtr& batch,
    vector_size_t row,
    column_index_t channel) const {
  const auto& source = batch->childAt(channel);
  auto value = BaseVector::create(source->type(), 1, pool());
  value->copy(source.get(), /*targetIndex=*/0, row, /*count=*/1);
  return value;
}

RowVectorPtr SegmentPatch::newOutput(vector_size_t rows) const {
  auto output = BaseVector::create(outputType_, rows, pool());
  auto rowOutput = std::dynamic_pointer_cast<RowVector>(output);
  VELOX_CHECK_NOT_NULL(rowOutput, "SegmentPatch output must be a row vector");
  return rowOutput;
}

void SegmentPatch::publishTail(
    const RowVectorPtr& batch,
    vector_size_t row) {
  state_->publishTail(
      keyOf(batch, row), segOf(batch, row), valueOf(batch, row, tailValueChannel_));
}

void SegmentPatch::publishPendingTail() {
  VELOX_CHECK(hasPendingBucket_);
  state_->publishTail(pendingKey_, pendingSeg_, pendingTail_);
}

void SegmentPatch::holdRow(
    const RowVectorPtr& batch,
    vector_size_t row) {
  if (heldRows_ % kHeldChunkRows == 0) {
    heldChunks_.push_back(newOutput(kHeldChunkRows));
  }
  auto& chunk = heldChunks_.back();
  const vector_size_t chunkRow = heldRows_ % kHeldChunkRows;
  for (size_t channel = 0; channel < chunk->childrenSize(); ++channel) {
    chunk->childAt(channel)->copy(
        batch->childAt(channel).get(), chunkRow, row, /*count=*/1);
  }
  heldKeys_.push_back(
      SegmentPatchState::patchKey(keyOf(batch, row), segOf(batch, row)));
  ++heldRows_;
}

void SegmentPatch::addInput(RowVectorPtr input) {
  const auto rows = input->size();
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addInputVector(input->estimateFlatSize(), rows);
  }
  if (rows == 0) {
    return;
  }

  const auto& patchVector = input->childAt(patchChannel_);
  std::vector<vector_size_t> boundaries;
  for (vector_size_t row = 0; row < rows; ++row) {
    if (patchVector->isNullAt(row)) {
      boundaries.push_back(row);
    }
  }

  if (boundaries.empty()) {
    // Common path: no bucket starts here, forward the batch untouched (no
    // allocation, no copy) and keep tracking the bucket that is still open.
    pendingKey_ = keyOf(input, rows - 1);
    pendingSeg_ = segOf(input, rows - 1);
    pendingTail_ = valueOf(input, rows - 1, tailValueChannel_);
    hasPendingBucket_ = true;
    pendingOutput_.push_back(std::move(input));
    return;
  }

  // At least one bucket starts in this batch: rows before each boundary belong
  // to the bucket that ends there, boundary rows are deferred.
  auto output =
      newOutput(rows - static_cast<vector_size_t>(boundaries.size()));
  vector_size_t outputRow = 0;
  for (size_t i = 0; i < boundaries.size(); ++i) {
    const auto boundary = boundaries[i];
    if (boundary == 0) {
      if (hasPendingBucket_) {
        publishPendingTail();
      }
    } else {
      publishTail(input, boundary - 1);
    }
    holdRow(input, boundary);
    const vector_size_t from = i == 0 ? 0 : boundaries[i - 1] + 1;
    for (vector_size_t row = from; row < boundary; ++row) {
      for (size_t channel = 0; channel < output->childrenSize(); ++channel) {
        output->childAt(channel)->copy(
            input->childAt(channel).get(), outputRow, row, /*count=*/1);
      }
      ++outputRow;
    }
  }
  for (vector_size_t row = boundaries.back() + 1; row < rows; ++row) {
    for (size_t channel = 0; channel < output->childrenSize(); ++channel) {
      output->childAt(channel)->copy(
          input->childAt(channel).get(), outputRow, row, /*count=*/1);
    }
    ++outputRow;
  }
  VELOX_CHECK_EQ(outputRow, output->size());

  // The bucket that stays open is the one this batch's last row belongs to.
  pendingKey_ = keyOf(input, rows - 1);
  pendingSeg_ = segOf(input, rows - 1);
  pendingTail_ = valueOf(input, rows - 1, tailValueChannel_);
  hasPendingBucket_ = true;

  // A batch made of nothing but bucket starts has no pass-through rows: Velox
  // forbids returning an empty vector, so only queue non-empty output.
  if (output->size() > 0) {
    pendingOutput_.push_back(std::move(output));
  }
}

void SegmentPatch::noMoreInput() {
  Operator::noMoreInput();
  if (hasPendingBucket_) {
    publishPendingTail();
    hasPendingBucket_ = false;
  }
  producerFinished_ = true;
  state_->producerDone();
  noMoreInput_ = true;
}

BlockingReason SegmentPatch::isBlocked(ContinueFuture* future) {
  if (!pendingOutput_.empty() || done_) {
    return BlockingReason::kNotBlocked;
  }
  if (noMoreInput_ && !state_->ready()) {
    // Reuses the join-build blocking reason: the wait is a build-side
    // dependency of this pipeline, and the label keeps it visible in the
    // per-operator blocked-time stats.
    const auto reason = state_->waitReady(&future_);
    if (reason != BlockingReason::kNotBlocked) {
      *future = std::move(future_);
      return reason;
    }
  }
  return BlockingReason::kNotBlocked;
}

RowVectorPtr SegmentPatch::getOutput() {
  if (!pendingOutput_.empty()) {
    auto output = std::move(pendingOutput_.front());
    pendingOutput_.pop_front();
    {
      auto lockedStats = stats_.wlock();
      lockedStats->addOutputVector(output->estimateFlatSize(), output->size());
    }
    return output;
  }

  if (!noMoreInput_ || done_) {
    return nullptr;
  }
  if (state_->abandoned()) {
    // A producer stopped before publishing all tails: the deferred boundary
    // rows cannot be patched correctly, so drop them and stop. This only
    // happens when the pipeline is being torn down early (downstream limit or
    // task failure), where those rows are not needed any more.
    done_ = true;
    heldChunks_.clear();
    heldKeys_.clear();
    return nullptr;
  }
  if (!state_->ready()) {
    // The driver blocks in isBlocked() until the patch table is ready.
    return nullptr;
  }
  return emitHeldRows();
}

void SegmentPatch::close() {
  // A driver that is closed without draining its input (a downstream limit
  // stopped the pipeline, or the task is failing) will never call
  // noMoreInput(), so its producers must be released here or the peers that
  // wait for the patch table would stay blocked forever.
  if (!producerFinished_) {
    producerFinished_ = true;
    state_->producerAbandoned();
  }
  heldChunks_.clear();
  heldKeys_.clear();
  pendingOutput_.clear();
  Operator::close();
}

RowVectorPtr SegmentPatch::emitHeldRows() {
  if (emittedHeld_ >= heldRows_) {
    done_ = true;
    heldChunks_.clear();
    heldKeys_.clear();
    return nullptr;
  }

  const auto count = std::min(kEmitRows, heldRows_ - emittedHeld_);
  auto output = newOutput(count);
  for (vector_size_t i = 0; i < count; ++i) {
    const auto held = emittedHeld_ + i;
    auto& chunk = heldChunks_[held / kHeldChunkRows];
    const vector_size_t chunkRow = held % kHeldChunkRows;
    if (auto patch = state_->lookup(heldKeys_[held])) {
      chunk->childAt(patchChannel_)->copy(
          patch.get(), chunkRow, /*sourceIndex=*/0, /*count=*/1);
    }
    for (size_t channel = 0; channel < output->childrenSize(); ++channel) {
      output->childAt(channel)->copy(
          chunk->childAt(channel).get(), i, chunkRow, /*count=*/1);
    }
  }
  emittedHeld_ += count;

  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(output->estimateFlatSize(), count);
  }

  if (emittedHeld_ >= heldRows_) {
    done_ = true;
    heldChunks_.clear();
    heldKeys_.clear();
  }
  return output;
}

bool SegmentPatch::isFinished() {
  return done_ && pendingOutput_.empty();
}

} // namespace facebook::velox::exec
