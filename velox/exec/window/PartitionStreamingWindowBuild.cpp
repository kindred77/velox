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

#include "velox/exec/window/PartitionStreamingWindowBuild.h"

#include <algorithm>

namespace facebook::velox::exec::window {

namespace {

// Maps the reordered partition-key descriptors to original input channels.
std::vector<column_index_t> extractPartitionChannels(
    const std::vector<std::pair<column_index_t, core::SortOrder>>&
        partitionKeyInfo,
    const std::vector<column_index_t>& inputChannels) {
  std::vector<column_index_t> channels;
  channels.reserve(partitionKeyInfo.size());
  for (const auto& key : partitionKeyInfo) {
    channels.push_back(inputChannels[key.first]);
  }
  return channels;
}

// Returns true if 'row' starts a new partition relative to the previous input
// row (same vector) or to the captured key values of the previous vector.
bool isNewPartition(
    const RowVectorPtr& input,
    vector_size_t row,
    const SingleRowValues& previousPartitionKeyValues,
    const std::vector<column_index_t>& partitionKeyChannels) {
  if (row == 0) {
    return previousPartitionKeyValues.hasValue() &&
        !previousPartitionKeyValues.equals(input, row);
  }
  for (const auto channel : partitionKeyChannels) {
    if (!input->childAt(channel)->equalValueAt(
            input->childAt(channel).get(), row - 1, row)) {
      return true;
    }
  }
  return false;
}

} // namespace

PartitionStreamingWindowBuild::PartitionStreamingWindowBuild(
    const std::shared_ptr<const core::WindowNode>& windowNode,
    velox::memory::MemoryPool* pool,
    const common::SpillConfig* spillConfig,
    tsan_atomic<bool>* nonReclaimableSection)
    : WindowBuild(windowNode, pool, spillConfig, nonReclaimableSection),
      previousPartitionKeyValues_(
          extractPartitionChannels(partitionKeyInfo_, inputChannels_), pool),
      partitionKeyChannels_(
          extractPartitionChannels(partitionKeyInfo_, inputChannels_)) {
  initializeRowContainer(pool);
  initializeDecodedInputVectors();
}

void PartitionStreamingWindowBuild::buildNextPartition() {
  partitionStartRows_.push_back(sortedRowsBase_ + sortedRows_.size());
  sortedRows_.insert(sortedRows_.end(), inputRows_.begin(), inputRows_.end());
  inputRows_.clear();
}

void PartitionStreamingWindowBuild::addInput(RowVectorPtr input) {
  for (const auto channel : partitionKeyChannels_) {
    input->childAt(channel)->loadedVector();
  }
  for (auto i = 0; i < inputChannels_.size(); ++i) {
    decodedInputVectors_[i].decode(*input->childAt(inputChannels_[i]));
  }

  for (auto row = 0; row < input->size(); ++row) {
    if (isNewPartition(
            input,
            row,
            previousPartitionKeyValues_,
            partitionKeyChannels_)) {
      buildNextPartition();
    }

    char* newRow = data_->newRow();

    for (auto col = 0; col < input->childrenSize(); ++col) {
      data_->store(decodedInputVectors_[col], row, newRow, col);
    }

    inputRows_.push_back(newRow);
  }
  if (input->size() > 0) {
    previousPartitionKeyValues_.capture(input, input->size() - 1);
  }
}

void PartitionStreamingWindowBuild::noMoreInput() {
  buildNextPartition();
  previousPartitionKeyValues_.reset();

  // Help for last partition related calculations.
  partitionStartRows_.push_back(sortedRowsBase_ + sortedRows_.size());
}

void PartitionStreamingWindowBuild::compactConsumedRows(size_t numRows) {
  VELOX_DCHECK_LE(numRows, sortedRows_.size());
  data_->eraseRows(folly::Range<char**>(sortedRows_.data(), numRows));
  sortedRows_.erase(sortedRows_.cbegin(), sortedRows_.cbegin() + numRows);
  sortedRowsBase_ += numRows;
}

size_t PartitionStreamingWindowBuild::compactionRowThreshold() {
  if (compactionRowThreshold_ != 0) {
    return compactionRowThreshold_;
  }
  // Keep the rows of consumed partitions for at most ~1MB before freeing
  // them. Erasing (and reallocating) per partition costs O(pending rows) each
  // time, which is quadratic for the many-small-partitions shape this build
  // exists to serve; batching keeps that bookkeeping O(1) per partition while
  // still bounding the memory held for processed partitions. The estimate is
  // taken on first use (not in the constructor) because it is only meaningful
  // once rows were materialized; an empty container falls back to the fixed
  // row size, which is always available.
  const auto estimatedRowSize =
      data_->estimateRowSize().value_or(data_->fixedRowSize());
  const auto rowSize = std::max<int64_t>(1, estimatedRowSize);
  compactionRowThreshold_ =
      std::max<size_t>(1, (1ULL << 20) / static_cast<size_t>(rowSize));
  return compactionRowThreshold_;
}

std::shared_ptr<WindowPartition>
PartitionStreamingWindowBuild::nextPartition() {
  VELOX_CHECK_GT(
      partitionStartRows_.size(), 0, "No window partitions available");

  ++currentPartition_;
  VELOX_CHECK_LE(
      currentPartition_,
      partitionStartRows_.size() - 2,
      "All window partitions consumed");

  // Rows of the partitions consumed so far are still in the RowContainer:
  // free them once enough accumulated instead of on every partition. The
  // index is absolute, so no bookkeeping per partition is needed.
  const auto numConsumedRows =
      partitionStartRows_[currentPartition_] - sortedRowsBase_;
  if (numConsumedRows >= compactionRowThreshold()) {
    compactConsumedRows(numConsumedRows);
  }

  const auto partitionSize = partitionStartRows_[currentPartition_ + 1] -
      partitionStartRows_[currentPartition_];
  const auto partition = folly::Range(
      sortedRows_.data() + partitionStartRows_[currentPartition_] -
          sortedRowsBase_,
      partitionSize);

  if (reusablePartition_ == nullptr) {
    reusablePartition_ = std::make_shared<WindowPartition>(
        data_.get(), partition, inversedInputChannels_, sortKeyInfo_);
  } else {
    reusablePartition_->resetRows(partition);
  }
  return reusablePartition_;
}

bool PartitionStreamingWindowBuild::hasNextPartition() {
  return partitionStartRows_.size() > 0 &&
      currentPartition_ <
      static_cast<vector_size_t>(partitionStartRows_.size() - 2);
}

} // namespace facebook::velox::exec::window
