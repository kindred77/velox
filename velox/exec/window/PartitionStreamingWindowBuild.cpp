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
  partitionStartRows_.push_back(sortedRows_.size());
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
  partitionStartRows_.push_back(sortedRows_.size());
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

  // Erase previous partition.
  if (currentPartition_ > 0) {
    const auto numPreviousPartitionRows =
        partitionStartRows_[currentPartition_];
    data_->eraseRows(
        folly::Range<char**>(sortedRows_.data(), numPreviousPartitionRows));
    sortedRows_.erase(
        sortedRows_.cbegin(), sortedRows_.cbegin() + numPreviousPartitionRows);
    sortedRows_.shrink_to_fit();
    for (int i = currentPartition_; i < partitionStartRows_.size(); ++i) {
      partitionStartRows_[i] =
          partitionStartRows_[i] - numPreviousPartitionRows;
    }
  }

  const auto partitionSize = partitionStartRows_[currentPartition_ + 1] -
      partitionStartRows_[currentPartition_];
  const auto partition = folly::Range(
      sortedRows_.data() + partitionStartRows_[currentPartition_],
      partitionSize);

  return std::make_shared<WindowPartition>(
      data_.get(), partition, inversedInputChannels_, sortKeyInfo_);
}

bool PartitionStreamingWindowBuild::hasNextPartition() {
  return partitionStartRows_.size() > 0 &&
      currentPartition_ <
      static_cast<vector_size_t>(partitionStartRows_.size() - 2);
}

} // namespace facebook::velox::exec::window
