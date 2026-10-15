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
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>

namespace facebook::velox::exec::window {

namespace {

struct PartitionStreamingBuildStats {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> rows{0};
  std::atomic<uint64_t> partitions{0};
  std::atomic<uint64_t> decodeNanos{0};
  std::atomic<uint64_t> combinedNanos{0};
  std::atomic<uint64_t> sampledMidRows{0};
  std::atomic<uint64_t> sampledMidPartitionNanos{0};
  std::atomic<uint64_t> sampledMidAllocateNanos{0};
  std::atomic<uint64_t> storeNanos{0};
  std::atomic<uint64_t> bookkeepingNanos{0};
  std::atomic<uint64_t> builds{0};
  std::atomic<uint64_t> sortedRowsCapacityBytes{0};
  std::atomic<uint64_t> inputRowsCapacityBytes{0};
  std::atomic<uint64_t> partitionStartsCapacityBytes{0};
  const bool enabled = [] {
    const char* value = std::getenv("GPORCA_WINDOW_PARTITION_STATS");
    return value != nullptr && std::atoi(value) != 0;
  }();

  ~PartitionStreamingBuildStats() {
    if (enabled) {
      std::cerr << "[window-partition-build] calls=" << calls.load()
                << " rows=" << rows.load()
                << " partitions=" << partitions.load()
                << " decode_ns=" << decodeNanos.load()
                << " combined_ns=" << combinedNanos.load()
                << " sampled_mid_rows=" << sampledMidRows.load()
                << " sampled_mid_partition_ns="
                << sampledMidPartitionNanos.load()
                << " sampled_mid_allocate_ns=" << sampledMidAllocateNanos.load()
                << " store_ns=" << storeNanos.load()
                << " bookkeeping_ns=" << bookkeepingNanos.load()
                << " builds=" << builds.load()
                << " sorted_rows_capacity_bytes="
                << sortedRowsCapacityBytes.load()
                << " input_rows_capacity_bytes=" << inputRowsCapacityBytes.load()
                << " partition_starts_capacity_bytes="
                << partitionStartsCapacityBytes.load() << "\n";
    }
  }
};

PartitionStreamingBuildStats& partitionStreamingBuildStats() {
  static PartitionStreamingBuildStats stats;
  return stats;
}

uint64_t elapsedNanos(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

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

/// Stores the rows of one input batch column-by-column instead of cell-by-cell
/// (performance_tuning 20260922, P0-b1), so the per-cell type dispatch and
/// per-row column-stats calls collapse into one pass per column. The same rows
/// with the same values are stored in the same partition order; only the store
/// granularity changes. Set GPORCA_WINDOW_BATCH_STORE=0 for a one-line rollback
/// to the cell-by-cell path.
bool windowBatchStore() {
  static const bool enabled = [] {
    const char* value = std::getenv("GPORCA_WINDOW_BATCH_STORE");
    return value == nullptr || std::atoi(value) != 0;
  }();
  return enabled;
}

bool stealPartitionRows() {
  static const bool enabled = [] {
    const char* value = std::getenv("GPORCA_WINDOW_STEAL_PARTITION_ROWS");
    return value != nullptr && std::atoi(value) != 0;
  }();
  return enabled;
}

} // namespace

PartitionStreamingWindowBuild::PartitionStreamingWindowBuild(
    const std::shared_ptr<const core::WindowNode>& windowNode,
    velox::memory::MemoryPool* pool,
    const common::SpillConfig* spillConfig,
    tsan_atomic<bool>* nonReclaimableSection)
    : WindowBuild(windowNode, pool, spillConfig, nonReclaimableSection),
      previousPartitionKeyValues_(
          extractPartitionChannels(partitionKeyInfo_, inputChannels_),
          pool),
      partitionKeyChannels_(
          extractPartitionChannels(partitionKeyInfo_, inputChannels_)) {
  initializeRowContainer(pool);
  initializeDecodedInputVectors();
}

void PartitionStreamingWindowBuild::buildNextPartition() {
  partitionStartRows_.push_back(sortedRowsBase_ + sortedRows_.size());
  if (stealPartitionRows() && sortedRows_.empty()) {
    sortedRows_.swap(inputRows_);
    // inputRows_ now owns any empty retained buffer from sortedRows_. It cannot
    // help grow the next partition, so release it instead of carrying two
    // pointer buffers through output.
    std::vector<char*>().swap(inputRows_);
    return;
  }
  sortedRows_.insert(sortedRows_.end(), inputRows_.begin(), inputRows_.end());
  inputRows_.clear();
}

void PartitionStreamingWindowBuild::addInput(RowVectorPtr input) {
  auto& stats = partitionStreamingBuildStats();
  const auto decodeStart = stats.enabled
      ? std::chrono::steady_clock::now()
      : std::chrono::steady_clock::time_point{};
  for (const auto channel : partitionKeyChannels_) {
    input->childAt(channel)->loadedVector();
  }
  for (auto i = 0; i < inputChannels_.size(); ++i) {
    decodedInputVectors_[i].decode(*input->childAt(inputChannels_[i]));
  }
  if (stats.enabled) {
    ++stats.calls;
    stats.rows += input->size();
    stats.decodeNanos += elapsedNanos(decodeStart);
  }

  if (windowBatchStore() && input->size() > 0) {
    std::vector<char*> newRows(input->size());
    std::vector<vector_size_t> partitionStarts;
    const auto combinedStart = stats.enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    uint64_t sampledMidRows = 0;
    uint64_t sampledMidPartitionNanos = 0;
    uint64_t sampledMidAllocateNanos = 0;
    const auto sampledMidRow = input->size() / 2;
    for (vector_size_t row = 0; row < input->size(); ++row) {
      bool startsPartition;
      if (stats.enabled && row == sampledMidRow) {
        const auto start = std::chrono::steady_clock::now();
        startsPartition = isNewPartition(
            input, row, previousPartitionKeyValues_, partitionKeyChannels_);
        sampledMidPartitionNanos += elapsedNanos(start);
      } else {
        startsPartition = isNewPartition(
            input, row, previousPartitionKeyValues_, partitionKeyChannels_);
      }
      if (startsPartition) {
        partitionStarts.push_back(row);
      }
      if (stats.enabled && row == sampledMidRow) {
        const auto start = std::chrono::steady_clock::now();
        newRows[row] = data_->newRow();
        sampledMidAllocateNanos += elapsedNanos(start);
        ++sampledMidRows;
      } else {
        newRows[row] = data_->newRow();
      }
    }
    if (stats.enabled) {
      stats.partitions += partitionStarts.size();
      stats.combinedNanos += elapsedNanos(combinedStart);
      stats.sampledMidRows += sampledMidRows;
      stats.sampledMidPartitionNanos += sampledMidPartitionNanos;
      stats.sampledMidAllocateNanos += sampledMidAllocateNanos;
    }
    const auto storeStart = stats.enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    for (auto col = 0; col < input->childrenSize(); ++col) {
      data_->store(
          decodedInputVectors_[col],
          folly::Range<char**>(newRows.data(), newRows.size()),
          col);
    }
    if (stats.enabled) {
      stats.storeNanos += elapsedNanos(storeStart);
    }
    // Rows before a partition start belong to the partition that is being
    // closed; the order of 'partitionStartRows_' is unchanged.
    const auto bookkeepingStart = stats.enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    size_t begin = 0;
    for (const auto start : partitionStarts) {
      if (start > begin) {
        inputRows_.insert(
            inputRows_.end(), newRows.begin() + begin, newRows.begin() + start);
      }
      buildNextPartition();
      begin = start;
    }
    inputRows_.insert(inputRows_.end(), newRows.begin() + begin, newRows.end());
    previousPartitionKeyValues_.capture(input, input->size() - 1);
    if (stats.enabled) {
      stats.bookkeepingNanos += elapsedNanos(bookkeepingStart);
    }
    return;
  }

  for (auto row = 0; row < input->size(); ++row) {
    if (isNewPartition(
            input, row, previousPartitionKeyValues_, partitionKeyChannels_)) {
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

  auto& stats = partitionStreamingBuildStats();
  if (stats.enabled) {
    const auto sortedRowsCapacityBytes =
        sortedRows_.capacity() * sizeof(char*);
    const auto inputRowsCapacityBytes = inputRows_.capacity() * sizeof(char*);
    const auto partitionStartsCapacityBytes =
        partitionStartRows_.capacity() * sizeof(vector_size_t);
    ++stats.builds;
    stats.sortedRowsCapacityBytes += sortedRowsCapacityBytes;
    stats.inputRowsCapacityBytes += inputRowsCapacityBytes;
    stats.partitionStartsCapacityBytes += partitionStartsCapacityBytes;
    std::cerr << "[window-partition-capacity] sorted_rows_capacity_bytes=" +
            std::to_string(sortedRowsCapacityBytes) +
            " input_rows_capacity_bytes=" +
            std::to_string(inputRowsCapacityBytes) +
            " partition_starts_capacity_bytes=" +
            std::to_string(partitionStartsCapacityBytes) + "\n";
  }
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
