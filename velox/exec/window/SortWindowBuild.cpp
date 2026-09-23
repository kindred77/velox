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

#include "velox/exec/window/SortWindowBuild.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/Window.h"
#include "velox/common/base/BitUtil.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace facebook::velox::exec::window {

namespace {

// The group build packs the row index into the low 32 bits of its sort word.
constexpr size_t kGroupBuildMaxRows = 0xFFFFFFFFULL;
// Rows inspected by the adaptive gate before choosing the build strategy.
constexpr size_t kGroupBuildSampleRows = 4096;
// Below this input size sorting is cheap enough that the extra grouping pass
// does not pay off.
constexpr size_t kGroupBuildMinRows = 64 * 1024;
// The group build replaces the global prefix sort with a hash grouping plus a
// per-partition sort. It wins when partitions are small: the global sort spends
// most of its comparisons ordering rows of *different* partitions. Enable it
// only when the sampled distinct partition ratio implies an average partition
// of at most 16 rows; with fewer, larger partitions the normalized-key prefix
// sort stays the cheaper path. This is a data-driven runtime guard with a
// conservative fallback and needs no statistics.
constexpr double kGroupBuildMinDistinctRatio = 1.0 / 16.0;

// Runtime stat exposing which build strategy was chosen (0 = global prefix
// sort, 1 = hash group build) so debug task stats stay attributable.
constexpr std::string_view kWindowBuildShape{"windowBuildShape"};
constexpr int64_t kWindowBuildShapePrefixSort = 0;
constexpr int64_t kWindowBuildShapeGroup = 1;

// Kill switch, mirroring the other execution-layer flags: default on, set
// GPORCA_WINDOW_GROUP_BUILD=0 to force the historical prefix-sort path.
bool windowGroupBuildEnabled() {
  static const bool enabled = [] {
    const char* flag = std::getenv("GPORCA_WINDOW_GROUP_BUILD");
    return flag == nullptr || std::strcmp(flag, "0") != 0;
  }();
  return enabled;
}

// Hashes the partition-key columns of a stored row. Integral keys are hashed
// with a single fused pass over the row storage; other key types fall back to
// RowContainer::hash per column so the mechanism stays type-generic.
struct PartitionKeyHasher {
  PartitionKeyHasher(
      const RowContainer* data,
      const std::vector<std::pair<column_index_t, core::SortOrder>>&
          partitionKeys)
      : data_(data) {
    channels_.reserve(partitionKeys.size());
    columns_.reserve(partitionKeys.size());
    kinds_.reserve(partitionKeys.size());
    for (const auto& key : partitionKeys) {
      channels_.push_back(key.first);
      columns_.push_back(data_->columnAt(key.first));
      kinds_.push_back(data_->keyTypes()[key.first]->kind());
    }
    for (const auto kind : kinds_) {
      if (kind != TypeKind::TINYINT && kind != TypeKind::SMALLINT &&
          kind != TypeKind::INTEGER && kind != TypeKind::BIGINT) {
        integral_ = false;
        break;
      }
    }
  }

  uint64_t operator()(const char* row) const {
    if (!integral_) {
      char* mutableRow = const_cast<char*>(row);
      folly::Range<char**> single(&mutableRow, 1);
      uint64_t hash = 0;
      for (size_t k = 0; k < columns_.size(); ++k) {
        data_->hash(channels_[k], single, k > 0, &hash);
      }
      return hash;
    }
    uint64_t hash = 0;
    for (size_t k = 0; k < columns_.size(); ++k) {
      uint64_t valueHash;
      if (RowContainer::isNullAt(row, columns_[k])) {
        valueHash = BaseVector::kNullHash;
      } else {
        switch (kinds_[k]) {
          case TypeKind::TINYINT:
            valueHash = static_cast<uint64_t>(
                data_->readValueAt<int8_t>(row, columns_[k].offset()));
            break;
          case TypeKind::SMALLINT:
            valueHash = static_cast<uint64_t>(
                data_->readValueAt<int16_t>(row, columns_[k].offset()));
            break;
          case TypeKind::INTEGER:
            valueHash = static_cast<uint64_t>(
                data_->readValueAt<int32_t>(row, columns_[k].offset()));
            break;
          default:
            valueHash = static_cast<uint64_t>(
                data_->readValueAt<int64_t>(row, columns_[k].offset()));
            break;
        }
      }
      hash = bits::hashMix(hash, valueHash);
    }
    return bits::hashMix(hash, 0x9e3779b97f4a7c15ULL);
  }

 private:
  const RowContainer* data_;
  std::vector<column_index_t> channels_;
  std::vector<RowColumn> columns_;
  std::vector<TypeKind> kinds_;
  bool integral_{true};
};

std::vector<CompareFlags> makeCompareFlags(
    int32_t numPartitionKeys,
    const std::vector<core::SortOrder>& sortingOrders) {
  std::vector<CompareFlags> compareFlags;
  compareFlags.reserve(numPartitionKeys + sortingOrders.size());

  for (auto i = 0; i < numPartitionKeys; ++i) {
    compareFlags.push_back({});
  }

  for (const auto& order : sortingOrders) {
    compareFlags.push_back(
        {order.isNullsFirst(), order.isAscending(), false /*equalsOnly*/});
  }

  return compareFlags;
}
} // namespace

SortWindowBuild::SortWindowBuild(
    const std::shared_ptr<const core::WindowNode>& node,
    velox::memory::MemoryPool* pool,
    common::PrefixSortConfig&& prefixSortConfig,
    const common::SpillConfig* spillConfig,
    tsan_atomic<bool>* nonReclaimableSection,
    folly::Synchronized<OperatorStats>* opStats,
    exec::SpillStats* spillStats)
    : WindowBuild(node, pool, spillConfig, nonReclaimableSection),
      numPartitionKeys_{node->partitionKeys().size()},
      compareFlags_{makeCompareFlags(numPartitionKeys_, node->sortingOrders())},
      pool_(pool),
      prefixSortConfig_(prefixSortConfig),
      opStats_(opStats),
      spillStats_(spillStats),
      sortedRows_(0, memory::StlAllocator<char*>(*pool)),
      partitionStartRows_(0, memory::StlAllocator<char*>(*pool)) {
  VELOX_CHECK_NOT_NULL(pool_);
  VELOX_CHECK_NOT_NULL(opStats_);
  initializeRowContainer(pool_);
  initializeDecodedInputVectors();
  allKeyInfo_.reserve(partitionKeyInfo_.size() + sortKeyInfo_.size());
  allKeyInfo_.insert(
      allKeyInfo_.cend(), partitionKeyInfo_.begin(), partitionKeyInfo_.end());
  allKeyInfo_.insert(
      allKeyInfo_.cend(), sortKeyInfo_.begin(), sortKeyInfo_.end());
  partitionStartRows_.resize(0);
}

void SortWindowBuild::addInput(RowVectorPtr input) {
  for (auto i = 0; i < inputChannels_.size(); ++i) {
    decodedInputVectors_[i].decode(*input->childAt(inputChannels_[i]));
  }

  ensureInputFits(input);

  // Add all the rows into the RowContainer.
  for (auto row = 0; row < input->size(); ++row) {
    addDecodedInputRow(decodedInputVectors_, row);
  }
}

void SortWindowBuild::addDecodedInputRow(
    std::vector<DecodedVector>& decodedInputVectors,
    vector_size_t row) {
  char* newRow = data_->newRow();

  for (auto col = 0; col < inputChannels_.size(); ++col) {
    data_->store(decodedInputVectors[col], row, newRow, col);
  }

  numRows_++;
}

void SortWindowBuild::ensureInputFits(const RowVectorPtr& input) {
  if (spillConfig_ == nullptr) {
    // Spilling is disabled.
    return;
  }

  if (data_->numRows() == 0) {
    // Nothing to spill.
    return;
  }

  // Test-only spill path.
  if (testingTriggerSpill(pool_->name())) {
    spill();
    return;
  }

  auto [freeRows, outOfLineFreeBytes] = data_->freeSpace();
  const auto outOfLineBytes =
      data_->stringAllocator().retainedSize() - outOfLineFreeBytes;
  const auto outOfLineBytesPerRow = outOfLineBytes / data_->numRows();

  const auto currentUsage = data_->pool()->usedBytes();
  const auto minReservationBytes =
      currentUsage * spillConfig_->minSpillableReservationPct / 100;
  const auto availableReservationBytes = data_->pool()->availableReservation();
  const auto incrementBytes =
      data_->sizeIncrement(input->size(), outOfLineBytesPerRow * input->size());

  // First to check if we have sufficient minimal memory reservation.
  if (availableReservationBytes >= minReservationBytes) {
    if ((freeRows > input->size()) &&
        (outOfLineBytes == 0 ||
         outOfLineFreeBytes >= outOfLineBytesPerRow * input->size())) {
      // Enough free rows for input rows and enough variable length free space.
      return;
    }
  }

  // Check if we can increase reservation. The increment is the largest of twice
  // the maximum increment from this input and 'spillableReservationGrowthPct_'
  // of the current memory usage.
  const auto targetIncrementBytes = std::max<int64_t>(
      incrementBytes * 2,
      currentUsage * spillConfig_->spillableReservationGrowthPct / 100);
  {
    memory::ReclaimableSectionGuard guard(nonReclaimableSection_);
    if (data_->pool()->maybeReserve(targetIncrementBytes)) {
      return;
    }
  }

  LOG(WARNING) << "Failed to reserve " << succinctBytes(targetIncrementBytes)
               << " for memory pool " << data_->pool()->name()
               << ", root pool: " << data_->pool()->root()->name()
               << ", used: " << succinctBytes(data_->pool()->usedBytes())
               << ", reservation: "
               << succinctBytes(data_->pool()->reservedBytes())
               << ", root pool reservation: "
               << succinctBytes(data_->pool()->root()->reservedBytes());
}

void SortWindowBuild::ensureSortFits() {
  // Check if spilling is enabled or not.
  if (spillConfig_ == nullptr) {
    return;
  }

  // Test-only spill path.
  if (testingTriggerSpill(pool_->name())) {
    spill();
    return;
  }

  if (spiller_ != nullptr) {
    return;
  }

  // The memory for std::vector sorted rows, `partitionStartRows_` and prefix
  // sort required buffer.
  uint64_t sortBufferToReserve =
      numRows_ * (sizeof(char*) + sizeof(vector_size_t)) +
      PrefixSort::maxRequiredBytes(
          data_.get(), compareFlags_, prefixSortConfig_, pool_);
  {
    memory::ReclaimableSectionGuard guard(nonReclaimableSection_);
    if (pool_->maybeReserve(sortBufferToReserve)) {
      return;
    }
  }

  LOG(WARNING) << fmt::format(
      "Failed to reserve {} for sort window build from memory pool {}, usage: {}, reservation: {}",
      succinctBytes(sortBufferToReserve),
      pool_->name(),
      succinctBytes(pool_->usedBytes()),
      succinctBytes(pool_->reservedBytes()));
}

void SortWindowBuild::setupSpiller() {
  VELOX_CHECK_NULL(spiller_);
  const auto sortingKeys = SpillState::makeSortingKeys(compareFlags_);
  spiller_ = std::make_unique<SortInputSpiller>(
      data_.get(), inputType_, sortingKeys, spillConfig_, spillStats_);
}

void SortWindowBuild::spill() {
  if (spiller_ == nullptr) {
    setupSpiller();
  }

  spiller_->spill();
  data_->clear();
  data_->pool()->release();
}

std::optional<exec::SpillStats> SortWindowBuild::spilledStats() const {
  if (spiller_ == nullptr) {
    return std::nullopt;
  }
  return spiller_->stats();
}

// Use double front and back search algorithm to find next partition start row.
// It is more efficient than linear or binary search.
// This algorithm is described at
// https://medium.com/@insomniocode/search-algorithm-double-front-and-back-20f5f28512e7
vector_size_t SortWindowBuild::findNextPartitionStartRow(vector_size_t start) {
  auto partitionCompare = [&](const char* lhs, const char* rhs) -> bool {
    return compareRowsWithKeys(lhs, rhs, partitionKeyInfo_);
  };

  auto left = start;
  auto right = left + 1;
  auto lastPosition = sortedRows_.size();
  while (right < lastPosition) {
    auto distance = 1;
    for (; distance < lastPosition - left; distance *= 2) {
      right = left + distance;
      if (partitionCompare(sortedRows_[left], sortedRows_[right]) != 0) {
        lastPosition = right;
        break;
      }
    }
    left += distance / 2;
    right = left + 1;
  }
  return right;
}

void SortWindowBuild::computePartitionStartRows() {
  partitionStartRows_.reserve(numRows_);

  // Using a sequential traversal to find changing partitions.
  // This algorithm is inefficient and can be changed
  // i) Use a binary search kind of strategy.
  // ii) If we use a Hashtable instead of a full sort then the count
  // of rows in the partition can be directly used.
  partitionStartRows_.push_back(0);

  VELOX_CHECK_GT(sortedRows_.size(), 0);

  vector_size_t start = 0;
  while (start < sortedRows_.size()) {
    auto next = findNextPartitionStartRow(start);
    partitionStartRows_.push_back(next);
    start = next;
  }
}

void SortWindowBuild::sortPartitions() {
  sortedRows_.resize(numRows_);
  RowContainerIterator iter;
  data_->listRows(&iter, numRows_, sortedRows_.data());

  const size_t numRows = sortedRows_.size();
  const PartitionKeyHasher hashPartitionKeys(data_.get(), partitionKeyInfo_);

  // Adaptive gate: group by partition-key hash when the shape is "many small
  // partitions", otherwise keep the global prefix sort. The sample is strided
  // over the container so it does not depend on insertion clustering.
  const bool useGroupBuild = [&]() {
    if (!windowGroupBuildEnabled() || numRows < kGroupBuildMinRows ||
        numRows > kGroupBuildMaxRows) {
      return false;
    }
    const size_t sampleSize = std::min(numRows, kGroupBuildSampleRows);
    const size_t step = std::max<size_t>(1, numRows / sampleSize);
    std::vector<uint64_t> sampleHashes;
    sampleHashes.reserve(sampleSize);
    for (size_t i = 0; i < numRows && sampleHashes.size() < sampleSize;
         i += step) {
      sampleHashes.push_back(hashPartitionKeys(sortedRows_[i]));
    }
    std::sort(sampleHashes.begin(), sampleHashes.end());
    sampleHashes.erase(
        std::unique(sampleHashes.begin(), sampleHashes.end()),
        sampleHashes.end());
    return static_cast<double>(sampleHashes.size()) / sampleSize >=
        kGroupBuildMinDistinctRatio;
  }();

  const auto recordBuildShape = [&](int64_t shape) {
    auto lockedStats = opStats_->wlock();
    lockedStats->runtimeStats[std::string(kWindowBuildShape)] =
        RuntimeMetric(shape);
  };

  if (useGroupBuild) {
    // Group rows by partition key instead of running one global sort:
    //   1) hash the partition keys of every row,
    //   2) sort "leading hash bits | row index" words (8 bytes per row, half
    //      the traffic of sorting (key, pointer) pairs),
    //   3) split equal-hash runs with the full key comparison - so hash
    //      collisions stay correct - and order the rows inside each partition.
    std::vector<uint64_t> sortWords(numRows);
    for (size_t i = 0; i < numRows; ++i) {
      const auto hash = hashPartitionKeys(sortedRows_[i]);
      sortWords[i] =
          (hash & 0xFFFFFFFF00000000ULL) | static_cast<uint64_t>(i);
    }
    std::sort(sortWords.begin(), sortWords.end());

    std::vector<char*> originalOrder(sortedRows_.begin(), sortedRows_.end());
    for (size_t i = 0; i < numRows; ++i) {
      const auto source = static_cast<uint32_t>(sortWords[i]);
      sortedRows_[i] = originalOrder[source];
    }

    const auto sortPartitionRows = [&](size_t begin, size_t end) {
      if (end - begin <= 16) {
        // Micro-partitions dominate in the gated shape, so insertion sort
        // avoids the per-comparison call overhead of std::sort.
        for (size_t i = begin + 1; i < end; ++i) {
          char* row = sortedRows_[i];
          size_t pos = i;
          while (pos > begin &&
                 compareRowsWithKeys(row, sortedRows_[pos - 1], allKeyInfo_)) {
            sortedRows_[pos] = sortedRows_[pos - 1];
            --pos;
          }
          sortedRows_[pos] = row;
        }
      } else {
        std::sort(
            sortedRows_.begin() + begin,
            sortedRows_.begin() + end,
            [&](const char* lhs, const char* rhs) {
              return compareRowsWithKeys(lhs, rhs, allKeyInfo_);
            });
      }
    };

    partitionStartRows_.clear();
    partitionStartRows_.push_back(0);
    size_t runStart = 0;
    while (runStart < numRows) {
      const uint64_t runKey = sortWords[runStart] >> 32;
      size_t runEnd = runStart + 1;
      while (runEnd < numRows && (sortWords[runEnd] >> 32) == runKey) {
        ++runEnd;
      }
      if (runEnd - runStart > 1) {
        sortPartitionRows(runStart, runEnd);
        for (size_t i = runStart + 1; i < runEnd; ++i) {
          if (compareRowsWithKeys(
                  sortedRows_[i - 1], sortedRows_[i], partitionKeyInfo_)) {
            partitionStartRows_.push_back(i);
          }
        }
      }
      // Rows in different hash runs always belong to different partitions, so
      // every run boundary is also a partition boundary.
      if (runEnd < numRows) {
        partitionStartRows_.push_back(runEnd);
      }
      runStart = runEnd;
    }
    partitionStartRows_.push_back(numRows);
    recordBuildShape(kWindowBuildShapeGroup);
    return;
  }

  // This is a very inefficient but easy implementation to order the input rows
  // by partition keys + sort keys.
  // Sort the pointers to the rows in RowContainer (data_) instead of sorting
  // the rows.
  PrefixSort::sort(
      data_.get(), compareFlags_, prefixSortConfig_, pool_, sortedRows_);

  computePartitionStartRows();
  recordBuildShape(kWindowBuildShapePrefixSort);
}

void SortWindowBuild::noMoreInput() {
  if (numRows_ == 0) {
    return;
  }

  ensureSortFits();

  if (spiller_ != nullptr) {
    // Spill remaining data to avoid running out of memory while sort-merging
    // spilled data.
    spill();

    VELOX_CHECK_NULL(merge_);
    SpillPartitionSet spillPartitionSet;
    spiller_->finishSpill(spillPartitionSet);
    VELOX_CHECK_EQ(spillPartitionSet.size(), 1);
    merge_ = spillPartitionSet.begin()->second->createOrderedReader(
        *spillConfig_, pool_, spillStats_);
  } else {
    // At this point we have seen all the input rows. The operator is
    // being prepared to output rows now.
    // To prepare the rows for output in SortWindowBuild they need to
    // be separated into partitions and sort by ORDER BY keys within
    // the partition. This will order the rows for getOutput().
    sortPartitions();
  }

  // Releases the unused memory reservation after procesing input.
  pool_->release();
}

void SortWindowBuild::loadNextPartitionBatchFromSpill() {
  // Check if current partition batch still has available partitions. If so,
  // return directly.
  if (currentPartition_ < static_cast<int>(partitionStartRows_.size() - 2)) {
    return;
  }

  const int minReadBatchRows = spillConfig_->windowMinReadBatchRows;
  sortedRows_.clear();
  sortedRows_.reserve(minReadBatchRows);
  data_->clear();
  partitionStartRows_.clear();
  partitionStartRows_.reserve(minReadBatchRows);
  partitionStartRows_.push_back(0);
  currentPartition_ = -1;
  numSpillReadBatches_++;

  // Load at least #minReadBatchRows rows and a complete partition. The rows
  // might contain multiple partitions. Record the partition boundaries as
  // inMemory case. In this way, the logic of getting window partitions would be
  // identical between inMemory and spill.
  for (;;) {
    auto next = merge_->next();
    if (next == nullptr) {
      partitionStartRows_.push_back(sortedRows_.size());
      break;
    }

    bool newPartition = false;
    if (!sortedRows_.empty()) {
      CompareFlags compareFlags =
          CompareFlags::equality(CompareFlags::NullHandlingMode::kNullAsValue);

      for (auto i = 0; i < numPartitionKeys_; ++i) {
        if (data_->compare(
                sortedRows_.back(),
                data_->columnAt(i),
                next->decoded(i),
                next->currentIndex(),
                compareFlags)) {
          newPartition = true;
          break;
        }
      }
    }

    if (newPartition) {
      partitionStartRows_.push_back(sortedRows_.size());
      if (sortedRows_.size() >= minReadBatchRows) {
        break;
      }
    }

    auto* newRow = data_->newRow();
    for (auto i = 0; i < inputChannels_.size(); ++i) {
      data_->store(next->decoded(i), next->currentIndex(), newRow, i);
    }
    sortedRows_.push_back(newRow);
    next->pop();
  }

  // No more partition batches. All data is consumed.
  if (sortedRows_.empty()) {
    partitionStartRows_.clear();
    numSpillReadBatches_--;

    auto lockedOpStats = opStats_->wlock();
    lockedOpStats
        ->runtimeStats[std::string(Window::kWindowSpillReadNumBatches)] =
        RuntimeMetric(numSpillReadBatches_);
  }
}

std::shared_ptr<WindowPartition> SortWindowBuild::nextPartition() {
  VELOX_CHECK(!partitionStartRows_.empty(), "No window partitions available");

  currentPartition_++;
  VELOX_CHECK_LE(
      currentPartition_,
      partitionStartRows_.size() - 2,
      "All window partitions consumed");

  // There is partition data available now.
  auto partitionSize = partitionStartRows_[currentPartition_ + 1] -
      partitionStartRows_[currentPartition_];
  auto partition = folly::Range(
      sortedRows_.data() + partitionStartRows_[currentPartition_],
      partitionSize);
  // Reuse one WindowPartition across partitions: shapes with a high number
  // of small partitions otherwise pay a heap allocation and vector copies
  // for every partition (Q591: 8.85M partitions).
  if (reusablePartition_ == nullptr) {
    reusablePartition_ = std::make_shared<WindowPartition>(
        data_.get(), partition, inversedInputChannels_, sortKeyInfo_);
  } else {
    reusablePartition_->resetRows(partition);
  }
  return reusablePartition_;
}

bool SortWindowBuild::hasNextPartition() {
  if (merge_ != nullptr) {
    loadNextPartitionBatchFromSpill();
  }

  return partitionStartRows_.size() > 0 &&
      currentPartition_ < static_cast<int>(partitionStartRows_.size() - 2);
}
} // namespace facebook::velox::exec::window
