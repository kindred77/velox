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

#include "SortBuffer.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/Spiller.h"

#include <cstdlib>
#include <iostream>
#include <atomic>

namespace facebook::velox::exec {

namespace {

/// Prototype for the [Win] 20260923 sort-row bulk allocation (tracking doc
/// section 13/14): allocate the rows of one input batch with a single
/// AllocationPool call instead of one call per row. Default off; set
/// GPORCA_SORT_ROW_BULK_ALLOC=1 to enable it for A/B measurement.
bool sortRowBulkAlloc() {
  static const bool enabled = [] {
    const char* value = std::getenv("GPORCA_SORT_ROW_BULK_ALLOC");
    return value != nullptr && std::atoi(value) != 0;
  }();
  return enabled;
}

/// my_gporca prototype: keys-only compact sort (see KeysOnlySort.h). Default
/// on; set GPORCA_SORT_KEYS_ONLY=0 for the one-line rollback to the regular
/// (row container + row pointer) sort.
bool keysOnlySortEnv() {
  static const bool enabled = [] {
    const char* value = std::getenv("GPORCA_SORT_KEYS_ONLY");
    return value == nullptr || std::atoi(value) != 0;
  }();
  return enabled;
}

/// Diagnostic probe (default off, zero overhead unless enabled): counts how
/// many SortBuffers take the keys-only fast path and how many rows they encode
/// in this process; printed once at exit. GPORCA_SORT_KEYS_ONLY_STATS=1.
struct KeysOnlySortStats {
  std::atomic<uint64_t> buffers{0};
  std::atomic<uint64_t> rows{0};
  const bool enabled = [] {
    const char* value = std::getenv("GPORCA_SORT_KEYS_ONLY_STATS");
    return value != nullptr && std::atoi(value) != 0;
  }();

  ~KeysOnlySortStats() {
    if (enabled) {
      std::cerr << "[sortkeysonly] sortBuffers=" << buffers.load()
                << " rows=" << rows.load() << "\n";
    }
  }
};

KeysOnlySortStats& keysOnlySortStats() {
  static KeysOnlySortStats stats;
  return stats;
}

} // namespace

SortBuffer::SortBuffer(
    const RowTypePtr& input,
    const std::vector<column_index_t>& sortColumnIndices,
    const std::vector<CompareFlags>& sortCompareFlags,
    velox::memory::MemoryPool* pool,
    tsan_atomic<bool>* nonReclaimableSection,
    common::PrefixSortConfig prefixSortConfig,
    const common::SpillConfig* spillConfig,
    exec::SpillStats* spillStats)
    : input_(input),
      sortCompareFlags_(sortCompareFlags),
      pool_(pool),
      nonReclaimableSection_(nonReclaimableSection),
      prefixSortConfig_(prefixSortConfig),
      spillConfig_(spillConfig),
      spillStats_(spillStats),
      sortedRows_(0, memory::StlAllocator<char*>(*pool)) {
  VELOX_CHECK_GE(input_->children().size(), sortCompareFlags_.size());
  VELOX_CHECK_GT(sortCompareFlags_.size(), 0);
  VELOX_CHECK_EQ(sortColumnIndices.size(), sortCompareFlags_.size());
  VELOX_CHECK_NOT_NULL(nonReclaimableSection_);

  std::vector<TypePtr> sortedColumnTypes;
  std::vector<TypePtr> nonSortedColumnTypes;
  std::vector<std::string> sortedSpillColumnNames;
  std::vector<TypePtr> sortedSpillColumnTypes;
  sortedColumnTypes.reserve(sortColumnIndices.size());
  nonSortedColumnTypes.reserve(input->size() - sortColumnIndices.size());
  sortedSpillColumnNames.reserve(input->size());
  sortedSpillColumnTypes.reserve(input->size());
  std::unordered_set<column_index_t> sortedChannelSet;
  // Sorted key columns.
  for (column_index_t i = 0; i < sortColumnIndices.size(); ++i) {
    columnMap_.emplace_back(IdentityProjection(i, sortColumnIndices.at(i)));
    sortedColumnTypes.emplace_back(input_->childAt(sortColumnIndices.at(i)));
    sortedSpillColumnTypes.emplace_back(
        input_->childAt(sortColumnIndices.at(i)));
    sortedSpillColumnNames.emplace_back(input->nameOf(sortColumnIndices.at(i)));
    sortedChannelSet.emplace(sortColumnIndices.at(i));
  }
  // Non-sorted key columns.
  for (column_index_t i = 0, nonSortedIndex = sortCompareFlags_.size();
       i < input_->size();
       ++i) {
    if (sortedChannelSet.count(i) != 0) {
      continue;
    }
    columnMap_.emplace_back(nonSortedIndex++, i);
    nonSortedColumnTypes.emplace_back(input_->childAt(i));
    sortedSpillColumnTypes.emplace_back(input_->childAt(i));
    sortedSpillColumnNames.emplace_back(input->nameOf(i));
  }

  data_ = std::make_unique<RowContainer>(
      sortedColumnTypes, nonSortedColumnTypes, /*useListRowIndex=*/true, pool_);
  spillerStoreType_ =
      ROW(std::move(sortedSpillColumnNames), std::move(sortedSpillColumnTypes));

  // Keys-only compact sort eligibility: env on, spilling disabled, every input
  // column is a sort key and all key types are fixed-width integers.
  keysOnlyEligible_ = keysOnlySortEnv() && spillConfig_ == nullptr &&
      input_->size() == sortColumnIndices.size();
  if (keysOnlyEligible_) {
    std::vector<TypePtr> keyTypes;
    keyTypes.reserve(sortColumnIndices.size());
    for (const auto index : sortColumnIndices) {
      const auto& type = input_->childAt(index);
      if (!keysonly::supportedKind(type->kind())) {
        keysOnlyEligible_ = false;
        break;
      }
      keyTypes.emplace_back(type);
    }
    if (keysOnlyEligible_) {
      const std::vector<bool> columnHasNulls(keyTypes.size(), false);
      const std::vector<std::optional<uint32_t>> maxStringLengths(
          keyTypes.size(), std::nullopt);
      std::vector<TypeKind> keyKinds;
      keyKinds.reserve(keyTypes.size());
      for (const auto& type : keyTypes) {
        keyKinds.emplace_back(type->kind());
      }
      keysOnlyPlan_.emplace(keysonly::KeysOnlyPlan{
          PrefixSortLayout::generate(
              keyTypes,
              columnHasNulls,
              sortCompareFlags_,
              prefixSortConfig_.maxNormalizedKeyBytes,
              prefixSortConfig_.maxStringPrefixLength,
              maxStringLengths),
          std::move(keyKinds)});
      // Only the all-normalized form is supported by the prototype.
      if (!keysOnlyPlan_->layout.hasNormalizedKeys ||
          keysOnlyPlan_->layout.hasNonNormalizedKey ||
          keysOnlyPlan_->layout.numNormalizedKeys !=
              keysOnlyPlan_->keyKinds.size()) {
        keysOnlyPlan_.reset();
        keysOnlyEligible_ = false;
      }
    }
  }
}

bool SortBuffer::keysOnlyEnabled() const {
  return keysOnlyEligible_ && keysOnlyPlan_.has_value();
}

SortBuffer::~SortBuffer() {
  pool_->release();
}

void SortBuffer::addInput(const VectorPtr& input) {
  velox::common::testutil::TestValue::adjust(
      "facebook::velox::exec::SortBuffer::addInput", this);

  VELOX_CHECK(!noMoreInput_);
  ensureInputFits(input);

  if (keysOnlyEnabled()) {
    if (keysOnlyActive_) {
      if (keysOnlyAddInput(input)) {
        numInputRows_ += input->size();
        return;
      }
      // The batch cannot be encoded (nulls); rewrite what we have into rows and
      // continue on the regular path.
      keysOnlyFallback();
    } else if (keysOnlyAddInput(input)) {
      keysOnlyActive_ = true;
      numInputRows_ += input->size();
      return;
    } else {
      // The first batch cannot use the fast path; keep the regular one for the
      // whole sort.
      keysOnlyEligible_ = false;
    }
  }

  storeRows(input);
  numInputRows_ += input->size();
}

// Stores one batch of rows in 'data_'. The caller accounts the row count.
void SortBuffer::storeRows(const VectorPtr& input) {
  const SelectivityVector allRows(input->size());
  const auto numRows = input->size();
  std::vector<char*> rows(numRows);
  if (sortRowBulkAlloc()) {
    // Prototype: one allocation for the whole batch, row addresses derived
    // arithmetically (see RowContainer::newRows).
    char* first = data_->newRows(numRows);
    const uint32_t stride = data_->rowStride();
    for (auto row = 0; row < numRows; ++row) {
      rows[row] = first + static_cast<uint64_t>(row) * stride;
    }
  } else {
    for (int row = 0; row < numRows; ++row) {
      rows[row] = data_->newRow();
    }
  }
  const auto* inputRow = input->as<RowVector>();
  for (const auto& columnProjection : columnMap_) {
    DecodedVector decoded(
        *inputRow->childAt(columnProjection.outputChannel), allRows);
    data_->store(
        decoded,
        folly::Range(rows.data(), input->size()),
        columnProjection.inputChannel);
  }
}

bool SortBuffer::keysOnlyAddInput(const VectorPtr& input) {
  const auto* inputRow = input->as<RowVector>();
  const vector_size_t numRows = input->size();
  if (numRows == 0) {
    return true;
  }
  // The prototype layout reserves no null bytes; refuse batches with nulls so
  // the caller can fall back to the regular path.
  const auto numColumns = inputRow->children().size();
  for (column_index_t i = 0; i < numColumns; ++i) {
    if (inputRow->childAt(i)->mayHaveNulls()) {
      return false;
    }
  }
  const auto stride = keysOnlyPlan_->layout.normalizedBufferSize;
  auto chunk = AlignedBuffer::allocate<char>(
      static_cast<uint64_t>(numRows) * stride, pool_);
  char* entries = chunk->asMutable<char>();

  const SelectivityVector allRows(numRows);
  std::vector<std::unique_ptr<DecodedVector>> decoded;
  std::vector<const DecodedVector*> keyViews;
  decoded.reserve(columnMap_.size());
  keyViews.reserve(columnMap_.size());
  // Sort key i is input column columnMap_[i].outputChannel; keys-only shapes
  // have no non-sorted columns so columnMap_ covers every input column.
  for (const auto& columnProjection : columnMap_) {
    decoded.emplace_back(std::make_unique<DecodedVector>(
        *inputRow->childAt(columnProjection.outputChannel), allRows));
    keyViews.emplace_back(decoded.back().get());
  }
  for (vector_size_t row = 0; row < numRows; ++row) {
    keysonly::encodeRow(
        *keysOnlyPlan_,
        keyViews,
        row,
        entries + static_cast<uint64_t>(row) * stride);
  }
  keysOnlyChunks_.emplace_back(std::move(chunk));
  keysOnlyEntriesBytes_ += static_cast<uint64_t>(numRows) * stride;
  auto& stats = keysOnlySortStats();
  if (stats.enabled) {
    if (keysOnlyChunks_.size() == 1) {
      ++stats.buffers;
    }
    stats.rows += numRows;
  }
  return true;
}

void SortBuffer::keysOnlyNoMoreInput() {
  const auto stride = keysOnlyPlan_->layout.normalizedBufferSize;
  const uint64_t totalBytes = numInputRows_ * stride;
  VELOX_CHECK_EQ(keysOnlyEntriesBytes_, totalBytes);
  const auto numPages = memory::AllocationTraits::numPages(totalBytes);
  pool_->allocateContiguous(numPages, keysOnlyEntries_);
  char* buffer = keysOnlyEntries_.data<char>();
  uint64_t offset = 0;
  for (const auto& chunk : keysOnlyChunks_) {
    std::memcpy(buffer + offset, chunk->as<char>(), chunk->size());
    offset += chunk->size();
  }
  VELOX_CHECK_EQ(offset, totalBytes);
  keysOnlyChunks_.clear();
  keysOnlyEntriesBytes_ = 0;
  updateEstimatedOutputRowSize();

  auto swapBuffer = AlignedBuffer::allocate<char>(stride, pool_);
  prefixsort::PrefixSortRunner sortRunner(
      stride, swapBuffer->asMutable<char>());
  sortRunner.quickSort(
      buffer,
      buffer + totalBytes,
      [&](const char* left, const char* right) {
        // PrefixSortRunner uses a three-way comparator (-1/0/1).
        return keysonly::compareEntries(*keysOnlyPlan_, left, right);
      });
}

void SortBuffer::keysOnlyGetOutput() {
  const auto stride = keysOnlyPlan_->layout.normalizedBufferSize;
  const char* entries = keysOnlyEntries_.data<char>();
  // Sort key i lives in input column columnMap_[i].outputChannel and is
  // returned in that same output child (see the columnMap_ construction).
  std::vector<VectorPtr> outputs;
  outputs.reserve(columnMap_.size());
  for (const auto& columnProjection : columnMap_) {
    outputs.emplace_back(output_->childAt(columnProjection.outputChannel));
  }
  const auto batchRows = output_->size();
  for (vector_size_t row = 0; row < batchRows; ++row) {
    keysonly::decodeEntry(
        *keysOnlyPlan_,
        entries + (numOutputRows_ + row) * stride,
        outputs,
        row);
  }
  numOutputRows_ += batchRows;
}

void SortBuffer::keysOnlyFallback() {
  const auto stride = keysOnlyPlan_->layout.normalizedBufferSize;
  // Entries still live in 'keysOnlyChunks_'; stage them in one buffer so the
  // decode loop can index rows directly.
  std::vector<char> staged(keysOnlyEntriesBytes_);
  uint64_t offset = 0;
  for (const auto& chunk : keysOnlyChunks_) {
    std::memcpy(staged.data() + offset, chunk->as<char>(), chunk->size());
    offset += chunk->size();
  }
  const char* entries = staged.data();

  constexpr uint64_t kBatchRows = 1024;
  for (uint64_t row = 0; row < numInputRows_; row += kBatchRows) {
    const auto batchRows = static_cast<vector_size_t>(
        std::min<uint64_t>(kBatchRows, numInputRows_ - row));
    auto batch = BaseVector::create(input_, batchRows, pool_);
    auto* batchRow = batch->as<RowVector>();
    std::vector<VectorPtr> outputs;
    outputs.reserve(columnMap_.size());
    for (const auto& columnProjection : columnMap_) {
      outputs.emplace_back(
          batchRow->childAt(columnProjection.outputChannel));
    }
    for (vector_size_t i = 0; i < batchRows; ++i) {
      keysonly::decodeEntry(
          *keysOnlyPlan_, entries + (row + i) * stride, outputs, i);
    }
    storeRows(batch);
  }
  keysOnlyChunks_.clear();
  keysOnlyEntriesBytes_ = 0;
  keysOnlyActive_ = false;
  keysOnlyEligible_ = false;
}

void SortBuffer::noMoreInput() {
  velox::common::testutil::TestValue::adjust(
      "facebook::velox::exec::SortBuffer::noMoreInput", this);
  VELOX_CHECK(!noMoreInput_);
  VELOX_CHECK_NULL(outputSpiller_);

  // It may trigger spill, make sure it's triggered before noMoreInput_ is set.
  ensureSortFits();

  noMoreInput_ = true;

  // No data.
  if (numInputRows_ == 0) {
    return;
  }

  if (keysOnlyActive_) {
    keysOnlyNoMoreInput();
    return;
  }

  if (inputSpiller_ == nullptr) {
    VELOX_CHECK_EQ(numInputRows_, data_->numRows());
    updateEstimatedOutputRowSize();
    // Sort the pointers to the rows in RowContainer (data_) instead of sorting
    // the rows.
    // TODO: Reuse 'RowContainer::rowPointers_'.
    sortedRows_.resize(numInputRows_);
    RowContainerIterator iter;
    data_->listRows(&iter, numInputRows_, sortedRows_.data());
    PrefixSort::sort(
        data_.get(), sortCompareFlags_, prefixSortConfig_, pool_, sortedRows_);
  } else {
    // Spill the remaining in-memory state to disk if spilling has been
    // triggered on this sort buffer. This is to simplify query OOM prevention
    // when producing output as we don't support to spill during that stage as
    // for now.
    spill();

    finishSpill();
  }

  // Releases the unused memory reservation after procesing input.
  pool_->release();
}

RowVectorPtr SortBuffer::getOutput(vector_size_t maxOutputRows) {
  SCOPE_EXIT {
    pool_->release();
  };

  VELOX_CHECK(noMoreInput_);

  if (numOutputRows_ == numInputRows_) {
    return nullptr;
  }
  VELOX_CHECK_GT(maxOutputRows, 0);
  VELOX_CHECK_GT(numInputRows_, numOutputRows_);
  const vector_size_t batchSize =
      std::min<uint64_t>(numInputRows_ - numOutputRows_, maxOutputRows);
  ensureOutputFits(batchSize);
  prepareOutput(batchSize);
  if (hasSpilled()) {
    getOutputWithSpill();
  } else {
    getOutputWithoutSpill();
  }
  return std::move(output_);
}

bool SortBuffer::hasSpilled() const {
  if (inputSpiller_ != nullptr) {
    VELOX_CHECK_NULL(outputSpiller_);
    return true;
  }
  return outputSpiller_ != nullptr;
}

void SortBuffer::spill() {
  VELOX_CHECK_NOT_NULL(
      spillConfig_, "spill config is null when SortBuffer spill is called");

  // Check if sort buffer is empty or not, and skip spill if it is empty.
  if (data_->numRows() == 0) {
    return;
  }
  updateEstimatedOutputRowSize();

  if (sortedRows_.empty()) {
    spillInput();
  } else {
    spillOutput();
  }
}

std::optional<uint64_t> SortBuffer::estimateOutputRowSize() const {
  return estimatedOutputRowSize_;
}

void SortBuffer::ensureInputFits(const VectorPtr& input) {
  // Check if spilling is enabled or not.
  if (spillConfig_ == nullptr) {
    return;
  }

  const int64_t numRows = data_->numRows();
  if (numRows == 0) {
    // 'data_' is empty. Nothing to spill.
    return;
  }

  auto [freeRows, outOfLineFreeBytes] = data_->freeSpace();
  const auto outOfLineBytes =
      data_->stringAllocator().retainedSize() - outOfLineFreeBytes;
  const int64_t flatInputBytes = input->estimateFlatSize();

  // Test-only spill path.
  if (numRows > 0 && testingTriggerSpill(pool_->name())) {
    spill();
    return;
  }

  const auto currentMemoryUsage = pool_->usedBytes();
  const auto minReservationBytes =
      currentMemoryUsage * spillConfig_->minSpillableReservationPct / 100;
  const auto availableReservationBytes = pool_->availableReservation();
  const int64_t estimatedIncrementalBytes =
      data_->sizeIncrement(input->size(), outOfLineBytes ? flatInputBytes : 0);
  if (availableReservationBytes > minReservationBytes) {
    // If we have enough free rows for input rows and enough variable length
    // free space for the vector's flat size, no need for spilling.
    if (freeRows > input->size() &&
        (outOfLineBytes == 0 || outOfLineFreeBytes >= flatInputBytes)) {
      return;
    }

    // If the current available reservation in memory pool is 2X the
    // estimatedIncrementalBytes, no need to spill.
    if (availableReservationBytes > 2 * estimatedIncrementalBytes) {
      return;
    }
  }

  // Try reserving targetIncrementBytes more in memory pool, if succeed, no
  // need to spill.
  const auto targetIncrementBytes = std::max<int64_t>(
      estimatedIncrementalBytes * 2,
      currentMemoryUsage * spillConfig_->spillableReservationGrowthPct / 100);
  {
    memory::ReclaimableSectionGuard guard(nonReclaimableSection_);
    if (pool_->maybeReserve(targetIncrementBytes)) {
      return;
    }
  }
  LOG(WARNING) << "Failed to reserve " << succinctBytes(targetIncrementBytes)
               << " for memory pool " << pool()->name()
               << ", root pool: " << pool()->root()->name()
               << ", used: " << succinctBytes(pool()->usedBytes())
               << ", reservation: " << succinctBytes(pool()->reservedBytes())
               << ", root pool reservation: "
               << succinctBytes(pool()->root()->reservedBytes());
}

void SortBuffer::ensureOutputFits(vector_size_t batchSize) {
  VELOX_CHECK_GT(batchSize, 0);
  // Check if spilling is enabled or not.
  if (spillConfig_ == nullptr) {
    return;
  }

  // Test-only spill path.
  if (testingTriggerSpill(pool_->name())) {
    spill();
    return;
  }

  if (!estimatedOutputRowSize_.has_value() || hasSpilled()) {
    return;
  }

  const uint64_t outputBufferSizeToReserve =
      estimatedOutputRowSize_.value() * batchSize * 1.2;
  {
    memory::ReclaimableSectionGuard guard(nonReclaimableSection_);
    if (pool_->maybeReserve(outputBufferSizeToReserve)) {
      return;
    }
  }
  LOG(WARNING) << "Failed to reserve "
               << succinctBytes(outputBufferSizeToReserve)
               << " for memory pool " << pool_->name()
               << ", root pool: " << pool_->root()->name()
               << ", used: " << succinctBytes(pool_->usedBytes())
               << ", reservation: " << succinctBytes(pool_->reservedBytes())
               << ", root pool reservation: "
               << succinctBytes(pool_->root()->reservedBytes());
}

void SortBuffer::ensureSortFits() {
  // Check if spilling is enabled or not.
  if (spillConfig_ == nullptr) {
    return;
  }

  // Test-only spill path.
  if (testingTriggerSpill(pool_->name())) {
    spill();
    return;
  }

  if (numInputRows_ == 0 || inputSpiller_ != nullptr) {
    return;
  }

  // The memory for std::vector sorted rows and prefix sort required buffer.
  const auto sortBufferToReserve =
      numInputRows_ * sizeof(char*) +
      PrefixSort::maxRequiredBytes(
          data_.get(), sortCompareFlags_, prefixSortConfig_, pool_);
  {
    memory::ReclaimableSectionGuard guard(nonReclaimableSection_);
    if (pool_->maybeReserve(sortBufferToReserve)) {
      return;
    }
  }

  LOG(WARNING) << fmt::format(
      "Failed to reserve {} for memory pool {}, usage: {}, reservation: {}",
      succinctBytes(sortBufferToReserve),
      pool_->name(),
      succinctBytes(pool_->usedBytes()),
      succinctBytes(pool_->reservedBytes()));
}

void SortBuffer::updateEstimatedOutputRowSize() {
  const auto optionalRowSize = data_->estimateRowSize();
  if (!optionalRowSize.has_value() || optionalRowSize.value() == 0) {
    return;
  }

  const auto rowSize = optionalRowSize.value();
  if (!estimatedOutputRowSize_.has_value()) {
    estimatedOutputRowSize_ = rowSize;
  } else if (rowSize > estimatedOutputRowSize_.value()) {
    estimatedOutputRowSize_ = rowSize;
  }
}

void SortBuffer::spillInput() {
  if (inputSpiller_ == nullptr) {
    VELOX_CHECK(!noMoreInput_);
    const auto sortingKeys = SpillState::makeSortingKeys(sortCompareFlags_);
    inputSpiller_ = std::make_unique<SortInputSpiller>(
        data_.get(), spillerStoreType_, sortingKeys, spillConfig_, spillStats_);
  }
  inputSpiller_->spill();
  data_->clear();
}

void SortBuffer::spillOutput() {
  if (hasSpilled()) {
    // Already spilled.
    return;
  }
  if (numOutputRows_ == sortedRows_.size()) {
    // All the output has been produced.
    return;
  }

  outputSpiller_ = std::make_unique<SortOutputSpiller>(
      data_.get(), spillerStoreType_, spillConfig_, spillStats_);
  auto spillRows = SpillerBase::SpillRows(
      sortedRows_.begin() + numOutputRows_,
      sortedRows_.end(),
      *memory::spillMemoryPool());
  outputSpiller_->spill(spillRows);
  data_->clear();
  sortedRows_.clear();
  sortedRows_.shrink_to_fit();
  // Finish right after spilling as the output spiller only spills at most
  // once.
  finishSpill();
}

void SortBuffer::prepareOutput(vector_size_t batchSize) {
  if (output_ != nullptr) {
    VectorPtr output = std::move(output_);
    BaseVector::prepareForReuse(output, batchSize);
    output_ = std::static_pointer_cast<RowVector>(output);
  } else {
    output_ = std::static_pointer_cast<RowVector>(
        BaseVector::create(input_, batchSize, pool_));
  }

  if (hasSpilled()) {
    spillSources_.resize(batchSize);
    spillSourceRows_.resize(batchSize);
    prepareOutputWithSpill();
  }

  VELOX_CHECK_GT(output_->size(), 0);
  VELOX_CHECK_LE(output_->size() + numOutputRows_, numInputRows_);
}

void SortBuffer::getOutputWithoutSpill() {
  if (keysOnlyActive_) {
    keysOnlyGetOutput();
    return;
  }
  VELOX_DCHECK_EQ(numInputRows_, sortedRows_.size());
  for (const auto& columnProjection : columnMap_) {
    data_->extractColumn(
        sortedRows_.data() + numOutputRows_,
        output_->size(),
        columnProjection.inputChannel,
        output_->childAt(columnProjection.outputChannel));
  }
  numOutputRows_ += output_->size();
}

void SortBuffer::getOutputWithSpill() {
  VELOX_CHECK_NOT_NULL(spillMerger_);
  VELOX_DCHECK_EQ(sortedRows_.size(), 0);

  int32_t outputRow = 0;
  int32_t outputSize = 0;
  bool isEndOfBatch = false;
  while (outputRow + outputSize < output_->size()) {
    SpillMergeStream* stream = spillMerger_->next();
    VELOX_CHECK_NOT_NULL(stream);

    spillSources_[outputSize] = &stream->current();
    spillSourceRows_[outputSize] = stream->currentIndex(&isEndOfBatch);
    ++outputSize;
    if (FOLLY_UNLIKELY(isEndOfBatch)) {
      // The stream is at end of input batch. Need to copy out the rows before
      // fetching next batch in 'pop'.
      gatherCopy(
          output_.get(),
          outputRow,
          outputSize,
          spillSources_,
          spillSourceRows_,
          columnMap_);
      outputRow += outputSize;
      outputSize = 0;
    }
    // Advance the stream.
    stream->pop();
  }
  VELOX_CHECK_EQ(outputRow + outputSize, output_->size());

  if (FOLLY_LIKELY(outputSize != 0)) {
    gatherCopy(
        output_.get(),
        outputRow,
        outputSize,
        spillSources_,
        spillSourceRows_,
        columnMap_);
  }

  numOutputRows_ += output_->size();
}

void SortBuffer::finishSpill() {
  VELOX_CHECK_NULL(spillMerger_);
  VELOX_CHECK(spillPartitionSet_.empty());
  VELOX_CHECK_EQ(
      !!(outputSpiller_ != nullptr) + !!(inputSpiller_ != nullptr),
      1,
      "inputSpiller_ {}, outputSpiller_ {}",
      inputSpiller_ == nullptr ? "set" : "null",
      outputSpiller_ == nullptr ? "set" : "null");
  if (inputSpiller_ != nullptr) {
    VELOX_CHECK(!inputSpiller_->finalized());
    inputSpiller_->finishSpill(spillPartitionSet_);
  } else {
    VELOX_CHECK(!outputSpiller_->finalized());
    outputSpiller_->finishSpill(spillPartitionSet_);
  }
  VELOX_CHECK_EQ(spillPartitionSet_.size(), 1);
}

void SortBuffer::prepareOutputWithSpill() {
  VELOX_CHECK(hasSpilled());
  if (spillMerger_ != nullptr) {
    VELOX_CHECK(spillPartitionSet_.empty());
    return;
  }

  VELOX_CHECK_EQ(spillPartitionSet_.size(), 1);
  spillMerger_ = spillPartitionSet_.begin()->second->createOrderedReader(
      *spillConfig_, pool(), spillStats_);
  spillPartitionSet_.clear();
}
} // namespace facebook::velox::exec
