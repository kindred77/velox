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

#include "velox/exec/ContainerRowSerde.h"
#include "velox/exec/KeysOnlySort.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/PrefixSort.h"
#include "velox/exec/RowContainer.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec {
class SortInputSpiller;
class SortOutputSpiller;

/// A utility class to accumulate data inside and output the sorted result.
/// Spilling would be triggered if spilling is enabled and memory usage exceeds
/// limit.
class SortBuffer {
 public:
  SortBuffer(
      const RowTypePtr& input,
      const std::vector<column_index_t>& sortColumnIndices,
      const std::vector<CompareFlags>& sortCompareFlags,
      velox::memory::MemoryPool* pool,
      tsan_atomic<bool>* nonReclaimableSection,
      common::PrefixSortConfig prefixSortConfig,
      const common::SpillConfig* spillConfig = nullptr,
      exec::SpillStats* spillStats = nullptr);

  ~SortBuffer();

  void addInput(const VectorPtr& input);

  /// Indicates no more input and triggers either of:
  ///  - In-memory sorting on rows stored in 'data_' if spilling is not enabled.
  ///  - Finish spilling and setup the sort merge reader for the un-spilling
  ///  processing for the output.
  void noMoreInput();

  /// Returns the sorted output rows in batch.
  RowVectorPtr getOutput(vector_size_t maxOutputRows);

  /// Indicates if this sort buffer can spill or not.
  bool canSpill() const {
    return spillConfig_ != nullptr;
  }

  /// Invoked to spill all the rows from 'data_'.
  void spill();

  memory::MemoryPool* pool() const {
    return pool_;
  }

  std::optional<uint64_t> estimateOutputRowSize() const;

 private:
  // Ensures there is sufficient memory reserved to process 'input'.
  void ensureInputFits(const VectorPtr& input);

  // Reserves memory for output processing. If reservation cannot be increased,
  // spills enough to make output fit.
  void ensureOutputFits(vector_size_t outputBatchSize);

  // Reserves memory for sort. If reservation cannot be increased, spills enough
  // to make output fit.
  void ensureSortFits();

  void updateEstimatedOutputRowSize();

  // Invoked to initialize or reset the reusable output buffer to get output.
  void prepareOutput(vector_size_t outputBatchSize);

  // Invoked to initialize reader to read the spilled data from storage for
  // output processing.
  void prepareOutputWithSpill();

  void getOutputWithoutSpill();

  void getOutputWithSpill();

  // Spill during input stage.
  void spillInput();

  // Spill during output stage.
  void spillOutput();

  // Finish spill, and we shouldn't get any rows from non-spilled partition as
  // there is only one hash partition for SortBuffer.
  void finishSpill();

  // Returns true if the sort buffer has spilled, regardless of during input or
  // output processing. If spilled() is true, it means the sort buffer is in
  // minimal memory mode and could not be spilled further.
  bool hasSpilled() const;

  const RowTypePtr input_;

  const std::vector<CompareFlags> sortCompareFlags_;

  velox::memory::MemoryPool* const pool_;

  // The flag is passed from the associated operator such as OrderBy or
  // TableWriter to indicate if this sort buffer object is under non-reclaimable
  // execution section or not.
  tsan_atomic<bool>* const nonReclaimableSection_;

  // Configuration settings for prefix-sort.
  const common::PrefixSortConfig prefixSortConfig_;

  const common::SpillConfig* const spillConfig_;

  exec::SpillStats* const spillStats_;

  // The column projection map between 'input_' and 'spillerStoreType_' as sort
  // buffer stores the sort columns first in 'data_'.
  std::vector<IdentityProjection> columnMap_;

  // Indicates no more input. Once it is set, addInput() can't be called on this
  // sort buffer object.
  bool noMoreInput_ = false;

  // The number of received input rows.
  uint64_t numInputRows_ = 0;

  // Used to store the input data in row format.
  std::unique_ptr<RowContainer> data_;

  std::vector<char*, memory::StlAllocator<char*>> sortedRows_;

  // The data type of the rows stored in 'data_' and spilled on disk. The
  // sort key columns are stored first then the non-sorted data columns.
  RowTypePtr spillerStoreType_;

  std::unique_ptr<SortInputSpiller> inputSpiller_;

  std::unique_ptr<SortOutputSpiller> outputSpiller_;

  SpillPartitionSet spillPartitionSet_;

  // Used to merge the sorted runs from in-memory rows and spilled rows on disk.
  std::unique_ptr<TreeOfLosers<SpillMergeStream>> spillMerger_;

  // Records the source rows to copy to 'output_' in order.
  std::vector<const RowVector*> spillSources_;

  std::vector<vector_size_t> spillSourceRows_;

  // Reusable output vector.
  RowVectorPtr output_;

  // Estimated size of a single output row by using the max
  // 'data_->estimateRowSize()' across all accumulated data set.
  std::optional<uint64_t> estimatedOutputRowSize_{};

  // The number of rows that has been returned.
  uint64_t numOutputRows_{0};

  // ==== yihudb prototype: keys-only compact sort ========================
  // When every input column is a sort key and all key types are fixed-width
  // integers, sort normalized key entries directly instead of materializing
  // (row + key + row pointer) per row. Gated by GPORCA_SORT_KEYS_ONLY=1 and
  // only used when spilling is disabled; falls back to the regular path as
  // soon as an input batch carries nulls or the env is off. See KeysOnlySort.h
  // and SortBuffer.cpp for the mechanics.
  bool keysOnlyEligible_{false};
  bool keysOnlyActive_{false};
  std::optional<keysonly::KeysOnlyPlan> keysOnlyPlan_{};
  std::vector<BufferPtr> keysOnlyChunks_;
  memory::ContiguousAllocation keysOnlyEntries_;
  uint64_t keysOnlyEntriesBytes_{0};

  // Returns true when the keys-only fast path is enabled for this sort buffer
  // (env + shape check).
  bool keysOnlyEnabled() const;

  // Appends one batch of encoded key entries. Returns false when the batch
  // cannot be encoded (nulls present).
  bool keysOnlyAddInput(const VectorPtr& input);

  // Sorts the accumulated key entries.
  void keysOnlyNoMoreInput();

  // Decodes the next batch of sorted entries into 'output_'.
  void keysOnlyGetOutput();

  // Rewrites every accumulated entry back into 'data_' rows and disables the
  // fast path; used when a later batch cannot be encoded.
  void keysOnlyFallback();

  // Stores one input batch in 'data_' (regular path).
  void storeRows(const VectorPtr& input);
};
} // namespace facebook::velox::exec
