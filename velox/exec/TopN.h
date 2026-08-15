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

#include "velox/exec/Operator.h"
#include "velox/exec/RowContainer.h"

namespace facebook::velox::exec {

class TopN : public Operator {
 public:
  TopN(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::TopNNode>& topNNode);

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  void noMoreInput() override;

  // Publish the current k-th sort-key bound as a dynamic filter to the input
  // scan. No-op when the top-N is not full yet or the sort key is not a
  // BigintRange-compatible type.
  void publishDynamicFilter();

  BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 private:
  const int32_t count_;

  bool finished_ = false;
  uint32_t numRowsReturned_ = 0;

  std::vector<column_index_t> sortingKeyColumns_;
  std::vector<column_index_t> nonKeyColumns_;

  // As the inputs are added to TopN operator, we use topRows_ (a priority
  // queue) to keep track of the pointers to rows stored in the
  // RowContainer (data_). We only update the RowContainer if a row is a
  // candidate for top rows. Otherwise, we will discard the row.
  // Since we use a priority queue for TopN, we perform
  // O(total_rows * logN) comparisons and require O(N) space.
  // Once all inputs are available, we copy the final set of rows to the
  // vector (rows_) in correct order. We use this vector along with the
  // RowContainer to generate the TopN's output.
  std::unique_ptr<RowContainer> data_;
  RowComparator comparator_;
  std::priority_queue<char*, std::vector<char*>, RowComparator> topRows_;

  // Ring-buffer fast path for streams that arrive monotonically in the
  // eviction direction: every new row sorts before-or-equal to the previous
  // row (e.g. `order by id desc` over a file stored in ascending id order).
  // For such a stream the top-N is exactly the last N rows seen, so the
  // priority queue can be replaced by an O(1)-per-row ring buffer instead of
  // an O(log N) heap update per row. The first row that sorts after the
  // previous one breaks the monotonic prefix: the ring (which still holds the
  // exact top-N of the prefix) is rebuilt into the priority queue and the
  // incremental path takes over. The decision is a pure performance
  // optimization; correctness does not depend on the stream being monotonic.
  bool monotonic_{true};
  // Last N rows of the monotonic prefix, in arrival order (ring buffer;
  // ringHead_ is the index of the oldest row).
  std::vector<char*> ring_;
  size_t ringHead_{0};
  // Row of the most recently seen input row (the newest ring entry). Its
  // memory stays valid until a later ring advance reuses it, which cannot
  // happen before N more rows.
  char* lastSeenRow_{nullptr};
  // Compact ring-buffer storage (alternative to the RowContainer-backed ring
  // above): when every output column is fixed-width, monotonic rows are kept
  // as raw images (1 null byte + value per column) in a preallocated ring
  // instead of round-tripping through the RowContainer on every row. The
  // per-row RowContainer cost (newRow/initializeFields/store) dominates for
  // 100M+ row monotonic streams (e.g. reverse top-N over a large ordered
  // table: `order by id, discount limit 10 offset 99999980`), while the
  // compact ring is a plain memcpy per column. Rows are materialized into the
  // RowContainer only when the ring exits (disorder break or noMoreInput),
  // i.e. at most count_ rows. Correctness does not depend on the compact path.
  bool compactRing_{false};
  // Per-column layout of a compact slot: null-flag byte then fixed-width value.
  struct CompactColumn {
    size_t nullOffset;
    size_t valueOffset;
    size_t valueSize;
  };
  std::vector<CompactColumn> compactColumns_;
  size_t compactRowSize_{0};
  std::vector<char> compactRingStorage_;
  // Index of the oldest compact slot (next one to be overwritten).
  size_t compactHead_{0};
  // Number of valid slots (0..count_).
  size_t compactCount_{0};
  char* lastSeenSlot_{nullptr};
  // Sort orders of the sorting keys, kept for the compact comparator.
  std::vector<core::SortOrder> sortingOrders_;
  // Upper bound on compact ring allocation; larger counts use the
  // RowContainer-backed ring to avoid a big upfront allocation.
  static constexpr size_t kMaxCompactRingBytes = 64 << 20;

  char* compactSlot(size_t i) {
    return compactRingStorage_.data() + i * compactRowSize_;
  }

  const char* compactSlot(size_t i) const {
    return compactRingStorage_.data() + i * compactRowSize_;
  }

  // Writes the i-th row of the decoded input vectors into the compact slot.
  void compactStore(char* slot, vector_size_t row);

  // Returns <0 if the input row sorts before the compact slot row (mirrors
  // RowComparator::compare(decodedVectors_, row, other)).
  int32_t compactCompareToDecoded(char* slot, vector_size_t row) const;

  // Returns <0 if 'a' sorts before 'b' (both compact slots).
  int32_t compactCompareSlots(const char* a, const char* b) const;

  // Converts the compact ring into RowContainer rows (in arrival order),
  // fills ring_ and switches back to the RowContainer-backed ring.
  void materializeCompactRing();

  // Moves the ring contents into the priority queue and disables the ring
  // fast path (used when the stream stops being monotonic).
  void rebuildHeapFromRing();

  // Enables the compact ring if the output columns are all fixed-width and the
  // ring allocation is bounded.
  void maybeEnableCompactRing();
  // Dynamic-filter producer: when enabled (GPORCA decided the input scan is
  // ordered by the sort key and the filter is not statically pushable), the
  // operator publishes its current k-th sort-key bound as a runtime filter
  // (BigintRange) to the scan, so row groups that can no longer improve the
  // top-N are pruned.
  bool dynamicFilterProducer_{false};
  // Last published upper bound of the sort key (ASC top-N: rows with a larger
  // key cannot enter); used to avoid re-publishing an unchanged bound.
  int64_t lastPublishedBound_{0};
  bool published_{false};
  // Number of input rows seen so far, and how many of them broke the
  // monotonic prefix. When the disorder rate is high (disorderRows_ * count_
  // exceeds totalRows_), the O(N) worst scan on every out-of-order row costs
  // more than the heap's O(1) discard path, so the ring falls back to the
  // priority queue.
  int64_t totalRows_{0};
  int64_t disorderRows_{0};
  std::vector<char*> rows_;

  std::vector<DecodedVector> decodedVectors_;
  vector_size_t outputBatchSize_;
};
} // namespace facebook::velox::exec
