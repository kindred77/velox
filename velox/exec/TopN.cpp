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
#include <folly/container/F14Map.h>

#include "velox/exec/ContainerRowSerde.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/TopN.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {
TopN::TopN(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::TopNNode>& topNNode)
    : Operator(
          driverCtx,
          topNNode->outputType(),
          operatorId,
          topNNode->id(),
          OperatorType::kTopN),
      count_(topNNode->count()),
      data_(std::make_unique<RowContainer>(outputType_->children(), pool())),
      comparator_(
          outputType_,
          topNNode->sortingKeys(),
          topNNode->sortingOrders(),
          data_.get()),
      topRows_(comparator_),
      decodedVectors_(outputType_->children().size()) {
  const auto numColumns{outputType_->children().size()};
  const auto numSortingKeys{topNNode->sortingKeys().size()};
  sortingKeyColumns_.reserve(numSortingKeys);
  std::vector<bool> isSortingKey(numColumns);
  for (const auto& key : topNNode->sortingKeys()) {
    sortingKeyColumns_.emplace_back(exprToChannel(key.get(), outputType_));
    isSortingKey[sortingKeyColumns_.back()] = true;
  }
  if (numColumns > numSortingKeys) {
    nonKeyColumns_.reserve(numColumns - numSortingKeys);
    for (column_index_t i = 0; i < numColumns; ++i) {
      if (!isSortingKey[i]) {
        nonKeyColumns_.emplace_back(i);
      }
    }
  }
}

void TopN::addInput(RowVectorPtr input) {
  for (const auto col : sortingKeyColumns_) {
    decodedVectors_[col].decode(*input->childAt(col));
  }

  const bool hasNonKeyColumn{!nonKeyColumns_.empty()};
  // Maps passed rows of 'data_' to the corresponding input row number. These
  // input rows of non-key columns are later stored into data_.
  folly::F14FastMap<void*, vector_size_t> passedRows;
  for (auto row = 0; row < input->size(); ++row) {
    ++totalRows_;
    char* newRow = nullptr;
    if (monotonic_) {
      // Ring-buffer fast path: while every new row sorts before-or-equal to
      // the previous one, the top-N is exactly the last N rows seen. Keep
      // them in a ring with O(1) work per row instead of an O(log N) heap
      // update (the heap churns when the input is monotonic in the eviction
      // direction, e.g. `order by id desc` over a file stored in ascending
      // id order).
      if (lastSeenRow_ == nullptr ||
          comparator_.compare(decodedVectors_, row, lastSeenRow_) <= 0) {
        newRow = (ring_.size() < count_)
                     ? data_->newRow()
                     : data_->initializeRow(
                           ring_[ringHead_], true /* reuse */);
        data_->initializeFields(newRow);
        for (const auto col : sortingKeyColumns_) {
          data_->store(decodedVectors_[col], row, newRow, col);
        }
        if (ring_.size() < count_) {
          ring_.push_back(newRow);
        } else {
          ring_[ringHead_] = newRow;
          ringHead_ = (ringHead_ + 1) % count_;
        }
        lastSeenRow_ = newRow;
        if (hasNonKeyColumn) {
          passedRows[newRow] = row;
        }
        continue;
      }
      // The stream stopped being strictly monotone. The ring still holds the
      // exact top-N of the prefix. If the new row is worse than (or equal to)
      // the ring's worst (the current k-th best), it cannot be in the top-N:
      // discard it and keep the ring, so the O(1) fast path can continue
      // (the invariant "ring == top-N of the prefix" is preserved because
      // later rows are even better). Only when the new row is better than the
      // worst does the ring bookkeeping (oldest == worst) break and the heap
      // fallback below is required.
      if (ring_.size() == count_) {
        ++disorderRows_;
        char* worst = ring_[0];
        for (size_t i = 1; i < ring_.size(); ++i) {
          if (comparator_.compare(ring_[i], worst) > 0) {
            worst = ring_[i];
          }
        }
        if (!comparator_(decodedVectors_, row, worst)) {
          if (disorderRows_ > totalRows_ / count_) {
            // The disorder rate is too high: the O(N) worst scan on every
            // out-of-order row costs more than the heap's O(1) discard path.
            // Fall back to the priority queue below.
            monotonic_ = false;
            for (const auto r : ring_) {
              topRows_.push(r);
            }
            ring_.clear();
            ringHead_ = 0;
            lastSeenRow_ = nullptr;
          } else {
            // Discard the row and keep the ring.
            continue;
          }
        }
      }
      // The stream stopped being monotonic: the ring still holds the exact
      // top-N of the prefix. Rebuild the priority queue from it and continue
      // with the incremental path (the current row sorts after every ring
      // row, so it is discarded there).
      monotonic_ = false;
      for (const auto r : ring_) {
        topRows_.push(r);
      }
      ring_.clear();
      ringHead_ = 0;
      lastSeenRow_ = nullptr;
    }
    if (topRows_.size() < count_) {
      newRow = data_->newRow();
    } else {
      char* topRow = topRows_.top();

      if (!comparator_(decodedVectors_, row, topRow)) {
        continue;
      }
      topRows_.pop();
      // Reuse the topRow's memory.
      newRow = data_->initializeRow(topRow, true /* reuse */);
    }

    data_->initializeFields(newRow);
    for (const auto col : sortingKeyColumns_) {
      data_->store(decodedVectors_[col], row, newRow, col);
    }

    topRows_.push(newRow);
    if (hasNonKeyColumn) {
      passedRows[newRow] = row;
    }
  }

  if (hasNonKeyColumn && !passedRows.empty()) {
    for (const auto col : nonKeyColumns_) {
      decodedVectors_[col].decode(*input->childAt(col));
      for (const auto [dataRow, inputRow] : passedRows) {
        data_->store(
            decodedVectors_[col],
            inputRow,
            reinterpret_cast<char*>(dataRow),
            col);
      }
    }
  }
}

RowVectorPtr TopN::getOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  const auto numRowsToReturn = std::min<vector_size_t>(
      outputBatchSize_, rows_.size() - numRowsReturned_);
  VELOX_CHECK_GT(numRowsToReturn, 0);

  auto result = BaseVector::create<RowVector>(
      outputType_, numRowsToReturn, operatorCtx_->pool());

  for (auto i = 0; i < outputType_->size(); ++i) {
    data_->extractColumn(
        rows_.data() + numRowsReturned_,
        numRowsToReturn,
        i,
        result->childAt(i));
  }
  numRowsReturned_ += numRowsToReturn;
  finished_ = (numRowsReturned_ == rows_.size());
  return result;
}

void TopN::noMoreInput() {
  Operator::noMoreInput();
  // Drain the ring-buffer fast path, if still active.
  if (monotonic_) {
    for (const auto r : ring_) {
      topRows_.push(r);
    }
    ring_.clear();
    ringHead_ = 0;
    lastSeenRow_ = nullptr;
    monotonic_ = false;
  }
  if (topRows_.empty()) {
    finished_ = true;
    return;
  }
  rows_.resize(topRows_.size());
  for (auto i = rows_.size(); i > 0; --i) {
    rows_[i - 1] = topRows_.top();
    topRows_.pop();
  }

  outputBatchSize_ = outputBatchRows(data_->estimateRowSize());
}

bool TopN::isFinished() {
  return finished_;
}
} // namespace facebook::velox::exec
