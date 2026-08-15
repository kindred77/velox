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
#include "velox/type/Filter.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {

namespace {

// Fixed-width primitive kinds supported by the compact ring storage. The row
// image of these types is a plain memcpy-able value and their comparison
// matches RowContainer's (both go through SimpleVector::comparePrimitiveAsc).
bool isCompactSupportedKind(TypeKind kind) {
  switch (kind) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::HUGEINT:
      return true;
    default:
      return false;
  }
}

template <TypeKind Kind>
using CompactValue = typename KindToFlatVector<Kind>::HashRowType;

template <TypeKind Kind>
CompactValue<Kind> readCompactValue(const char* raw) {
  CompactValue<Kind> value;
  memcpy(&value, raw, sizeof(value));
  return value;
}

template <TypeKind Kind>
void writeDecodedValue(
    const DecodedVector& decoded,
    vector_size_t row,
    char* dst) {
  const auto value = decoded.valueAt<CompactValue<Kind>>(row);
  memcpy(dst, &value, sizeof(value));
}

template <TypeKind Kind>
void writeCompactValue(
    const char* raw,
    FlatVector<CompactValue<Kind>>* flat,
    vector_size_t row,
    bool isNull) {
  if (isNull) {
    flat->setNull(row, true);
  } else {
    flat->set(row, readCompactValue<Kind>(raw));
  }
}

template <TypeKind Kind>
int compareRawValues(const char* a, const char* b) {
  return SimpleVector<CompactValue<Kind>>::comparePrimitiveAsc(
      readCompactValue<Kind>(a), readCompactValue<Kind>(b));
}

template <TypeKind Kind>
int compareRawToDecoded(
    const char* raw,
    const DecodedVector& decoded,
    vector_size_t row) {
  return SimpleVector<CompactValue<Kind>>::comparePrimitiveAsc(
      readCompactValue<Kind>(raw), decoded.valueAt<CompactValue<Kind>>(row));
}

// Fills a temporary flat vector of 'type' from compact slots. Used only when
// materializing the ring back into the RowContainer through store(), which
// keeps column stats (null counts) consistent.
template <TypeKind Kind>
void fillCompactColumn(
    const char* const* slots,
    size_t n,
    size_t nullOffset,
    size_t valueOffset,
    const TypePtr& type,
    VectorPtr& out,
    memory::MemoryPool* pool) {
  using T = CompactValue<Kind>;
  auto vec = BaseVector::create(type, n, pool);
  auto* flat = vec->as<FlatVector<T>>();
  for (size_t i = 0; i < n; ++i) {
    const char* slot = slots[i];
    writeCompactValue<Kind>(slot + valueOffset, flat, i, slot[nullOffset] != 0);
  }
  out = std::move(vec);
}

} // namespace

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
      dynamicFilterProducer_(topNNode->dynamicFilterProducer()),
      data_(std::make_unique<RowContainer>(outputType_->children(), pool())),
      comparator_(
          outputType_,
          topNNode->sortingKeys(),
          topNNode->sortingOrders(),
          data_.get()),
      topRows_(comparator_),
      sortingOrders_(topNNode->sortingOrders()),
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

  // Compact ring eligibility: every output column must be a fixed-width
  // primitive supported by the compact layout, and the ring allocation must
  // be bounded (large top-N counts keep the RowContainer-backed ring).
  bool compactEligible = true;
  size_t rowSize = 0;
  compactColumns_.reserve(numColumns);
  for (column_index_t i = 0; i < numColumns; ++i) {
    const auto& type = outputType_->childAt(i);
    if (!type->isFixedWidth() || !isCompactSupportedKind(type->kind())) {
      compactEligible = false;
      break;
    }
    compactColumns_.push_back(
        CompactColumn{rowSize, rowSize + 1, type->cppSizeInBytes()});
    rowSize += 1 + compactColumns_.back().valueSize;
  }
  if (compactEligible && rowSize > 0 &&
      rowSize * static_cast<size_t>(count_) <= kMaxCompactRingBytes) {
    compactRowSize_ = rowSize;
  } else {
    compactColumns_.clear();
  }
}

void TopN::maybeEnableCompactRing() {
  if (compactRing_ || compactRowSize_ == 0) {
    return;
  }
  compactRingStorage_.resize(static_cast<size_t>(count_) * compactRowSize_);
  compactHead_ = 0;
  compactCount_ = 0;
  lastSeenSlot_ = nullptr;
  compactRing_ = true;
}

void TopN::compactStore(char* slot, vector_size_t row) {
  for (column_index_t col = 0; col < outputType_->size(); ++col) {
    const auto& cc = compactColumns_[col];
    const bool isNull = decodedVectors_[col].isNullAt(row);
    slot[cc.nullOffset] = isNull ? 1 : 0;
    if (!isNull) {
      VELOX_DYNAMIC_TYPE_DISPATCH(
          writeDecodedValue,
          outputType_->childAt(col)->kind(),
          decodedVectors_[col],
          row,
          slot + cc.valueOffset);
    }
  }
}

int32_t TopN::compactCompareToDecoded(char* slot, vector_size_t row) const {
  // Mirrors RowComparator::compare(decodedVectors_, row, other): returns <0
  // when the input row sorts before the stored (compact) row.
  for (size_t i = 0; i < sortingKeyColumns_.size(); ++i) {
    const auto col = sortingKeyColumns_[i];
    const auto& so = sortingOrders_[i];
    const auto& cc = compactColumns_[col];
    const bool slotNull = slot[cc.nullOffset] != 0;
    const bool inputNull = decodedVectors_[col].isNullAt(row);
    int result;
    if (slotNull) {
      result = inputNull ? 0 : (so.isNullsFirst() ? -1 : 1);
    } else if (inputNull) {
      result = so.isNullsFirst() ? 1 : -1;
    } else {
      result = VELOX_DYNAMIC_TYPE_DISPATCH(
          compareRawToDecoded,
          outputType_->childAt(col)->kind(),
          slot + cc.valueOffset,
          decodedVectors_[col],
          row);
      if (!so.isAscending()) {
        result = -result;
      }
    }
    if (result != 0) {
      return -result;
    }
  }
  return 0;
}

int32_t TopN::compactCompareSlots(const char* a, const char* b) const {
  for (size_t i = 0; i < sortingKeyColumns_.size(); ++i) {
    const auto col = sortingKeyColumns_[i];
    const auto& so = sortingOrders_[i];
    const auto& cc = compactColumns_[col];
    const bool aNull = a[cc.nullOffset] != 0;
    const bool bNull = b[cc.nullOffset] != 0;
    int result;
    if (aNull) {
      result = bNull ? 0 : (so.isNullsFirst() ? -1 : 1);
    } else if (bNull) {
      result = so.isNullsFirst() ? 1 : -1;
    } else {
      result = VELOX_DYNAMIC_TYPE_DISPATCH(
          compareRawValues,
          outputType_->childAt(col)->kind(),
          a + cc.valueOffset,
          b + cc.valueOffset);
      if (!so.isAscending()) {
        result = -result;
      }
    }
    if (result != 0) {
      return result;
    }
  }
  return 0;
}

void TopN::materializeCompactRing() {
  VELOX_CHECK(compactRing_);
  const size_t n = compactCount_;
  // Arrival order starts at the oldest slot: compactHead_ once the ring is
  // full, otherwise slot 0.
  const size_t start = (n == static_cast<size_t>(count_)) ? compactHead_ : 0;
  std::vector<char*> newRows(n);
  for (size_t i = 0; i < n; ++i) {
    newRows[i] = data_->newRow();
    data_->initializeFields(newRows[i]);
  }
  // Build a temporary flat vector per column from the compact slots and store
  // it through the public RowContainer store() path. Writing rows directly
  // would bypass the column stats (null counts) that extractColumn and other
  // consumers rely on.
  std::vector<VectorPtr> flatCols(outputType_->size());
  for (column_index_t col = 0; col < outputType_->size(); ++col) {
    const auto& cc = compactColumns_[col];
    std::vector<const char*> slots(n);
    for (size_t i = 0; i < n; ++i) {
      slots[i] = compactSlot((start + i) % count_);
    }
    VELOX_DYNAMIC_TYPE_DISPATCH(
        fillCompactColumn,
        outputType_->childAt(col)->kind(),
        slots.data(),
        n,
        cc.nullOffset,
        cc.valueOffset,
        outputType_->childAt(col),
        flatCols[col],
        pool());
  }
  for (column_index_t col = 0; col < outputType_->size(); ++col) {
    DecodedVector decoded;
    decoded.decode(*flatCols[col]);
    data_->store(decoded, folly::Range<char**>(newRows.data(), n), col);
  }
  for (auto row : newRows) {
    ring_.push_back(row);
  }
  ringHead_ = 0;
  std::vector<char>().swap(compactRingStorage_);
  compactHead_ = 0;
  compactCount_ = 0;
  lastSeenSlot_ = nullptr;
  compactRing_ = false;
}

void TopN::rebuildHeapFromRing() {
  for (const auto r : ring_) {
    topRows_.push(r);
  }
  ring_.clear();
  ringHead_ = 0;
  lastSeenRow_ = nullptr;
  monotonic_ = false;
}

void TopN::addInput(RowVectorPtr input) {
  for (const auto col : sortingKeyColumns_) {
    decodedVectors_[col].decode(*input->childAt(col));
  }
  if (monotonic_ && !compactRing_) {
    maybeEnableCompactRing();
  }
  if (compactRing_) {
    // The compact ring stores every column immediately, so non-key columns
    // must be decoded here as well (the RowContainer-backed path decodes them
    // later via 'passedRows').
    for (const auto col : nonKeyColumns_) {
      decodedVectors_[col].decode(*input->childAt(col));
    }
  }

  const bool hasNonKeyColumn{!nonKeyColumns_.empty()};
  // Maps passed rows of 'data_' to the corresponding input row number. These
  // input rows of non-key columns are later stored into data_.
  folly::F14FastMap<void*, vector_size_t> passedRows;
  for (auto row = 0; row < input->size(); ++row) {
    ++totalRows_;
    char* newRow = nullptr;
    if (monotonic_) {
      if (compactRing_) {
        // Compact ring: same monotonicity invariant as the RowContainer ring
        // below, but rows are stored as raw fixed-width images (memcpy per
        // column) instead of a per-row RowContainer round-trip. The ring is
        // materialized into the RowContainer only when it exits.
        if (lastSeenSlot_ == nullptr ||
            compactCompareToDecoded(lastSeenSlot_, row) <= 0) {
          char* slot = compactSlot(compactHead_);
          compactStore(slot, row);
          lastSeenSlot_ = slot;
          compactHead_ = (compactHead_ + 1) % count_;
          compactCount_ =
              std::min(compactCount_ + 1, static_cast<size_t>(count_));
          continue;
        }
        // The stream stopped being strictly monotone. The compact ring still
        // holds the exact top-N of the prefix; mirror the RowContainer ring
        // disorder handling below (worst scan + disorder-rate fallback).
        bool rebuildFromRing = false;
        if (compactCount_ == static_cast<size_t>(count_)) {
          ++disorderRows_;
          char* worst = compactSlot(compactHead_);
          for (size_t i = 1; i < compactCount_; ++i) {
            char* candidate = compactSlot((compactHead_ + i) % count_);
            if (compactCompareSlots(candidate, worst) > 0) {
              worst = candidate;
            }
          }
          if (!(compactCompareToDecoded(worst, row) < 0)) {
            if (disorderRows_ > totalRows_ / count_) {
              // Disorder rate too high: fall back to the priority queue.
              rebuildFromRing = true;
            } else {
              // Discard the row and keep the ring.
              continue;
            }
          } else {
            // New row is better than the ring worst: ring bookkeeping breaks,
            // rebuild the heap from the ring.
            rebuildFromRing = true;
          }
        } else {
          // Ring not full yet: rebuild the heap from the ring.
          rebuildFromRing = true;
        }
        if (rebuildFromRing) {
          materializeCompactRing();
          rebuildHeapFromRing();
        }
      } else {
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
              : data_->initializeRow(ring_[ringHead_], true /* reuse */);
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
              rebuildHeapFromRing();
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
        rebuildHeapFromRing();
      }
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
  publishDynamicFilter();
}

void TopN::publishDynamicFilter() {
  if (!dynamicFilterProducer_) {
    return;
  }
  // A valid k-th bound exists only once the top-N is full.
  char* worst = nullptr;
  const bool compact = compactRing_;
  if (monotonic_) {
    if (compact) {
      if (compactCount_ < static_cast<size_t>(count_)) {
        return;
      }
      worst = compactSlot(compactHead_);
    } else {
      if (ring_.size() < static_cast<size_t>(count_)) {
        return;
      }
      worst = ring_[ringHead_];
    }
  } else {
    if (topRows_.size() < static_cast<size_t>(count_)) {
      return;
    }
    worst = topRows_.top();
  }
  // Only BigintRange-compatible sort keys are supported (integer/date).
  const auto& keyType = outputType_->childAt(sortingKeyColumns_[0]);
  if (!keyType->isBigint() && !keyType->isInteger() && !keyType->isSmallint() &&
      !keyType->isDate()) {
    return;
  }
  const auto offset = data_->columnAt(sortingKeyColumns_[0]).offset();
  const char* raw = compact
      ? worst + compactColumns_[sortingKeyColumns_[0]].valueOffset
      : worst + offset;
  int64_t kth = 0;
  if (keyType->isBigint()) {
    memcpy(&kth, raw, sizeof(int64_t));
  } else if (keyType->isDate() || keyType->isInteger()) {
    int32_t v;
    memcpy(&v, raw, sizeof(int32_t));
    kth = v;
  } else if (keyType->isSmallint()) {
    int16_t v;
    memcpy(&v, raw, sizeof(int16_t));
    kth = v;
  } else {
    return;
  }
  if (published_ && kth == lastPublishedBound_) {
    return;
  }
  lastPublishedBound_ = kth;
  published_ = true;

  auto* driver = operatorCtx_->driver();
  driver->pushdownFilters(
      this,
      {sortingKeyColumns_[0]},
      [kth](column_index_t, common::FilterPtr& filter) {
        // ASC top-N: rows whose sort key exceeds the k-th worst can no longer
        // enter the top-N; publish an upper bound so the scan prunes row
        // groups whose minimum key is above it. The final correctness is
        // still arbitrated by this operator.
        filter = std::make_shared<common::BigintRange>(
            std::numeric_limits<int64_t>::min(), kth, true);
        return true;
      });
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
    if (compactRing_) {
      materializeCompactRing();
    }
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
