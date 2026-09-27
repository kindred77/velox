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

#include <atomic>
#include <cstdlib>
#include <cstring>

#include "velox/common/base/SimdUtil.h"
#include "velox/expression/EvalCtx.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ConstantVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox::exec {

namespace detail {

/// Why the fused materialization did not apply (diagnostics only).
enum class MaterializeSkip {
  kNone,
  kKind,
  kTypeMismatch,
  kDisabled,
  kSize,
  kEncoding,
};

/// Diagnostic counters for the [Win] 20260923 performance study; enabled with
/// GPORCA_CONST_MATERIALIZE_STATS=1 (default off, two stderr lines at exit).
struct ConstantMaterializeStats {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> preserveCalls{0};
  std::atomic<uint64_t> applied{0};
  std::atomic<uint64_t> appliedRows{0};
  std::atomic<uint64_t> skipKind{0};
  std::atomic<uint64_t> skipTypeMismatch{0};
  std::atomic<uint64_t> skipDisabled{0};
  std::atomic<uint64_t> skipSize{0};
  std::atomic<uint64_t> skipEncoding{0};
};

inline ConstantMaterializeStats& constantMaterializeStats() {
  static ConstantMaterializeStats stats;
  return stats;
}

inline void reportConstantMaterializeStats() {
  const auto& stats = constantMaterializeStats();
  if (stats.calls.load() == 0) {
    return;
  }
  ::fprintf(
      stderr,
      "[constmat] calls=%llu preserve_calls=%llu applied=%llu"
      " applied_rows=%llu\n",
      static_cast<unsigned long long>(stats.calls.load()),
      static_cast<unsigned long long>(stats.preserveCalls.load()),
      static_cast<unsigned long long>(stats.applied.load()),
      static_cast<unsigned long long>(stats.appliedRows.load()));
  ::fprintf(
      stderr,
      "[constmat] skip_kind=%llu skip_type=%llu skip_disabled=%llu"
      " skip_size=%llu skip_encoding=%llu\n",
      static_cast<unsigned long long>(stats.skipKind.load()),
      static_cast<unsigned long long>(stats.skipTypeMismatch.load()),
      static_cast<unsigned long long>(stats.skipDisabled.load()),
      static_cast<unsigned long long>(stats.skipSize.load()),
      static_cast<unsigned long long>(stats.skipEncoding.load()));
}

inline bool constantMaterializeStatsEnabled() {
  static const bool enabled =
      ::getenv("GPORCA_CONST_MATERIALIZE_STATS") != nullptr;
  return enabled;
}

inline bool& constantMaterializeStatsRegistered() {
  static bool registered = false;
  return registered;
}

/// Accounts one constant materialization (only when the probe is enabled).
inline void accountConstantMaterialize(
    bool preservePath,
    bool applied,
    vector_size_t selected,
    MaterializeSkip skip) {
  if (!constantMaterializeStatsEnabled()) {
    return;
  }
  auto& stats = constantMaterializeStats();
  if (!constantMaterializeStatsRegistered()) {
    constantMaterializeStatsRegistered() = true;
    std::atexit(reportConstantMaterializeStats);
  }
  stats.calls.fetch_add(1, std::memory_order_relaxed);
  if (preservePath) {
    stats.preserveCalls.fetch_add(1, std::memory_order_relaxed);
  }
  if (applied) {
    stats.applied.fetch_add(1, std::memory_order_relaxed);
    stats.appliedRows.fetch_add(
        static_cast<uint64_t>(selected), std::memory_order_relaxed);
  }
  auto bump = [&](MaterializeSkip kind, std::atomic<uint64_t>& counter) {
    if (skip == kind) {
      counter.fetch_add(1, std::memory_order_relaxed);
    }
  };
  bump(MaterializeSkip::kKind, stats.skipKind);
  bump(MaterializeSkip::kTypeMismatch, stats.skipTypeMismatch);
  bump(MaterializeSkip::kDisabled, stats.skipDisabled);
  bump(MaterializeSkip::kSize, stats.skipSize);
  bump(MaterializeSkip::kEncoding, stats.skipEncoding);
}

/// Prototype switch (2026-09-27, [Win] 20260923 P3 candidate B3): enabled by
/// default; `GPORCA_CONST_MATERIALIZE=0` restores the two-pass copy-on-write
/// path inside `EvalCtx::moveOrCopyResult`.
inline bool constantMaterializeEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("GPORCA_CONST_MATERIALIZE");
    return value == nullptr || std::atoi(value) != 0;
  }();
  return enabled;
}

/// Returns 'count' (at most 64) selection bits of 'bits' starting at 'row'.
inline int32_t laneBits(const uint64_t* bits, int32_t row, int32_t count) {
  const int32_t word = row >> 6;
  const int32_t shift = row & 63;
  uint64_t mask = bits[word] >> shift;
  if (shift + count > 64) {
    mask |= bits[word + 1] << (64 - shift);
  }
  return static_cast<int32_t>(mask & bits::lowMask(count));
}

/// out[row] = selected(row) ? value : base[row], for row in [begin, end).
/// 'base' may be 'out' itself.
template <typename T>
inline void blendConstant(
    T* out,
    const T* base,
    const uint64_t* bits,
    int32_t begin,
    int32_t end,
    T value) {
  constexpr int32_t kLanes = xsimd::batch<T>::size;
  const auto valueBatch = xsimd::broadcast<T>(value);
  int32_t row = begin;
  for (; row + kLanes <= end; row += kLanes) {
    const int32_t laneMask = laneBits(bits, row, kLanes);
    if (laneMask == static_cast<int32_t>(bits::lowMask(kLanes))) {
      // Fully selected chunk: no need to read the preserved values.
      valueBatch.store_unaligned(out + row);
      continue;
    }
    const auto oldValues = xsimd::batch<T>::load_unaligned(base + row);
    const auto mask = simd::fromBitMask<T>(laneMask);
    xsimd::select(mask, valueBatch, oldValues).store_unaligned(out + row);
  }
  for (; row < end; ++row) {
    out[row] = bits::isBitSet(bits, row) ? value : base[row];
  }
}

/// out[row] = selected(row) ? trueValue : falseValue, for row in [begin, end).
template <typename T>
inline void selectConstant(
    T* out,
    const uint64_t* bits,
    int32_t begin,
    int32_t end,
    T falseValue,
    T trueValue) {
  constexpr int32_t kLanes = xsimd::batch<T>::size;
  const auto falseBatch = xsimd::broadcast<T>(falseValue);
  const auto trueBatch = xsimd::broadcast<T>(trueValue);
  int32_t row = begin;
  for (; row + kLanes <= end; row += kLanes) {
    const int32_t laneMask = laneBits(bits, row, kLanes);
    if (laneMask == 0) {
      falseBatch.store_unaligned(out + row);
      continue;
    }
    if (laneMask == static_cast<int32_t>(bits::lowMask(kLanes))) {
      trueBatch.store_unaligned(out + row);
      continue;
    }
    xsimd::select(
        simd::fromBitMask<T>(laneMask), trueBatch, falseBatch)
        .store_unaligned(out + row);
  }
  for (; row < end; ++row) {
    out[row] = bits::isBitSet(bits, row) ? trueValue : falseValue;
  }
}

template <typename T>
inline bool materializeConstantTyped(
    const VectorPtr& constant,
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result,
    MaterializeSkip& skip) {
  // The merge below only rewrites rows of the existing result; results that
  // need to grow stay with the generic path.
  if (result->size() < rows.end()) {
    // The result grows to the end of the selection.  A constant result keeps
    // its value for every row (its length is only metadata), but a flat result
    // has no values for the new rows, so that case stays with the generic path.
    if (!result->isConstantEncoding()) {
      skip = MaterializeSkip::kSize;
      return false;
    }
  }
  // The preserved rows are either a constant (a previous branch of a CASE
  // expression materialized a literal) or a flat vector.
  const bool preserveConstant = result->isConstantEncoding();
  if (!preserveConstant && !result->isFlatEncoding()) {
    skip = MaterializeSkip::kEncoding;
    return false;
  }
  const ConstantVector<T>* oldConstant =
      preserveConstant ? result->asUnchecked<ConstantVector<T>>() : nullptr;
  const T* oldValues =
      preserveConstant
      ? nullptr
      : result->asUnchecked<FlatVector<T>>()->rawValues();
  if (!preserveConstant && oldValues == nullptr) {
    // The generic path maps a flat vector without a values buffer to all
    // nulls; leave that interpretation to it.
    skip = MaterializeSkip::kEncoding;
    return false;
  }
  // Null flags of a ConstantVector are not exposed through rawNulls(); they are
  // all-null or all-not-null.
  const bool oldIsNull = oldConstant != nullptr && oldConstant->isNullAt(0);

  const auto value = constant->asUnchecked<ConstantVector<T>>()->valueAt(0);
  const bool isConstantNull = constant->isNullAt(0);
  const auto oldSize = result->size();
  const auto targetSize = std::max<vector_size_t>(oldSize, rows.end());
  const auto* bits = rows.allBits();
  const int32_t begin = rows.begin();
  const int32_t end = rows.end();
  const uint64_t* oldNulls = result->rawNulls();

  if (result.use_count() == 1 && result->isFlatEncoding()) {
    // Uniquely owned: rows outside the selection window already hold the
    // preserved values, only the window needs the constant.
    auto* values = result->asUnchecked<FlatVector<T>>()->mutableRawValues();
    blendConstant(values, values, bits, begin, end, value);
  } else {
    VectorPtr merged = context.getVector(result->type(), targetSize);
    auto* out = merged->asUnchecked<FlatVector<T>>()->mutableRawValues();
    // Rows outside the selection window keep the old values.  The bits of a
    // SelectivityVector outside [begin(), end()) are not meaningful, so those
    // ranges are not tested bit by bit.
    auto fillPreserved = [&](int32_t from, int32_t to) {
      const auto count = to - from;
      if (count <= 0) {
        return;
      }
      if (preserveConstant) {
        std::fill_n(out + from, count, oldConstant->valueAt(0));
      } else {
        std::memcpy(out + from, oldValues + from, count * sizeof(T));
      }
    };
    fillPreserved(end, targetSize);
    fillPreserved(0, begin);
    if (preserveConstant) {
      selectConstant(
          out,
          bits,
          begin,
          end,
          oldConstant->valueAt(0),
          value);
    } else {
      blendConstant(out, oldValues, bits, begin, end, value);
    }
    result = std::move(merged);
  }

  if (oldNulls != nullptr || oldIsNull || isConstantNull) {
    BufferPtr nulls = AlignedBuffer::allocate<bool>(
        targetSize, result->pool(), bits::kNotNull);
    auto* rawNulls = nulls->asMutable<uint64_t>();
    if (oldIsNull) {
      // Every row of a null constant is null, including rows beyond its length
      // when the constant grew to the end of the selection.
      std::memset(rawNulls, bits::kNullByte, bits::nbytes(targetSize));
    } else if (oldNulls != nullptr) {
      // The old vector may be shorter than the merged one when the constant
      // result grew to the end of the selection.
      std::memcpy(rawNulls, oldNulls, bits::nbytes(oldSize));
    }
    if (isConstantNull) {
      rows.setNulls(rawNulls);
    } else {
      rows.clearNulls(rawNulls);
    }
    result->setNulls(nulls);
  }
  return true;
}

} // namespace detail

/// Writes 'constant' into the rows of 'rows' of a partially populated
/// '*result', keeping the rows outside 'rows'.
///
/// The generic path (`EvalCtx::moveOrCopyResult`) makes the result writable
/// first - `BaseVector::ensureWritable` copies the rows *outside* `rows` into a
/// fresh vector when the existing one is shared (copy on write) - and then
/// copies the constant into the rows of `rows`.  Both passes iterate the
/// selection one set bit at a time, so materializing a constant costs two
/// scalar loops per batch.  For literals inside CASE/IF-like expressions, which
/// are evaluated per batch over the whole scan, this is a significant share of
/// the projection cost (measured on [Win] 20260927: ~11% of the CPU of a 300M
/// row global aggregation with six `case when <cond> then <const> else <const>`
/// projections).
///
/// The rows outside 'rows' are either a constant (a previous branch of the CASE
/// expression materialized a literal) or a flat vector (a previous branch
/// computed values), so both passes collapse into a single pass that writes the
/// constant for the selection and the preserved values elsewhere, applying the
/// selection as a SIMD blend instead of iterating per set bit.
///
/// Applies only when the result keeps its rows outside 'rows' (the same
/// condition that selects the copy-on-write path in `moveOrCopyResult`) and the
/// merge is a plain 32/64-bit integer fill; every other case (other types,
/// all-null or values-less sources, encoded/complex vectors, results that need
/// to grow, empty selections) falls back to the generic path, which remains the
/// single source of truth for the semantics.  Disable with
/// GPORCA_CONST_MATERIALIZE=0.
///
/// Returns true when '*result' was rewritten by this function; false means the
/// caller must use the generic path.
inline bool materializeConstantOverRows(
    const VectorPtr& constant,
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  const bool preservePath = result != nullptr && !result->isLazy() &&
      rows.hasSelections() && constant->isConstantEncoding() &&
      context.resultShouldBePreserved(result, rows);
  auto skip = detail::MaterializeSkip::kNone;
  bool applied = false;
  if (preservePath) {
    if (!detail::constantMaterializeEnabled()) {
      skip = detail::MaterializeSkip::kDisabled;
    } else if (*result->type() != *constant->type()) {
      skip = detail::MaterializeSkip::kTypeMismatch;
    } else {
      switch (result->typeKind()) {
        case TypeKind::INTEGER:
          applied = detail::materializeConstantTyped<int32_t>(
              constant, rows, context, result, skip);
          break;
        case TypeKind::BIGINT:
          applied = detail::materializeConstantTyped<int64_t>(
              constant, rows, context, result, skip);
          break;
        default:
          skip = detail::MaterializeSkip::kKind;
          break;
      }
    }
  }
  if (detail::constantMaterializeStatsEnabled()) {
    detail::accountConstantMaterialize(
        preservePath, applied, rows.countSelected(), skip);
  }
  return applied;
}

} // namespace facebook::velox::exec
