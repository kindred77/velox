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
#include "velox/expression/BooleanMix.h"

#include <cstdlib>

namespace facebook::velox::exec {

namespace {

/// my_gporca prototype (2026-09-26, [Win] 20260923 section 18, candidate C2):
/// batch the per-row bit gather of dictionary-encoded booleans. `getFlatBool`
/// held 21% of Q638 (called once per AND child); the helper below packs 8 rows
/// into one byte for dense selections and drops the per-row null test when the
/// decoded base has no nulls. Flipped to default-on on 2026-09-26 ([Win]
/// 20260923 section 19: same-binary A/B Q638 -13.0%/-13.6% reproducible,
/// `getFlatBool` Exc 21.75% -> 0.06% with the batched gather at 6.93%);
/// `GPORCA_BOOL_BATCH_GATHER=0` is the one-line rollback.
bool useBatchedBoolGather() {
  static const bool enabled = []() {
    const char* env = std::getenv("GPORCA_BOOL_BATCH_GATHER");
    if (env != nullptr && *env != '\0') {
      return std::atoi(env) != 0;
    }
    return true;
  }();
  return enabled;
}

/// Gathers one bit per row from the decoded dictionary values. `valuesToSet`
/// must be zero-initialized and `indices` non-null (same assumptions as the
/// per-row loop it replaces).
void gatherDictionaryBits(
    const uint64_t* values,
    const vector_size_t* indices,
    const SelectivityVector& activeRows,
    uint64_t* valuesToSet) {
  auto* outBytes = reinterpret_cast<uint8_t*>(valuesToSet);
  auto bitOf = [&](int32_t row) {
    const auto index = indices[row];
    return static_cast<uint8_t>((values[index >> 6] >> (index & 63)) & 1);
  };
  if (activeRows.isAllSelected()) {
    int32_t row = activeRows.begin();
    const int32_t end = activeRows.end();
    // Leading rows up to byte alignment, then 8 rows per byte.
    for (; row < end && (row & 7) != 0; ++row) {
      outBytes[row >> 3] |= static_cast<uint8_t>(bitOf(row) << (row & 7));
    }
    for (; row + 8 <= end; row += 8) {
      uint8_t byte = 0;
      for (int32_t k = 0; k < 8; ++k) {
        byte |= static_cast<uint8_t>(bitOf(row + k) << k);
      }
      outBytes[row >> 3] |= byte;
    }
    for (; row < end; ++row) {
      outBytes[row >> 3] |= static_cast<uint8_t>(bitOf(row) << (row & 7));
    }
    return;
  }
  activeRows.applyToSelected([&](int32_t row) {
    outBytes[row >> 3] |= static_cast<uint8_t>(bitOf(row) << (row & 7));
  });
}

/// Checks if bits in specified positions are all set, all unset or mixed.
BooleanMix refineBooleanMixNonNull(
    const uint64_t* bits,
    const SelectivityVector& rows) {
  int32_t first = bits::findFirstBit(bits, rows.begin(), rows.end());
  if (first < 0) {
    return BooleanMix::kAllFalse;
  }
  if (first == rows.begin() && bits::isAllSet(bits, rows.begin(), rows.end())) {
    return BooleanMix::kAllTrue;
  }
  return BooleanMix::kMixNonNull;
}
} // namespace

// Return a BooleanMix representing the status of boolean values in vector. If
// vector contains a mix of true and false, extract the boolean values to a raw
// buffer valuesOut. valuesOut may point to a raw buffer possessed by vector.
// nullsOut remain unchanged if there is no null in vector. tempValues and
// tempNulls may or may not be set by this function.
BooleanMix getFlatBool(
    BaseVector* vector,
    const SelectivityVector& activeRows,
    EvalCtx& context,
    BufferPtr* tempValues,
    BufferPtr* tempNulls,
    bool mergeNullsToValues,
    const uint64_t** valuesOut,
    const uint64_t** nullsOut) {
  VELOX_CHECK_EQ(vector->typeKind(), TypeKind::BOOLEAN);
  const auto size = activeRows.end();
  switch (vector->encoding()) {
    case VectorEncoding::Simple::FLAT: {
      auto values =
          vector->asUnchecked<FlatVector<bool>>()->rawValues<uint64_t>();
      if (!values) {
        return BooleanMix::kAllNull;
      }
      auto nulls = vector->rawNulls();
      if (nulls && mergeNullsToValues) {
        uint64_t* mergedValues;
        BaseVector::ensureBuffer<bool>(
            size, context.pool(), tempValues, &mergedValues);

        // NOTE: false bit in 'nulls' indicate null.
        bits::andBits(
            mergedValues, values, nulls, activeRows.begin(), activeRows.end());

        bits::andBits(
            mergedValues,
            activeRows.asRange().bits(),
            activeRows.begin(),
            activeRows.end());

        *valuesOut = mergedValues;
        return refineBooleanMixNonNull(mergedValues, activeRows);
      }
      *valuesOut = values;
      if (!mergeNullsToValues) {
        *nullsOut = nulls;
      }
      return nulls ? BooleanMix::kMix
                   : refineBooleanMixNonNull(values, activeRows);
    }
    case VectorEncoding::Simple::CONSTANT: {
      if (vector->isNullAt(0)) {
        return BooleanMix::kAllNull;
      }
      return vector->asUnchecked<ConstantVector<bool>>()->valueAt(0)
          ? BooleanMix::kAllTrue
          : BooleanMix::kAllFalse;
    }
    default: {
      uint64_t* nullsToSet = nullptr;
      uint64_t* valuesToSet = nullptr;
      if (vector->mayHaveNulls() && !mergeNullsToValues) {
        BaseVector::ensureBuffer<bool>(
            size, context.pool(), tempNulls, &nullsToSet);
        memset(nullsToSet, bits::kNotNullByte, bits::nbytes(size));
      }
      BaseVector::ensureBuffer<bool>(
          size, context.pool(), tempValues, &valuesToSet);
      memset(valuesToSet, 0, bits::nbytes(size));
      DecodedVector decoded(*vector, activeRows);
      auto values = decoded.data<uint64_t>();
      auto nulls = decoded.nulls(&activeRows);
      auto indices = decoded.indices();
      // my_gporca prototype (C2, default off): with no base nulls the per-row
      // null test and the per-bit read-modify-write collapse into one
      // byte-batched gather (8 rows per byte for dense selections).
      if (useBatchedBoolGather() && nulls == nullptr && values != nullptr &&
          indices != nullptr) {
        gatherDictionaryBits(values, indices, activeRows, valuesToSet);
      } else {
        activeRows.applyToSelected([&](int32_t i) {
          auto index = indices[i];
          bool isNull = nulls && bits::isBitNull(nulls, i);
          if (!isNull && bits::isBitSet(values, index)) {
            bits::setBit(valuesToSet, i);
          }
          if (nullsToSet && isNull) {
            bits::setNull(nullsToSet, i);
          }
        });
      }
      if (!mergeNullsToValues) {
        *nullsOut = nullsToSet;
      }
      *valuesOut = valuesToSet;
      return nullsToSet ? BooleanMix::kMix
                        : refineBooleanMixNonNull(valuesToSet, activeRows);
    }
  }
}
} // namespace facebook::velox::exec
