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

#include <cstring>
#include <type_traits>

#include "velox/exec/PrefixSort.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

// Keys-only compact sort helpers (yihudb prototype, env
// GPORCA_SORT_KEYS_ONLY=1, default off; see SortBuffer.*).
//
// When every input column of a sort is one of its sort keys and all key types
// are fixed-width integers, the SortBuffer can sort normalized key entries
// directly instead of materializing (row + normalized key + row pointer) per
// row. This file provides the vector->entry encoder and entry->vector decoder
// used by that path. The normalized byte layout is produced by the same
// PrefixSortLayout::generate() and PrefixSortEncoder::encodeNoNulls() the
// regular path uses, so both paths sort identical byte sequences.
namespace facebook::velox::exec::keysonly {

/// Key types supported by the prototype (fixed-width integers; DATE and other
/// integer-backed logical types are TypeKind::INTEGER).
inline bool supportedKind(TypeKind kind) {
  return kind == TypeKind::SMALLINT || kind == TypeKind::INTEGER ||
      kind == TypeKind::BIGINT;
}

/// Layout plus the key kinds, which PrefixSortLayout does not retain.
struct KeysOnlyPlan {
  PrefixSortLayout layout;
  std::vector<TypeKind> keyKinds;
};

/// Alignment padding plus the byte-order inversion applied to the whole
/// normalized key area (involution, so the same function decodes).
FOLLY_ALWAYS_INLINE void bitsSwapByWord(uint64_t* address, int32_t bytes) {
  while (bytes != 0) {
    *address = __builtin_bswap64(*address);
    ++address;
    bytes -= 8;
  }
}

FOLLY_ALWAYS_INLINE void zeroPadding(
    const PrefixSortLayout& layout,
    char* entry) {
  if (layout.numPaddingBytes > 0) {
    simd::memset(
        entry + layout.normalizedBufferSize - layout.numPaddingBytes,
        0,
        layout.numPaddingBytes);
  }
}

/// Encodes one row's keys (from decoded input columns) into 'entry'.
/// The caller must have verified that every key column carries no nulls.
FOLLY_ALWAYS_INLINE void encodeRow(
    const KeysOnlyPlan& plan,
    const std::vector<const DecodedVector*>& keys,
    vector_size_t row,
    char* entry) {
  const auto& layout = plan.layout;
  for (uint32_t i = 0; i < layout.numNormalizedKeys; ++i) {
    switch (plan.keyKinds[i]) {
      case TypeKind::SMALLINT: {
        layout.encoders[i].encodeNoNulls(
            keys[i]->valueAt<int16_t>(row),
            entry + layout.prefixOffsets[i],
            layout.encodeSizes[i]);
        break;
      }
      case TypeKind::INTEGER: {
        layout.encoders[i].encodeNoNulls(
            keys[i]->valueAt<int32_t>(row),
            entry + layout.prefixOffsets[i],
            layout.encodeSizes[i]);
        break;
      }
      case TypeKind::BIGINT: {
        layout.encoders[i].encodeNoNulls(
            keys[i]->valueAt<int64_t>(row),
            entry + layout.prefixOffsets[i],
            layout.encodeSizes[i]);
        break;
      }
      default:
        VELOX_UNSUPPORTED(
            "keys-only sort does not support key type {}",
            TypeKindName::toName(plan.keyKinds[i]));
    }
  }
  zeroPadding(layout, entry);
  bitsSwapByWord(
      reinterpret_cast<uint64_t*>(entry), layout.normalizedBufferSize);
}

/// Inverse of PrefixSortEncoder::encodeNoNulls for one integer key. 'scratch'
/// holds the un-byte-swapped normalized key area.
template <typename U>
FOLLY_ALWAYS_INLINE U bswapSized(U value) {
  if constexpr (sizeof(U) == 2) {
    return __builtin_bswap16(value);
  } else if constexpr (sizeof(U) == 4) {
    return __builtin_bswap32(value);
  } else {
    return __builtin_bswap64(value);
  }
}

template <typename T>
FOLLY_ALWAYS_INLINE T decodeTypedKey(
    const KeysOnlyPlan& plan,
    uint32_t index,
    const char* scratch) {
  using Unsigned = std::make_unsigned_t<T>;
  Unsigned stored;
  std::memcpy(
      &stored,
      scratch + plan.layout.prefixOffsets[index],
      sizeof(Unsigned));
  // encodeNoNulls stores bswap(value ^ signBit) at the key offset; the whole
  // normalized area is then word-swapped (undone by the caller).
  Unsigned value = bswapSized(stored);
  if (!plan.layout.compareFlags[index].ascending) {
    value = static_cast<Unsigned>(~value);
  }
  const auto signBit =
      static_cast<Unsigned>(Unsigned(1) << (8 * sizeof(T) - 1));
  return static_cast<T>(static_cast<Unsigned>(value ^ signBit));
}

/// Decodes one entry into the output children (flat vectors), writing row
/// 'row'. The caller must have verified there are no nulls.
FOLLY_ALWAYS_INLINE void decodeEntry(
    const KeysOnlyPlan& plan,
    const char* entry,
    const std::vector<VectorPtr>& outputs,
    vector_size_t row) {
  const auto& layout = plan.layout;
  char scratch[128];
  VELOX_CHECK_LE(layout.normalizedBufferSize, sizeof(scratch));
  std::memcpy(scratch, entry, layout.normalizedBufferSize);
  bitsSwapByWord(
      reinterpret_cast<uint64_t*>(scratch), layout.normalizedBufferSize);
  for (uint32_t i = 0; i < layout.numNormalizedKeys; ++i) {
    switch (plan.keyKinds[i]) {
      case TypeKind::SMALLINT: {
        auto* out = outputs[i]->as<FlatVector<int16_t>>();
        out->set(row, decodeTypedKey<int16_t>(plan, i, scratch));
        break;
      }
      case TypeKind::INTEGER: {
        auto* out = outputs[i]->as<FlatVector<int32_t>>();
        out->set(row, decodeTypedKey<int32_t>(plan, i, scratch));
        break;
      }
      case TypeKind::BIGINT: {
        auto* out = outputs[i]->as<FlatVector<int64_t>>();
        out->set(row, decodeTypedKey<int64_t>(plan, i, scratch));
        break;
      }
      default:
        VELOX_UNSUPPORTED(
            "keys-only sort does not support key type {}",
            TypeKindName::toName(plan.keyKinds[i]));
    }
  }
}

/// Word-wise comparison of two entries, identical to PrefixSort's
/// compareAllNormalizedKeys.
FOLLY_ALWAYS_INLINE int32_t compareEntries(
    const KeysOnlyPlan& plan,
    const char* left,
    const char* right) {
  auto* l = reinterpret_cast<const uint64_t*>(left);
  auto* r = reinterpret_cast<const uint64_t*>(right);
  int32_t bytes = plan.layout.normalizedBufferSize;
  while (bytes != 0) {
    if (*l != *r) {
      return *l > *r ? 1 : -1;
    }
    ++l;
    ++r;
    bytes -= 8;
  }
  return 0;
}

} // namespace facebook::velox::exec::keysonly
