#pragma once

#include <cstdint>
#include <vector>

#include "velox/common/base/BitUtil.h"

namespace facebook::velox::exec {

/// Compact keys-only distinct set for a single fixed-width grouping key
/// (int32/int64/date).  It stores keys directly in an open-addressing table,
/// avoiding the per-row RowContainer bookkeeping used by generic hash
/// aggregation.  Zero and null are tracked separately so a zero key can be
/// used as the empty-slot sentinel without an extra payload array.
///
/// Prototype primitive behind YIHUDB_KEYS_ONLY_DEDUP; no spill support.  The
/// caller must only use it when spilling is disabled for the aggregation.
class KeyOnlyDedupSet {
 public:
  explicit KeyOnlyDedupSet(size_t initialCapacity = 1 << 16)
      : slots_(normalizeCapacity(initialCapacity), 0) {}

  /// Inserts one value.  Returns true if it was not present.
  bool insert(uint64_t value, bool isNull) {
    if (isNull) {
      if (nullPresent_) {
        return false;
      }
      nullPresent_ = true;
      ++size_;
      return true;
    }
    if (value == 0) {
      if (zeroPresent_) {
        return false;
      }
      zeroPresent_ = true;
      ++size_;
      return true;
    }
    if ((size_ + 1) * 10 >= slots_.size() * 7) {
      rehash(slots_.size() * 2);
    }
    size_t slot = hashKey(value) & (slots_.size() - 1);
    while (slots_[slot] != 0) {
      if (slots_[slot] == value) {
        return false;
      }
      slot = (slot + 1) & (slots_.size() - 1);
    }
    slots_[slot] = value;
    ++size_;
    return true;
  }

  size_t size() const {
    return size_;
  }

  uint64_t allocatedBytes() const {
    return slots_.capacity() * sizeof(uint64_t);
  }

  size_t slotCount() const {
    return slots_.size();
  }

  bool keyAtSlot(size_t slot, uint64_t& value) const {
    value = slots_[slot];
    return value != 0;
  }

  void clear() {
    std::fill(slots_.begin(), slots_.end(), 0);
    zeroPresent_ = false;
    nullPresent_ = false;
    size_ = 0;
  }

  bool zeroPresent() const {
    return zeroPresent_;
  }

  bool nullPresent() const {
    return nullPresent_;
  }

 private:
  static size_t normalizeCapacity(size_t capacity) {
    size_t normalized = 16;
    while (normalized < capacity) {
      normalized <<= 1;
    }
    return normalized;
  }

  static uint64_t hashKey(uint64_t value) {
    // SplitMix64 finalizer: integer keys are already unique, so a cheap
    // avalanche is enough to spread the low/high bits over the table.
    uint64_t z = value + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

  void rehash(size_t newCapacity) {
    std::vector<uint64_t> old = std::move(slots_);
    slots_.assign(normalizeCapacity(newCapacity), 0);
    for (uint64_t value : old) {
      if (value == 0) {
        continue;
      }
      size_t slot = hashKey(value) & (slots_.size() - 1);
      while (slots_[slot] != 0) {
        slot = (slot + 1) & (slots_.size() - 1);
      }
      slots_[slot] = value;
    }
  }

  std::vector<uint64_t> slots_;
  bool zeroPresent_{false};
  bool nullPresent_{false};
  size_t size_{0};
};

} // namespace facebook::velox::exec
