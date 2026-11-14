#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace facebook::velox::exec {

/// Compact keys-only distinct set for a single fixed-width integer/date key.
///
/// Small inputs use an inline open-addressing hash table.  Once enough distinct
/// keys have been observed, a dense range is promoted to a bitmap (one bit per
/// value between the observed min and max), which is much smaller and more
/// cache friendly for id/date-like keys.  If the range grows beyond the bitmap
/// budget (or values fall far outside it), the set falls back to the hash table.
///
/// Prototype primitive behind YIHUDB_KEYS_ONLY_DEDUP; no spill support.  The
/// caller must only use it when spilling is disabled or when it can materialize
/// the set back to the generic row-container path.
class KeyOnlyDedupSet {
 public:
  /// Resumable iteration over the distinct keys.  Emits zero and null first
  /// (when present), then either the bitmap bits or the hash slots.
  class Iterator {
   public:
    Iterator() = default;

   private:
    friend class KeyOnlyDedupSet;
    size_t slot_{0};
    size_t bit_{0};
    uint8_t phase_{0};
  };

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

    if (bitmapMode_) {
      if (value >= bitmapMin_ && value - bitmapMin_ < bitmapBits_) {
        return setBit(value - bitmapMin_);
      }
      if (!growBitmap(value)) {
        convertBitmapToHash();
        // Fall through to the hash insertion below.
      } else {
        return setBit(value - bitmapMin_);
      }
    }

    if ((size_ + 1) * 10 >= slots_.size() * 7) {
      rehash(slots_.size() * 2);
    }
    const size_t slot = hashKey(value) & (slots_.size() - 1);
    size_t probe = slot;
    while (slots_[probe] != 0) {
      if (slots_[probe] == value) {
        return false;
      }
      probe = (probe + 1) & (slots_.size() - 1);
    }
    slots_[probe] = value;
    ++size_;
    if (!promotionDisabled_ && size_ >= kPromoteKeys) {
      promoteToBitmap();
    }
    return true;
  }

  size_t size() const {
    return size_;
  }

  uint64_t allocatedBytes() const {
    if (bitmapMode_) {
      return bitmap_.capacity() * sizeof(uint64_t);
    }
    return slots_.capacity() * sizeof(uint64_t);
  }

  /// Emits the next distinct key.  Returns false when exhausted.
  bool next(Iterator& iterator, uint64_t& value, bool& isNull) const {
    if (iterator.phase_ == 0) {
      iterator.phase_ = 1;
      if (zeroPresent_) {
        value = 0;
        isNull = false;
        return true;
      }
    }
    if (iterator.phase_ == 1) {
      iterator.phase_ = 2;
      if (nullPresent_) {
        value = 0;
        isNull = true;
        return true;
      }
    }
    if (iterator.phase_ == 2) {
      if (bitmapMode_) {
        while (iterator.bit_ < bitmapBits_) {
          const size_t word = iterator.bit_ >> 6;
          if (bitmap_[word] == 0) {
            iterator.bit_ = (word + 1) << 6;
            continue;
          }
          while (iterator.bit_ < bitmapBits_) {
            const size_t currentWord = iterator.bit_ >> 6;
            if ((bitmap_[currentWord] >> (iterator.bit_ & 63)) & 1ULL) {
              value = bitmapMin_ + iterator.bit_;
              ++iterator.bit_;
              isNull = false;
              return true;
            }
            ++iterator.bit_;
          }
        }
      } else {
        while (iterator.slot_ < slots_.size()) {
          const uint64_t key = slots_[iterator.slot_++];
          if (key != 0) {
            value = key;
            isNull = false;
            return true;
          }
        }
      }
      iterator.phase_ = 3;
    }
    return false;
  }

  void clear() {
    std::fill(slots_.begin(), slots_.end(), 0);
    bitmap_.clear();
    bitmapMode_ = false;
    promotionDisabled_ = false;
    size_ = 0;
    zeroPresent_ = false;
    nullPresent_ = false;
  }

 private:
  static constexpr size_t kPromoteKeys = 1 << 14;
  // 512M bits = 64MB worst case; dense id/date ranges fit comfortably.
  static constexpr size_t kMaxBitmapBits = 1ULL << 29;

  static size_t normalizeCapacity(size_t capacity) {
    size_t normalized = 16;
    while (normalized < capacity) {
      normalized <<= 1;
    }
    return normalized;
  }

  static uint64_t hashKey(uint64_t value) {
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

  bool setBit(uint64_t index) {
    const size_t word = index >> 6;
    const uint64_t mask = 1ULL << (index & 63);
    if ((bitmap_[word] & mask) != 0) {
      return false;
    }
    bitmap_[word] |= mask;
    bitmapMax_ = std::max(bitmapMax_, bitmapMin_ + index);
    ++size_;
    return true;
  }

  /// Promotes the hash table to a bitmap when the observed key range is dense
  /// enough.  The bitmap covers at least the sampled range; later values may
  /// grow it, and an excessive range falls back to the hash table.
  void promoteToBitmap() {
    uint64_t minValue = UINT64_MAX;
    uint64_t maxValue = 0;
    bool any = false;
    for (uint64_t value : slots_) {
      if (value == 0) {
        continue;
      }
      any = true;
      minValue = std::min(minValue, value);
      maxValue = std::max(maxValue, value);
    }
    if (!any) {
      return;
    }
    const uint64_t range = maxValue - minValue + 1;
    if (range > kMaxBitmapBits) {
      promotionDisabled_ = true;
      return;
    }
    size_t bits = 64;
    while (bits < range) {
      bits <<= 1;
    }
    bitmap_.assign((bits + 63) >> 6, 0);
    bitmapMin_ = minValue;
    bitmapMax_ = maxValue;
    bitmapBits_ = bits;
    for (uint64_t value : slots_) {
      if (value != 0) {
        bitmap_[(value - bitmapMin_) >> 6] |=
            1ULL << ((value - bitmapMin_) & 63);
      }
    }
    bitmapMode_ = true;
    std::vector<uint64_t>().swap(slots_);
  }

  /// Grows the bitmap to cover 'value' with power-of-two capacity.  Returns
  /// false when the required range exceeds the bitmap budget.
  bool growBitmap(uint64_t value) {
    uint64_t newMin = std::min(bitmapMin_, value);
    uint64_t newMax = std::max(bitmapMax_, value);
    const uint64_t required = newMax - newMin + 1;
    if (required > kMaxBitmapBits) {
      return false;
    }
    size_t bits = bitmapBits_;
    while (bits < required) {
      bits <<= 1;
    }
    if (bits == bitmapBits_ && newMin == bitmapMin_) {
      return true;
    }
    std::vector<uint64_t> grown((bits + 63) >> 6, 0);
    const uint64_t shift = bitmapMin_ - newMin;
    for (size_t i = 0; i < bitmapBits_; ++i) {
      if ((bitmap_[i >> 6] >> (i & 63)) & 1ULL) {
        const uint64_t target = i + shift;
        grown[target >> 6] |= 1ULL << (target & 63);
      }
    }
    bitmap_.swap(grown);
    bitmapMin_ = newMin;
    bitmapMax_ = newMax;
    bitmapBits_ = bits;
    return true;
  }

  void convertBitmapToHash() {
    size_t capacity = normalizeCapacity(std::max<size_t>(16, size_ * 2));
    std::vector<uint64_t> slots(capacity, 0);
    for (size_t i = 0; i < bitmapBits_; ++i) {
      if ((bitmap_[i >> 6] >> (i & 63)) & 1ULL) {
        const uint64_t value = bitmapMin_ + i;
        size_t slot = hashKey(value) & (capacity - 1);
        while (slots[slot] != 0) {
          slot = (slot + 1) & (capacity - 1);
        }
        slots[slot] = value;
      }
    }
    slots_.swap(slots);
    bitmap_.clear();
    bitmapMode_ = false;
    promotionDisabled_ = true;
  }

  std::vector<uint64_t> slots_;
  std::vector<uint64_t> bitmap_;
  uint64_t bitmapMin_{0};
  uint64_t bitmapMax_{0};
  size_t bitmapBits_{0};
  bool bitmapMode_{false};
  bool promotionDisabled_{false};
  bool zeroPresent_{false};
  bool nullPresent_{false};
  size_t size_{0};
};

} // namespace facebook::velox::exec
