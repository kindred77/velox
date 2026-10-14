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

#include "velox/common/memory/SubPageAllocator.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/MmapAllocator.h"

namespace facebook::velox::memory {
namespace {

constexpr MachinePageCount kSlabPages = 16;
constexpr size_t kSlabBytes = kSlabPages * AllocationTraits::kPageSize;
constexpr uint32_t kBlockQuantum = MemoryAllocator::kMinAlignment;
constexpr uint32_t kMinBlockBytes = 2 * kBlockQuantum;

uint64_t roundUp(uint64_t value, uint32_t multiple) {
  const auto remainder = value % multiple;
  return remainder == 0 ? value : value + multiple - remainder;
}

uint32_t blockSizeFor(uint64_t bytes, uint16_t alignment) {
  // Each block begins at a multiple of its size, and block sizes are multiples
  // of the requested alignment. Hence only the aligned header prefix, rather
  // than a worst-case alignment gap, is needed before the user bytes.
  const auto userOffset = roundUp(2 * sizeof(void*), alignment);
  const auto required = userOffset + bytes;
  VELOX_CHECK_LE(required, std::numeric_limits<uint32_t>::max());
  return std::max<uint32_t>(
      kMinBlockBytes,
      static_cast<uint32_t>(roundUp(required, alignment)));
}

} // namespace

class SubPageAllocator::SizeClass {
 private:
  struct Slab;

  struct FreeBlock {
    FreeBlock* next;
  };

  struct AllocationHeader {
    Slab* slab;
    FreeBlock* block;
  };

  struct alignas(max_align_t) Slab {
    SizeClass* owner;
    uint8_t* base;
    FreeBlock* freeHead;
    uint32_t usedBlocks;
    uint32_t totalBlocks;
    Slab* previousAvailable;
    Slab* nextAvailable;
    Slab* previousAll;
    Slab* nextAll;
  };

 public:
  SizeClass(
      MmapAllocator& owner,
      uint32_t blockBytes,
      bool trackAllocationPeaks)
      : owner_(owner),
        blockBytes_(blockBytes),
        trackAllocationPeaks_(trackAllocationPeaks) {}

  ~SizeClass() {
    while (allHead_ != nullptr) {
      releaseSlab(allHead_);
    }
  }

  void* allocate(uint64_t bytes, uint16_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (availableHead_ == nullptr && !addSlab()) {
      return nullptr;
    }

    auto* slab = availableHead_;
    if (slab == emptySlab_) {
      emptySlab_ = nullptr;
    }
    auto* block = slab->freeHead;
    VELOX_CHECK_NOT_NULL(block);
    slab->freeHead = block->next;
    ++slab->usedBlocks;
    if (trackAllocationPeaks_) {
      ++allocationCount_;
      peakUsedBlocks_ = std::max(peakUsedBlocks_, slab->usedBlocks);
    }
    if (slab->freeHead == nullptr) {
      removeAvailable(slab);
    }

    const auto blockAddress = reinterpret_cast<uintptr_t>(block);
    const auto userAddress =
        (blockAddress + sizeof(AllocationHeader) + alignment - 1) &
        ~(static_cast<uintptr_t>(alignment) - 1);
    auto* header = reinterpret_cast<AllocationHeader*>(
        userAddress - sizeof(AllocationHeader));
    VELOX_CHECK_LE(userAddress + bytes, blockAddress + blockBytes_);
    header->slab = slab;
    header->block = block;
    return reinterpret_cast<void*>(userAddress);
  }

  static void free(void* address) noexcept {
    auto* header = reinterpret_cast<AllocationHeader*>(address) - 1;
    auto* slab = header->slab;
    slab->owner->freeBlock(slab, header->block);
  }

 private:
  bool addSlab() {
    auto* base = owner_.allocateSubPageSlab();
    if (base == nullptr) {
      return false;
    }

    auto* slab = new (base)
        Slab{this, base, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr};
    const auto blocksBegin = roundUp(
        reinterpret_cast<uintptr_t>(base) + sizeof(Slab), blockBytes_);
    const auto slabEnd = reinterpret_cast<uintptr_t>(base) + kSlabBytes;
    slab->totalBlocks =
        static_cast<uint32_t>((slabEnd - blocksBegin) / blockBytes_);
    VELOX_CHECK_GT(slab->totalBlocks, 0);

    for (uint32_t index = 0; index < slab->totalBlocks; ++index) {
      auto* block = reinterpret_cast<FreeBlock*>(
          blocksBegin + static_cast<uintptr_t>(index) * blockBytes_);
      block->next = slab->freeHead;
      slab->freeHead = block;
    }
    addAll(slab);
    addAvailable(slab);
    if (trackAllocationPeaks_) {
      ++slabCount_;
      peakSlabCount_ = std::max(peakSlabCount_, slabCount_);
    }
    return true;
  }

  void freeBlock(Slab* slab, FreeBlock* block) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool wasFull = slab->freeHead == nullptr;
    block->next = slab->freeHead;
    slab->freeHead = block;
    VELOX_DCHECK_GT(slab->usedBlocks, 0);
    --slab->usedBlocks;
    if (wasFull) {
      addAvailable(slab);
    }
    if (slab->usedBlocks == 0) {
      if (emptySlab_ == nullptr) {
        emptySlab_ = slab;
      } else if (emptySlab_ != slab) {
        releaseSlab(slab);
      }
    }
  }

  void addAvailable(Slab* slab) {
    VELOX_DCHECK_NULL(slab->previousAvailable);
    VELOX_DCHECK_NULL(slab->nextAvailable);
    slab->nextAvailable = availableHead_;
    if (availableHead_ != nullptr) {
      availableHead_->previousAvailable = slab;
    }
    availableHead_ = slab;
  }

  void removeAvailable(Slab* slab) {
    if (slab->previousAvailable != nullptr) {
      slab->previousAvailable->nextAvailable = slab->nextAvailable;
    } else if (availableHead_ == slab) {
      availableHead_ = slab->nextAvailable;
    } else {
      return;
    }
    if (slab->nextAvailable != nullptr) {
      slab->nextAvailable->previousAvailable = slab->previousAvailable;
    }
    slab->previousAvailable = nullptr;
    slab->nextAvailable = nullptr;
  }

  void addAll(Slab* slab) {
    slab->nextAll = allHead_;
    if (allHead_ != nullptr) {
      allHead_->previousAll = slab;
    }
    allHead_ = slab;
  }

  void removeAll(Slab* slab) {
    if (slab->previousAll != nullptr) {
      slab->previousAll->nextAll = slab->nextAll;
    } else {
      allHead_ = slab->nextAll;
    }
    if (slab->nextAll != nullptr) {
      slab->nextAll->previousAll = slab->previousAll;
    }
  }

  void releaseSlab(Slab* slab) noexcept {
    if (emptySlab_ == slab) {
      emptySlab_ = nullptr;
    }
    removeAvailable(slab);
    removeAll(slab);
    void* address = slab->base;
    slab->~Slab();
    if (trackAllocationPeaks_) {
      VELOX_DCHECK_GT(slabCount_, 0);
      --slabCount_;
    }
    owner_.freeSubPageSlab(address);
  }

 public:
  std::string toString() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (allocationCount_ == 0) {
      return {};
    }
    std::stringstream out;
    out << "[subpage block " << blockBytes_ << ": allocations "
        << allocationCount_ << " peak-blocks-per-slab " << peakUsedBlocks_
        << " peak-slabs " << peakSlabCount_ << " peak-backing-bytes "
        << static_cast<uint64_t>(peakSlabCount_) * kSlabBytes << "]";
    return out.str();
  }

 private:

  MmapAllocator& owner_;
  const uint32_t blockBytes_;
  const bool trackAllocationPeaks_;
  mutable std::mutex mutex_;
  Slab* availableHead_{nullptr};
  Slab* allHead_{nullptr};
  Slab* emptySlab_{nullptr};
  uint64_t allocationCount_{0};
  uint32_t peakUsedBlocks_{0};
  uint32_t slabCount_{0};
  uint32_t peakSlabCount_{0};
};

SubPageAllocator::SubPageAllocator(
    MmapAllocator& owner,
    uint32_t maxBytes,
    bool trackAllocationPeaks)
    : owner_(owner), maxBytes_(maxBytes) {
  VELOX_CHECK_GT(maxBytes_, 0);
  VELOX_CHECK_LE(maxBytes_, AllocationTraits::kPageSize);
  VELOX_CHECK_LE(maxBytes_, std::numeric_limits<uint16_t>::max());
  // freeBytes() receives the size but not the allocation alignment. Therefore
  // every valid alignment for a handled size must stay on the sub-page path so
  // allocation and free routing cannot disagree. A valid non-zero request is
  // a multiple of its alignment, hence alignment is at most maxBytes_.
  const auto maxAlignment = static_cast<uint16_t>(
      std::max<uint32_t>(MemoryAllocator::kMinAlignment, maxBytes_));
  const auto largestBlock = blockSizeFor(maxBytes_, maxAlignment);
  for (uint32_t blockBytes = kMinBlockBytes; blockBytes <= largestBlock;
       blockBytes += kBlockQuantum) {
    sizeClasses_.push_back(std::make_unique<SizeClass>(
        owner_, blockBytes, trackAllocationPeaks));
  }
}

SubPageAllocator::~SubPageAllocator() = default;

bool SubPageAllocator::handles(uint64_t bytes, uint16_t /*alignment*/) const {
  return bytes > 0 && bytes <= maxBytes_;
}

void* SubPageAllocator::allocate(uint64_t bytes, uint16_t alignment) {
  return sizeClass(bytes, alignment).allocate(bytes, alignment);
}

void SubPageAllocator::free(void* address) noexcept {
  SizeClass::free(address);
}

std::string SubPageAllocator::toString() const {
  std::stringstream out;
  for (const auto& sizeClass : sizeClasses_) {
    auto stats = sizeClass->toString();
    if (!stats.empty()) {
      out << stats << '\n';
    }
  }
  return out.str();
}

SubPageAllocator::SizeClass& SubPageAllocator::sizeClass(
    uint64_t bytes,
    uint16_t alignment) {
  const auto blockBytes = blockSizeFor(bytes, alignment);
  const auto index =
      (blockBytes - kMinBlockBytes) / kBlockQuantum;
  VELOX_CHECK_LT(index, sizeClasses_.size());
  return *sizeClasses_[index];
}

} // namespace facebook::velox::memory
