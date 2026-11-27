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

#include "velox/common/memory/AllocationPool.h"
#include <folly/ScopeGuard.h>
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/CheckedArithmetic.h"
#if defined(_MSC_VER)
#include "velox/common/base/Builtins.h"
#endif
#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/MemoryAllocator.h"

namespace facebook::velox::memory {
namespace {
// Shared by the allocation path and its conservative preauthorization bound.
constexpr int64_t kMinimumLargeMapBytes = 16 * AllocationTraits::kHugePageSize;
} // namespace


folly::Range<char*> AllocationPool::rangeAt(int32_t index) const {
  if (index >= 0 && index < ranges_.size()) {
    const auto range = ranges_[index];
    return index == ranges_.size() - 1
        ? folly::Range<char*>(range.data(), currentOffset_)
        : range;
  }
  VELOX_FAIL("Out of range index for rangeAt(): {}", index);
}

void AllocationPool::clear() {
  ranges_.clear();
  allocations_.clear();
  largeAllocations_.clear();
  startOfRun_ = nullptr;
  bytesInRun_ = 0;
  currentOffset_ = 0;
  usedBytes_ = 0;
}

uint64_t AllocationPool::fixedAllocationBytesUpperBound(
    uint64_t count, uint64_t bytes, int32_t alignment) const {
  if (count == 0) return 0;
  VELOX_CHECK_GT(bytes, 0);
  VELOX_CHECK_GT(alignment, 0);
  const uint64_t align = alignment;
  VELOX_CHECK(bits::isPowerOfTwo(align));
  VELOX_CHECK_LE(align, uint64_t(AllocationTraits::kPageSize));
  const uint64_t span = checkedPlus(bytes, align - 1);
  const uint64_t stride = span & ~(align - 1);

  // Only whole entries already backed AND addressable can avoid new backing.
  // Subtracting an unusable tail would undercount large entries at run ends.
  const uint64_t available = std::max<int64_t>(
      0, std::min(freeBytes(), freeAddressableBytes()));
  const uint64_t padding = startOfRun_ == nullptr ? 0
      : alignmentPadding(startOfRun_ + currentOffset_, alignment);
  uint64_t freeEntries = 0;
  if (available >= padding && available - padding >= bytes) {
    freeEntries = 1 + (available - padding - bytes) / stride;
  }
  if (count <= freeEntries) return 0;
  count -= freeEntries;
  // Keep total payload representable even when the small phase can cover it.
  (void)checkedMultiply(count, stride);
  constexpr uint64_t huge = AllocationTraits::kHugePageSize;
  VELOX_CHECK(!pool_->sizeClasses().empty());
  const uint64_t largestSmall =
      AllocationTraits::pageBytes(pool_->sizeClasses().back());
  VELOX_CHECK_GT(largestSmall, 0);
  uint64_t smallBytes = 0;
  if (span <= largestSmall && hugePageThreshold_ > usedBytes_ &&
      largeAllocations_.empty()) {
    // newRunImpl requests a single run with minSizeClass equal to its page
    // count. The first supported class at least that large is its backing.
    // Malloc-backed runs need not be page aligned: reserve worst-case initial
    // padding before counting complete entries in each future run.
    const auto pages = std::max<MachinePageCount>(
        kMinPages, AllocationTraits::numPages(span));
    const auto& classes = pool_->sizeClasses();
    const auto sizeClass = std::lower_bound(classes.begin(), classes.end(), pages);
    VELOX_CHECK(sizeClass != classes.end());
    const uint64_t runBytes = AllocationTraits::pageBytes(*sizeClass);
    const uint64_t entriesPerSmallRun = 1 + (runBytes - span) / stride;
    const uint64_t neededRuns = 1 + (count - 1) / entriesPerSmallRun;
    const uint64_t untilLarge = hugePageThreshold_ - usedBytes_;
    const uint64_t allowedRuns = 1 + (untilLarge - 1) / runBytes;
    const uint64_t smallRuns = std::min(neededRuns, allowedRuns);
    smallBytes = checkedMultiply(smallRuns, runBytes);
    if (smallRuns == neededRuns) return smallBytes;
    count -= checkedMultiply(smallRuns, entriesPerSmallRun);
  } else if (span <= largestSmall && hugePageThreshold_ > usedBytes_) {
    // A large run can still grow without newRunImpl even below the threshold.
    // Keep the conservative small-phase allowance as well: once its address
    // range ends, this configuration may switch back to small runs.
    const uint64_t smallCap = checkedPlus(
        uint64_t(hugePageThreshold_ - usedBytes_), largestSmall);
    smallBytes = count > smallCap / largestSmall
        ? smallCap : checkedMultiply(count, largestSmall);
  }

  const uint64_t payload = checkedMultiply(count, stride);
  // Every fresh large map has at least this many addressable bytes after
  // alignment at both ends. Backing can include a skipped aligned prefix and
  // a final growth rounding, each smaller than one huge page. An existing
  // partially filled large run can contribute another such allowance.
  constexpr uint64_t minRange = kMinimumLargeMapBytes - 2 * huge;
  const uint64_t entriesPerRun = std::max<uint64_t>(1, minRange / stride);
  uint64_t largeRuns = 1 + (count - 1) / entriesPerRun;
  if (!largeAllocations_.empty()) largeRuns = checkedPlus(largeRuns, uint64_t(1));
  const uint64_t bound = checkedPlus(payload, checkedMultiply(largeRuns, 2 * huge));
  return checkedPlus(smallBytes, bound);
}

char* AllocationPool::allocateFixed(uint64_t bytes, int32_t alignment) {
  VELOX_CHECK_GT(bytes, 0, "Cannot allocate zero bytes");
  if (freeAddressableBytes() >= bytes && alignment == 1) {
    auto* result = startOfRun_ + currentOffset_;
    maybeGrowLastAllocation(bytes);
    return result;
  }
  VELOX_CHECK_EQ(
      __builtin_popcount(alignment), 1, "Alignment can only be power of 2");

  auto numPages = AllocationTraits::numPages(bytes + alignment - 1);

  if (freeAddressableBytes() == 0) {
    newRunImpl(numPages);
  } else {
    auto alignedBytes = bytes + alignmentPadding(firstFreeInRun(), alignment);
    if (freeAddressableBytes() < alignedBytes) {
      newRunImpl(numPages);
    }
  }
  currentOffset_ += alignmentPadding(firstFreeInRun(), alignment);
  VELOX_CHECK_LE(bytes + currentOffset_, bytesInRun_);
  auto* result = startOfRun_ + currentOffset_;
  VELOX_CHECK_EQ(reinterpret_cast<uintptr_t>(result) % alignment, 0);
  maybeGrowLastAllocation(bytes);
  return result;
}

void AllocationPool::maybeGrowLastAllocation(uint64_t bytesRequested) {
  const auto updateOffset = currentOffset_ + bytesRequested;
  if (updateOffset > endOfReservedRun()) {
    VELOX_CHECK_GT(bytesInRun_, AllocationTraits::kHugePageSize);
    const auto bytesToReserve = bits::roundUp(
        updateOffset - endOfReservedRun(), AllocationTraits::kHugePageSize);
    largeAllocations_.back().grow(AllocationTraits::numPages(bytesToReserve));
    usedBytes_ += bytesToReserve;
  }
  // Only update currentOffset_ once it points to valid data.
  currentOffset_ = updateOffset;
}

void AllocationPool::newRunImpl(MachinePageCount numPages) {
  // Reserve metadata before acquiring backing. If allocation fails, the old
  // current run and its cursor remain intact. No per-entry hot-path update is
  // needed: only a run transition freezes its used extent.
  if (!ranges_.empty()) {
    ranges_.back() = folly::Range<char*>(startOfRun_, currentOffset_);
  }
  ranges_.emplace_back();
  auto rollbackRange = folly::makeGuard([&] { ranges_.pop_back(); });
  if (usedBytes_ >= hugePageThreshold_ ||
      numPages > pool_->sizeClasses().back()) {
    // At least 16 huge pages, no more than kMaxMmapBytes. The next is
    // double the previous. Because the previous is a hair under the
    // power of two because of fractional pages at ends of allocation,
    // add an extra huge page size.
    int64_t nextSize = std::min(
        kMaxMmapBytes,
        std::max<int64_t>(
            kMinimumLargeMapBytes,
            bits::nextPowerOfTwo(
                usedBytes_ + AllocationTraits::kHugePageSize)));
    // Round 'numPages' to no of pages in huge page. Allocating this plus an
    // extra huge page guarantees that 'numPages' worth of contiguous aligned
    // huge pages will be founfd in the allocation.
    numPages = bits::roundUp(numPages, AllocationTraits::numPagesInHugePage());
    if (AllocationTraits::pageBytes(numPages) +
            AllocationTraits::kHugePageSize >
        nextSize) {
      // Extra large single request.
      nextSize = AllocationTraits::pageBytes(numPages) +
          AllocationTraits::kHugePageSize;
    }

    ContiguousAllocation largeAlloc;
    const MachinePageCount pagesToAlloc =
        AllocationTraits::numPagesInHugePage();
    pool_->allocateContiguous(
        pagesToAlloc, largeAlloc, AllocationTraits::numPages(nextSize));

    auto range = largeAlloc.hugePageRange().value();
    largeAllocations_.emplace_back(std::move(largeAlloc));
    startOfRun_ = range.data();
    bytesInRun_ = range.size();
    ranges_.back() = folly::Range<char*>(startOfRun_, size_t{0});
    currentOffset_ = 0;
    usedBytes_ += AllocationTraits::pageBytes(pagesToAlloc);
    rollbackRange.dismiss();
    return;
  }

  Allocation allocation;
  auto roundedPages = std::max<int32_t>(kMinPages, numPages);
  pool_->allocateNonContiguous(roundedPages, allocation, roundedPages);
  VELOX_CHECK_EQ(allocation.numRuns(), 1);
  allocations_.push_back(std::move(allocation));
  startOfRun_ = allocations_.back().runAt(0).data<char>();
  bytesInRun_ = allocations_.back().runAt(0).numBytes();
  ranges_.back() = folly::Range<char*>(startOfRun_, size_t{0});
  currentOffset_ = 0;
  usedBytes_ += bytesInRun_;
  rollbackRange.dismiss();
}

void AllocationPool::newRun(int64_t preferredSize) {
  newRunImpl(AllocationTraits::numPages(preferredSize));
}

} // namespace facebook::velox::memory
