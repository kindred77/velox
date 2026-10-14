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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "velox/common/memory/MemoryAllocator.h"

namespace facebook::velox::memory {

class MmapAllocator;

/// Packs small byte allocations into page-backed slabs. Slab backing remains
/// accounted by MmapAllocator and an empty slab is returned immediately to its
/// page cache. This avoids both a full machine page per small object and the
/// process heap fragmentation caused by delegating small requests to malloc.
class SubPageAllocator final {
 public:
  SubPageAllocator(
      MmapAllocator& owner,
      uint32_t maxBytes,
      bool trackAllocationPeaks);

  ~SubPageAllocator();

  bool handles(uint64_t bytes, uint16_t alignment) const;

  void* allocate(uint64_t bytes, uint16_t alignment);

  void free(void* address) noexcept;

  std::string toString() const;

 private:
  class SizeClass;

  SizeClass& sizeClass(uint64_t bytes, uint16_t alignment);

  MmapAllocator& owner_;
  const uint32_t maxBytes_;
  std::vector<std::unique_ptr<SizeClass>> sizeClasses_;
};

} // namespace facebook::velox::memory
