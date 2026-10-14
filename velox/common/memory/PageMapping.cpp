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

#include "velox/common/memory/PageMapping.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/mman.h>
#endif

namespace facebook::velox::memory {

void* PageMapping::reserve(size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
#ifdef _WIN32
  return ::VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
#else
  auto* address = ::mmap(
      nullptr,
      bytes,
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0);
  return address == MAP_FAILED ? nullptr : address;
#endif
}

bool PageMapping::commit(void* address, size_t bytes) {
  if (address == nullptr || bytes == 0) {
    return false;
  }
#ifdef _WIN32
  return ::VirtualAlloc(address, bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
  // Anonymous mmap is demand-paged. The first access supplies backing pages.
  return true;
#endif
}

bool PageMapping::discard(void* address, size_t bytes) {
  if (address == nullptr || bytes == 0) {
    return false;
  }
#ifdef _WIN32
  return ::VirtualFree(address, bytes, MEM_DECOMMIT) != 0;
#else
  return ::madvise(address, bytes, MADV_DONTNEED) == 0;
#endif
}

bool PageMapping::release(void* address, size_t bytes) {
  if (address == nullptr || bytes == 0) {
    return false;
  }
#ifdef _WIN32
  return ::VirtualFree(address, 0, MEM_RELEASE) != 0;
#else
  return ::munmap(address, bytes) == 0;
#endif
}

} // namespace facebook::velox::memory
