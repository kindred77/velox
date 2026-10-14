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

#include <cstddef>

namespace facebook::velox::memory {

/// Cross-platform virtual-memory primitives used by page-backed allocators.
/// Reserving address space never commits the full range on Windows. Callers
/// must commit a range before access and may discard it while retaining the
/// reservation for later reuse.
class PageMapping final {
 public:
  static void* reserve(size_t bytes);

  static bool commit(void* address, size_t bytes);

  static bool discard(void* address, size_t bytes);

  static bool release(void* address, size_t bytes);
};

} // namespace facebook::velox::memory
