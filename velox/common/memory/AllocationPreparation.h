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
#include <exception>
#include <limits>

namespace facebook::velox::memory {
class MemoryPool;

/// A reversible preparation could not obtain this complete credit amount.
/// This is not an allocator failure and must never authorize replay of updates.
class AllocationPreparationDemand final : public std::exception {
 public:
  explicit AllocationPreparationDemand(uint64_t required, const MemoryPool* pool = nullptr)
      : required_(required), pool_(pool) {}
  uint64_t requiredBytes() const { return required_; }
  const MemoryPool* pool() const { return pool_; }
  const char* what() const noexcept override {
    return "Reversible allocation preparation needs additional credit";
  }
 private:
  uint64_t required_;
  const MemoryPool* pool_;
};

/// Refines the existing credit scope; creates neither a budget nor a ledger.
/// Requires a credit scope on this pool. All preparation must be owned and
/// discardable on unwind. Missing credit grows through the same controller;
/// on refusal, discard preparation before requesting the complete amount.
/// End this guard before consuming aggregate input.
/// A nested, ordinary credit scope remains strict.
class AllocationPreparationScope {
 public:
  // Bounds this scope's total granted credit, including any initial grant.
  // Refusal unwinds discardable work before allocating the denied block.
  explicit AllocationPreparationScope(MemoryPool* pool,
      uint64_t maxBytes = std::numeric_limits<uint64_t>::max());
  ~AllocationPreparationScope();
  uint64_t grantedBytes() const;
  AllocationPreparationScope(const AllocationPreparationScope&) = delete;
  AllocationPreparationScope& operator=(const AllocationPreparationScope&) = delete;
 private:
  void* credit_;
  bool previous_;
  uint64_t previousLimit_;
};
} // namespace facebook::velox::memory
