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

#include <velox/exec/HashBitRange.h>
#include <velox/exec/VectorHasher.h>
#include "velox/core/PlanNode.h"

namespace facebook::velox::exec {

/// Calculates partition number for each row of the specified vector using a
/// hash function. The constructor with hashBitRange parameter requires both
/// hashBitRange and keyChannels to be non-empty. The constructor with
/// numPartitions allows the keyChannels argument to be empty. If keyChannels is
/// empty, then the resulting partition number of partition() will always be
/// zero.
class HashPartitionFunction : public core::PartitionFunction {
 public:
  /// 'segmentChannel' (when set) marks a *segment* column of a segmented
  /// window's (partition key..., segment) bucket exchange: all key channels
  /// are hashed as usual, but the segment value rotates the landing driver so
  /// that one key's segments spread over the drivers instead of being placed
  /// independently. Every (key, segment) bucket still lands on exactly one
  /// driver, so partitioned consumers keep their co-location invariant.
  HashPartitionFunction(
      bool localExchange,
      int numPartitions,
      const RowTypePtr& inputType,
      const std::vector<column_index_t>& keyChannels,
      const std::vector<VectorPtr>& constValues = {},
      std::optional<column_index_t> segmentChannel = std::nullopt);

  HashPartitionFunction(
      const HashBitRange& hashBitRange,
      const RowTypePtr& inputType,
      const std::vector<column_index_t>& keyChannels,
      const std::vector<VectorPtr>& constValues = {},
      std::optional<column_index_t> segmentChannel = std::nullopt);

  ~HashPartitionFunction() override = default;

  std::optional<uint32_t> partition(
      const RowVector& input,
      std::vector<uint32_t>& partitions) override;

  int numPartitions() const {
    return numPartitions_;
  }

 private:
  void init(
      const RowTypePtr& inputType,
      const std::vector<column_index_t>& keyChannels,
      const std::vector<VectorPtr>& constValues);

  const bool localExchange_;
  const int numPartitions_;
  const std::optional<HashBitRange> hashBitRange_ = std::nullopt;
  const std::optional<column_index_t> segmentChannel_ = std::nullopt;
  /// Index of the segment column inside 'hashers_', -1 when disabled.
  int segmentHasherIndex_{-1};
  std::vector<std::unique_ptr<VectorHasher>> hashers_;

  // Reusable memory.
  SelectivityVector rows_;
  raw_vector<uint64_t> hashes_;
  /// Key-only hash snapshot used by the segment rotation.
  raw_vector<uint64_t> segmentKeyHashes_;
};

/// Factory class to create HashPartitionFunction
/// 'keyChannels' stores the index of keys to partition on, if the key is a
/// constant, use index 'kConstantChannel' to indicate so and store the constant
/// value as a base vector in 'constValues'
/// The 'constValues' size is less than or equal to 'keyChannels' size
class HashPartitionFunctionSpec : public core::PartitionFunctionSpec {
 public:
  HashPartitionFunctionSpec(
      RowTypePtr inputType,
      std::vector<column_index_t> keyChannels,
      std::vector<VectorPtr> constValues = {},
      std::optional<column_index_t> segmentChannel = std::nullopt)
      : inputType_{std::move(inputType)},
        keyChannels_{std::move(keyChannels)},
        constValues_{std::move(constValues)},
        segmentChannel_{segmentChannel} {}

  std::unique_ptr<core::PartitionFunction> create(
      int numPartitions,
      bool localExchange) const override;

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static core::PartitionFunctionSpecPtr deserialize(
      const folly::dynamic& obj,
      void* context);

 private:
  const RowTypePtr inputType_;
  const std::vector<column_index_t> keyChannels_;
  const std::vector<VectorPtr> constValues_;
  /// Set for segmented-window bucket exchanges; see HashPartitionFunction.
  const std::optional<column_index_t> segmentChannel_;
};
} // namespace facebook::velox::exec
