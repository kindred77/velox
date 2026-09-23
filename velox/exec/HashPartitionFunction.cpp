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
#include <velox/exec/HashPartitionFunction.h>
#include <velox/exec/VectorHasher.h>

#include "velox/common/base/XxHashInline.h"
#include "velox/vector/DecodedVector.h"

#include <algorithm>
#include <cstdlib>

namespace facebook::velox::exec {
namespace {
// Gets the hash value for local exchange with given 'rawHash'. 'rawHash'
// is the value computed by this hash function which is used for remote
// shuffle across stages like for Prestissimo.
static inline uint32_t localExchangeHash(uint32_t rawHash) {
  // Mix the bits so we don't use the same hash used to distribute between
  // stages.
  bits::reverseBits(reinterpret_cast<uint8_t*>(&rawHash), sizeof(rawHash));
  return XXH32(&rawHash, sizeof(rawHash), 0);
}

/// Integer columns the segment rotation can read. The check happens once, on
/// the plan type, so a single operator instance never mixes layouts.
bool isRotatableSegmentType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BIGINT:
    case TypeKind::INTEGER:
    case TypeKind::SMALLINT:
    case TypeKind::TINYINT:
      return true;
    default:
      return false;
  }
}

int64_t intValueAt(const DecodedVector& vector, vector_size_t row) {
  switch (vector.base()->type()->kind()) {
    case TypeKind::BIGINT:
      return vector.valueAt<int64_t>(row);
    case TypeKind::INTEGER:
      return vector.valueAt<int32_t>(row);
    case TypeKind::SMALLINT:
      return vector.valueAt<int16_t>(row);
    default:
      // Only reachable for the types admitted by isRotatableSegmentType().
      return vector.valueAt<int8_t>(row);
  }
}

/// One-line fallback switch for the segmented-window bucket rotation.
bool segmentRotationEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("GPORCA_SEG_BUCKET_ROTATE");
    return value == nullptr || std::atoi(value) != 0;
  }();
  return enabled;
}
} // namespace

HashPartitionFunction::HashPartitionFunction(
    bool localExchange,
    int numPartitions,
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& keyChannels,
    const std::vector<VectorPtr>& constValues,
    std::optional<column_index_t> segmentChannel)
    : localExchange_{localExchange},
      numPartitions_{numPartitions},
      segmentChannel_{segmentChannel} {
  init(inputType, keyChannels, constValues);
}

HashPartitionFunction::HashPartitionFunction(
    const HashBitRange& hashBitRange,
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& keyChannels,
    const std::vector<VectorPtr>& constValues,
    std::optional<column_index_t> segmentChannel)
    : localExchange_{false},
      numPartitions_{hashBitRange.numPartitions()},
      hashBitRange_(hashBitRange),
      segmentChannel_{segmentChannel} {
  VELOX_CHECK_GT(hashBitRange.numPartitions(), 0);
  VELOX_CHECK(!keyChannels.empty());
  init(inputType, keyChannels, constValues);
}

void HashPartitionFunction::init(
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& keyChannels,
    const std::vector<VectorPtr>& constValues) {
  hashers_.reserve(keyChannels.size());
  size_t constChannel{0};
  for (const auto channel : keyChannels) {
    if (channel != kConstantChannel) {
      hashers_.emplace_back(
          VectorHasher::create(inputType->childAt(channel), channel));
    } else {
      const auto& constValue = constValues[constChannel++];
      hashers_.emplace_back(VectorHasher::create(constValue->type(), channel));
      hashers_.back()->precompute(*constValue);
    }
  }
  if (segmentChannel_.has_value()) {
    segmentHasherIndex_ = -1;
    if (*segmentChannel_ < inputType->size() &&
        isRotatableSegmentType(inputType->childAt(*segmentChannel_))) {
      for (size_t index = 0; index < hashers_.size(); ++index) {
        if (hashers_[index]->channel() == *segmentChannel_) {
          // The key-only hash is snapshotted before this hasher runs, so the
          // segment has to be preceded by at least one key column.
          segmentHasherIndex_ = index > 0 ? static_cast<int>(index) : -1;
          break;
        }
      }
    }
  }
}

std::optional<uint32_t> HashPartitionFunction::partition(
    const RowVector& input,
    std::vector<uint32_t>& partitions) {
  if (hashers_.empty()) {
    return 0u;
  }

  const auto size = input.size();
  rows_.resize(size);
  rows_.setAll();

  // Segmented-window bucket exchange: a segment column rotates the landing
  // driver so that one key's segments spread over the drivers instead of being
  // placed independently. Every (key, segment) bucket still lands on exactly
  // one driver, so local sort/window consumers keep their co-location
  // invariant and only the assignment changes.
  const bool rotate = segmentRotationEnabled() && segmentHasherIndex_ >= 0 &&
      localExchange_ && !hashBitRange_.has_value();
  if (rotate) {
    segmentKeyHashes_.resize(size);
  }

  hashes_.resize(size);
  for (auto i = 0; i < hashers_.size(); ++i) {
    if (rotate && static_cast<int>(i) == segmentHasherIndex_) {
      // Snapshot the key-only hash state before the segment column is mixed
      // in: the per-key offset must not depend on the segment.
      std::copy(
          hashes_.begin(), hashes_.end(), segmentKeyHashes_.begin());
    }
    auto& hasher = hashers_[i];
    if (hasher->channel() != kConstantChannel) {
      hashers_[i]->decode(*input.childAt(hasher->channel()), rows_);
      hashers_[i]->hash(rows_, i > 0, hashes_);
    } else {
      hashers_[i]->hashPrecomputed(rows_, i > 0, hashes_);
    }
  }

  partitions.resize(size);
  if (rotate) {
    // Reuse the decoded segment column the hasher already produced: it was
    // decoded over the same rows, so the batch cost is the rotation arithmetic
    // alone. A null segment value cannot be read as an integer; keep such rows
    // on a deterministic per-key slot instead of mixing placements inside one
    // bucket. The segmented-window gate proves the order key is NULL-free, so
    // the null case is defensive only.
    const auto& segmentRows = hashers_[segmentHasherIndex_]->decodedVector();
    constexpr uint32_t kNullSegment = 0xFFFFFFFFu;
    for (auto i = 0; i < size; ++i) {
      const auto segment = segmentRows.isNullAt(i)
          ? kNullSegment
          : static_cast<uint32_t>(intValueAt(segmentRows, i));
      const auto base =
          localExchangeHash(static_cast<uint32_t>(segmentKeyHashes_[i]));
      partitions[i] = (base + segment) % numPartitions_;
    }
  } else if (hashBitRange_.has_value()) {
    if (localExchange_) {
      for (auto i = 0; i < size; ++i) {
        partitions[i] = hashBitRange_->partition(localExchangeHash(hashes_[i]));
      }
    } else {
      for (auto i = 0; i < size; ++i) {
        partitions[i] = hashBitRange_->partition(hashes_[i]);
      }
    }
  } else {
    if (localExchange_) {
      for (auto i = 0; i < size; ++i) {
        partitions[i] = localExchangeHash(hashes_[i]) % numPartitions_;
      }
    } else {
      for (auto i = 0; i < size; ++i) {
        partitions[i] = hashes_[i] % numPartitions_;
      }
    }
  }

  return std::nullopt;
}

std::unique_ptr<core::PartitionFunction> HashPartitionFunctionSpec::create(
    int numPartitions,
    bool localExchange) const {
  return std::make_unique<exec::HashPartitionFunction>(
      localExchange,
      numPartitions,
      inputType_,
      keyChannels_,
      constValues_,
      segmentChannel_);
}

std::string HashPartitionFunctionSpec::toString() const {
  std::ostringstream keys;
  size_t constIndex = 0;
  for (auto i = 0; i < keyChannels_.size(); ++i) {
    if (i > 0) {
      keys << ", ";
    }
    auto channel = keyChannels_[i];
    if (channel == kConstantChannel) {
      keys << "\"" << constValues_[constIndex++]->toString(0) << "\"";
    } else {
      keys << inputType_->nameOf(channel);
    }
  }

  return fmt::format("HASH({})", keys.str());
}

folly::dynamic HashPartitionFunctionSpec::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "HashPartitionFunctionSpec";
  obj["inputType"] = inputType_->serialize();
  obj["keyChannels"] = ISerializable::serialize(keyChannels_);
  std::vector<velox::core::ConstantTypedExpr> constValues;
  constValues.reserve(constValues_.size());
  for (const auto& value : constValues_) {
    VELOX_CHECK_NOT_NULL(value);
    constValues.emplace_back(value);
  }
  obj["constants"] = ISerializable::serialize(constValues);
  if (segmentChannel_.has_value()) {
    obj["segmentChannel"] = *segmentChannel_;
  }
  return obj;
}

// static
core::PartitionFunctionSpecPtr HashPartitionFunctionSpec::deserialize(
    const folly::dynamic& obj,
    void* context) {
  const auto keys = ISerializable::deserialize<std::vector<column_index_t>>(
      obj["keyChannels"], context);
  const auto constTypeExprs =
      ISerializable::deserialize<std::vector<velox::core::ConstantTypedExpr>>(
          obj["constants"], context);

  auto* pool = static_cast<memory::MemoryPool*>(context);
  std::vector<VectorPtr> constValues;
  constValues.reserve(constTypeExprs.size());
  for (const auto& value : constTypeExprs) {
    constValues.emplace_back(value->toConstantVector(pool));
  }
  std::optional<column_index_t> segmentChannel;
  if (obj.count("segmentChannel") != 0 && !obj["segmentChannel"].isNull()) {
    segmentChannel = static_cast<column_index_t>(obj["segmentChannel"].asInt());
  }
  return std::make_shared<HashPartitionFunctionSpec>(
      ISerializable::deserialize<RowType>(obj["inputType"]),
      keys,
      constValues,
      segmentChannel);
}
} // namespace facebook::velox::exec
