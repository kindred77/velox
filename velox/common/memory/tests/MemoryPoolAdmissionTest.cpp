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

#include <future>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cstring>
#include <cstdlib>
#include <gtest/gtest.h>
#include <folly/ScopeGuard.h>
#include "velox/common/base/AdmissionController.h"
#include "velox/buffer/Buffer.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/AggregationMasks.h"
#include "velox/vector/FlatVector.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/AllocationPool.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/common/memory/AllocationPreparation.h"

namespace facebook::velox::memory {
class MemoryPoolAdmissionTest : public testing::Test {
 protected:
  MemoryPoolAdmissionTest() : manager_(MemoryManager::Options{}) {
    root_ = manager_.addRootPool("admission-test");
  }
  std::shared_ptr<common::AdmissionController> budget(uint64_t bytes) {
    common::AdmissionController::Config config;
    config.maxLimit = bytes;
    return std::make_shared<common::AdmissionController>(config);
  }
  VectorPtr nestedDictionary(vector_size_t size, MemoryPool* source) {
    auto base = BaseVector::create(BIGINT(), size, source);
    auto* values = base->asFlatVector<int64_t>();
    for (vector_size_t i = 0; i < size; ++i) values->set(i, i * 3);
    values->setNull(3, true);
    auto indices = AlignedBuffer::allocate<vector_size_t>(size, source);
    auto outerIndices = AlignedBuffer::allocate<vector_size_t>(size, source);
    for (vector_size_t i = 0; i < size; ++i) {
      indices->asMutable<vector_size_t>()[i] = size - 1 - i;
      outerIndices->asMutable<vector_size_t>()[i] = (i + 7) % size;
    }
    auto inner = BaseVector::wrapInDictionary(nullptr, indices, size, base);
    inner->setNull(7, true);
    auto outer = BaseVector::wrapInDictionary(nullptr, outerIndices, size, inner);
    outer->setNull(11, true);
    return outer;
  }
  VectorPtr hasherInput(const TypePtr& type, vector_size_t size, MemoryPool* source) {
    auto input = BaseVector::create(type, size, source);
    for (vector_size_t i = 0; i < size; ++i) {
      if (type->kind() == TypeKind::BIGINT) {
        input->asFlatVector<int64_t>()->set(i, int64_t(i) * 1000000007);
      } else if (type->kind() == TypeKind::TIMESTAMP) {
        input->asFlatVector<Timestamp>()->set(i, Timestamp(int64_t(i) * 100000007, (i % 1000) * 1000000));
      } else {
        const auto text = std::string("value-") + std::to_string(i) + std::string(100, 'x');
        input->asFlatVector<StringView>()->set(i, StringView(text));
      }
    }
    input->setNull(3, true);
    return input;
  }
  std::vector<uint64_t> valueIds(exec::VectorHasher& hasher, const VectorPtr& input) {
    SelectivityVector rows(input->size());
    raw_vector<uint64_t> result(input->size());
    hasher.decode(*input, rows);
    (void)hasher.computeValueIds(rows, result);
    hasher.enableValueIds(1, 0);
    hasher.decode(*input, rows);
    const bool mapped = hasher.computeValueIds(rows, result);
    EXPECT_TRUE(mapped);
    if (!mapped) return {};
    return std::vector<uint64_t>(result.begin(), result.end());
  }
  MemoryManager manager_;
  std::shared_ptr<MemoryPool> root_;
};

TEST_F(MemoryPoolAdmissionTest, asyncCreditWaitsForRealFreeAndRejectsWrongPoolBudget) {
  auto shared = budget(64);
  auto first = root_->addLeafChild("async-first");
  auto second = root_->addLeafChild("async-second");
  first->setAllocationBudget(shared);
  second->setAllocationBudget(shared);
  auto* held = first->allocate(64);
  auto pending = shared->reserveAsync(64);
  auto ready = pending.takeFuture();
  EXPECT_FALSE(ready.isReady());
  EXPECT_EQ(second->usedBytes(), 0);
  first->release();
  EXPECT_FALSE(ready.isReady());
  first->free(held, 64);
  EXPECT_TRUE(ready.isReady());
  void* output = nullptr;
  second->withAllocationCredit(std::move(pending), [&] { output = second->allocate(32); });
  EXPECT_EQ(shared->currentResourceUsage(), 64);
  second->free(output, 32);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
  auto other = root_->addLeafChild("async-other-budget");
  auto otherBudget = budget(64);
  other->setAllocationBudget(otherBudget);
  auto wrong = shared->reserveAsync(64);
  EXPECT_THROW(other->withAllocationCredit(std::move(wrong), [] {}), std::exception);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
  EXPECT_EQ(otherBudget->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, asyncCreditRejectsPendingAndRefundsUnusedOnException) {
  auto shared = budget(128);
  auto pool = root_->addLeafChild("async-exception");
  pool->setAllocationBudget(shared);
  auto* held = pool->allocate(128);
  auto pending = shared->reserveAsync(64);
  auto ready = pending.takeFuture();
  bool invoked = false;
  EXPECT_THROW(pool->withAllocationCredit(std::move(pending), [&] { invoked = true; }),
      std::exception);
  EXPECT_FALSE(invoked);
  EXPECT_TRUE(ready.isReady());
  EXPECT_EQ(shared->currentResourceUsage(), 128);
  pool->free(held, 128);
  auto granted = shared->reserveAsync(128);
  void* output = nullptr;
  EXPECT_THROW(pool->withAllocationCredit(std::move(granted), [&] {
    output = pool->allocate(64);
    throw std::runtime_error("operation after allocation");
  }), std::runtime_error);
  EXPECT_EQ(pool->usedBytes(), 64);
  EXPECT_EQ(shared->currentResourceUsage(), 64);
  pool->free(output, 64);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, lookupDecoderMaterializationUsesWorkPool) {
  constexpr vector_size_t size = 4096;
  auto source = root_->addLeafChild("decoder-source");
  auto work = root_->addLeafChild("decoder-work");
  auto shared = budget(1 << 20);
  work->setAllocationBudget(shared);
  auto outer = nestedDictionary(size, source.get());
  const auto inputBytes = source->usedBytes();
  std::vector<std::unique_ptr<exec::VectorHasher>> hashers;
  auto lookup = std::make_unique<exec::HashLookup>(hashers, work.get());
  auto& decoded = lookup->scratchMemory.decoded;
  decoded.decode(*outer);
  EXPECT_GT(work->usedBytes(), 0);
  EXPECT_EQ(shared->currentResourceUsage(), work->usedBytes());
  EXPECT_EQ(source->usedBytes(), inputBytes);
  for (vector_size_t i = 0; i < size; ++i) {
    EXPECT_EQ(decoded.isNullAt(i), outer->isNullAt(i));
    if (!decoded.isNullAt(i)) {
      EXPECT_EQ(decoded.valueAt<int64_t>(i), (size - 1 - (i + 7) % size) * 3);
    }
  }
  lookup.reset();
  EXPECT_EQ(work->usedBytes(), 0);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, decoderOwnershipSurvivesMoveAndRetainsStorageForReuse) {
  constexpr vector_size_t size = 4096;
  auto source = root_->addLeafChild("decoder-move-source");
  auto work = root_->addLeafChild("decoder-move-work");
  auto shared = budget(1 << 20);
  work->setAllocationBudget(shared);
  auto input = nestedDictionary(size, source.get());
  const auto inputBytes = source->usedBytes();
  {
    DecodedVector unmanaged;
    unmanaged.decode(*input);
    EXPECT_EQ(work->usedBytes(), 0);
    EXPECT_EQ(source->usedBytes(), inputBytes);
  }
  std::optional<DecodedVector> moved;
  {
    DecodedVector owned(work.get());
    owned.decode(*input);
    const auto paid = work->usedBytes();
    EXPECT_GT(paid, 0);
    EXPECT_EQ(shared->currentResourceUsage(), paid);
    owned.decode(*input);
    EXPECT_EQ(work->usedBytes(), paid);
    moved.emplace(std::move(owned));
    EXPECT_EQ(work->usedBytes(), paid);
    owned.decode(*input);
    EXPECT_GT(work->usedBytes(), paid);
    EXPECT_EQ(shared->currentResourceUsage(), work->usedBytes());
  }
  EXPECT_GT(work->usedBytes(), 0);
  for (vector_size_t i = 0; i < size; ++i) {
    EXPECT_EQ(moved->isNullAt(i), input->isNullAt(i));
    if (!moved->isNullAt(i)) {
      EXPECT_EQ(moved->valueAt<int64_t>(i), (size - 1 - (i + 7) % size) * 3);
    }
  }
  moved.reset();
  EXPECT_EQ(work->usedBytes(), 0);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, stlAllocatorDefaultResourceMatchesStandardGrowth) {
  StlAllocator<uint64_t> system;
  StlAllocator<uint64_t> sameSystem;
  EXPECT_TRUE(system == sameSystem);
  std::vector<uint64_t, StlAllocator<uint64_t>> values;
  std::vector<uint64_t> standard;
  for (uint64_t i = 0; i < 4096; ++i) {
    values.push_back(i);
    standard.push_back(i);
    EXPECT_EQ(values.capacity(), standard.capacity());
  }
  EXPECT_TRUE(std::equal(values.begin(), values.end(), standard.begin()));
  EXPECT_THROW(StlAllocator<uint64_t>(static_cast<MemoryPool*>(nullptr)), std::exception);
  EXPECT_THROW(system.allocate(std::numeric_limits<size_t>::max()), std::exception);
  struct alignas(128) AlignedValue { char bytes[128]; };
  StlAllocator<AlignedValue> aligned;
  auto* memory = aligned.allocate(2);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(memory) % alignof(AlignedValue), 0);
  aligned.deallocate(memory, 2);
}

TEST_F(MemoryPoolAdmissionTest, hasherValueDomainsUseOwnerAndPreserveDefaultIds) {
  for (const auto& type : std::vector<TypePtr>{BIGINT(), TIMESTAMP(), VARCHAR(), VARBINARY()}) {
    SCOPED_TRACE(type->toString());
    auto source = root_->addLeafChild("value-domain-input");
    auto work = root_->addLeafChild("value-domain-work");
    auto shared = budget(1 << 20);
    work->setAllocationBudget(shared);
    auto input = hasherInput(type, 256, source.get());
    const auto inputBytes = source->usedBytes();
    auto owned = exec::VectorHasher::create(type, 0, work.get());
    auto unmanaged = exec::VectorHasher::create(type, 0);
    const auto expected = valueIds(*unmanaged, input);
    EXPECT_EQ(valueIds(*owned, input), expected);
    EXPECT_GT(work->usedBytes(), 0);
    EXPECT_EQ(shared->currentResourceUsage(), work->usedBytes());
    EXPECT_EQ(source->usedBytes(), inputBytes);
    const auto paid = work->usedBytes();
    EXPECT_EQ(valueIds(*owned, input), expected);
    EXPECT_EQ(work->usedBytes(), paid);
    owned->resetStats();
    EXPECT_EQ(shared->currentResourceUsage(), work->usedBytes());
    owned.reset();
    EXPECT_EQ(work->usedBytes(), 0);
    EXPECT_EQ(shared->currentResourceUsage(), 0);
  }
}

TEST_F(MemoryPoolAdmissionTest, copiedAndMergedStringDomainsKeepPeerPayloadCharged) {
  auto inputPool = root_->addLeafChild("peer-domain-input");
  auto sourcePool = root_->addLeafChild("peer-domain-source");
  auto sourceBudget = budget(1 << 20);
  sourcePool->setAllocationBudget(sourceBudget);
  auto input = hasherInput(VARCHAR(), 256, inputPool.get());
  auto source = exec::VectorHasher::create(VARCHAR(), 0, sourcePool.get());
  const auto expected = valueIds(*source, input);
  const auto sourceBytes = sourcePool->usedBytes();
  EXPECT_GT(sourceBytes, 0);
  for (bool merge : {false, true}) {
    SCOPED_TRACE(merge);
    auto destinationPool = root_->addLeafChild("peer-domain-destination");
    auto destinationBudget = budget(1 << 20);
    destinationPool->setAllocationBudget(destinationBudget);
    auto destination = exec::VectorHasher::create(VARCHAR(), 0, destinationPool.get());
    auto reference = exec::VectorHasher::create(VARCHAR(), 0);
    if (merge) {
      auto seed = BaseVector::create(VARCHAR(), 1, inputPool.get());
      const std::string text(100, 's');
      seed->asFlatVector<StringView>()->set(0, StringView(text));
      (void)valueIds(*destination, seed);
      (void)valueIds(*reference, seed);
    }
    const auto initialBytes = destinationPool->usedBytes();
    // An empty destination takes the original private copyStatsFrom path.
    destination->merge(*source, 100000);
    reference->merge(*source, 100000);
    EXPECT_GT(destinationPool->usedBytes(), initialBytes);
    EXPECT_EQ(valueIds(*destination, input), valueIds(*reference, input));
    EXPECT_EQ(sourcePool->usedBytes(), sourceBytes);
    EXPECT_EQ(sourceBudget->currentResourceUsage(), sourceBytes);
    EXPECT_EQ(destinationBudget->currentResourceUsage(), destinationPool->usedBytes());
    destination.reset();
    EXPECT_EQ(destinationBudget->currentResourceUsage(), 0);
    EXPECT_EQ(sourceBudget->currentResourceUsage(), sourceBytes);
  }
  source.reset();
  EXPECT_EQ(sourceBudget->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, flatDomainCopyKeepsDestinationOwnerAndRetainedBacking) {
  auto sourcePool = root_->addLeafChild("flat-copy-source");
  auto destinationPool = root_->addLeafChild("flat-copy-destination");
  auto sourceBudget = budget(1 << 20), destinationBudget = budget(1 << 20);
  sourcePool->setAllocationBudget(sourceBudget);
  destinationPool->setAllocationBudget(destinationBudget);
  std::optional<exec::FlatValueIds> source(std::in_place, sourcePool.get());
  source->insert(0);
  EXPECT_GE(sourcePool->stats().numAllocs, 2);
  for (int64_t i = 1; i < 4096; ++i) source->insert(i * 1000000007);
  {
    exec::FlatValueIds destination(destinationPool.get());
    destination = *source;
    const auto retained = destinationPool->usedBytes();
    EXPECT_GT(retained, 0);
    source.reset();
    EXPECT_EQ(sourceBudget->currentResourceUsage(), 0);
    for (int64_t i = 0; i < 4096; ++i) {
      EXPECT_EQ(destination.find(i * 1000000007), i + 1);
    }
    destination.clear();
    EXPECT_EQ(destinationPool->usedBytes(), retained);
    EXPECT_EQ(destinationBudget->currentResourceUsage(), retained);
    EXPECT_EQ(destination.insert(-1).first, 1);
    EXPECT_EQ(destinationPool->usedBytes(), retained);
  }
  EXPECT_EQ(destinationBudget->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, longStringPayloadUsesOwnerAfterInputIsReleased) {
  auto inputPool = root_->addLeafChild("large-string-input");
  auto work = root_->addLeafChild("large-string-work");
  auto shared = budget(256 << 10);
  work->setAllocationBudget(shared);
  auto owned = exec::VectorHasher::create(VARCHAR(), 0, work.get());
  const auto initial = work->usedBytes();
  EXPECT_GT(initial, 0);
  const std::string text((64 << 10) + 17, 'x');
  auto input = BaseVector::create(VARCHAR(), 1, inputPool.get());
  input->asFlatVector<StringView>()->set(0, StringView(text));
  const auto expected = valueIds(*owned, input);
  const auto paid = work->usedBytes();
  EXPECT_GE(paid - initial, text.size());
  input.reset();
  EXPECT_EQ(inputPool->usedBytes(), 0);
  EXPECT_EQ(shared->currentResourceUsage(), paid);
  input = BaseVector::create(VARCHAR(), 1, inputPool.get());
  input->asFlatVector<StringView>()->set(0, StringView(text));
  EXPECT_EQ(valueIds(*owned, input), expected);
  EXPECT_EQ(work->usedBytes(), paid);
  owned->resetStats();
  EXPECT_LT(work->usedBytes(), paid);
  EXPECT_EQ(shared->currentResourceUsage(), work->usedBytes());
  owned.reset();
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, valueDomainBudgetFailureDiscardsStateWithoutLeakingUnits) {
  auto inputPool = root_->addLeafChild("refused-domain-input");
  auto input = BaseVector::create(VARCHAR(), 1, inputPool.get());
  const std::string text((64 << 10) + 17, 'x');
  input->asFlatVector<StringView>()->set(0, StringView(text));
  for (uint64_t limit : {8192, 8200, 8256, 8448, 16384}) {
    SCOPED_TRACE(limit);
    auto work = root_->addLeafChild("refused-domain-work");
    auto shared = budget(limit);
    work->setAllocationBudget(shared);
    auto owned = exec::VectorHasher::create(VARCHAR(), 0, work.get());
    EXPECT_THROW((void)valueIds(*owned, input), std::exception);
    EXPECT_LE(shared->currentResourceUsage(), limit);
    EXPECT_EQ(shared->currentResourceUsage(), work->usedBytes());
    // Discard the failed operation's state; never replay a partly updated aggregate.
    owned.reset();
    EXPECT_EQ(shared->currentResourceUsage(), 0);
    EXPECT_EQ(work->usedBytes(), 0);
  }
}

TEST_F(MemoryPoolAdmissionTest, rowWiseMultipartStringScratchUsesHasherOwner) {
  struct Row { uint8_t nulls{0}; StringView value; };
  for (int mode : {0, 1, 2}) {
    SCOPED_TRACE(mode);
    const bool multipart = mode == 2;
    std::string text = multipart ? std::string((128 << 10) + 17, 'x')
                                : (mode == 1 ? std::string(64, 'c') : "tiny");
    if (multipart) text[128] = '\0';
    auto source = root_->addLeafChild("row-string-source");
    auto work = root_->addLeafChild("row-string-work");
    auto shared = budget(1 << 20);
    work->setAllocationBudget(shared);
    auto input = BaseVector::create(VARCHAR(), 1, source.get());
    input->asFlatVector<StringView>()->set(0, StringView(text));
    auto hasher = exec::VectorHasher::create(VARCHAR(), 0, work.get());
    const auto expected = valueIds(*hasher, input);
    HashStringAllocator allocator(source.get());
    Row row;
    if (multipart) {
      ByteOutputStream stream(&allocator);
      (void)allocator.newWrite(stream, 32);
      stream.appendStringView(text);
      const auto positions = allocator.finishWrite(stream, 0);
      ASSERT_TRUE(positions.first.header->isContinued());
      row.value = StringView(positions.first.position, text.size());
    } else if (mode == 1) {
      allocator.copyMultipart(StringView(text), reinterpret_cast<char*>(&row), offsetof(Row, value));
      ASSERT_GE(HashStringAllocator::headerOf(row.value.data())->size(), row.value.size());
    } else row.value = StringView(text);
    auto* group = reinterpret_cast<char*>(&row);
    raw_vector<uint64_t> result(1);
    const auto before = work->stats();
    EXPECT_TRUE(hasher->computeValueIdsForRows(&group, 1, offsetof(Row, value),
        offsetof(Row, nulls), 1, result));
    EXPECT_EQ(result[0], expected[0]);
    EXPECT_EQ(work->usedBytes(), before.usedBytes);
    EXPECT_EQ(shared->currentResourceUsage(), before.usedBytes);
    if (multipart) {
      EXPECT_GT(work->stats().numAllocs, before.numAllocs);
      EXPECT_GE(work->stats().peakBytes, before.usedBytes + text.size());
    } else EXPECT_EQ(work->stats().numAllocs, before.numAllocs);
    row.nulls = 1;
    const auto allocations = work->stats().numAllocs;
    EXPECT_TRUE(hasher->computeValueIdsForRows(&group, 1, offsetof(Row, value),
        offsetof(Row, nulls), 1, result));
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(work->stats().numAllocs, allocations);
    hasher.reset();
    EXPECT_EQ(shared->currentResourceUsage(), 0);
  }
}

TEST_F(MemoryPoolAdmissionTest, multipartStorageUsesAllocatorAndPreservesOldDataOnRefusal) {
  auto source = root_->addLeafChild("multipart-storage-source");
  auto work = root_->addLeafChild("multipart-storage-work");
  auto shared = budget(64 << 10);
  work->setAllocationBudget(shared);
  HashStringAllocator allocator(source.get());
  const std::string text((128 << 10) + 17, 'b');
  ByteOutputStream stream(&allocator);
  (void)allocator.newWrite(stream, 32);
  stream.appendStringView(text);
  const auto positions = allocator.finishWrite(stream, 0);
  ASSERT_TRUE(positions.first.header->isContinued());
  const StringView multipart(positions.first.position, text.size());
  std::string unmanaged;
  EXPECT_EQ(HashStringAllocator::contiguousString(multipart, unmanaged), StringView(text));
  EXPECT_EQ(work->usedBytes(), 0);
  using Storage = std::basic_string<char, std::char_traits<char>, StlAllocator<char>>;
  {
    Storage storage{StlAllocator<char>(work.get())};
    storage.resize(32 << 10, 'a');
    const auto before = work->usedBytes();
    EXPECT_THROW(HashStringAllocator::contiguousString(multipart, storage), std::exception);
    EXPECT_EQ(storage.size(), 32 << 10);
    EXPECT_TRUE(std::all_of(storage.begin(), storage.end(), [](char c) { return c == 'a'; }));
    EXPECT_EQ(work->usedBytes(), before);
    EXPECT_EQ(shared->currentResourceUsage(), before);
  }
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, rowContainerReadScratchUsesCallerCredit) {
  constexpr uint64_t limit = 16 << 20;
  auto shared = budget(limit);
  auto source = root_->addLeafChild("row-input");
  auto data = root_->addLeafChild("row-data");
  auto caller = root_->addLeafChild("row-reader");
  data->setAllocationBudget(shared);
  caller->setAllocationBudget(shared);
  auto input = BaseVector::create(VARCHAR(), 3, source.get());
  std::string text((128 << 10) + 17, 'x');
  text[128] = '\0';
  input->asFlatVector<StringView>()->set(0, StringView(text));
  text.back() = 'y';
  input->asFlatVector<StringView>()->set(1, StringView(text));
  input->setNull(2, true);
  DecodedVector decoded(*input);
  exec::RowContainer container({VARCHAR()}, data.get());
  std::vector<char*> rows;
  for (int i = 0; i < 3; ++i) {
    rows.push_back(container.newRow());
    container.store(decoded, i, rows.back(), 0);
  }
  StringView stored;
  std::memcpy(&stored, rows[0] + container.columnAt(0).offset(), sizeof(stored));
  ASSERT_TRUE(HashStringAllocator::headerOf(stored.data())->isContinued());
  uint64_t expected[3];
  container.hash(0, folly::Range<char**>(rows.data(), rows.size()), false, expected);
  const auto resident = data->usedBytes();
  const auto dataAllocs = data->stats().numAllocs;
  const auto callerAllocs = caller->stats().numAllocs;
  const auto charged = shared->currentResourceUsage();
  ASSERT_TRUE(caller->tryWithAllocationCredit(limit - charged, [&] {
    EXPECT_LT(container.compare(rows[0], rows[1], 0), 0);
    EXPECT_LT(container.compare(rows[0], container.columnAt(0), decoded, 1), 0);
    uint64_t actual[3];
    container.hash(0, folly::Range<char**>(rows.data(), rows.size()), false, actual);
    for (int i = 0; i < 3; ++i) EXPECT_EQ(actual[i], expected[i]);
    EXPECT_EQ(caller->usedBytes(), 0);
  }));
  EXPECT_GT(caller->stats().numAllocs, callerAllocs);
  EXPECT_EQ(data->usedBytes(), resident);
  EXPECT_EQ(data->stats().numAllocs, dataAllocs);
  EXPECT_EQ(shared->currentResourceUsage(), charged);
  ASSERT_TRUE(caller->tryWithAllocationCredit(64 << 10, [&] {
    EXPECT_THROW(container.compare(rows[0], rows[1], 0), std::exception);
    EXPECT_EQ(caller->usedBytes(), 0);
  }));
  EXPECT_EQ(shared->currentResourceUsage(), charged);
  EXPECT_EQ(data->usedBytes(), resident);
  EXPECT_EQ(data->stats().numAllocs, dataAllocs);
  EXPECT_LT(container.compare(rows[0], rows[1], 0), 0);
}

TEST_F(MemoryPoolAdmissionTest, contiguousRowReadsNeedNoScratchCredit) {
  auto source = root_->addLeafChild("contiguous-input");
  auto data = root_->addLeafChild("contiguous-data");
  auto caller = root_->addLeafChild("contiguous-reader");
  auto shared = budget(1 << 20);
  caller->setAllocationBudget(shared);
  auto input = BaseVector::create(VARCHAR(), 3, source.get());
  input->asFlatVector<StringView>()->set(0, StringView("inline"));
  const std::string text(100, 'x');
  input->asFlatVector<StringView>()->set(1, StringView(text));
  input->setNull(2, true);
  DecodedVector decoded(*input);
  exec::RowContainer container({VARCHAR()}, data.get());
  std::vector<char*> rows;
  for (int i = 0; i < 3; ++i) {
    rows.push_back(container.newRow());
    container.store(decoded, i, rows.back(), 0);
  }
  const auto allocations = data->stats().numAllocs;
  ASSERT_TRUE(caller->tryWithAllocationCredit(0, [&] {
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(container.compare(rows[i], rows[i], 0), 0);
      EXPECT_EQ(container.compare(rows[i], container.columnAt(0), decoded, i), 0);
    }
    uint64_t actual[3];
    container.hash(0, folly::Range<char**>(rows.data(), rows.size()), false, actual);
    EXPECT_EQ(actual[0], folly::hasher<StringView>()(StringView("inline")));
    EXPECT_EQ(actual[1], folly::hasher<StringView>()(StringView(text)));
    EXPECT_EQ(actual[2], BaseVector::kNullHash);
  }));
  EXPECT_EQ(caller->stats().numAllocs, 0);
  EXPECT_EQ(data->stats().numAllocs, allocations);
}

TEST_F(MemoryPoolAdmissionTest, scratchCreditOwnerIsScopedAndThreadLocal) {
  auto shared = budget(1 << 20);
  auto outer = root_->addLeafChild("scratch-outer");
  auto inner = root_->addLeafChild("scratch-inner");
  outer->setAllocationBudget(shared);
  inner->setAllocationBudget(shared);
  EXPECT_EQ(MemoryPool::currentAllocationCreditPool(), nullptr);
  ASSERT_TRUE(outer->tryWithAllocationCredit(1 << 18, [&] {
    EXPECT_EQ(MemoryPool::currentAllocationCreditPool(), outer.get());
    EXPECT_EQ(std::async(std::launch::async, [] {
      return MemoryPool::currentAllocationCreditPool();
    }).get(), nullptr);
    EXPECT_THROW(inner->tryWithAllocationCredit(1 << 18, [&] {
      EXPECT_EQ(MemoryPool::currentAllocationCreditPool(), inner.get());
      throw std::runtime_error("scope unwind");
    }), std::runtime_error);
    EXPECT_EQ(MemoryPool::currentAllocationCreditPool(), outer.get());
  }));
  EXPECT_EQ(MemoryPool::currentAllocationCreditPool(), nullptr);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, aggregationMaskBitsAndDecoderUseWorkPool) {
  constexpr vector_size_t size = 8192;
  auto shared = budget(1 << 20);
  auto source = root_->addLeafChild("mask-source");
  auto work = root_->addLeafChild("mask-work");
  work->setAllocationBudget(shared);
  auto base = BaseVector::create(BOOLEAN(), size, source.get());
  for (int i = 0; i < size; ++i) base->asFlatVector<bool>()->set(i, i % 3 == 0);
  base->setNull(3, true);
  auto indices = AlignedBuffer::allocate<vector_size_t>(size, source.get());
  auto outerIndices = AlignedBuffer::allocate<vector_size_t>(size, source.get());
  for (int i = 0; i < size; ++i) {
    indices->asMutable<vector_size_t>()[i] = size - 1 - i;
    outerIndices->asMutable<vector_size_t>()[i] = (i + 7) % size;
  }
  auto inner = BaseVector::wrapInDictionary(nullptr, indices, size, base);
  inner->setNull(7, true);
  auto outer = BaseVector::wrapInDictionary(nullptr, outerIndices, size, inner);
  outer->setNull(11, true);
  auto input = std::make_shared<RowVector>(source.get(), ROW({"m"}, {BOOLEAN()}),
      nullptr, size, std::vector<VectorPtr>{outer});
  SelectivityVector rows(size);
  rows.setValidRange(100, 200, false);
  rows.updateBounds();
  {
    exec::AggregationMasks masks({0, 0, std::nullopt}, work.get());
    exec::AggregationMasks reference({0, 0, std::nullopt});
    reference.addInput(input, rows);
    masks.addInput(input, rows);
    EXPECT_GT(work->usedBytes(), size * sizeof(vector_size_t));
    EXPECT_EQ(masks.activeRows(0), masks.activeRows(1));
    EXPECT_EQ(masks.activeRows(2), nullptr);
    const auto allocations = work->stats().numAllocs;
    const auto charge = work->usedBytes();
    for (int i = 0; i < size; ++i) {
      const auto innerRow = (i + 7) % size;
      const auto baseRow = size - 1 - innerRow;
      const bool expected = rows.isValid(i) && i != 11 && innerRow != 7 &&
          baseRow != 3 && baseRow % 3 == 0;
      EXPECT_EQ(masks.activeRows(0)->isValid(i), expected);
      EXPECT_EQ(masks.activeRows(0)->isValid(i), reference.activeRows(0)->isValid(i));
    }
    masks.addInput(input, rows);
    EXPECT_EQ(work->stats().numAllocs, allocations);
    EXPECT_EQ(work->usedBytes(), charge);
    EXPECT_EQ(shared->currentResourceUsage(), charge);
  }
  EXPECT_EQ(work->usedBytes(), 0);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, selectionBitsUseDestinationOwner) {
  auto shared = budget(1 << 20);
  auto first = root_->addLeafChild("selection-first");
  auto second = root_->addLeafChild("selection-second");
  first->setAllocationBudget(shared);
  second->setAllocationBudget(shared);
  {
    SelectivityVector source(131073, true, first.get());
    source.setValid(65, false);
    source.updateBounds();
    EXPECT_GT(first->usedBytes(), 0);
    SelectivityVector dest(0, false, second.get());
    dest = source;
    EXPECT_GT(second->usedBytes(), 0);
    EXPECT_FALSE(dest.isValid(65));
    EXPECT_EQ(dest.countSelected(), source.countSelected());
    const auto firstCharge = first->usedBytes();
    const auto secondCharge = second->usedBytes();
    SelectivityVector independent(source);
    EXPECT_EQ(first->usedBytes(), firstCharge);
    EXPECT_EQ(second->usedBytes(), secondCharge);
    EXPECT_EQ(independent.countSelected(), source.countSelected());
    SelectivityVector moved(std::move(source));
    EXPECT_EQ(first->usedBytes(), firstCharge);
    dest = std::move(moved);
    EXPECT_EQ(second->usedBytes(), secondCharge);
    EXPECT_EQ(dest.countSelected(), independent.countSelected());
    EXPECT_EQ(SelectivityVector::empty().size(), 0);
    dest = SelectivityVector::empty();
    EXPECT_EQ(dest.size(), 0);
    EXPECT_EQ(second->usedBytes(), secondCharge);
    dest.resize(1);
    EXPECT_EQ(dest.countSelected(), 1);
    EXPECT_EQ(second->usedBytes(), secondCharge); // Retained until real free.
  }
  EXPECT_EQ(first->usedBytes(), 0);
  EXPECT_EQ(second->usedBytes(), 0);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, selectionRefusalPreservesSelectionAndRealCharge) {
  auto shared = budget(4096);
  auto work = root_->addLeafChild("selection-refusal");
  work->setAllocationBudget(shared);
  {
    SelectivityVector rows(65, true, work.get());
    rows.setValid(32, false);
    rows.updateBounds();
    const auto charged = work->usedBytes();
    EXPECT_GT(charged, 0);
    EXPECT_THROW(rows.resize(1 << 20), std::exception);
    EXPECT_EQ(rows.size(), 65);
    EXPECT_EQ(rows.countSelected(), 64);
    EXPECT_FALSE(rows.isValid(32));
    EXPECT_EQ(work->usedBytes(), charged);
    SelectivityVector large(1 << 20);
    EXPECT_THROW(rows = large, std::exception);
    EXPECT_EQ(rows.size(), 65);
    EXPECT_EQ(rows.countSelected(), 64);
    EXPECT_EQ(work->usedBytes(), charged);
    EXPECT_EQ(shared->currentResourceUsage(), charged);
  }
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, selectionValueCopyOutlivesSourcePool) {
  std::optional<SelectivityVector> copy;
  auto owner = root_->addLeafChild("selection-copy-source");
  {
    SelectivityVector rows(8193, false, owner.get());
    rows.setValid(8192, true);
    rows.updateBounds();
    copy.emplace(rows);
    EXPECT_GT(owner->usedBytes(), 0);
  }
  EXPECT_EQ(owner->usedBytes(), 0);
  owner.reset();
  EXPECT_EQ(copy->size(), 8193);
  EXPECT_EQ(copy->begin(), 8192);
  EXPECT_EQ(copy->end(), 8193);
  EXPECT_EQ(copy->countSelected(), 1);
}

TEST_F(MemoryPoolAdmissionTest, fixedRowDemandAdmitsSmallFirstBatch) {
  auto work = root_->addLeafChild("row-demand-small");
  auto shared = budget(64 << 10);
  work->setAllocationBudget(shared);
  exec::RowContainer rows({BIGINT()}, work.get());
  EXPECT_EQ(rows.fixedRowBackingBytesUpperBound(0), 0);
  const auto demand = rows.fixedRowBackingBytesUpperBound(1);
  EXPECT_LE(demand, 64 << 10);
  bool consumed = false;
  const bool accepted = work->tryWithAllocationCredit(demand, [&] {
    consumed = true;
    EXPECT_NE(rows.newRow(), nullptr);
  });
  EXPECT_TRUE(accepted);
  EXPECT_TRUE(consumed);
  EXPECT_LE(work->usedBytes(), demand);
  rows.clear();
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, fixedRowDemandCoversLayoutRunsAndFreeRows) {
  for (const int alignment : {1, 8, 64}) {
    for (const bool normalized : {false, true}) {
      auto work = root_->addLeafChild("row-demand-layout");
      auto shared = budget(32 << 20);
      work->setAllocationBudget(shared);
      std::vector<exec::Accumulator> accumulators;
      accumulators.emplace_back(true, 177, false, alignment, BIGINT(),
          [](folly::Range<char**>, VectorPtr&) {}, [](folly::Range<char**>) {});
      exec::RowContainer rows({BIGINT()}, true, accumulators, {}, false, false,
          false, false, normalized, false, work.get());
      std::vector<char*> allocated;
      for (const uint64_t count : {1, 127, 8193, 2333}) {
        const auto before = work->usedBytes();
        const auto demand = rows.fixedRowBackingBytesUpperBound(count);
        ASSERT_TRUE(work->tryWithAllocationCredit(demand, [&] {
          for (uint64_t i = 0; i < count; ++i) allocated.push_back(rows.newRow());
        }));
        EXPECT_LE(work->usedBytes() - before, demand);
        EXPECT_EQ(shared->currentResourceUsage(), work->usedBytes());
      }
      constexpr size_t reuse = 512;
      rows.eraseRows(folly::Range<char**>(allocated.data(), reuse));
      EXPECT_EQ(rows.fixedRowBackingBytesUpperBound(reuse), 0);
      const auto before = work->usedBytes();
      const auto allocations = work->stats().numAllocs;
      ASSERT_TRUE(work->tryWithAllocationCredit(0, [&] {
        for (size_t i = 0; i < reuse; ++i) EXPECT_NE(rows.newRow(), nullptr);
      }));
      EXPECT_EQ(work->usedBytes(), before);
      EXPECT_EQ(work->stats().numAllocs, allocations);
      rows.disableNormalizedKeys();
      const auto demand = rows.fixedRowBackingBytesUpperBound(2000);
      ASSERT_TRUE(work->tryWithAllocationCredit(demand, [&] {
        for (int i = 0; i < 2000; ++i) EXPECT_NE(rows.newRow(), nullptr);
      }));
      EXPECT_LE(work->usedBytes() - before, demand);
      rows.clear();
      EXPECT_EQ(shared->currentResourceUsage(), 0);
    }
  }
}

TEST_F(MemoryPoolAdmissionTest, fixedRowDemandRejectsUnboundedIndexAndOverflow) {
  auto work = root_->addLeafChild("row-demand-unsupported");
  exec::RowContainer indexed({BIGINT()}, {}, true, work.get());
  EXPECT_THROW(indexed.fixedRowBackingBytesUpperBound(1), std::exception);
  exec::RowContainer rows({BIGINT()}, work.get());
  EXPECT_THROW(rows.fixedRowBackingBytesUpperBound(UINT64_MAX), std::exception);
  EXPECT_EQ(work->usedBytes(), 0);
}

TEST_F(MemoryPoolAdmissionTest, actualBytesAcrossPoolsAndReservationRelease) {
  auto shared = budget(128);
  auto a = root_->addLeafChild("a"), b = root_->addLeafChild("b");
  a->setAllocationBudget(shared); b->setAllocationBudget(shared);
  void* x = a->allocate(64); void* y = b->allocate(64); void* extra = nullptr;
  auto cleanup = folly::makeGuard([&] { a->free(x, 64); b->free(y, 64); if (extra) b->free(extra, 64); });
  EXPECT_EQ(shared->currentResourceUsage(), 128);
  a->release(); b->release();
  EXPECT_EQ(shared->currentResourceUsage(), 128) << "unused reservation release is not actual free";
  bool refused = false;
  try { extra = b->allocate(64); } catch (const std::exception&) { refused = true; }
  EXPECT_TRUE(refused);
  EXPECT_EQ(a->usedBytes() + b->usedBytes(), 128);
}

TEST_F(MemoryPoolAdmissionTest, preauthorizationExceptionAndActualFree) {
  auto shared = budget(128);
  auto pool = root_->addLeafChild("a"); pool->setAllocationBudget(shared);
  void* held = nullptr;
  auto cleanup = folly::makeGuard([&] { if (held) pool->free(held, 64); });
  EXPECT_THROW(pool->tryWithAllocationCredit(128, [&] {
    EXPECT_EQ(shared->currentResourceUsage(), 128);
    held = pool->allocate(64);
    throw std::runtime_error("after actual allocation");
  }), std::runtime_error);
  EXPECT_EQ(shared->currentResourceUsage(), 64) << "only unspent credit rolls back";
  bool called = false;
  EXPECT_FALSE(pool->tryWithAllocationCredit(128, [&] { called = true; }));
  EXPECT_FALSE(called);
  pool->free(held, 64); held = nullptr;
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, creditReusesActualFreeButCannotBeStolenByAnotherThread) {
  auto shared = budget(128);
  auto pool = root_->addLeafChild("a"); pool->setAllocationBudget(shared);
  EXPECT_TRUE(pool->tryWithAllocationCredit(128, [&] {
    auto thief = std::async(std::launch::async, [&] {
      try { auto* p = pool->allocate(64); pool->free(p, 64); return false; }
      catch (const std::exception&) { return true; }
    });
    EXPECT_TRUE(thief.get());
    for (int i = 0; i < 4; ++i) {
      auto* p = pool->allocate(128);
      EXPECT_EQ(shared->currentResourceUsage(), 128);
      pool->free(p, 128);
      EXPECT_EQ(shared->currentResourceUsage(), 128) << "freed credit remains reserved until scope exit";
    }
  }));
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, transferKeepsChargeAndRejectsUnaccountedDomain) {
  auto shared = budget(128);
  auto a = root_->addLeafChild("a"), b = root_->addLeafChild("b"), plain = root_->addLeafChild("plain");
  a->setAllocationBudget(shared); b->setAllocationBudget(shared);
  auto* p = a->allocate(64);
  auto* owner = a.get();
  auto cleanup = folly::makeGuard([&] { owner->free(p, 64); });
  if (a->transferTo(b.get(), p, 64)) owner = b.get();
  else ADD_FAILURE() << "same budget transfer must retain existing zero-copy path";
  EXPECT_EQ(shared->currentResourceUsage(), 64);
  const bool crossed = b->transferTo(plain.get(), p, 64);
  if (crossed) owner = plain.get();
  EXPECT_FALSE(crossed) << "moving pool stats cannot silently return a live allocation's charge";
  EXPECT_EQ(shared->currentResourceUsage(), 64);
}

TEST_F(MemoryPoolAdmissionTest, pageReplacementFailureSettlesDestroyedOldAllocation) {
  auto shared = budget(16ULL << 20);
  auto pool = root_->addLeafChild("a"); pool->setAllocationBudget(shared);
  Allocation noncontiguous; ContiguousAllocation contiguous;
  auto cleanup = folly::makeGuard([&] {
    manager_.allocator()->testingClearFailureInjection();
    if (!noncontiguous.empty()) pool->freeNonContiguous(noncontiguous);
    if (!contiguous.empty()) pool->freeContiguous(contiguous);
  });
  pool->allocateNonContiguous(4, noncontiguous);
  EXPECT_EQ(shared->currentResourceUsage(), noncontiguous.byteSize());
  manager_.allocator()->testingSetFailureInjection(MemoryAllocator::InjectedFailure::kAllocate, true);
  EXPECT_THROW(pool->allocateNonContiguous(8, noncontiguous), std::exception);
  EXPECT_TRUE(noncontiguous.empty());
  EXPECT_EQ(shared->currentResourceUsage(), 0);
  manager_.allocator()->testingClearFailureInjection();
  pool->allocateContiguous(4, contiguous);
  EXPECT_EQ(shared->currentResourceUsage(), contiguous.size());
  manager_.allocator()->testingSetFailureInjection(MemoryAllocator::InjectedFailure::kAllocate, true);
  EXPECT_THROW(pool->allocateContiguous(8, contiguous), std::exception);
  EXPECT_TRUE(contiguous.empty());
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, reallocBudgetRefusalPreservesOldChargeAndExternalAccounting) {
  auto shared = budget(128);
  auto pool = root_->addLeafChild("a"); pool->setAllocationBudget(shared);
  auto* p = pool->allocate(64);
  int64_t heldSize = 64;
  auto cleanup = folly::makeGuard([&] { manager_.allocator()->testingClearFailureInjection(); pool->free(p, heldSize); });
  EXPECT_EQ(shared->currentResourceUsage(), 64);
  bool refused = false;
  try { p = pool->reallocate(p, 64, 128); heldSize = 128; }
  catch (const std::exception&) { refused = true; }
  EXPECT_TRUE(refused);
  EXPECT_EQ(pool->usedBytes(), 64);
  EXPECT_EQ(shared->currentResourceUsage(), 64);
  manager_.allocator()->testingClearFailureInjection();
  pool->reportExternalAllocation(33);
  EXPECT_EQ(shared->currentResourceUsage(), 97);
  pool->reportExternalFree(33);
  EXPECT_EQ(shared->currentResourceUsage(), 64);
}

TEST_F(MemoryPoolAdmissionTest, successfulPageSizeMixReplacementAndGrow) {
  auto shared = budget(1ULL << 20);
  auto pool = root_->addLeafChild("a"); pool->setAllocationBudget(shared);
  Allocation pages; ContiguousAllocation contiguous;
  auto cleanup = folly::makeGuard([&] {
    if (!pages.empty()) pool->freeNonContiguous(pages);
    if (!contiguous.empty()) pool->freeContiguous(contiguous);
  });
  pool->allocateNonContiguous(7, pages, 4);
  EXPECT_EQ(pages.byteSize(), manager_.allocator()->allocationBytes(7, 4));
  EXPECT_EQ(shared->currentResourceUsage(), pages.byteSize());
  pool->allocateNonContiguous(2, pages);
  EXPECT_EQ(shared->currentResourceUsage(), pages.byteSize());
  pool->allocateContiguous(2, contiguous, 8);
  pool->growContiguous(4, contiguous);
  EXPECT_EQ(shared->currentResourceUsage(), pages.byteSize() + contiguous.size());
  pool->allocateContiguous(1, contiguous);
  EXPECT_EQ(shared->currentResourceUsage(), pages.byteSize() + contiguous.size());
  pool->freeNonContiguous(pages); pool->freeContiguous(contiguous);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, pageReservationExceptionAlsoSettlesFreedOldBlock) {
  auto shared = budget(64ULL << 20);
  auto smallRoot = manager_.addRootPool("small", 8ULL << 20);
  auto pool = smallRoot->addLeafChild("a"); pool->setAllocationBudget(shared);
  Allocation pages; ContiguousAllocation contiguous;
  auto cleanup = folly::makeGuard([&] {
    if (!pages.empty()) pool->freeNonContiguous(pages);
    if (!contiguous.empty()) pool->freeContiguous(contiguous);
  });
  pool->allocateNonContiguous(2, pages);
  EXPECT_GT(shared->currentResourceUsage(), 0);
  EXPECT_THROW(pool->allocateNonContiguous(4096, pages), std::exception);
  EXPECT_TRUE(pages.empty());
  EXPECT_EQ(shared->currentResourceUsage(), 0);
  pool->allocateContiguous(2, contiguous);
  EXPECT_GT(shared->currentResourceUsage(), 0);
  EXPECT_THROW(pool->allocateContiguous(4096, contiguous), std::exception);
  EXPECT_TRUE(contiguous.empty());
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, bothReallocModesPreserveDataAndCharge) {
  const bool saved = FLAGS_velox_enable_inplace_realloc;
  auto restore = folly::makeGuard([&] { FLAGS_velox_enable_inplace_realloc = saved; });
  auto shared = budget(4096);
  auto pool = root_->addLeafChild("a"); pool->setAllocationBudget(shared);
  for (bool inplace : {false, true}) {
    FLAGS_velox_enable_inplace_realloc = inplace;
    void* p = pool->allocate(64); int64_t size = 64;
    auto cleanup = folly::makeGuard([&] { if (p) pool->free(p, size); });
    std::memset(p, 0x5a, 64);
    p = pool->reallocate(p, size, 128); size = 128;
    EXPECT_EQ(shared->currentResourceUsage(), pool->usedBytes());
    for (int i = 0; i < 64; ++i) EXPECT_EQ(static_cast<unsigned char*>(p)[i], 0x5a);
    p = pool->reallocate(p, size, 16); size = 16;
    EXPECT_EQ(shared->currentResourceUsage(), pool->usedBytes());
    for (int i = 0; i < 16; ++i) EXPECT_EQ(static_cast<unsigned char*>(p)[i], 0x5a);
    pool->free(p, size); p = nullptr;
    EXPECT_EQ(shared->currentResourceUsage(), 0);
  }
}

TEST_F(MemoryPoolAdmissionTest, alignedZeroAndBudgetOwnerLifetime) {
  auto shared = budget(4096); std::weak_ptr<common::AdmissionController> owner = shared;
  auto pool = root_->addLeafChild("a"); pool->setAllocationBudget(shared); shared.reset();
  ASSERT_FALSE(owner.expired());
  auto* zero = pool->allocateZeroFilled(4, 64);
  auto* aligned = pool->allocateAligned(129, 256);
  auto cleanup = folly::makeGuard([&] {
    if (zero) pool->free(zero, 256);
    if (aligned) pool->freeAligned(aligned, 129, 256);
  });
  EXPECT_EQ(owner.lock()->currentResourceUsage(), pool->usedBytes());
  for (int i = 0; i < 256; ++i) EXPECT_EQ(static_cast<unsigned char*>(zero)[i], 0);
  pool->free(zero, 256); zero = nullptr;
  pool->freeAligned(aligned, 129, 256); aligned = nullptr;
  EXPECT_EQ(owner.lock()->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, reversiblePreparationRefinesOnlyItsOwnCredit) {
  auto pool = root_->addLeafChild("preparation");
  auto shared = budget(4096);
  pool->setAllocationBudget(shared);
  uint64_t required = 0;
  ASSERT_TRUE(shared->tryAccept(4096 - 64));
  auto pressure = folly::makeGuard([&] { shared->release(4096 - 64); });
  try {
    pool->tryWithAllocationCredit(64, [&] {
      AllocationPreparationScope preparation(pool.get());
      auto* p = pool->allocate(64);
      auto cleanup = folly::makeGuard([&] { pool->free(p, 64); });
      (void)pool->allocate(128);
    });
  } catch (const AllocationPreparationDemand& demand) {
    required = demand.requiredBytes();
  }
  EXPECT_EQ(required, 192);
  EXPECT_EQ(pool->usedBytes(), 0);
  EXPECT_EQ(shared->currentResourceUsage(), 4096 - 64);
  shared->release(4096 - 64);
  pressure.dismiss();
  EXPECT_TRUE(pool->tryWithAllocationCredit(256, [&] {
    AllocationPreparationScope preparation(pool.get());
    EXPECT_THROW(pool->tryWithAllocationCredit(0, [&] {
      (void)pool->allocate(64);
    }), VeloxRuntimeError);
  }));
  EXPECT_EQ(shared->currentResourceUsage(), 0);
  void* first = nullptr;
  void* second = nullptr;
  auto cleanup = folly::makeGuard([&] {
    if (first) pool->free(first, 64);
    if (second) pool->free(second, 128);
  });
  EXPECT_TRUE(pool->tryWithAllocationCredit(0, [&] {
    AllocationPreparationScope preparation(pool.get());
    first = pool->allocate(64);
    second = pool->allocate(128);
    EXPECT_EQ(preparation.grantedBytes(), required);
  }));
  EXPECT_EQ(shared->currentResourceUsage(), 192);
}

TEST_F(MemoryPoolAdmissionTest, reservedRowsAreNotLiveAndNeedNoUpdateCredit) {
  auto pool = root_->addLeafChild("reserved-rows");
  auto shared = budget(1 << 20);
  pool->setAllocationBudget(shared);
  exec::RowContainer rows({BIGINT()}, {}, false, pool.get());
  const auto bound = rows.fixedRowBackingBytesUpperBound(64);
  EXPECT_TRUE(pool->tryWithAllocationCredit(bound, [&] { rows.reserveRows(64); }));
  EXPECT_EQ(rows.numRows(), 0);
  const auto resident = pool->usedBytes();
  EXPECT_TRUE(pool->tryWithAllocationCredit(0, [&] {
    for (int i = 0; i < 64; ++i) (void)rows.newRow();
  }));
  EXPECT_EQ(rows.numRows(), 64);
  EXPECT_EQ(pool->usedBytes(), resident);
  rows.clear();
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, nestedCreditsAndUnsignedLimit) {
  auto shared = budget(std::numeric_limits<uint64_t>::max());
  auto a = root_->addLeafChild("a"), b = root_->addLeafChild("b");
  a->setAllocationBudget(shared); b->setAllocationBudget(shared);
  EXPECT_TRUE(a->tryWithAllocationCredit(256, [&] {
    auto* p = a->allocate(64);
    auto cleanup = folly::makeGuard([&] { if (p) a->free(p, 64); });
    EXPECT_THROW(b->tryWithAllocationCredit(128, [&] {
      auto* q = b->allocate(128); b->free(q, 128);
      throw std::runtime_error("inner scope");
    }), std::runtime_error);
    EXPECT_EQ(shared->currentResourceUsage(), 256);
    EXPECT_TRUE(a->tryWithAllocationCredit(64, [&] { a->free(p, 64); p = nullptr; }));
    EXPECT_EQ(shared->currentResourceUsage(), 192);
    auto* q = a->allocate(192); a->free(q, 192);
  }));
  EXPECT_EQ(shared->currentResourceUsage(), 0);
  EXPECT_TRUE(a->tryWithAllocationCredit(std::numeric_limits<uint64_t>::max(), [&] {
    auto* q = a->allocate(64); a->free(q, 64);
    EXPECT_EQ(shared->currentResourceUsage(), std::numeric_limits<uint64_t>::max());
  }));
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}


TEST_F(MemoryPoolAdmissionTest, byteBackendFailuresKeepOldAllocationAndRefundNewCharge) {
  const bool saved = FLAGS_velox_enable_inplace_realloc;
  auto restore = folly::makeGuard([&] {
    FLAGS_velox_enable_inplace_realloc = saved;
    manager_.allocator()->testingClearFailureInjection();
  });
  auto shared = budget(1024);
  for (bool inplace : {false, true}) {
    manager_.allocator()->testingClearFailureInjection();
    FLAGS_velox_enable_inplace_realloc = inplace;
    auto pool = root_->addLeafChild(inplace ? "inplace" : "legacy");
    pool->setAllocationBudget(shared);
    void* p = pool->allocate(64); int64_t heldSize = 64;
    auto cleanup = folly::makeGuard([&] { pool->free(p, heldSize); });
    std::memset(p, 0x4e, 64);
    manager_.allocator()->testingSetFailureInjection(MemoryAllocator::InjectedFailure::kAllocate, true);
    bool failed = false;
    try { p = pool->reallocate(p, heldSize, 128); heldSize = 128; }
    catch (const std::exception&) { failed = true; }
    EXPECT_TRUE(failed) << "real backend failure injection must cover both realloc branches";
    EXPECT_EQ(shared->currentResourceUsage(), 64);
    EXPECT_EQ(pool->usedBytes(), 64);
    EXPECT_EQ(manager_.allocator()->totalUsedBytes(), 64);
    for (int i = 0; i < 64; ++i) EXPECT_EQ(static_cast<unsigned char*>(p)[i], 0x4e);
    for (int operation = 0; operation < 3; ++operation) {
      void* unexpected = nullptr; failed = false;
      try {
        if (operation == 0) unexpected = pool->allocate(128);
        if (operation == 1) unexpected = pool->allocateZeroFilled(4, 64);
        if (operation == 2) unexpected = pool->allocateAligned(129, 256);
      } catch (const std::exception&) { failed = true; }
      if (unexpected) {
        if (operation == 2) pool->freeAligned(unexpected, 129, 256);
        else pool->free(unexpected, operation == 0 ? 128 : 256);
      }
      EXPECT_TRUE(failed);
      EXPECT_EQ(shared->currentResourceUsage(), 64);
      EXPECT_EQ(pool->usedBytes(), 64);
      EXPECT_EQ(manager_.allocator()->totalUsedBytes(), 64);
    }
  }
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}


TEST_F(MemoryPoolAdmissionTest, fixedRunBoundAdmitsSmallRequestsWithoutLargeMapCredit) {
  struct Case { uint64_t count; uint64_t bytes; int32_t alignment; };
  const Case cases[] = {{1, 32, 8}, {2048, 32, 8}, {2049, 32, 8},
      {8192, 32, 8}, {3, 60000, 4096}, {3, 65536, 4096}, {3, 200000, 8}};
  for (const auto& c : cases) {
    SCOPED_TRACE(testing::Message() << "count=" << c.count << " bytes=" << c.bytes);
    const auto pages = std::max<MachinePageCount>(AllocationPool::kMinPages,
        AllocationTraits::numPages(c.bytes + c.alignment - 1));
    const auto runBytes = manager_.allocator()->allocationBytes(pages, pages);
    const auto stride = (c.bytes + c.alignment - 1) & ~(uint64_t(c.alignment) - 1);
    const auto entries = 1 + (runBytes - c.bytes - (c.alignment - 1)) / stride;
    const auto expected = (1 + (c.count - 1) / entries) * runBytes;
    auto shared = budget(expected);
    auto pool = root_->addLeafChild("small-runs");
    pool->setAllocationBudget(shared);
    AllocationPool runs(pool.get());
    runs.setHugePageThreshold(1 << 20);
    const auto bound = runs.fixedAllocationBytesUpperBound(c.count, c.bytes, c.alignment);
    EXPECT_EQ(bound, expected);
    EXPECT_NO_THROW(EXPECT_TRUE(pool->tryWithAllocationCredit(bound, [&] {
      for (uint64_t i = 0; i < c.count; ++i) {
        auto* value = runs.allocateFixed(c.bytes, c.alignment);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(value) % c.alignment, 0);
      }
    })));
    EXPECT_LE(pool->usedBytes(), expected);
    EXPECT_EQ(shared->currentResourceUsage(), pool->usedBytes());
    runs.clear();
    EXPECT_EQ(shared->currentResourceUsage(), 0);
  }
}

TEST_F(MemoryPoolAdmissionTest, fixedRunBoundIncludesSmallToHugeTransition) {
  auto shared = budget(64ULL << 20);
  auto pool = root_->addLeafChild("runs"); pool->setAllocationBudget(shared);
  AllocationPool runs(pool.get());
  constexpr uint64_t count = 32768, bytes = 32;
  const auto bound = runs.fixedAllocationBytesUpperBound(count, bytes, 8);
  for (uint64_t i = 0; i < count; ++i) (void)runs.allocateFixed(bytes, 8);
  const auto actual = pool->usedBytes();
  EXPECT_GE(bound, actual);
  EXPECT_EQ(shared->currentResourceUsage(), actual);
  std::cout << "[fixed-run-bound] count=" << count << " bound=" << bound
            << " actual=" << actual << std::endl;
  runs.clear();
  EXPECT_EQ(shared->currentResourceUsage(), 0);
  EXPECT_NO_THROW(EXPECT_TRUE(pool->tryWithAllocationCredit(bound, [&] {
    for (uint64_t i = 0; i < count; ++i) (void)runs.allocateFixed(bytes, 8);
  })));
}


TEST_F(MemoryPoolAdmissionTest, fixedRunBoundCoversExistingTailAlignmentAndLargeRequests) {
  struct Case { uint64_t count; uint64_t bytes; int32_t alignment; };
  const Case cases[] = {{128, 1, 1}, {8193, 24, 8}, {32768, 32, 8},
      {65537, 64, 16}, {8193, 4096, 4096}, {3, (3ULL << 20) + 17, 8},
      {2, (40ULL << 20) + 17, 8}, {1500000, 37, 16}};
  auto shared = budget(256ULL << 20);
  auto pool = root_->addLeafChild("runs"); pool->setAllocationBudget(shared);
  for (const auto& c : cases) for (bool prefill : {false, true}) {
    SCOPED_TRACE(testing::Message() << "count=" << c.count << " bytes=" << c.bytes
        << " alignment=" << c.alignment << " prefill=" << prefill);
    AllocationPool runs(pool.get());
    if (prefill) (void)runs.allocateFixed(c.bytes, c.alignment);
    const auto before = pool->usedBytes();
    const auto bound = runs.fixedAllocationBytesUpperBound(c.count, c.bytes, c.alignment);
    EXPECT_NO_THROW(EXPECT_TRUE(pool->tryWithAllocationCredit(bound, [&] {
      for (uint64_t i = 0; i < c.count; ++i) (void)runs.allocateFixed(c.bytes, c.alignment);
    })));
    EXPECT_LE(pool->usedBytes() - before, bound);
    EXPECT_EQ(shared->currentResourceUsage(), pool->usedBytes());
    std::cout << "[fixed-run-bound-matrix] bytes=" << c.bytes << " count=" << c.count
        << " prefill=" << prefill << " bound=" << bound
        << " growth=" << pool->usedBytes() - before << std::endl;
    runs.clear();
    EXPECT_EQ(shared->currentResourceUsage(), 0);
  }
}

TEST_F(MemoryPoolAdmissionTest, fixedRunBoundCoversThresholdOvershootAndExistingTail) {
  struct Case { uint64_t count; uint64_t bytes; int32_t alignment; };
  const Case cases[] = {{8193, 37, 16}, {17, 1, 4096}, {5, 65536, 4096}};
  auto shared = budget(64ULL << 20);
  auto pool = root_->addLeafChild("threshold-runs");
  pool->setAllocationBudget(shared);
  for (int64_t threshold : {1, 65535, 65536, 65537, 262143, 262144, 262145}) {
    for (const auto& c : cases) for (bool prefill : {false, true}) {
      SCOPED_TRACE(testing::Message() << "threshold=" << threshold
          << " bytes=" << c.bytes << " alignment=" << c.alignment
          << " prefill=" << prefill);
      AllocationPool runs(pool.get());
      runs.setHugePageThreshold(threshold);
      if (prefill) (void)runs.allocateFixed(1);
      const auto before = pool->usedBytes();
      const auto bound = runs.fixedAllocationBytesUpperBound(c.count, c.bytes, c.alignment);
      EXPECT_TRUE(pool->tryWithAllocationCredit(bound, [&] {
        for (uint64_t i = 0; i < c.count; ++i) (void)runs.allocateFixed(c.bytes, c.alignment);
      }));
      EXPECT_LE(pool->usedBytes() - before, bound);
      EXPECT_EQ(shared->currentResourceUsage(), pool->usedBytes());
      runs.clear();
      EXPECT_EQ(shared->currentResourceUsage(), 0);
    }
  }
}

TEST_F(MemoryPoolAdmissionTest, fixedRunBoundCoversGrowingLargeRunBelowThreshold) {
  auto shared = budget(64ULL << 20);
  auto pool = root_->addLeafChild("growing-large-run");
  pool->setAllocationBudget(shared);
  AllocationPool runs(pool.get());
  runs.setHugePageThreshold(64LL << 20);
  (void)runs.allocateFixed((2ULL << 20) + 1);
  (void)runs.allocateFixed((2ULL << 20) - 1);
  const auto before = pool->usedBytes();
  const auto bound = runs.fixedAllocationBytesUpperBound(1, 32, 8);
  EXPECT_GE(bound, AllocationTraits::kHugePageSize);
  EXPECT_NO_THROW(EXPECT_TRUE(pool->tryWithAllocationCredit(bound, [&] {
    (void)runs.allocateFixed(32, 8);
  })));
  EXPECT_LE(pool->usedBytes() - before, bound);
  EXPECT_EQ(shared->currentResourceUsage(), pool->usedBytes());
  runs.clear();
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, fixedRunBoundRejectsOverflowAndUnsupportedAlignmentBeforeAllocation) {
  auto pool = root_->addLeafChild("runs");
  AllocationPool runs(pool.get());
  EXPECT_EQ(runs.fixedAllocationBytesUpperBound(0, 0), 0);
  EXPECT_THROW(runs.fixedAllocationBytesUpperBound(std::numeric_limits<uint64_t>::max(), 64, 8), std::exception);
  EXPECT_THROW(runs.fixedAllocationBytesUpperBound(1, 0), std::exception);
  EXPECT_THROW(runs.fixedAllocationBytesUpperBound(1, 32, 3), std::exception);
  EXPECT_THROW(runs.fixedAllocationBytesUpperBound(1, 32, 8192), std::exception);
  EXPECT_EQ(pool->usedBytes(), 0);
}

TEST_F(MemoryPoolAdmissionTest, alignedBufferRequestMatchesActualAllocation) {
  auto pool = root_->addLeafChild("buffer-demand");
  auto shared = budget(8 * 1024 * 1024);
  pool->setAllocationBudget(shared);
  for (bool exact : {false, true}) {
    for (size_t count : {size_t(0), size_t(1), size_t(7), size_t(8),
                         size_t(9), size_t(1023), size_t(32769)}) {
      const auto request = AlignedBuffer::allocationSize<int64_t>(
          count, pool.get(), exact);
      const auto charge = bits::roundUp(request, pool->alignment());
      BufferPtr held;
      EXPECT_TRUE(pool->tryWithAllocationCredit(charge, [&] {
        held = AlignedBuffer::allocate<int64_t>(
            count, pool.get(), std::nullopt, exact);
      }));
      ASSERT_NE(held, nullptr);
      EXPECT_EQ(held->size(), count * sizeof(int64_t));
      EXPECT_EQ(pool->usedBytes(), charge);
      EXPECT_EQ(shared->currentResourceUsage(), charge);
      held.reset();
      EXPECT_EQ(shared->currentResourceUsage(), 0);
    }
  }
  EXPECT_ANY_THROW(AlignedBuffer::allocationSize<int64_t>(
      std::numeric_limits<size_t>::max(), pool.get()));
  EXPECT_ANY_THROW(AlignedBuffer::allocationSize<char>(
      std::numeric_limits<size_t>::max(), pool.get(), true));
  EXPECT_EQ(pool->usedBytes(), 0);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, bufferGrowthKeepsHeldOutputCharged) {
  auto pool = root_->addLeafChild("held-output");
  auto shared = budget(1024 * 1024);
  pool->setAllocationBudget(shared);
  auto output = AlignedBuffer::allocate<int64_t>(8, pool.get(), int64_t(73));
  auto consumer = output;
  const auto oldCharge = pool->usedBytes();
  const auto demand = bits::roundUp(
      AlignedBuffer::allocationSize<int64_t>(4097, pool.get()),
      pool->alignment());
  EXPECT_TRUE(pool->tryWithAllocationCredit(demand, [&] {
    AlignedBuffer::reallocate<int64_t>(&output, 4097);
  }));
  EXPECT_NE(output.get(), consumer.get());
  EXPECT_EQ(pool->usedBytes(), oldCharge + demand);
  EXPECT_EQ(shared->currentResourceUsage(), oldCharge + demand);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(consumer->as<int64_t>()[i], 73);
    EXPECT_EQ(output->as<int64_t>()[i], 73);
  }
  output.reset();
  EXPECT_EQ(shared->currentResourceUsage(), oldCharge);
  consumer.reset();
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, bitPackedBufferDemandAndGrowth) {
  auto pool = root_->addLeafChild("bit-packed-output");
  auto shared = budget(1024 * 1024);
  pool->setAllocationBudget(shared);
  for (bool exact : {false, true}) {
    for (size_t count : {size_t(0), size_t(1), size_t(7), size_t(8),
                         size_t(9), size_t(32769)}) {
      const auto request = AlignedBuffer::allocationSize<bool>(
          count, pool.get(), exact);
      auto output = AlignedBuffer::allocate<bool>(count, pool.get(), true, exact);
      EXPECT_EQ(request, AlignedBuffer::allocationSize<char>(
          bits::nbytes(count), pool.get(), exact));
      EXPECT_EQ(pool->usedBytes(), bits::roundUp(request, pool->alignment()));
      EXPECT_EQ(output->size(), bits::nbytes(count));
    }
  }
  for (bool held : {false, true}) {
    auto output = AlignedBuffer::allocate<bool>(9, pool.get(), true);
    BufferPtr consumer = held ? output : nullptr;
    const auto oldCharge = pool->usedBytes();
    const auto demand = bits::roundUp(
        AlignedBuffer::allocationSize<bool>(32769, pool.get()),
        pool->alignment());
    EXPECT_TRUE(pool->tryWithAllocationCredit(demand, [&] {
      AlignedBuffer::reallocate<bool>(&output, 32769);
    }));
    EXPECT_EQ(pool->usedBytes(), demand + (held ? oldCharge : 0));
    EXPECT_EQ(shared->currentResourceUsage(), pool->usedBytes());
    for (size_t i = 0; i < 9; ++i) {
      EXPECT_TRUE(bits::isBitSet(output->as<uint64_t>(), i));
      if (held) EXPECT_TRUE(bits::isBitSet(consumer->as<uint64_t>(), i));
    }
  }
  EXPECT_EQ(pool->usedBytes(), 0);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, bitPackedCountOverflowDoesNotMutateBuffer) {
  auto pool = root_->addLeafChild("bit-count-overflow");
  EXPECT_ANY_THROW(AlignedBuffer::allocate<bool>(
      std::numeric_limits<size_t>::max(), pool.get()));
  EXPECT_ANY_THROW(AlignedBuffer::allocationSize<bool>(
      std::numeric_limits<size_t>::max(), pool.get()));
  auto output = AlignedBuffer::allocate<bool>(9, pool.get(), true);
  const auto oldSize = output->size();
  auto* old = output.get();
  EXPECT_ANY_THROW(AlignedBuffer::reallocate<bool>(
      &output, std::numeric_limits<size_t>::max()));
  EXPECT_EQ(output.get(), old);
  EXPECT_EQ(output->size(), oldSize);
  for (size_t i = 0; i < 9; ++i) {
    EXPECT_TRUE(bits::isBitSet(output->as<uint64_t>(), i));
  }
}

TEST_F(MemoryPoolAdmissionTest, newGroupIndicesUseOriginalLookupPool) {
  auto pool = root_->addLeafChild("new-group-indices");
  auto shared = budget(1024 * 1024);
  pool->setAllocationBudget(shared);
  std::vector<std::unique_ptr<exec::VectorHasher>> hashers;
  {
    exec::HashLookup lookup(hashers, pool.get());
    std::vector<vector_size_t> original;
    for (vector_size_t i = 0; i < 4096; ++i) {
      lookup.newGroups.push_back(i);
      original.push_back(i);
    }
    EXPECT_EQ(lookup.newGroups.size(), original.size());
    EXPECT_TRUE(std::equal(lookup.newGroups.begin(), lookup.newGroups.end(),
                           original.begin()));
    EXPECT_LE(pool->usedBytes(), bits::roundUp(
        original.capacity() * sizeof(vector_size_t), pool->alignment()))
        << "accounting must not require a larger indexing representation";
    ASSERT_GT(pool->usedBytes(), 0)
        << "new group indices must not bypass the operator pool";
    EXPECT_EQ(shared->currentResourceUsage(), pool->usedBytes());
    const auto backing = pool->usedBytes();
    lookup.newGroups.clear();
    EXPECT_EQ(pool->usedBytes(), backing)
        << "clearing logical entries does not free backing";
    lookup.newGroups.shrink_to_fit();
    EXPECT_EQ(pool->usedBytes(), 0);
    EXPECT_EQ(shared->currentResourceUsage(), 0);
  }
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, lookupHashScratchUsesOriginalPool) {
  auto pool = root_->addLeafChild("lookup-hash-scratch");
  auto shared = budget(1024 * 1024);
  pool->setAllocationBudget(shared);
  std::vector<std::unique_ptr<exec::VectorHasher>> hashers;
  {
    exec::HashLookup lookup(hashers, pool.get());
    lookup.scratchMemory.hashes.resize(4096);
    ASSERT_GT(pool->usedBytes(), 0)
        << "lookup hash scratch must not use the process allocator";
    EXPECT_EQ(shared->currentResourceUsage(), pool->usedBytes());
    lookup.scratchMemory.hashes.clear();
    EXPECT_GT(shared->currentResourceUsage(), 0);
  }
  EXPECT_EQ(pool->usedBytes(), 0);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, movedNewGroupWorkspaceKeepsAccountingDomain) {
  auto pool = root_->addLeafChild("new-group-conversion");
  auto shared = budget(1024 * 1024);
  pool->setAllocationBudget(shared);
  std::vector<std::unique_ptr<exec::VectorHasher>> hashers;
  {
    exec::HashLookup lookup(hashers, pool.get());
    for (vector_size_t i = 0; i < 4096; ++i) lookup.newGroups.push_back(i);
    const auto savedCharge = pool->usedBytes();
    auto saved = std::move(lookup.newGroups);
    lookup.newGroups.clear();
    for (vector_size_t i = 0; i < 1024; ++i) lookup.newGroups.push_back(i + 4096);
    EXPECT_GT(pool->usedBytes(), savedCharge)
        << "conversion workspace must retain its original pool after move";
    EXPECT_EQ(shared->currentResourceUsage(), pool->usedBytes());
    for (vector_size_t i = 0; i < 1024; ++i) {
      EXPECT_EQ(lookup.newGroups[i], i + 4096);
      EXPECT_EQ(saved[i], i);
    }
    lookup.newGroups = std::move(saved);
    EXPECT_EQ(lookup.newGroups.size(), 4096);
    EXPECT_EQ(pool->usedBytes(), savedCharge);
  }
  EXPECT_EQ(pool->usedBytes(), 0);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, hashStringStandaloneMetadataUsesOwner) {
  for (const size_t count : {1, 2, 5, 17, 64}) for (const size_t bytes : {1, 64, 65}) {
    auto work = root_->addLeafChild("string-standalone-metadata");
    auto shared = budget(1 << 20); work->setAllocationBudget(shared);
    HashStringAllocator allocator(work.get());
    std::vector<void*> blocks;
    for (size_t i = 0; i < count; ++i) blocks.push_back(allocator.allocateFromPool(bytes));
    EXPECT_GT(work->usedBytes(), blocks.size() * bytes);
    EXPECT_EQ(allocator.retainedSize(), work->usedBytes());
    EXPECT_EQ(shared->currentResourceUsage(), work->usedBytes());
    for (auto* block : blocks) allocator.freeToPool(block, bytes);
    EXPECT_EQ(allocator.retainedSize(), work->usedBytes());
    allocator.clear();
    EXPECT_EQ(allocator.retainedSize(), 0);
    EXPECT_EQ(work->usedBytes(), 0);
    EXPECT_EQ(shared->currentResourceUsage(), 0);
  }
}

TEST_F(MemoryPoolAdmissionTest, hashStringGrowthRefusalPreservesExistingBlocks) {
  for (const uint64_t limit : {64, 128, 192, 256, 320, 384, 512, 1024, 2048, 4096, 1 << 20}) {
    SCOPED_TRACE(limit);
    auto work = root_->addLeafChild("string-growth-refusal");
    auto shared = budget(limit); work->setAllocationBudget(shared);
    HashStringAllocator allocator(work.get());
    std::vector<void*> blocks;
    for (int i = 0; i < 64; ++i) {
      const auto before = work->usedBytes();
      const auto retained = allocator.retainedSize();
      try {
        auto* block = allocator.allocateFromPool(64);
        std::memset(block, i, 64); blocks.push_back(block);
      } catch (const std::exception&) {
        EXPECT_EQ(work->usedBytes(), before);
        EXPECT_EQ(allocator.retainedSize(), retained);
        EXPECT_EQ(shared->currentResourceUsage(), before);
        break;
      }
    }
    if (limit == (1 << 20)) EXPECT_EQ(blocks.size(), 64);
    else EXPECT_LT(blocks.size(), 64);
    for (size_t i = 0; i < blocks.size(); ++i) {
      for (size_t j = 0; j < 64; ++j)
        EXPECT_EQ(static_cast<unsigned char*>(blocks[i])[j], i);
      allocator.freeToPool(blocks[i], 64);
    }
    allocator.clear();
    EXPECT_EQ(work->usedBytes(), 0);
    EXPECT_EQ(shared->currentResourceUsage(), 0);
  }
  auto work = root_->addLeafChild("string-header-allocation");
  auto shared = budget(4 << 20); work->setAllocationBudget(shared);
  HashStringAllocator allocator(work.get());
  for (const int bytes : {HashStringAllocator::kMaxAlloc + 17, (1 << 20) + 13}) {
    auto* block = allocator.allocate(bytes);
    EXPECT_EQ(allocator.retainedSize(), work->usedBytes());
    allocator.free(block);
    EXPECT_EQ(allocator.retainedSize(), work->usedBytes());
  }
  allocator.clear();
  EXPECT_THROW(allocator.allocateFromPool(std::numeric_limits<uint64_t>::max()), std::exception);
  EXPECT_EQ(work->usedBytes(), 0);
  EXPECT_EQ(shared->currentResourceUsage(), 0);
}

TEST_F(MemoryPoolAdmissionTest, hashStringFrozenStandaloneRefusalIsAtomic) {
  // Isolate the legacy orphan: a rejected call returns no handle to clean up.
  EXPECT_EXIT({
    auto work = root_->addLeafChild("string-frozen-refusal");
    auto shared = budget(1 << 20); work->setAllocationBudget(shared);
    HashStringAllocator allocator(work.get());
    bool raised = false;
    try { allocator.freezeAndExecute([&] { allocator.allocateFromPool(64); }); }
    catch (const std::exception&) { raised = true; }
    const bool valid = raised && work->usedBytes() == 0 && shared->currentResourceUsage() == 0;
    std::_Exit(valid ? 0 : 1);
  }, testing::ExitedWithCode(0), "");
}

TEST_F(MemoryPoolAdmissionTest, hashStringRegistrationRefusalRefundsPayload) {
  EXPECT_EXIT({
    auto work = root_->addLeafChild("string-registration-refusal");
    // Payload alone fits; its mandatory registration must also be admitted.
    auto shared = budget(64); work->setAllocationBudget(shared);
    HashStringAllocator allocator(work.get());
    bool raised = false;
    try { allocator.allocateFromPool(64); }
    catch (const std::exception&) { raised = true; }
    const bool valid = raised && allocator.retainedSize() == 0 &&
        work->usedBytes() == 0 && shared->currentResourceUsage() == 0;
    std::_Exit(valid ? 0 : 1);
  }, testing::ExitedWithCode(0), "");
}

} // namespace facebook::velox::memory
