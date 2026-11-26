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

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/ConstantVector.h"
#include "velox/vector/FlatMapVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/VectorUtil.h"
#include "velox/vector/SequenceVector.h"

namespace facebook::velox::test {
namespace {
class VectorRetentionContractTest : public testing::Test {
 protected:
  memory::MemoryManager manager_{memory::MemoryManager::Options{}};
  std::shared_ptr<memory::MemoryPool> root_{manager_.addRootPool("retention-test")};
  std::shared_ptr<memory::MemoryPool> pool_{root_->addLeafChild("vectors")};

  VectorPtr integers(vector_size_t size = 1) {
    auto vector = BaseVector::create(INTEGER(), size, pool_.get());
    for (vector_size_t i = 0; i < size; ++i) vector->asFlatVector<int32_t>()->set(i, i);
    return vector;
  }

  BufferPtr index(vector_size_t value) {
    auto buffer = AlignedBuffer::allocate<vector_size_t>(1, pool_.get());
    buffer->asMutable<vector_size_t>()[0] = value;
    return buffer;
  }
};

TEST_F(VectorRetentionContractTest, flatMapCountsAllDataBuffersAndStrings) {
  auto keys = integers();
  auto values = BaseVector::create(VARCHAR(), 1, pool_.get());
  const std::string text(131072, 'x');
  values->asFlatVector<StringView>()->set(0, StringView(text));
  auto nulls = allocateNulls(1, pool_.get());
  auto inMap = allocateNulls(1, pool_.get());
  auto map = std::make_shared<FlatMapVector>(pool_.get(), MAP(INTEGER(), VARCHAR()), nulls,
      1, keys, std::vector<VectorPtr>{values}, std::vector<BufferPtr>{inMap});
  const uint64_t expected = nulls->capacity() + keys->retainedSize() +
      values->retainedSize() + inMap->capacity();
  uint64_t strings = 0;
  EXPECT_EQ(map->retainedSize(strings), expected);
  EXPECT_EQ(strings, values->asFlatVector<StringView>()->stringBufferSize());
  EXPECT_EQ(retainedVectorBytes(*map), expected);
  EXPECT_GT(expected, 65536);
}

TEST_F(VectorRetentionContractTest, sequenceCountsLocalNullBuffer) {
  auto values = integers();
  auto lengths = index(1);
  auto vector = std::make_shared<SequenceVector<int32_t>>(pool_.get(), 1, values, lengths);
  auto nulls = allocateNulls(1, pool_.get());
  vector->setNulls(nulls);
  const auto expected = values->retainedSize() + lengths->capacity() + nulls->capacity();
  EXPECT_EQ(vector->retainedSize(), expected);
  EXPECT_EQ(retainedVectorBytes(*vector), expected);
}

TEST_F(VectorRetentionContractTest, sequenceLengthViewIsUnknown) {
  auto lengths = index(1);
  auto vector = std::make_shared<SequenceVector<int32_t>>(pool_.get(), 1, integers(),
      Buffer::slice<vector_size_t>(lengths, 0, 1, pool_.get()));
  EXPECT_FALSE(retainedVectorBytes(*vector).has_value());
}

TEST_F(VectorRetentionContractTest, nullConstantCountsOwnNullBuffer) {
  auto vector = BaseVector::createNullConstant(INTEGER(), 1, pool_.get());
  ASSERT_TRUE(vector->nulls());
  EXPECT_GE(vector->retainedSize(), vector->nulls()->capacity());
  auto bytes = retainedVectorBytes(*vector);
  ASSERT_TRUE(bytes.has_value());
  EXPECT_GE(*bytes, vector->nulls()->capacity());
}

TEST_F(VectorRetentionContractTest, zeroOwnedBuffersAreKnown) {
  RowVector vector(pool_.get(), ROW(std::vector<std::string>{}, std::vector<TypePtr>{}),
      nullptr, 1, std::vector<VectorPtr>{});
  EXPECT_EQ(retainedVectorBytes(vector), 0);
}

TEST_F(VectorRetentionContractTest, flatSliceCannotHideParentRetention) {
  auto parent = integers(262144);
  auto slice = parent->slice(0, 1);
  ASSERT_TRUE(slice->values()->isView());
  ASSERT_GT(parent->retainedSize(), 65536);
  EXPECT_FALSE(retainedVectorBytes(*slice).has_value());
  auto copy = BaseVector::create(INTEGER(), 1, pool_.get());
  copy->copy(slice.get(), 0, 0, 1);
  EXPECT_EQ(retainedVectorBytes(*copy), copy->retainedSize());
}

TEST_F(VectorRetentionContractTest, dictionaryIndexViewIsUnknown) {
  auto indices = index(0);
  auto view = Buffer::slice<vector_size_t>(indices, 0, 1, pool_.get());
  auto vector = BaseVector::wrapInDictionary(nullptr, view, 1, integers());
  EXPECT_FALSE(retainedVectorBytes(*vector).has_value());
}

TEST_F(VectorRetentionContractTest, nullBufferViewIsUnknown) {
  auto vector = integers();
  auto nulls = allocateNulls(128, pool_.get());
  vector->setNulls(Buffer::slice<bool>(nulls, 0, 1, pool_.get()));
  EXPECT_FALSE(retainedVectorBytes(*vector).has_value());
}

TEST_F(VectorRetentionContractTest, arrayBufferViewsAreUnknown) {
  for (bool offsetsView : {false, true}) {
    auto offsets = index(0), sizes = index(1);
    if (offsetsView) offsets = Buffer::slice<vector_size_t>(offsets, 0, 1, pool_.get());
    else sizes = Buffer::slice<vector_size_t>(sizes, 0, 1, pool_.get());
    auto vector = std::make_shared<ArrayVector>(pool_.get(), ARRAY(INTEGER()), nullptr,
        1, offsets, sizes, integers());
    EXPECT_FALSE(retainedVectorBytes(*vector).has_value());
  }
}

TEST_F(VectorRetentionContractTest, mapBufferViewsAreUnknown) {
  auto offsets = index(0), sizes = index(1);
  auto vector = std::make_shared<MapVector>(pool_.get(), MAP(INTEGER(), INTEGER()), nullptr,
      1, Buffer::slice<vector_size_t>(offsets, 0, 1, pool_.get()), sizes, integers(), integers());
  EXPECT_FALSE(retainedVectorBytes(*vector).has_value());
}

TEST_F(VectorRetentionContractTest, flatMapInMapViewIsUnknown) {
  auto inMap = allocateNulls(1, pool_.get());
  auto vector = std::make_shared<FlatMapVector>(pool_.get(), MAP(INTEGER(), INTEGER()), nullptr,
      1, integers(), std::vector<VectorPtr>{integers()},
      std::vector<BufferPtr>{Buffer::slice<bool>(inMap, 0, 1, pool_.get())});
  EXPECT_FALSE(retainedVectorBytes(*vector).has_value());
}

TEST_F(VectorRetentionContractTest, stringBackingViewIsUnknown) {
  auto vector = BaseVector::create(VARCHAR(), 1, pool_.get());
  const std::string text(512, 'x');
  auto flat = vector->asFlatVector<StringView>();
  flat->set(0, StringView(text));
  auto buffers = flat->stringBuffers();
  buffers[0] = Buffer::slice<char>(buffers[0], 0, buffers[0]->size(), pool_.get());
  flat->setStringBuffers(std::move(buffers));
  EXPECT_FALSE(retainedVectorBytes(*vector).has_value());
  auto constant = std::make_shared<ConstantVector<StringView>>(pool_.get(), 1, 0, vector);
  // ConstantVector copies scalar strings into owned storage during construction.
  ASSERT_TRUE(constant->getStringBuffer());
  ASSERT_FALSE(constant->getStringBuffer()->isView());
  EXPECT_EQ(retainedVectorBytes(*constant), constant->retainedSize());
}
} // namespace
} // namespace facebook::velox::test
