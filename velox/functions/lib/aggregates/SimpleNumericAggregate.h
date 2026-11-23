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

#include "velox/exec/Aggregate.h"
#include "velox/vector/AggregationHook.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/LazyVector.h"
#include "velox/vector/VectorTypeUtils.h"

#include <vector>
#include <optional>

namespace facebook::velox::functions::aggregate {

template <typename TInput, typename TAccumulator, typename TResult>
class SimpleNumericAggregate : public exec::Aggregate {
 protected:
  template <typename TOutput = TAccumulator>
  bool prepareFixedNumericInput(
      bool rawInput, const SelectivityVector& rows,
      const std::vector<VectorPtr>& args, const VectorPtr& output) {
    return rawInput && args.size() == 1 && isFixedSize() &&
        isDirectlyDecodedScalar(*args[0]) && (output->size() > 0 || !rows.hasSelections()) &&
        output->template as<FlatVector<TOutput>>() != nullptr &&
        hasFixedWidthVectorBacking(*output);
  }

  void setAllocatorInternal(HashStringAllocator* allocator) override {
    auto* pool = allocator ? allocator->pool() : nullptr;
    if (pushdownScratch_->indices.get_allocator().pool != pool) {
      pushdownScratch_.emplace(pool);
    }
    exec::Aggregate::setAllocatorInternal(allocator);
  }

  explicit SimpleNumericAggregate(TypePtr resultType) : Aggregate(resultType) {}

 public:
  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

 protected:
  template <typename T>
  static constexpr bool kMayPushdown = !std::is_same_v<T, int128_t> &&
      !std::is_same_v<T, Timestamp> && !std::is_same_v<T, UnknownValue>;

  // TData is either TAccumulator or TResult, which in most cases are the same,
  // but for sum(real) can differ.
  template <typename TData = TResult, typename ExtractOneValue>
  void doExtractValues(
      char** groups,
      int32_t numGroups,
      VectorPtr* result,
      ExtractOneValue extractOneValue) {
    VELOX_CHECK_EQ((*result)->encoding(), VectorEncoding::Simple::FLAT);
    auto vector = (*result)->as<FlatVector<TData>>();
    VELOX_CHECK(
        vector,
        "Unexpected type of the result vector: {}",
        (*result)->type()->toString());
    VELOX_CHECK_EQ(vector->elementSize(), sizeof(TData));
    vector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(vector);
    if constexpr (std::is_same_v<TData, bool>) {
      uint64_t* rawValues = vector->template mutableRawValues<uint64_t>();
      for (int32_t i = 0; i < numGroups; ++i) {
        char* group = groups[i];
        if (isNull(group)) {
          vector->setNull(i, true);
        } else {
          clearNull(rawNulls, i);
          bits::setBit(rawValues, i, extractOneValue(group));
        }
      }
    } else {
      TData* rawValues = vector->mutableRawValues();
      for (int32_t i = 0; i < numGroups; ++i) {
        char* group = groups[i];
        if (isNull(group)) {
          vector->setNull(i, true);
        } else {
          clearNull(rawNulls, i);
          rawValues[i] = extractOneValue(group);
        }
      }
    }
  }

  // TData is used to store the updated group states. It can be either
  // TAccumulator or TResult, which in most cases are the same, but for
  // sum(real) can differ. TValue is used to decode the update input 'args'.
  // It can be either TAccumulator or TInput, which is most cases are the same
  // but for sum(real) can differ.
  template <
      bool tableHasNulls,
      typename TData = TResult,
      typename TValue = TInput,
      typename UpdateSingleValue>
  void updateGroups(
      char** groups,
      const SelectivityVector& rows,
      const VectorPtr& arg,
      UpdateSingleValue updateSingleValue,
      bool mayPushdown) {
    // Types without an aggregation hook must always materialize lazy input.
    // Callers can request pushdown for the operator as a whole even when a
    // particular aggregate type (e.g. Timestamp or int128_t) does not support
    // it.
    // Copied decoding scratch belongs to the accumulator's resource; borrowed
    // input buffers keep their source ownership. Standalone callers may omit it.
    DecodedVector decoded(allocator_ ? allocator_->pool() : nullptr);
    decoded.decode(*arg, rows, !mayPushdown || !kMayPushdown<TData>);
    if constexpr (kMayPushdown<TData>) {
      if (mayPushdown &&
          decoded.base()->encoding() == VectorEncoding::Simple::LAZY &&
          !arg->type()->isDecimal()) {
        auto* lazy = decoded.base()->asChecked<const LazyVector>();
        if (lazy->supportsHook()) {
          auto prepared = preparePushdownInput(decoded, rows, groups);
          velox::aggregate::SimpleCallableHook<TData, UpdateSingleValue> hook(
              exec::Aggregate::offset_,
              exec::Aggregate::nullByte_,
              exec::Aggregate::nullMask_,
              prepared.groups,
              &this->exec::Aggregate::numNulls_,
              updateSingleValue);
          lazy->load(prepared.rows, &hook);
          return;
        }
        decoded.decode(*arg, rows);
      }
    }

    if (decoded.isConstantMapping()) {
      if (!decoded.isNullAt(0)) {
        auto value = decoded.valueAt<TValue>(0);
        rows.applyToSelected([&](vector_size_t i) {
          updateNonNullValue<tableHasNulls, TData>(
              groups[i], TData(value), updateSingleValue);
        });
      }
    } else if (decoded.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decoded.isNullAt(i)) {
          return;
        }
        updateNonNullValue<tableHasNulls, TData>(
            groups[i], TData(decoded.valueAt<TValue>(i)), updateSingleValue);
      });
    } else if (decoded.isIdentityMapping() && !std::is_same_v<TValue, bool>) {
      auto data = decoded.data<TValue>();
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<tableHasNulls, TData>(
            groups[i], TData(data[i]), updateSingleValue);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<tableHasNulls, TData>(
            groups[i], TData(decoded.valueAt<TValue>(i)), updateSingleValue);
      });
    }
  }

  // TData is used to store the updated group state. It can be either
  // TAccumulator or TResult, which in most cases are the same, but for
  // sum(real) can differ. TValue is used to decode the update input 'args'.
  // It can be either TAccumulator or TInput, which is most cases are the same
  // but for sum(real) can differ.
  template <
      typename TData = TResult,
      typename TValue = TInput,
      typename UpdateSingle,
      typename UpdateDuplicate>
  void updateOneGroup(
      char* group,
      const SelectivityVector& rows,
      const VectorPtr& arg,
      UpdateSingle updateSingleValue,
      UpdateDuplicate updateDuplicateValues,
      bool mayPushdown,
      TData initialValue) {
    // See updateGroups: unsupported hook types cannot leave a LazyVector
    // undecoded even if the operator enables aggregate pushdown.
    DecodedVector decoded(allocator_ ? allocator_->pool() : nullptr);
    decoded.decode(*arg, rows, !mayPushdown || !kMayPushdown<TData>);
    if constexpr (kMayPushdown<TData>) {
      if (mayPushdown &&
          decoded.base()->encoding() == VectorEncoding::Simple::LAZY &&
          !arg->type()->isDecimal()) {
        auto* lazy = decoded.base()->asChecked<const LazyVector>();
        if (lazy->supportsHook()) {
          // Single-group (scalar) aggregation: every row maps to the one
          // group.  Mirroring updateGroups, let the LazyVector accumulate
          // directly during decode instead of materializing the column; a
          // per-row group array pointing at the single group is safe for any
          // row numbering (yihudb scalar-aggregate pushdown, 2026-08-24).
          auto prepared = preparePushdownInput(decoded, rows, nullptr);
          GroupPointers groups(prepared.rows.size(), group,
              pushdownScratch_->groups.get_allocator());
          velox::aggregate::SimpleCallableHook<TData, UpdateSingle> hook(
              exec::Aggregate::offset_,
              exec::Aggregate::nullByte_,
              exec::Aggregate::nullMask_,
              groups.data(),
              &this->exec::Aggregate::numNulls_,
              updateSingleValue);
          lazy->load(prepared.rows, &hook);
          return;
        }
        decoded.decode(*arg, rows);
      }
    }

    // Do row by row if not all rows are selected.
    if (decoded.isConstantMapping()) {
      if (!decoded.isNullAt(0)) {
        updateDuplicateValues(
            initialValue,
            TData(decoded.valueAt<TValue>(0)),
            rows.countSelected());
        updateNonNullValue<true, TData>(group, initialValue, updateSingleValue);
      }
    } else if (decoded.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decoded.isNullAt(i)) {
          return;
        }
        updateNonNullValue<true, TData>(
            group, TData(decoded.valueAt<TValue>(i)), updateSingleValue);
      });
    } else if (decoded.isIdentityMapping() && !std::is_same_v<TValue, bool>) {
      auto data = decoded.data<TValue>();
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<true, TData>(
            group, TData(data[i]), updateSingleValue);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<true, TData>(
            group, TData(decoded.valueAt<TValue>(i)), updateSingleValue);
      });
    }
  }

  template <typename THook>
  void
  pushdown(char** groups, const SelectivityVector& rows, const VectorPtr& arg) {
    DecodedVector decoded(allocator_ ? allocator_->pool() : nullptr);
    decoded.decode(*arg, rows, false);
    auto prepared = preparePushdownInput(decoded, rows, groups);
    THook hook(
        exec::Aggregate::offset_,
        exec::Aggregate::nullByte_,
        exec::Aggregate::nullMask_,
        prepared.groups,
        &this->exec::Aggregate::numNulls_);
    auto* lazy = decoded.base()->asChecked<const LazyVector>();
    VELOX_CHECK(lazy->supportsHook());
    lazy->load(prepared.rows, &hook);
  }

 private:
  using GroupPointers = std::vector<char*, memory::StlAllocator<char*>>;
  struct PushdownScratch {
    explicit PushdownScratch(memory::MemoryPool* pool)
        : indices(pool ? memory::StlAllocator<vector_size_t>(pool)
                       : memory::StlAllocator<vector_size_t>{}),
          groups(pool ? memory::StlAllocator<char*>(pool)
                      : memory::StlAllocator<char*>{}) {}
    std::vector<vector_size_t, memory::StlAllocator<vector_size_t>> indices;
    GroupPointers groups;
  };
  std::optional<PushdownScratch> pushdownScratch_{std::in_place, nullptr};

  struct PushdownInput {
    RowSet rows;
    char** groups;
  };

  PushdownInput preparePushdownInput(
      const DecodedVector& decoded, const SelectivityVector& rows,
      char** groups) {
    const auto* indices = decoded.indices();
    if (rows.isAllSelected()) {
      return {RowSet(indices, rows.end()), groups};
    }
    auto& scratch = *pushdownScratch_;
    const auto count = rows.countSelected();
    scratch.indices.resize(count);
    if (groups) scratch.groups.resize(count);
    vector_size_t position = 0;
    // ValueHook receives positions in the selected RowSet. Compact its group
    // pointers with the source indices, before loading can update SQL state.
    rows.applyToSelected([&](vector_size_t row) {
      scratch.indices[position] = indices[row];
      if (groups) scratch.groups[position] = groups[row];
      ++position;
    });
    return {RowSet(scratch.indices.data(), count),
            groups ? scratch.groups.data() : nullptr};
  }

  // TData is either TAccumulator or TResult, which in most cases are the same,
  // but for sum(real) can differ.
  template <
      bool tableHasNulls,
      typename TDataType = TAccumulator,
      typename Update>
  inline void
  updateNonNullValue(char* group, TDataType value, Update updateValue) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    updateValue(*exec::Aggregate::value<TDataType>(group), value);
  }
};

} // namespace facebook::velox::functions::aggregate
