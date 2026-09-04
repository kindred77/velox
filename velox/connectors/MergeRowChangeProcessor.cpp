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

#include "velox/connectors/MergeRowChangeProcessor.h"

#include <limits>

#include "velox/common/base/Exceptions.h"
#include "velox/type/Type.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::connector {
namespace {

const RowVector* asMergeRowVector(const VectorPtr& mergeRowChild) {
  VELOX_CHECK_NOT_NULL(mergeRowChild, "merge_row column is null.");
  const auto loaded = mergeRowChild->loadedVector();
  const auto* rowVector = loaded->as<RowVector>();
  VELOX_CHECK_NOT_NULL(
      rowVector,
      "merge_row column must be a flat RowVector, got encoding: {}",
      static_cast<int>(loaded->encoding()));
  return rowVector;
}

const FlatVector<int8_t>* asOperationVector(const RowVector& mergeRow) {
  const auto numMergeFields = mergeRow.childrenSize();
  VELOX_CHECK_GE(
      numMergeFields,
      2,
      "merge_row must have at least operation + case_number fields, got {}",
      numMergeFields);
  const auto& operationChild =
      mergeRow.childAt(static_cast<column_index_t>(numMergeFields - 2));
  VELOX_CHECK_NOT_NULL(operationChild, "operation field of merge_row is null.");
  const auto* operationVector =
      operationChild->loadedVector()->asFlatVector<int8_t>();
  VELOX_CHECK_NOT_NULL(
      operationVector,
      "operation field of merge_row must be a flat TINYINT vector.");
  return operationVector;
}

} // namespace

MergeRowChangeProcessor::MergeRowChangeProcessor(
    std::vector<TypePtr> targetColumnTypes,
    std::vector<std::string> outputColumnNames,
    TypePtr rowIdType,
    column_index_t targetRowIdChannel,
    column_index_t mergeRowChannel)
    : targetColumnTypes_(std::move(targetColumnTypes)),
      outputColumnNames_(std::move(outputColumnNames)),
      rowIdType_(std::move(rowIdType)),
      targetRowIdChannel_(targetRowIdChannel),
      mergeRowChannel_(mergeRowChannel),
      numTargetColumns_(targetColumnTypes_.size()),
      outputType_(
          buildOutputType(targetColumnTypes_, outputColumnNames_, rowIdType_)) {
  VELOX_CHECK_NOT_NULL(rowIdType_, "rowIdType is null.");
  VELOX_CHECK_EQ(
      outputColumnNames_.size(),
      targetColumnTypes_.size() + 3,
      "outputColumnNames size must be targetColumnTypes.size() + 3.");
  VELOX_CHECK_NE(
      targetRowIdChannel_,
      mergeRowChannel_,
      "targetRowIdChannel and mergeRowChannel must differ.");
}

RowTypePtr MergeRowChangeProcessor::buildOutputType(
    const std::vector<TypePtr>& targetColumnTypes,
    const std::vector<std::string>& outputColumnNames,
    const TypePtr& rowIdType) {
  VELOX_CHECK_NOT_NULL(rowIdType, "rowIdType is null.");
  VELOX_CHECK_EQ(
      outputColumnNames.size(),
      targetColumnTypes.size() + 3,
      "outputColumnNames size must be targetColumnTypes.size() + 3.");
  std::vector<TypePtr> types;
  types.reserve(targetColumnTypes.size() + 3);
  for (size_t i = 0; i < targetColumnTypes.size(); ++i) {
    VELOX_CHECK_NOT_NULL(
        targetColumnTypes[i], "targetColumnTypes[{}] is null.", i);
    types.push_back(targetColumnTypes[i]);
  }
  types.push_back(TINYINT());
  types.push_back(rowIdType);
  types.push_back(TINYINT());
  return ROW(outputColumnNames, std::move(types));
}

MergeRowChangeProcessor::OperationCounts
MergeRowChangeProcessor::countOperations(
    const FlatVector<int8_t>& operationVector,
    vector_size_t numRows) const {
  OperationCounts counts;
  for (vector_size_t i = 0; i < numRows; ++i) {
    VELOX_USER_CHECK(
        !operationVector.isNullAt(i),
        "merge_row operation field is null at position {}.",
        i);
    switch (operationVector.valueAt(i)) {
      case kDefaultCaseOperationNumber:
        break;
      case kInsertOperationNumber:
        ++counts.numInsert;
        break;
      case kDeleteOperationNumber:
        ++counts.numDelete;
        break;
      case kUpdateOperationNumber:
        ++counts.numUpdate;
        break;
      default:
        VELOX_USER_FAIL(
            "Unknown merge operation byte: {}",
            static_cast<int>(operationVector.valueAt(i)));
    }
  }
  return counts;
}

std::vector<VectorPtr> MergeRowChangeProcessor::allocateOutputChildren(
    vector_size_t numRows,
    memory::MemoryPool* pool) const {
  std::vector<VectorPtr> outputChildren;
  outputChildren.reserve(numTargetColumns_ + 3);
  for (const auto& type : targetColumnTypes_) {
    outputChildren.push_back(BaseVector::create(type, numRows, pool));
  }
  outputChildren.push_back(BaseVector::create(TINYINT(), numRows, pool));
  outputChildren.push_back(BaseVector::create(rowIdType_, numRows, pool));
  outputChildren.push_back(BaseVector::create(TINYINT(), numRows, pool));
  return outputChildren;
}

RowVectorPtr MergeRowChangeProcessor::transform(
    const RowVectorPtr& input,
    memory::MemoryPool* pool) const {
  VELOX_CHECK_NOT_NULL(input, "Input row vector is null.");
  VELOX_CHECK_NOT_NULL(pool, "Memory pool is null.");

  const vector_size_t inputPositions = input->size();
  if (inputPositions == 0) {
    return BaseVector::create<RowVector>(outputType_, 0, pool);
  }

  VELOX_CHECK_LT(
      targetRowIdChannel_,
      input->childrenSize(),
      "targetRowIdChannel out of range.");
  VELOX_CHECK_LT(
      mergeRowChannel_, input->childrenSize(), "mergeRowChannel out of range.");

  const auto& rowIdInput = input->childAt(targetRowIdChannel_);
  VELOX_CHECK_NOT_NULL(rowIdInput, "targetRowId column is null.");
  VELOX_CHECK_EQ(
      rowIdInput->size(), inputPositions, "targetRowId row count mismatch.");
  VELOX_CHECK(
      rowIdInput->type()->equivalent(*rowIdType_),
      "targetRowId type does not match rowIdType.");

  auto mergeRowChild = input->childAt(mergeRowChannel_);
  BaseVector::flattenVector(mergeRowChild);
  const auto* mergeRow = asMergeRowVector(mergeRowChild);
  VELOX_CHECK_EQ(
      mergeRow->size(), inputPositions, "merge_row row count mismatch.");
  VELOX_CHECK_GE(
      mergeRow->childrenSize(),
      numTargetColumns_ + 2,
      "merge_row must have at least {} fields, got {}",
      numTargetColumns_ + 2,
      mergeRow->childrenSize());
  for (size_t i = 0; i < numTargetColumns_; ++i) {
    VELOX_CHECK(
        mergeRow->childAt(static_cast<column_index_t>(i))
            ->type()
            ->equivalent(*targetColumnTypes_[i]),
        "merge_row target column {} type mismatch.",
        i);
  }

  const auto* operationVector = asOperationVector(*mergeRow);
  const OperationCounts counts =
      countOperations(*operationVector, inputPositions);
  const uint64_t totalOutput64 =
      counts.numInsert + counts.numDelete + 2 * counts.numUpdate;
  VELOX_CHECK_LE(
      totalOutput64,
      static_cast<uint64_t>(std::numeric_limits<vector_size_t>::max()),
      "Merge output row count exceeds vector_size_t.");
  const auto totalOutput = static_cast<vector_size_t>(totalOutput64);

  auto outputChildren = allocateOutputChildren(totalOutput, pool);
  const size_t operationOutIndex = numTargetColumns_;
  const size_t rowIdOutIndex = numTargetColumns_ + 1;
  const size_t insertFromUpdateOutIndex = numTargetColumns_ + 2;
  auto* operationOut =
      outputChildren[operationOutIndex]->asFlatVector<int8_t>();
  auto& rowIdOut = outputChildren[rowIdOutIndex];
  auto* insertFromUpdateOut =
      outputChildren[insertFromUpdateOutIndex]->asFlatVector<int8_t>();
  VELOX_CHECK_NOT_NULL(operationOut, "operation output is not flat.");
  VELOX_CHECK_NOT_NULL(
      insertFromUpdateOut, "insert_from_update output is not flat.");

  vector_size_t outIdx = 0;
  auto emitDeleteRow = [&](vector_size_t inputIdx) {
    for (size_t column = 0; column < numTargetColumns_; ++column) {
      outputChildren[column]->setNull(outIdx, true);
    }
    operationOut->set(outIdx, kDeleteOperationNumber);
    rowIdOut->copy(rowIdInput.get(), outIdx, inputIdx, 1);
    insertFromUpdateOut->set(outIdx, 0);
    ++outIdx;
  };
  auto emitInsertRow = [&](vector_size_t inputIdx, bool fromUpdate) {
    for (size_t column = 0; column < numTargetColumns_; ++column) {
      outputChildren[column]->copy(
          mergeRow->childAt(static_cast<column_index_t>(column)).get(),
          outIdx,
          inputIdx,
          1);
    }
    operationOut->set(outIdx, kInsertOperationNumber);
    rowIdOut->setNull(outIdx, true);
    insertFromUpdateOut->set(outIdx, fromUpdate ? 1 : 0);
    ++outIdx;
  };

  for (vector_size_t i = 0; i < inputPositions; ++i) {
    const int8_t operation = operationVector->valueAt(i);
    if (operation == kDeleteOperationNumber ||
        operation == kUpdateOperationNumber) {
      emitDeleteRow(i);
    }
    if (operation == kInsertOperationNumber ||
        operation == kUpdateOperationNumber) {
      emitInsertRow(i, operation == kUpdateOperationNumber);
    }
  }

  VELOX_CHECK_EQ(outIdx, totalOutput, "Merge output row count mismatch.");
  return std::make_shared<RowVector>(
      pool,
      outputType_,
      /*nulls=*/nullptr,
      totalOutput,
      std::move(outputChildren));
}

} // namespace facebook::velox::connector
