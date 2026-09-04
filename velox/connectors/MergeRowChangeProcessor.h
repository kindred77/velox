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
#include <string>
#include <vector>

#include "velox/vector/ComplexVector.h"

namespace facebook::velox::connector {

/// Connector-neutral page transform for the DELETE_ROW_AND_INSERT_ROW merge
/// convention. INSERT and DELETE produce one row, UPDATE produces DELETE then
/// INSERT, and DEFAULT produces no row. Row identity is opaque to this class.
///
/// The merge-row channel must be a ROW whose leading fields are target column
/// values and whose last two fields are operation TINYINT and case number.
/// Output is [target columns..., operation, row_id, insert_from_update].
class MergeRowChangeProcessor {
 public:
  static constexpr int8_t kInsertOperationNumber = 1;
  static constexpr int8_t kDeleteOperationNumber = 2;
  static constexpr int8_t kUpdateOperationNumber = 3;
  static constexpr int8_t kDefaultCaseOperationNumber = -1;

  MergeRowChangeProcessor(
      std::vector<TypePtr> targetColumnTypes,
      std::vector<std::string> outputColumnNames,
      TypePtr rowIdType,
      column_index_t targetRowIdChannel,
      column_index_t mergeRowChannel);

  const RowTypePtr& outputType() const {
    return outputType_;
  }

  RowVectorPtr transform(const RowVectorPtr& input, memory::MemoryPool* pool)
      const;

 private:
  struct OperationCounts {
    uint64_t numInsert{0};
    uint64_t numDelete{0};
    uint64_t numUpdate{0};
  };

  OperationCounts countOperations(
      const FlatVector<int8_t>& operationVector,
      vector_size_t numRows) const;

  std::vector<VectorPtr> allocateOutputChildren(
      vector_size_t numRows,
      memory::MemoryPool* pool) const;

  static RowTypePtr buildOutputType(
      const std::vector<TypePtr>& targetColumnTypes,
      const std::vector<std::string>& outputColumnNames,
      const TypePtr& rowIdType);

  const std::vector<TypePtr> targetColumnTypes_;
  const std::vector<std::string> outputColumnNames_;
  const TypePtr rowIdType_;
  const column_index_t targetRowIdChannel_;
  const column_index_t mergeRowChannel_;
  const size_t numTargetColumns_;
  const RowTypePtr outputType_;
};

} // namespace facebook::velox::connector
