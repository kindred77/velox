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

#include "velox/connectors/MergeRowChangeProcessor.h"

namespace facebook::velox::connector::hive::iceberg {

/// Iceberg adapter for the connector-neutral merge action transform. Iceberg
/// owns the planner-facing name while the row expansion stays shared.
class IcebergMergeProcessor {
 public:
  static constexpr int8_t kInsertOperationNumber =
      connector::MergeRowChangeProcessor::kInsertOperationNumber;
  static constexpr int8_t kDeleteOperationNumber =
      connector::MergeRowChangeProcessor::kDeleteOperationNumber;
  static constexpr int8_t kUpdateOperationNumber =
      connector::MergeRowChangeProcessor::kUpdateOperationNumber;
  static constexpr int8_t kDefaultCaseOperationNumber =
      connector::MergeRowChangeProcessor::kDefaultCaseOperationNumber;

  IcebergMergeProcessor(
      std::vector<TypePtr> targetColumnTypes,
      std::vector<std::string> outputColumnNames,
      TypePtr rowIdType,
      column_index_t targetRowIdChannel,
      column_index_t mergeRowChannel);

  const RowTypePtr& outputType() const {
    return processor_.outputType();
  }

  RowVectorPtr transform(const RowVectorPtr& input, memory::MemoryPool* pool)
      const {
    return processor_.transform(input, pool);
  }

 private:
  connector::MergeRowChangeProcessor processor_;
};

} // namespace facebook::velox::connector::hive::iceberg
