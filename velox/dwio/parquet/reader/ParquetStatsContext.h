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

#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/parquet/reader/SemanticVersion.h"
#include "velox/dwio/parquet/thrift/ParquetThrift.h"

namespace facebook::velox::parquet {

struct ParquetStatsContext : dwio::common::StatsContext {
 public:
  ParquetStatsContext() = default;

  ParquetStatsContext(const std::optional<SemanticVersion>& version)
      : parquetVersion(version) {}

  bool shouldIgnoreStatistics(thrift::Type type) const {
    if (!parquetVersion.has_value()) {
      // Unknown writer (e.g. files written by DuckDB). Numeric min/max
      // statistics are exact values and safe to use for row-group pruning;
      // only string statistics may be truncated, so keep them conservative.
      return type == thrift::Type::BYTE_ARRAY ||
          type == thrift::Type::FIXED_LEN_BYTE_ARRAY;
    }
    return parquetVersion->shouldIgnoreStatistics(type);
  }

 private:
  std::optional<SemanticVersion> parquetVersion;
};

} // namespace facebook::velox::parquet
