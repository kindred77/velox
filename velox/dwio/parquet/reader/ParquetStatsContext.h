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

#include <algorithm>
#include <cctype>

#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/parquet/reader/SemanticVersion.h"
#include "velox/dwio/parquet/thrift/ParquetThrift.h"

namespace facebook::velox::parquet {

struct ParquetStatsContext : dwio::common::StatsContext {
 public:
  ParquetStatsContext() = default;

  ParquetStatsContext(
      const std::optional<SemanticVersion>& version,
      const std::optional<std::string>& createdBy = {})
      : parquetVersion(version), createdBy_(createdBy) {}

  bool shouldIgnoreStatistics(thrift::Type type) const {
    if (!parquetVersion.has_value()) {
      // Unknown writer (e.g. files written by DuckDB). Numeric min/max
      // statistics are exact values and safe to use for row-group pruning;
      // string statistics may be truncated by some writers, so only trust
      // them for writers known to write untruncated values.
      if (type != thrift::Type::BYTE_ARRAY &&
          type != thrift::Type::FIXED_LEN_BYTE_ARRAY) {
        return false;
      }
      return !isKnownTrustedStringWriter();
    }
    return parquetVersion->shouldIgnoreStatistics(type);
  }

 private:
  // Case-insensitive match against writers known to write untruncated string
  // min/max statistics (e.g. DuckDB).
  bool isKnownTrustedStringWriter() const {
    if (!createdBy_.has_value()) {
      return false;
    }
    static constexpr const char* kTrustedWriters[] = {"duckdb"};
    std::string lower = *createdBy_;
    std::transform(
        lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
          return std::tolower(c);
        });
    for (const char* marker : kTrustedWriters) {
      if (lower.find(marker) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  std::optional<SemanticVersion> parquetVersion;
  std::optional<std::string> createdBy_;
};

} // namespace facebook::velox::parquet
