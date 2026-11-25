/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once
#include "velox/type/Type.h"

namespace facebook::velox {

// Logical declarations and execution display limits are independent. A missing
// precision/scale denotes an unconstrained declaration, never an inferred zero.
struct LogicalTypeDescription {
  std::string name;
  std::optional<uint32_t> precision;
  std::optional<int32_t> scale;
  std::optional<uint64_t> maximumTextBytes;
};

using StringValueAppender =
    void (*)(const Type&, const StringView&, std::string&);

// Optional additional capability of an existing CustomTypeFactory. It shares
// that registry's lifetime; register factories before starting query execution.
// Appenders must validate before modifying output and reject source aliasing.
class CustomTypeOutput {
 public:
  virtual ~CustomTypeOutput() = default;
  virtual LogicalTypeDescription logicalType(const Type& type) const = 0;
  virtual StringValueAppender stringValueAppender() const = 0;
};

const CustomTypeOutput* getCustomTypeOutput(const Type& type);
LogicalTypeDescription describeLogicalType(const Type& type);
// Null for ordinary strings. Row sinks can cache this once per column.
StringValueAppender getStringValueAppender(const Type& type);
std::string formatStringValue(const Type& type, const StringView& value);

} // namespace facebook::velox
