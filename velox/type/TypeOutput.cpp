/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "velox/type/TypeOutput.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox {
LogicalTypeDescription describeLogicalType(const Type& type) {
  if (const auto* capability = getCustomTypeOutput(type)) {
    return capability->logicalType(type);
  }
  LogicalTypeDescription result{type.name(), {}, {}, {}};
  if (type.isDecimal()) {
    const auto [precision, scale] = getDecimalPrecisionScale(type);
    result.precision = precision;
    result.scale = scale;
    result.maximumTextBytes = DecimalUtil::maxStringViewSize(precision, scale);
  }
  return result;
}
StringValueAppender getStringValueAppender(const Type& type) {
  const auto* capability = getCustomTypeOutput(type);
  return capability ? capability->stringValueAppender() : nullptr;
}
std::string formatStringValue(const Type& type, const StringView& value) {
  std::string result;
  if (const auto append = getStringValueAppender(type)) {
    append(type, value, result);
  } else {
    result.append(value.data(), value.size());
  }
  return result;
}
} // namespace facebook::velox
