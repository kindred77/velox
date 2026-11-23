#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <folly/Range.h>

#include "velox/core/PlanNode.h"
#include "velox/core/AggregationResourcePolicy.h"

namespace facebook::velox::exec {

using AggregateInputResourceContracts =
    folly::Range<std::optional<core::AggregateInputResourceContract>*>;
using ConstAggregateInputResourceContracts =
    folly::Range<const std::optional<core::AggregateInputResourceContract>*>;

/// Resolve from the registered model and bound signature without constructing
/// an aggregate or consuming data. Unsupported types/state protocols are unknown;
/// actual input encodings are checked during preparation. Known raw SINGLE
/// resources do not authorize chunking a SINGLE.
std::optional<core::AggregateInputResourceContract> getAggregateInputResourceContract(
    const std::string& name,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    const TypePtr& outputType);

} // namespace facebook::velox::exec
