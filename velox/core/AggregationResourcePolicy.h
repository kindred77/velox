#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <folly/dynamic.h>

namespace facebook::velox::core {

/// Captured host-local cap; hosts instantiate independent controllers for it.
struct AggregationResourceDomainConfig {
  uint64_t domainId{0};
  uint64_t maxBytes{0};
};

/// A registration promise, not permission to reorder SQL or emit duplicates.
/// FixedRowState describes a C-layout tuple of fixed primitive fields.
enum class AggregateInputResourceModel : uint8_t {
  kUnknown,
  kFixedScalarState,
  kFixedRowState,
};

/// Aggregate-component bounds only. Row flags, hash tables, input/output
/// vectors, allocation rounding and simultaneous residency are separate costs.
/// This contains no executor pointer and can be captured as planning facts.
struct AggregateInputResourceContract {
  static constexpr uint32_t kVersion = 1;
  uint32_t stateBytesUpperBound{0};
  uint32_t stateAlignmentUpperBound{0};
  uint32_t argumentCount{0};
  AggregateInputResourceModel model{AggregateInputResourceModel::kUnknown};
  bool operator==(const AggregateInputResourceContract&) const = default;
};

/// Planning authorization for one aggregation occurrence. This carries no
/// controller or pool pointer; resource domains are instantiated by each host.
struct AggregationResourcePolicy {
  static constexpr uint32_t kVersion = 1;
  uint32_t version{kVersion};
  uint64_t domainId{0};
  std::vector<std::optional<AggregateInputResourceContract>> inputResources;

  void validate() const;
  folly::dynamic serialize() const;
  static std::shared_ptr<const AggregationResourcePolicy> deserialize(
      const folly::dynamic& value);
};

} // namespace facebook::velox::core
