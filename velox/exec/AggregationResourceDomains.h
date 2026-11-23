#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include "velox/common/base/AdmissionController.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec {

/// Immutable host-query bindings. Only scalar domain ids travel in the plan;
/// these controllers and their charged pool backing stay on the worker.
class AggregationResourceDomains final {
 public:
  static constexpr const char* kRegistryKey = "aggregationResourceDomains";
  using Entry = std::pair<uint64_t, std::shared_ptr<common::AdmissionController>>;

  explicit AggregationResourceDomains(std::vector<Entry> entries) {
    for (auto& [id, controller] : entries) {
      VELOX_USER_CHECK(id > 0 && controller && controller->maxResourceUsage() > 0,
          "Invalid aggregation resource domain");
      VELOX_USER_CHECK(domains_.emplace(id, std::move(controller)).second,
          "Duplicate aggregation resource domain");
    }
  }

  const std::shared_ptr<common::AdmissionController>& controller(uint64_t id) const {
    const auto it = domains_.find(id);
    VELOX_USER_CHECK(it != domains_.end(), "Aggregation resource domain is not installed: {}", id);
    return it->second;
  }

 private:
  std::unordered_map<uint64_t, std::shared_ptr<common::AdmissionController>> domains_;
};

} // namespace facebook::velox::exec
