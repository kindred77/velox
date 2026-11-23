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
#include <deque>
#include <limits>
#include <string>
#include <mutex>
#include <memory>
#include "velox/common/future/VeloxPromise.h"

namespace facebook::velox::common {

class AdmissionReservation;

/// A generic admission controller that can be used to limit the number of
/// resources in use and can log metrics like resource usage, queued count,
/// queued wait times. When a calling thread's request for resources surpasses
/// the set limit, it will be placed in a FIFO queue. The thread must then wait
/// until sufficient resources are freed by other threads, addressing all
/// preceding requests in the queue, before its own request can be granted.
class AdmissionController : public std::enable_shared_from_this<AdmissionController> {
 public:
  struct Config {
    /// The maximum number of resource units that can be used at any given time.
    /// Set to a default value of max unit64 to signify unlimited limit.
    uint64_t maxLimit{std::numeric_limits<uint64_t>::max()};
    /// The metric name for resource usage. If not set, it will not be reported.
    /// Should be a registered as a average metric.
    std::string resourceUsageAvgMetric;
    /// The metric name for resource queued count. If not set, it will not be
    /// reported. Should be a registered as a count metric.
    std::string resourceQueuedCountMetric;
    /// The metric name for resource queued wait time. If not set, it will not
    /// be reported. Should be a registered as a histogram metric.
    std::string resourceQueuedTimeMsHistogramMetric;
  };
  explicit AdmissionController(const Config& config) : config_(config) {}

  // Accept can block until sufficient resources are freed by other threads.
  void accept(uint64_t resourceUnits);

  // Does not wait for capacity or enqueue. Positive requests cannot bypass waiters;
  // a zero request succeeds without consuming capacity. Returns false when
  // capacity is unavailable, including requests larger than the total limit.
  bool tryAccept(uint64_t resourceUnits);
  void release(uint64_t resourceUnits);

  /// Never blocks the calling thread. Requires shared ownership of this
  /// controller. The reservation owns any grant until claim() transfers it;
  /// cancellation/destruction refunds an unclaimed grant or removes a waiter.
  AdmissionReservation reserveAsync(uint64_t resourceUnits);

  uint64_t currentResourceUsage() const {
    std::lock_guard<std::mutex> l(mu_);
    return unitsUsed_;
  }

  uint64_t maxResourceUsage() const { return config_.maxLimit; }

 private:
  // Called under mu_; subtraction remains valid even at UINT64_MAX.
  bool canAdmitLocked(uint64_t resourceUnits) const {
    return resourceUnits <= config_.maxLimit - unitsUsed_;
  }

  friend class AdmissionReservation;
  struct Request;
  void drainQueue();
  void cancel(const std::shared_ptr<Request>& request);
  Config config_;
  mutable std::mutex mu_;
  uint64_t unitsUsed_{0};
  std::deque<std::shared_ptr<Request>> queue_;
  // Cancelled requests carry their own completion links; cancelling never
  // allocates a second queue node while the resource is under pressure.
  std::shared_ptr<Request> readyHead_;
  std::shared_ptr<Request> readyTail_;
  bool draining_{false};
};
/// Move-only ownership of a queued or granted admission request. A ready
/// future includes cancellation: callers must claim the grant before work.
/// Claim transfers responsibility for release to the consuming operation.
class AdmissionReservation {
 public:
  AdmissionReservation(AdmissionReservation&& other) noexcept;
  AdmissionReservation& operator=(AdmissionReservation&& other) noexcept;
  ~AdmissionReservation();

  ContinueFuture takeFuture();
  bool isGranted() const;
  bool belongsTo(const AdmissionController* owner) const;
  uint64_t claim();
  void cancel();

 private:
  friend class AdmissionController;
  AdmissionReservation(std::shared_ptr<AdmissionController> owner,
      std::shared_ptr<AdmissionController::Request> request);
  std::shared_ptr<AdmissionController> owner_;
  std::shared_ptr<AdmissionController::Request> request_;
};
} // namespace facebook::velox::common
