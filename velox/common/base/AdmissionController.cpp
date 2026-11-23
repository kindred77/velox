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

#include "velox/common/base/AdmissionController.h"
#include <algorithm>
#include <chrono>
#include <optional>
#include <folly/ScopeGuard.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/time/Timer.h"

namespace facebook::velox::common {

struct AdmissionController::Request {
  enum class State { kQueued, kGranted, kClaimed, kCancelled };
  explicit Request(uint64_t units, bool async = false)
      : unitsRequested(units), future(promise.getSemiFuture()),
        async(async), queuedAt(std::chrono::steady_clock::now()) {}
  const uint64_t unitsRequested;
  ContinuePromise promise;
  ContinueFuture future;
  State state{State::kQueued};
  std::shared_ptr<Request> nextReady;
  const bool async;
  const std::chrono::steady_clock::time_point queuedAt;
};


void AdmissionController::accept(uint64_t resourceUnits) {
  if (resourceUnits == 0) {
    return;
  }
  ContinueFuture future;
  uint64_t updatedValue = 0;
  VELOX_CHECK_LE(
      resourceUnits,
      config_.maxLimit,
      "A single request cannot exceed the max limit");
  {
    std::lock_guard<std::mutex> l(mu_);
    if (!queue_.empty() || !canAdmitLocked(resourceUnits)) {
      auto request = std::make_shared<Request>(resourceUnits);
      future = std::move(request->future);
      queue_.push_back(std::move(request));
    } else {
      updatedValue = unitsUsed_ += resourceUnits;
    }
  }
  if (!future.valid()) {
    // Only upadate if there was no wait, as the releasing thread is responsible
    // for updating the metric.
    if (!config_.resourceUsageAvgMetric.empty()) {
      RECORD_METRIC_VALUE(config_.resourceUsageAvgMetric, updatedValue);
    }
    return;
  }
  if (!config_.resourceQueuedCountMetric.empty()) {
    RECORD_METRIC_VALUE(config_.resourceQueuedCountMetric);
  }
  uint64_t waitTimeUs{0};
  {
    MicrosecondWallTimer timer(&waitTimeUs);
    future.wait();
  }
  if (!config_.resourceQueuedTimeMsHistogramMetric.empty()) {
    RECORD_HISTOGRAM_METRIC_VALUE(
        config_.resourceQueuedTimeMsHistogramMetric, waitTimeUs / 1'000);
  }
}

bool AdmissionController::tryAccept(uint64_t resourceUnits) {
  if (resourceUnits == 0) {
    return true;
  }
  uint64_t updatedValue;
  {
    std::lock_guard<std::mutex> l(mu_);
    if (!queue_.empty() || !canAdmitLocked(resourceUnits)) {
      return false;
    }
    updatedValue = unitsUsed_ += resourceUnits;
  }
  if (!config_.resourceUsageAvgMetric.empty()) {
    RECORD_METRIC_VALUE(config_.resourceUsageAvgMetric, updatedValue);
  }
  return true;
}

void AdmissionController::release(uint64_t resourceUnits) {
  {
    std::lock_guard<std::mutex> l(mu_);
    VELOX_CHECK_LE(resourceUnits, unitsUsed_,
        "Cannot release more units than have been acquired");
    unitsUsed_ -= resourceUnits;
  }
  drainQueue();
}

void AdmissionController::drainQueue() {
  // A continuation can drop every external owner, or release/cancel another
  // request inline. Keep this pump alive and let nested calls only change
  // state: both grants and cancellations complete with bounded stack depth.
  auto keepAlive = weak_from_this().lock();
  {
    std::lock_guard<std::mutex> l(mu_);
    if (draining_) return;
    draining_ = true;
  }
  auto resetDraining = folly::makeGuard([&] {
    std::lock_guard<std::mutex> l(mu_);
    draining_ = false;
  });
  while (true) {
    std::shared_ptr<Request> request;
    std::optional<ContinuePromise> ready;
    bool granted = false;
    uint64_t updatedValue;
    {
      std::lock_guard<std::mutex> l(mu_);
      if (readyHead_) {
        request = std::move(readyHead_);
        readyHead_ = std::move(request->nextReady);
        if (!readyHead_) readyTail_.reset();
      } else if (!queue_.empty() && canAdmitLocked(queue_.front()->unitsRequested)) {
        request = std::move(queue_.front());
        queue_.pop_front();
        unitsUsed_ += request->unitsRequested;
        request->state = Request::State::kGranted;
        granted = true;
      } else {
        draining_ = false;
        resetDraining.dismiss();
      }
      if (request) ready.emplace(std::move(request->promise));
      updatedValue = unitsUsed_;
    }
    if (!config_.resourceUsageAvgMetric.empty()) {
      RECORD_METRIC_VALUE(config_.resourceUsageAvgMetric, updatedValue);
    }
    if (!ready) return;
    if (granted && request->async && !config_.resourceQueuedTimeMsHistogramMetric.empty()) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - request->queuedAt).count();
      RECORD_HISTOGRAM_METRIC_VALUE(config_.resourceQueuedTimeMsHistogramMetric, elapsed);
    }
    ready->setValue();
  }
}

AdmissionReservation AdmissionController::reserveAsync(uint64_t resourceUnits) {
  auto owner = weak_from_this().lock();
  VELOX_CHECK(owner != nullptr, "Async admission requires a shared controller");
  VELOX_CHECK_LE(resourceUnits, config_.maxLimit,
      "A single request cannot exceed the max limit");
  auto request = std::make_shared<Request>(resourceUnits, true);
  std::optional<ContinuePromise> ready;
  uint64_t updatedValue;
  {
    std::lock_guard<std::mutex> l(mu_);
    if (resourceUnits == 0 || (queue_.empty() && canAdmitLocked(resourceUnits))) {
      unitsUsed_ += resourceUnits;
      request->state = Request::State::kGranted;
      ready.emplace(std::move(request->promise));
    } else {
      queue_.push_back(request);
    }
    updatedValue = unitsUsed_;
  }
  if (ready) {
    if (resourceUnits != 0 && !config_.resourceUsageAvgMetric.empty()) {
      RECORD_METRIC_VALUE(config_.resourceUsageAvgMetric, updatedValue);
    }
    ready->setValue();
  } else if (!config_.resourceQueuedCountMetric.empty()) {
    RECORD_METRIC_VALUE(config_.resourceQueuedCountMetric);
  }
  return AdmissionReservation(std::move(owner), std::move(request));
}

void AdmissionController::cancel(const std::shared_ptr<Request>& request) {
  {
    std::lock_guard<std::mutex> l(mu_);
    if (request->state == Request::State::kQueued) {
      const auto it = std::find(queue_.begin(), queue_.end(), request);
      VELOX_CHECK(it != queue_.end());
      queue_.erase(it);
      if (readyTail_) readyTail_->nextReady = request;
      else readyHead_ = request;
      readyTail_ = request;
    } else if (request->state == Request::State::kGranted) {
      VELOX_CHECK_LE(request->unitsRequested, unitsUsed_);
      unitsUsed_ -= request->unitsRequested;
    } else {
      return;
    }
    request->state = Request::State::kCancelled;
  }
  drainQueue();
}

AdmissionReservation::AdmissionReservation(
    std::shared_ptr<AdmissionController> owner,
    std::shared_ptr<AdmissionController::Request> request)
    : owner_(std::move(owner)), request_(std::move(request)) {}

AdmissionReservation::AdmissionReservation(AdmissionReservation&& other) noexcept
    : owner_(std::move(other.owner_)), request_(std::move(other.request_)) {}

AdmissionReservation& AdmissionReservation::operator=(AdmissionReservation&& other) noexcept {
  if (this != &other) {
    AdmissionReservation previous(std::move(*this));
    owner_ = std::move(other.owner_);
    request_ = std::move(other.request_);
  }
  return *this;
}

AdmissionReservation::~AdmissionReservation() { cancel(); }

ContinueFuture AdmissionReservation::takeFuture() {
  VELOX_CHECK(owner_ && request_);
  std::lock_guard<std::mutex> l(owner_->mu_);
  VELOX_CHECK(request_->future.valid(), "Admission future already taken");
  return std::move(request_->future);
}

bool AdmissionReservation::isGranted() const {
  if (!owner_ || !request_) return false;
  std::lock_guard<std::mutex> l(owner_->mu_);
  return request_->state == AdmissionController::Request::State::kGranted;
}

bool AdmissionReservation::belongsTo(const AdmissionController* owner) const {
  return owner_ && owner_.get() == owner && request_;
}

uint64_t AdmissionReservation::claim() {
  VELOX_CHECK(owner_ && request_);
  std::lock_guard<std::mutex> l(owner_->mu_);
  VELOX_CHECK(request_->state == AdmissionController::Request::State::kGranted,
      "Admission reservation has no unclaimed grant");
  request_->state = AdmissionController::Request::State::kClaimed;
  return request_->unitsRequested;
}

void AdmissionReservation::cancel() {
  // Cancellation completion may inline-destroy the reservation itself.
  auto owner = owner_;
  auto request = request_;
  if (owner && request) owner->cancel(request);
}
} // namespace facebook::velox::common
