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
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include "velox/common/base/VeloxException.h"
#include "velox/common/base/tests/GTestUtils.h"

using namespace facebook::velox;
namespace facebook::velox::common {
TEST(AdmissionController, basic) {
  const uint64_t kLimit = 100000;
  AdmissionController::Config config;
  config.maxLimit = kLimit;
  AdmissionController admissionController(config);
  EXPECT_EQ(admissionController.currentResourceUsage(), 0);

  admissionController.accept(100);
  EXPECT_EQ(admissionController.currentResourceUsage(), 100);

  admissionController.accept(100);
  EXPECT_EQ(admissionController.currentResourceUsage(), 200);

  admissionController.release(100);
  EXPECT_EQ(admissionController.currentResourceUsage(), 100);

  VELOX_ASSERT_THROW(
      admissionController.release(101),
      "Cannot release more units than have been acquired");

  VELOX_ASSERT_THROW(
      admissionController.accept(kLimit + 1),
      "A single request cannot exceed the max limit");
}

TEST(AdmissionController, overflowDoesNotOverAdmit) {
  AdmissionController::Config config;
  config.maxLimit = std::numeric_limits<uint64_t>::max();
  AdmissionController controller(config);
  const auto held = config.maxLimit - 1;
  controller.accept(held);
  std::promise<void> entered;
  auto waiter = std::async(std::launch::async, [&] {
    entered.set_value();
    controller.accept(3);
  });
  entered.get_future().wait();
  const bool admittedEarly =
      waiter.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
  EXPECT_FALSE(admittedEarly);
  if (admittedEarly) {
    // A wrapping implementation has lost its accounting; do not underflow it
    // again during cleanup and obscure the original failure.
    waiter.get();
    return;
  }
  EXPECT_EQ(controller.currentResourceUsage(), held);
  controller.release(1);
  EXPECT_EQ(waiter.wait_for(std::chrono::seconds(1)), std::future_status::timeout);
  EXPECT_EQ(controller.currentResourceUsage(), held - 1);
  controller.release(held - 1);
  waiter.get();
  EXPECT_EQ(controller.currentResourceUsage(), 3);
  controller.release(3);
  EXPECT_EQ(controller.currentResourceUsage(), 0);
}

TEST(AdmissionController, nonblockingCapacityAndZero) {
  AdmissionController::Config config;
  config.maxLimit = 6;
  AdmissionController controller(config);
  EXPECT_TRUE(controller.tryAccept(0));
  EXPECT_TRUE(controller.tryAccept(4));
  EXPECT_FALSE(controller.tryAccept(3));
  EXPECT_EQ(controller.currentResourceUsage(), 4);
  EXPECT_TRUE(controller.tryAccept(2));
  EXPECT_FALSE(controller.tryAccept(1));
  EXPECT_FALSE(controller.tryAccept(7));
  EXPECT_TRUE(controller.tryAccept(0));
  EXPECT_EQ(controller.currentResourceUsage(), 6);
  controller.release(4);
  EXPECT_TRUE(controller.tryAccept(4));
  controller.release(6);
  EXPECT_EQ(controller.currentResourceUsage(), 0);

  config.maxLimit = 0;
  AdmissionController zero(config);
  EXPECT_TRUE(zero.tryAccept(0));
  EXPECT_FALSE(zero.tryAccept(1));
}

TEST(AdmissionController, nonblockingUnsignedLimit) {
  AdmissionController::Config config;
  config.maxLimit = std::numeric_limits<uint64_t>::max();
  AdmissionController controller(config);
  EXPECT_TRUE(controller.tryAccept(config.maxLimit - 1));
  EXPECT_FALSE(controller.tryAccept(2));
  EXPECT_TRUE(controller.tryAccept(1));
  EXPECT_FALSE(controller.tryAccept(1));
  EXPECT_EQ(controller.currentResourceUsage(), config.maxLimit);
  controller.release(config.maxLimit);
  EXPECT_EQ(controller.currentResourceUsage(), 0);
}

TEST(AdmissionController, queuedRequestsCannotBeBypassed) {
  AdmissionController::Config config;
  config.maxLimit = 10;
  AdmissionController controller(config);
  controller.accept(8);
  std::promise<void> entered;
  auto head = std::async(std::launch::async, [&] {
    entered.set_value();
    controller.accept(3);
  });
  entered.get_future().wait();
  // Observe a real queued request without sleeps or test-only controller APIs.
  bool denied = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (!denied && std::chrono::steady_clock::now() < deadline) {
    denied = !controller.tryAccept(1);
    if (!denied) controller.release(1);
    std::this_thread::yield();
  }
  EXPECT_TRUE(denied);
  EXPECT_TRUE(controller.tryAccept(0));
  EXPECT_EQ(controller.currentResourceUsage(), 8);
  std::promise<void> tailEntered;
  auto tail = std::async(std::launch::async, [&] {
    tailEntered.set_value();
    controller.accept(1);
  });
  tailEntered.get_future().wait();
  EXPECT_EQ(tail.wait_for(std::chrono::seconds(1)), std::future_status::timeout);
  controller.release(8);
  head.get();
  tail.get();
  EXPECT_EQ(controller.currentResourceUsage(), 4);
  controller.release(4);
  EXPECT_EQ(controller.currentResourceUsage(), 0);
}

TEST(AdmissionController, nonblockingConcurrentResourceOwnership) {
  AdmissionController::Config config;
  config.maxLimit = 8;
  AdmissionController controller(config);
  std::atomic<uint64_t> live{0};
  std::atomic<uint64_t> acquired{0};
  std::atomic<bool> exceeded{false};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 4; ++worker) {
    workers.emplace_back([&, worker] {
      for (int attempt = 0; attempt < 2500; ++attempt) {
        const uint64_t amount = 1 + (worker + attempt) % 4;
        if (!controller.tryAccept(amount)) {
          std::this_thread::yield();
          continue;
        }
        if (live.fetch_add(amount) + amount > config.maxLimit) exceeded = true;
        ++acquired;
        std::this_thread::yield();
        live.fetch_sub(amount);
        controller.release(amount);
      }
    });
  }
  for (auto& worker : workers) worker.join();
  EXPECT_FALSE(exceeded.load());
  EXPECT_GT(acquired.load(), 0);
  EXPECT_EQ(live.load(), 0);
  EXPECT_EQ(controller.currentResourceUsage(), 0);
}

TEST(AdmissionController, multiThreaded) {
  // Ensure that resource usage never exceeds the limit set in the admission
  // controller.
  const uint64_t kLimit = 30;
  std::atomic_uint64_t currentUsage{0};
  AdmissionController::Config config;
  config.maxLimit = kLimit;
  AdmissionController admissionController(config);

  std::vector<std::thread> threads;
  for (int i = 0; i < 20; i++) {
    threads.push_back(std::thread([&]() {
      for (int j = 0; j < 10'000; j++) {
        uint64_t usageUnits = std::rand() % (kLimit + 2);
        if (usageUnits > kLimit) {
          VELOX_ASSERT_THROW(
              admissionController.accept(usageUnits),
              "A single request cannot exceed the max limit")
          continue;
        }
        admissionController.accept(usageUnits);
        uint64_t curr = currentUsage.fetch_add(usageUnits);
        ASSERT_LE(curr + usageUnits, kLimit);
        currentUsage.fetch_sub(usageUnits);
        admissionController.release(usageUnits);
      }
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}
} // namespace facebook::velox::common
