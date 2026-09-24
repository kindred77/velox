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

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

/// Temporary diagnostic probe for the [Win] 20260923 peel / dictionary study
/// (dev_tasks/performance_tuning/20260923_win): counts how often velox's
/// expression-layer dictionary peeling engages, how many dictionary levels it
/// collapses, whether the wrap can reuse the input indices or must compose them
/// per batch, and how much work the row translation / re-wrap moves. The study
/// uses these counts to size a mechanism (share the peel across the expressions
/// of one batch, or make the per-batch compose cheaper) before implementing it.
///
/// Disabled unless GPORCA_PEEL_STATS is set in the environment; one summary
/// line is printed to stderr at process exit. Remove or turn into a mechanism
/// once the study is done.
namespace facebook::velox::exec::peelstats {

inline bool enabled() {
  static const bool value = ::getenv("GPORCA_PEEL_STATS") != nullptr;
  return value;
}

struct Counters {
  std::atomic<uint64_t> peelSuccess{0};
  std::atomic<uint64_t> peelNotPeelable{0};
  std::atomic<uint64_t> peelConstWrap{0};
  std::atomic<uint64_t> peelDictWrap{0};
  std::atomic<uint64_t> levels1{0};
  std::atomic<uint64_t> levels2{0};
  std::atomic<uint64_t> levels3Plus{0};
  std::atomic<uint64_t> peelRows{0};
  std::atomic<uint64_t> peelFields{0};
  std::atomic<uint64_t> mayCache{0};
  std::atomic<uint64_t> wrapperReuse{0};
  std::atomic<uint64_t> wrapperCopy{0};
  std::atomic<uint64_t> innerRowsCalls{0};
  std::atomic<uint64_t> innerRows{0};
  std::atomic<uint64_t> resultWrapCalls{0};
  std::atomic<uint64_t> resultConstWrap{0};
};

inline Counters& counters() {
  static Counters counters;
  return counters;
}

inline void report() {
  auto& c = counters();
  ::fprintf(
      stderr,
      "[peelstats] peel_ok=%llu not_peelable=%llu dict_wrap=%llu const_wrap=%llu"
      " levels1=%llu levels2=%llu levels3plus=%llu peel_rows=%llu peel_fields=%llu"
      " may_cache=%llu wrapper_reuse=%llu wrapper_copy=%llu inner_rows_calls=%llu"
      " inner_rows=%llu result_wrap_calls=%llu result_const_wrap=%llu\n",
      static_cast<unsigned long long>(c.peelSuccess.load()),
      static_cast<unsigned long long>(c.peelNotPeelable.load()),
      static_cast<unsigned long long>(c.peelDictWrap.load()),
      static_cast<unsigned long long>(c.peelConstWrap.load()),
      static_cast<unsigned long long>(c.levels1.load()),
      static_cast<unsigned long long>(c.levels2.load()),
      static_cast<unsigned long long>(c.levels3Plus.load()),
      static_cast<unsigned long long>(c.peelRows.load()),
      static_cast<unsigned long long>(c.peelFields.load()),
      static_cast<unsigned long long>(c.mayCache.load()),
      static_cast<unsigned long long>(c.wrapperReuse.load()),
      static_cast<unsigned long long>(c.wrapperCopy.load()),
      static_cast<unsigned long long>(c.innerRowsCalls.load()),
      static_cast<unsigned long long>(c.innerRows.load()),
      static_cast<unsigned long long>(c.resultWrapCalls.load()),
      static_cast<unsigned long long>(c.resultConstWrap.load()));
}

/// Registers the exit-time report once, when the probe is enabled.
inline void init() {
  static const bool registered = []() {
    if (enabled()) {
      std::atexit(report);
    }
    return true;
  }();
  (void)registered;
}

} // namespace facebook::velox::exec::peelstats
