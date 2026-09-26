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
#include <cstdlib>
#include <limits>

#include "folly/CPortability.h"

#include "velox/functions/lib/CheckedArithmeticImpl.h"

namespace facebook::velox::functions::aggregate {

/// Prototypes for the SUM(bigint) overflow-check candidates of the [Win]
/// 20260923 perf cycle (P3-A1 / P3-A2, see
/// dev_tasks/performance_tuning/20260923_win/tasks/P3_agg_inner_loop.md).
///
/// Both gates default to OFF, so the default path is byte-identical to the
/// upstream behaviour; each has a one-line rollback (`=0`) and is only read
/// once per update call (never per row).
///
/// Why: on Windows `windows::builtin_add_overflow<int64_t>` is a software sign
/// test whose `&&` short-circuits, and the per-row group store keeps the
/// accumulator in memory. The offline micro-benchmark (tracking doc section 16,
/// tool `test/tools/sum_batch_overflow_bench.cc`) measures 4.77 ns/row for a
/// mixed-sign single-group sum and 4.50 ns/row for a 960-group hash table,
/// against 1.3 ns/row for a branchless check and 0.66-0.79 ns/row for a batch
/// accumulate with one range check per batch.

/// P3-A2: per-row overflow test without the short-circuit branch. Applies to
/// both the grouped and the single-group update paths.
/// Flipped to default-on on 2026-09-26 ([Win] 20260923 section 17: same-binary
/// A/B Q653 -7..8%, Q620 -13.5%, Q638 -11.7%, full 727/727 with the gate on);
/// `GPORCA_SUM_BRANCHLESS_CHECK=0` is the one-line rollback.
inline bool sumOverflowBranchlessCheckEnabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("GPORCA_SUM_BRANCHLESS_CHECK");
    if (env != nullptr && *env != '\0') {
      return std::atoi(env) != 0;
    }
    return true;
  }();
  return enabled;
}

/// P3-A1: batch accumulate the single-group input in registers with one range
/// check per batch (suspicious batches fall back to the canonical per-row
/// path).
/// Flipped to default-on on 2026-09-26 ([Win] 20260923 section 17: same-binary
/// A/B Q620 -18.8%, Q638 -14.4%, controls within noise, full 727/727 with the
/// gate on); `GPORCA_SUM_BATCH_ACCUMULATE=0` is the one-line rollback.
inline bool sumOverflowBatchAccumulateEnabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("GPORCA_SUM_BATCH_ACCUMULATE");
    if (env != nullptr && *env != '\0') {
      return std::atoi(env) != 0;
    }
    return true;
  }();
  return enabled;
}

/// Same contract as functions::checkedPlus<int64_t> but with a branchless
/// overflow test (`&` instead of `&&`). The cold overflow arm delegates to the
/// canonical helper so the error type and message stay identical.
FOLLY_ALWAYS_INLINE int64_t checkedPlusBranchlessInt64(int64_t a, int64_t b) {
  const int64_t result =
      static_cast<int64_t>(static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
  if (UNLIKELY(((a ^ b) >= 0) & ((result ^ a) < 0))) {
    return checkedPlus<int64_t>(a, b);
  }
  return result;
}

/// Mirrors the signed-overflow test of windows::builtin_add_overflow<int64_t>
/// without the branch (used by the batch bounds check below).
FOLLY_ALWAYS_INLINE bool addOverflowsInt64(int64_t a, int64_t b) {
  const int64_t result =
      static_cast<int64_t>(static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
  return ((a ^ b) >= 0) & ((result ^ a) < 0);
}

} // namespace facebook::velox::functions::aggregate
