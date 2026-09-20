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
#include <set>

#include "velox/expression/ExprConstants.h"
#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/expression/ExprUtils.h"
#include "velox/expression/SwitchRewrite.h"

#include <cstdlib>

namespace facebook::velox::expression {

namespace {

/// Kill switch for the nested-IF flattening. The rewrite is on by default;
/// set GPORCA_FLAT_CASE=0 to fall back to the original nested IF form (used for
/// A/B measurements and as a quick rollback).
bool flattenSwitchChains() {
  static const bool enabled = [] {
    const char* value = std::getenv("GPORCA_FLAT_CASE");
    return value == nullptr || *value == '\0' || *value != '0';
  }();
  return enabled;
}

bool isSwitchLike(const core::TypedExprPtr& expr) {
  if (!expr->isCallKind()) {
    return false;
  }
  const auto& name = expr->asUnchecked<core::CallTypedExpr>()->name();
  return name == kSwitch || name == kIf;
}

/// Appends the (condition, value) pairs of a chain of nested IF/SWITCH
/// expressions matching `resultType` to 'pairs' and returns the trailing ELSE
/// value, or nullptr when the chain has no ELSE clause.
core::TypedExprPtr flattenSwitchChain(
    const core::TypedExprPtr& expr,
    const TypePtr& resultType,
    std::vector<core::TypedExprPtr>& pairs) {
  if (!isSwitchLike(expr) || *expr->type() != *resultType) {
    return expr;
  }

  const auto& inputs = expr->inputs();
  const auto numInputs = inputs.size();
  const bool hasElse = numInputs % 2 == 1;
  const auto lastPair = hasElse ? numInputs - 2 : numInputs - 1;
  for (auto i = 0; i + 1 <= lastPair; i += 2) {
    pairs.push_back(inputs.at(i));
    pairs.push_back(inputs.at(i + 1));
  }

  if (!hasElse) {
    return nullptr;
  }

  const auto& elseValue = inputs.at(numInputs - 1);
  if (isSwitchLike(elseValue) && *elseValue->type() == *resultType) {
    return flattenSwitchChain(elseValue, resultType, pairs);
  }
  return elseValue;
}

} // namespace

core::TypedExprPtr SwitchRewrite::rewrite(const core::TypedExprPtr& expr) {
  if (!expr->isCallKind()) {
    return nullptr;
  }
  const auto* callExpr = expr->asUnchecked<core::CallTypedExpr>();
  if (!(callExpr->name() == kSwitch) && !(callExpr->name() == kIf)) {
    return nullptr;
  }

  const auto& inputs = expr->inputs();
  const auto numInputs = inputs.size();
  std::vector<core::TypedExprPtr> optimizedInputs;

  // Iterate over all `(condition, value)` pairs. Handle `else` value at the
  // end.
  for (auto i = 0; i < numInputs - 1; i += 2) {
    const auto& condition = inputs.at(i);
    const auto& value = inputs.at(i + 1);

    if (condition->isConstantKind()) {
      const auto* constCondition =
          condition->asUnchecked<core::ConstantTypedExpr>();
      if (auto boolCondition = constCondition->toBool()) {
        if (boolCondition.value()) {
          // If this condition is true and all conditions before this are false,
          // simplify `switch` expression to the corresponding `value`.
          if (optimizedInputs.empty()) {
            return value;
          }
          // If this condition is true and all conditions before this can only
          // be evaluated at runtime, make this the new `else` value and stop
          // checking further conditions.
          optimizedInputs.push_back(value);
          break;
        } else {
          // Skip false conditions.
          continue;
        }
      } else {
        // Skip NULL conditions.
        continue;
      }
    } else {
      optimizedInputs.push_back(condition);
      optimizedInputs.push_back(value);
    }
  }

  // Handle `else` value if present.
  if (optimizedInputs.size() % 2 == 0 && numInputs % 2 == 1) {
    const auto elseValue = inputs.at(numInputs - 1);
    // Return `else` value if there are no conditions.
    if (optimizedInputs.empty()) {
      return elseValue;
    }
    optimizedInputs.emplace_back(elseValue);
  }
  // Return NULL if there are no conditions and `else` value is not present.
  if (optimizedInputs.empty()) {
    return core::ConstantTypedExpr::makeNull(expr->type());
  }

  // Flatten a chain of nested IF/SWITCH expressions into a single SWITCH with
  // one (condition, value) pair per branch. Velox evaluates every switch
  // expression as an independent merge: a chain of N nested IFs allocates a
  // result vector and re-merges the whole batch at every level, while the
  // flattened form merges all branches into a single result vector. Branch
  // values must already have the result type, so mixed-type chains, which need
  // coercions, are left as they are.
  if (flattenSwitchChains() && optimizedInputs.size() % 2 == 1 &&
      optimizedInputs.size() > 1) {
    std::vector<core::TypedExprPtr> pairs(
        optimizedInputs.begin(), optimizedInputs.end() - 1);
    auto elseValue =
        flattenSwitchChain(optimizedInputs.back(), expr->type(), pairs);
    if (pairs.size() >= 4) {
      if (elseValue != nullptr) {
        pairs.push_back(elseValue);
      }
      return std::make_shared<core::CallTypedExpr>(
          expr->type(), std::move(pairs), kSwitch);
    }
  }

  return std::make_shared<core::CallTypedExpr>(
      expr->type(), std::move(optimizedInputs), callExpr->name());
}

void SwitchRewrite::registerRewrite() {
  expression::ExprRewriteRegistry::instance().registerRewrite(
      expression::SwitchRewrite::rewrite);
}

} // namespace facebook::velox::expression
