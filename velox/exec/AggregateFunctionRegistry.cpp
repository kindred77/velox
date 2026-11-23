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
#include "velox/exec/AggregateFunctionRegistry.h"

#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateCompanionSignatures.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/type/Type.h"
#include <stdexcept>

namespace facebook::velox::exec {

namespace {
template <typename Result, typename Visit>
std::optional<Result> bindRegisteredSignature(
    const std::vector<std::shared_ptr<AggregateFunctionSignature>>& signatures,
    const std::vector<TypePtr>& inputs,
    Visit visit) {
  for (const auto& signature : signatures) {
    SignatureBinder binder(*signature, inputs, TypeCoercer::defaults());
    if (binder.tryBind()) {
      if (auto result = visit(*signature, binder)) {
        return result;
      }
    }
  }
  return std::nullopt;
}

std::string makeSignatureNotSupportedError(
    const std::string& name,
    const std::vector<TypePtr>& argTypes,
    const std::vector<std::shared_ptr<AggregateFunctionSignature>>&
        signatures) {
  std::stringstream error;
  error << "Aggregate function signature is not supported: "
        << toString(name, argTypes)
        << ". Supported signatures: " << toString(signatures) << ".";
  return error.str();
}
} // namespace

std::optional<AggregateExecutionTypes> tryResolveAggregateExecutionTypes(
    const std::string& name, const std::vector<TypePtr>& inputs,
    const TypePtr& expectedResult) {
  const auto signatures = getAggregateFunctionSignatures(name);
  if (!signatures) return std::nullopt;
  return bindRegisteredSignature<AggregateExecutionTypes>(
      *signatures, inputs, [&](const auto& signature, auto& binder)
          -> std::optional<AggregateExecutionTypes> {
        auto intermediate = binder.tryResolveType(signature.intermediateType());
        auto result = binder.tryResolveReturnType();
        if (!intermediate || !result) return std::nullopt;
        auto merge = CompanionSignatures::hasSameIntermediateTypesAcrossSignatures(*signatures)
            ? CompanionSignatures::mergeExtractFunctionNameWithSuffix(name, signature.returnType())
            : CompanionSignatures::mergeExtractFunctionName(name);
        if (!expectedResult || expectedResult->equivalent(*result)) {
          return AggregateExecutionTypes{std::move(intermediate), std::move(result), std::move(merge)};
        }
        // An adapted final result must be registered for this exact state.
        if (const auto companions = getCompanionFunctionSignatures(name)) {
          for (const auto& entry : companions->mergeExtract) {
            for (const auto& candidate : entry.signatures) {
              const std::vector<TypePtr> stateInput{intermediate};
              SignatureBinder stateBinder(*candidate, stateInput, TypeCoercer::defaults());
              if (!stateBinder.tryBind()) continue;
              const auto extracted = stateBinder.tryResolveReturnType();
              if (extracted && expectedResult->equivalent(*extracted)) {
                return AggregateExecutionTypes{std::move(intermediate), expectedResult, entry.functionName, true};
              }
            }
          }
        }
        return std::nullopt;
      });
}

AggregateExecutionTypes resolveAggregateExecutionTypes(
    const std::string& name, const std::vector<TypePtr>& rawInputTypes,
    const TypePtr& expectedResult) {
  if (auto types = tryResolveAggregateExecutionTypes(name, rawInputTypes, expectedResult)) {
    return std::move(*types);
  }
  throw std::invalid_argument("aggregate has no registered execution signature: " + name);
}

TypePtr resolveResultType(
    const std::string& name,
    const std::vector<TypePtr>& argTypes) {
  if (auto signatures = getAggregateFunctionSignatures(name)) {
    for (const auto& signature : signatures.value()) {
      SignatureBinder binder(*signature, argTypes, TypeCoercer::defaults());
      if (binder.tryBind()) {
        return binder.tryResolveReturnType();
      }
    }

    VELOX_USER_FAIL(
        makeSignatureNotSupportedError(name, argTypes, signatures.value()));
  }

  VELOX_USER_FAIL("Aggregate function not registered: {}", name);
}

TypePtr resolveResultTypeWithCoercions(
    const std::string& name,
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions,
    const TypeCoercer& coercer) {
  coercions.clear();

  if (auto signatures = getAggregateFunctionSignatures(name)) {
    std::vector<FunctionSignaturePtr> baseSignatures(
        signatures.value().begin(), signatures.value().end());
    if (auto type = tryResolveReturnTypeWithCoercions(
            baseSignatures, argTypes, coercions, coercer)) {
      return type;
    }

    VELOX_USER_FAIL(
        makeSignatureNotSupportedError(name, argTypes, signatures.value()));
  }

  VELOX_USER_FAIL("Aggregate function not registered: {}", name);
}

std::optional<TypePtr> tryResolveIntermediateType(
    const std::vector<std::shared_ptr<AggregateFunctionSignature>>& signatures,
    const std::vector<TypePtr>& argTypes) {
  return bindRegisteredSignature<TypePtr>(signatures, argTypes,
      [](const auto& signature, auto& binder) -> std::optional<TypePtr> {
        return binder.tryResolveType(signature.intermediateType());
      });
}

TypePtr resolveIntermediateType(
    const std::string& name,
    const std::vector<TypePtr>& argTypes) {
  if (auto signatures = getAggregateFunctionSignatures(name)) {
    if (auto type = tryResolveIntermediateType(*signatures, argTypes)) {
      return *type;
    }

    VELOX_USER_FAIL(
        "Aggregate function signature is not supported: {}. Supported signatures: {}.",
        toString(name, argTypes),
        toString(signatures.value()));
  } else {
    VELOX_USER_FAIL("Aggregate function not registered: {}", name);
  }
}

std::vector<std::string> getAggregateFunctionNames() {
  std::vector<std::string> names;
  exec::aggregateFunctions().withRLock([&](const auto& map) {
    names.reserve(map.size());
    for (const auto& function : map) {
      names.push_back(function.first);
    }
  });

  return names;
}

} // namespace facebook::velox::exec
