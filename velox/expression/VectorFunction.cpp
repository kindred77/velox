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
#include "velox/expression/VectorFunction.h"
#include <algorithm>
#include "folly/Synchronized.h"
#include "velox/expression/SignatureBinder.h"

namespace facebook::velox::exec {

namespace {
template <typename TResult, typename TFunc>
std::optional<TResult> applyToVectorFunctionEntry(
    const std::string& name,
    TFunc applyFunc) {
  auto sanitizedName = sanitizeName(name);

  return vectorFunctionFactories().withRLock(
      [&](auto& functions) -> std::optional<TResult> {
        auto it = functions.find(sanitizedName);
        if (it == functions.end()) {
          return std::nullopt;
        }
        return applyFunc(sanitizedName, it->second);
      });
}

// The entry pointer is borrowed only while the registry read lock is held.
struct ExactVectorFunction {
  TypePtr returnType;
  const VectorFunctionEntry* entry;
};

std::optional<ExactVectorFunction> resolveExactVectorFunction(
    const std::vector<VectorFunctionEntry>& entries,
    const std::vector<TypePtr>& argTypes) {
  for (const auto& entry : entries) {
    for (const auto& signature : entry.signatures) {
      SignatureBinder binder(*signature, argTypes, TypeCoercer::defaults());
      if (binder.tryBind()) {
        return ExactVectorFunction{binder.tryResolveReturnType(), &entry};
      }
    }
  }
  return std::nullopt;
}

// Zip `inputTypes` and `constantInputs` vectors into a single
// vector of `VectorFunctionArg`.
std::vector<VectorFunctionArg> toVectorFunctionArgs(
    const std::vector<TypePtr>& inputTypes,
    const std::vector<VectorPtr>& constantInputs) {
  std::vector<VectorFunctionArg> args;
  args.reserve(inputTypes.size());

  for (vector_size_t i = 0; i < inputTypes.size(); ++i) {
    args.push_back({
        inputTypes[i],
        constantInputs.size() > i ? constantInputs[i] : nullptr,
    });
  }

  return args;
}
} // namespace

VectorFunctionMap& vectorFunctionFactories() {
  static VectorFunctionMap factories;
  return factories;
}

std::optional<VectorFunctionMetadata> getVectorFunctionMetadata(
    const std::string& name) {
  return applyToVectorFunctionEntry<VectorFunctionMetadata>(
      name,
      [&](const auto& /*name*/, const auto& entries) {
        auto metadata = entries.front().metadata;
        metadata.supportsFlattening &= entries.size() == 1;
        for (const auto& entry : entries) {
          metadata.deterministic &= entry.metadata.deterministic;
          metadata.defaultNullBehavior &= entry.metadata.defaultNullBehavior;
          if (metadata.owner != entry.metadata.owner) {
            metadata.owner = {};
          }
        }
        return metadata;
      });
}

std::optional<std::vector<FunctionSignaturePtr>> getVectorFunctionSignatures(
    const std::string& name) {
  return applyToVectorFunctionEntry<std::vector<FunctionSignaturePtr>>(
      name, [&](const auto& /*name*/, const auto& entries) {
        std::vector<FunctionSignaturePtr> signatures;
        for (const auto& entry : entries) {
          signatures.insert(
              signatures.end(), entry.signatures.begin(), entry.signatures.end());
        }
        return signatures;
      });
}

TypePtr resolveVectorFunction(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes) {
  if (auto outputTypeWithMetadata =
          resolveVectorFunctionWithMetadata(functionName, argTypes)) {
    return outputTypeWithMetadata->first;
  }

  return nullptr;
}

namespace {
bool hasCoercion(const std::vector<Coercion>& coercions) {
  for (const auto& coercion : coercions) {
    if (coercion.type != nullptr) {
      return true;
    }
  }

  return false;
}

bool sameCastPath(
    const std::vector<Coercion>& left,
    const std::vector<Coercion>& right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (size_t i = 0; i < left.size(); ++i) {
    if (bool(left[i].type) != bool(right[i].type)) {
      return false;
    }
    if (left[i].type &&
        (left[i].cost != right[i].cost ||
         *left[i].type != *right[i].type)) {
      return false;
    }
  }
  return true;
}
} // namespace

TypePtr resolveVectorFunctionWithCoercions(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions,
    const TypeCoercer& coercer) {
  if (auto result = resolveVectorFunctionWithMetadataWithCoercions(
          functionName, argTypes, coercions, coercer)) {
    return result->first;
  }

  return nullptr;
}

std::optional<std::pair<TypePtr, VectorFunctionMetadata>>
resolveVectorFunctionWithMetadata(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes) {
  return applyToVectorFunctionEntry<std::pair<TypePtr, VectorFunctionMetadata>>(
      functionName,
      [&](const auto& /*name*/, const auto& entries)
          -> std::optional<std::pair<TypePtr, VectorFunctionMetadata>> {
        if (auto resolved = resolveExactVectorFunction(entries, argTypes)) {
          return {{resolved->returnType, resolved->entry->metadata}};
        }
        return std::nullopt;
      });
}

std::optional<std::pair<TypePtr, VectorFunctionMetadata>>
resolveVectorFunctionWithMetadataWithCoercions(
    const std::string& functionName,
    const std::vector<TypePtr>& argTypes,
    std::vector<TypePtr>& coercions,
    const TypeCoercer& coercer) {
  coercions.clear();

  return applyToVectorFunctionEntry<std::pair<TypePtr, VectorFunctionMetadata>>(
      functionName,
      [&](const auto& /*name*/, const auto& entries)
          -> std::optional<std::pair<TypePtr, VectorFunctionMetadata>> {
        std::vector<std::pair<std::vector<Coercion>, ExactVectorFunction>> candidates;
        for (const auto& entry : entries) {
          for (const auto& signature : entry.signatures) {
            exec::SignatureBinder binder(*signature, argTypes, coercer);
            std::vector<Coercion> requiredCoercions;
            if (binder.tryBindWithCoercions(requiredCoercions)) {
              auto type = binder.tryResolveReturnType();
              VELOX_CHECK_NOT_NULL(type);
              if (!hasCoercion(requiredCoercions)) {
                coercions.resize(argTypes.size(), nullptr);
                return {{type, entry.metadata}};
              }

              ExactVectorFunction candidate{type, &entry};
              if (entries.size() > 1) {
                auto coercedTypes = argTypes;
                for (size_t i = 0; i < requiredCoercions.size(); ++i) {
                  if (requiredCoercions[i].type) {
                    coercedTypes[i] = requiredCoercions[i].type;
                  }
                }
                // A prior generic signature may match after explicit casts.
                // Plan the same provider that expression compilation will use.
                auto resolved = resolveExactVectorFunction(entries, coercedTypes);
                if (!resolved) {
                  continue;
                }
                candidate = *resolved;
                const bool duplicate = std::any_of(
                    candidates.begin(),
                    candidates.end(),
                    [&](const auto& existing) {
                      return existing.second.entry == candidate.entry &&
                          *existing.second.returnType == *candidate.returnType &&
                          sameCastPath(existing.first, requiredCoercions);
                    });
                if (duplicate) {
                  continue;
                }
              }
              candidates.emplace_back(std::move(requiredCoercions), candidate);
            }
          }
        }
        // Candidate null behavior belongs to the provider of that signature.
        auto index = Coercion::pickLowestCost(
            candidates, argTypes, [&](size_t candidateIndex) {
              return Coercion::CandidateMetadata{
                  .returnType = candidates[candidateIndex].second.returnType,
                  .nullOnNull = candidates[candidateIndex]
                                    .second.entry->metadata.defaultNullBehavior};
            });

        if (index) {
          const auto& requiredCoercions = candidates[index.value()].first;
          coercions.reserve(requiredCoercions.size());
          for (const auto& coercion : requiredCoercions) {
            coercions.push_back(coercion.type);
          }

          const auto& selected = candidates[index.value()].second;
          return {{selected.returnType, selected.entry->metadata}};
        }

        return std::nullopt;
      });
}

std::shared_ptr<VectorFunction> getVectorFunction(
    const std::string& name,
    const std::vector<TypePtr>& inputTypes,
    const std::vector<VectorPtr>& constantInputs,
    const core::QueryConfig& config) {
  auto functionWithMetadata =
      getVectorFunctionWithMetadata(name, inputTypes, constantInputs, config);
  if (!functionWithMetadata.has_value()) {
    return nullptr;
  }
  return functionWithMetadata->first;
}

std::optional<
    std::pair<std::shared_ptr<VectorFunction>, VectorFunctionMetadata>>
getVectorFunctionWithMetadata(
    const std::string& name,
    const std::vector<TypePtr>& inputTypes,
    const std::vector<VectorPtr>& constantInputs,
    const core::QueryConfig& config) {
  if (!constantInputs.empty()) {
    VELOX_CHECK_EQ(inputTypes.size(), constantInputs.size());
  }

  return applyToVectorFunctionEntry<
      std::pair<std::shared_ptr<VectorFunction>, VectorFunctionMetadata>>(
      name,
      [&](const auto& sanitizedName, const auto& entries)
          -> std::optional<std::pair<
              std::shared_ptr<VectorFunction>,
              VectorFunctionMetadata>> {
        if (auto resolved = resolveExactVectorFunction(entries, inputTypes)) {
          auto inputArgs = toVectorFunctionArgs(inputTypes, constantInputs);
          const auto& entry = *resolved->entry;
          return {
              {entry.factory(sanitizedName, inputArgs, config), entry.metadata}};
        }
        return std::nullopt;
      });
}

/// Registers a new vector function. When overwrite = true, previous functions
/// with the given name will be replaced.
/// Returns true iff an insertion actually happened
bool registerStatefulVectorFunction(
    std::string_view name,
    std::vector<FunctionSignaturePtr> signatures,
    VectorFunctionFactory factory,
    VectorFunctionMetadata metadata,
    bool overwrite) {
  auto sanitizedName = sanitizeName(name);
  // Construct before mutating the map. Initializer lists copy std::function
  // and can leave an empty group behind if construction throws after lookup.
  std::vector<VectorFunctionEntry> entries;
  entries.push_back(
      {std::move(signatures), std::move(factory), std::move(metadata)});
  return vectorFunctionFactories().withWLock([&](auto& functionMap) {
    if (overwrite) {
      functionMap.insert_or_assign(sanitizedName, std::move(entries));
      return true;
    }
    return functionMap.try_emplace(sanitizedName, std::move(entries)).second;
  });
}

bool registerStatefulVectorFunctionOverload(
    std::string_view name,
    std::vector<FunctionSignaturePtr> signatures,
    VectorFunctionFactory factory,
    VectorFunctionMetadata metadata) {
  if (signatures.empty() || !factory) {
    return false;
  }
  for (size_t i = 0; i < signatures.size(); ++i) {
    if (!signatures[i]) {
      return false;
    }
    for (size_t j = 0; j < i; ++j) {
      if (*signatures[i] == *signatures[j]) {
        return false;
      }
    }
  }
  auto sanitizedName = sanitizeName(name);
  return vectorFunctionFactories().withWLock([&](auto& functions) {
    auto it = functions.find(sanitizedName);
    if (it != functions.end()) {
      for (const auto& entry : it->second) {
        if (entry.metadata.companionFunction != metadata.companionFunction) {
          return false;
        }
        for (const auto& existing : entry.signatures) {
          for (const auto& signature : signatures) {
            if (*existing == *signature) {
              return false;
            }
          }
        }
      }
    }
    VectorFunctionEntry entry{std::move(signatures), std::move(factory), metadata};
    if (it != functions.end()) {
      it->second.push_back(std::move(entry));
    } else {
      // Build the nonempty group before insertion, including on allocation failure.
      std::vector<VectorFunctionEntry> entries;
      entries.push_back(std::move(entry));
      functions.emplace(sanitizedName, std::move(entries));
    }
    return true;
  });
}

// Returns true iff an insertion actually happened
bool registerVectorFunction(
    std::string_view name,
    std::vector<FunctionSignaturePtr> signatures,
    std::unique_ptr<VectorFunction> func,
    VectorFunctionMetadata metadata,
    bool overwrite) {
  std::shared_ptr<VectorFunction> sharedFunc = std::move(func);
  auto factory = [sharedFunc](
                     const auto& /*name*/,
                     const auto& /*vectorArg*/,
                     const auto& /*config*/) { return sharedFunc; };
  return registerStatefulVectorFunction(
      name, signatures, factory, metadata, overwrite);
}

} // namespace facebook::velox::exec
