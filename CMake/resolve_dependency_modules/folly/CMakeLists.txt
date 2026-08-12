# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
project(Folly)
cmake_minimum_required(VERSION 3.28)

velox_set_source(FastFloat)
velox_resolve_dependency(FastFloat CONFIG REQUIRED)

set(VELOX_FOLLY_BUILD_VERSION v2026.01.05.00)
set(
  VELOX_FOLLY_BUILD_SHA256_CHECKSUM
  8b41494b664fcde3d02f652d6a27381ec5e938ef58cf986d8ce4958d424af101
)
set(
  VELOX_FOLLY_SOURCE_URL
  "https://github.com/facebook/folly/releases/download/${VELOX_FOLLY_BUILD_VERSION}/folly-${VELOX_FOLLY_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(FOLLY)

message(STATUS "Building Folly from source")

if(gflags_SOURCE STREQUAL "BUNDLED")
  set(glog_patch && git apply ${CMAKE_CURRENT_LIST_DIR}/folly-gflags-glog.patch)
  # Together with the patch applied above prevents folly from test compiling a
  # snippet to find the right namespace (which would fail because gflags isn't
  # built yet)
  set(FOLLY_UNUSUAL_GFLAGS_NAMESPACE OFF)
  set(FOLLY_GFLAGS_NAMESPACE gflags)
endif()

FetchContent_Declare(
  folly
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../thirdpart_libs/folly_v2026.01.05.00
  #URL ${VELOX_FOLLY_SOURCE_URL}
  #URL_HASH ${VELOX_FOLLY_BUILD_SHA256_CHECKSUM}
  #PATCH_COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/folly-no-export.patch ${glog_patch}
  OVERRIDE_FIND_PACKAGE
  EXCLUDE_FROM_ALL
)

set(BUILD_SHARED_LIBS ${VELOX_BUILD_SHARED})

# Enable INT128 support
set(FOLLY_HAVE_INT128_T ON)

# vcpkg's glog (0.7+) requires consumers to define these macros, otherwise its
# headers fail with "#error <glog/logging.h> was not included correctly". The
# bundled folly build does not manage a glog::glog target that would carry them
# (its TU includes <glog/logging.h> directly, e.g. CancellationToken-inl.h), so
# provide them for the folly sub-build.
add_compile_definitions(GLOG_USE_GLOG_EXPORT GLOG_USE_GFLAGS)

FetchContent_MakeAvailable(folly)

# Folly::folly is not valid for FC but we want to match FindFolly
add_library(Folly::folly ALIAS folly)
add_library(Folly::follybenchmark ALIAS follybenchmark)

# With vcpkg (system) folly, the glog compile definitions reached velox targets
# transitively via Folly::folly -> glog::glog. The bundled folly target does not
# carry that link edge, so velox targets that include glog headers through folly
# (e.g. velox_encode) would compile without GLOG_USE_GLOG_EXPORT. Attach the
# same definitions to Folly::folly to keep behavior identical.
if(TARGET folly)
  set_property(
    TARGET folly
    APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS GLOG_USE_GLOG_EXPORT GLOG_USE_GFLAGS
  )
endif()

if(gflags_SOURCE STREQUAL "BUNDLED")
  add_dependencies(folly glog::glog gflags::gflags fmt::fmt)
endif()
