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
include_guard(GLOBAL)

set(VELOX_PROTOBUF_BUILD_VERSION 21.8)
set(
  VELOX_PROTOBUF_BUILD_SHA256_CHECKSUM
  83ad4faf95ff9cbece7cb9c56eb3ca9e42c3497b77001840ab616982c6269fb6
)
if(${VELOX_PROTOBUF_BUILD_VERSION} LESS 22.0)
  string(
    CONCAT
    VELOX_PROTOBUF_SOURCE_URL
    "https://github.com/protocolbuffers/protobuf/releases/download/"
    "v${VELOX_PROTOBUF_BUILD_VERSION}/protobuf-all-${VELOX_PROTOBUF_BUILD_VERSION}.tar.gz"
  )
else()
  velox_set_source(absl)
  velox_resolve_dependency(absl CONFIG REQUIRED)
  string(
    CONCAT
    VELOX_PROTOBUF_SOURCE_URL
    "https://github.com/protocolbuffers/protobuf/archive/"
    "v${VELOX_PROTOBUF_BUILD_VERSION}.tar.gz"
  )
endif()

velox_resolve_dependency_url(PROTOBUF)

message(STATUS "Building Protobuf from source")

FetchContent_Declare(
  protobuf
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../thirdpart_libs/protobuf-3.21.7
  #URL ${VELOX_PROTOBUF_SOURCE_URL}
  #URL_HASH ${VELOX_PROTOBUF_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE
  EXCLUDE_FROM_ALL
  SYSTEM
)

set(protobuf_BUILD_TESTS OFF)
set(protobuf_ABSL_PROVIDER "package")
# The bundled 3.21.7 protobuf calls find_package(ZLIB) and then puts
# ZLIB_INCLUDE_DIRECTORIES ahead of its own src/ in the directory include
# list. With vcpkg that path is the installed include root, so
# google/protobuf/*.h resolves to vcpkg protobuf 29.x headers while compiling
# the bundled sources (C++11 vs vcpkg absl's C++14+ requirement). Protobuf
# only uses zlib for gzip streams, which this engine does not need.
set(protobuf_WITH_ZLIB OFF CACHE BOOL "Build bundled protobuf without zlib" FORCE)
FetchContent_MakeAvailable(protobuf)
set(Protobuf_INCLUDE_DIRS ${protobuf_SOURCE_DIR}/src)

# protobuf's own targets propagate ${protobuf_SOURCE_DIR}/src as an INTERFACE
# include dir, which CMake emits as -isystem AFTER vcpkg's include dir. GCC then
# resolves google/protobuf/*.h from vcpkg's protobuf 29.3 headers instead of the
# bundled 3.21.7 headers, which shows up at link time as undefined
# 'ZeroCopyInputStream::ReadCord' (a virtual added in newer protobuf). The
# non-system include_directories(${Protobuf_INCLUDE_DIRS}) in velox CMakeLists
# already puts the bundled headers first; drop the duplicate interface includes.
foreach(_protobuf_target IN ITEMS libprotobuf libprotobuf-lite)
  if(TARGET ${_protobuf_target})
    set_property(TARGET ${_protobuf_target} PROPERTY INTERFACE_INCLUDE_DIRECTORIES "")
  endif()
endforeach()
unset(_protobuf_target)
