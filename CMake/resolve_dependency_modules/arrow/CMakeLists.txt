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
project(Arrow)

if(VELOX_ENABLE_ARROW)
  # Avoid issues in finding the boost headers and libraries
  # by setting the BOOST_ROOT to the install prefix.
  # The same logic is used in the setup script-common.sh to install boost.
  if(NOT DEFINED ENV{BOOST_ROOT} OR "$ENV{BOOST_ROOT}" STREQUAL "")
    if(DEFINED ENV{INSTALL_PREFIX} AND NOT "$ENV{INSTALL_PREFIX}" STREQUAL "")
      set(BOOST_ROOT "$ENV{INSTALL_PREFIX}")
    else()
      set(BOOST_ROOT "/usr/local")
    endif()
  else()
    set(BOOST_ROOT "$ENV{BOOST_ROOT}")
  endif()
  message(STATUS "Using BOOST_ROOT: ${BOOST_ROOT}")
  set(ARROW_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/arrow_ep")
  set(
    ARROW_CMAKE_ARGS
    -DARROW_PARQUET=OFF
    -DARROW_DEPENDENCY_SOURCE=AUTO
    # vcpkg static triplets (x64-linux / x64-windows-static-md) only ship
    # static zstd/lz4/snappy/zlib, but Arrow defaults ARROW_DEPENDENCY_USE_SHARED
    # to ON and then fails with "Zstandard target doesn't exist:
    # zstd::libzstd_shared". Force static dependencies to match. The per-dep
    # flags are set explicitly because a stale arrow_ep-build CMakeCache keeps
    # the old ON defaults even when ARROW_DEPENDENCY_USE_SHARED changes.
    -DARROW_DEPENDENCY_USE_SHARED=OFF
    -DARROW_ZSTD_USE_SHARED=OFF
    -DARROW_LZ4_USE_SHARED=OFF
    -DARROW_SNAPPY_USE_SHARED=OFF
    -DARROW_ZLIB_USE_SHARED=OFF
    -DARROW_WITH_LZ4=ON
    -DARROW_WITH_SNAPPY=ON
    -DARROW_WITH_ZLIB=ON
    -DARROW_WITH_ZSTD=ON
    -DARROW_JEMALLOC=OFF
    -DARROW_SIMD_LEVEL=NONE
    -DARROW_RUNTIME_SIMD_LEVEL=NONE
    -DARROW_WITH_UTF8PROC=OFF
    # Velox's own tests are disabled in this project (VELOX_BUILD_TESTING=OFF),
    # so arrow_testing is not needed. Building it with vcpkg Boost 1.87+ fails
    # because arrow 18's arrow/testing/process.cc uses the old boost::process
    # API, which Boost 1.88 exposes only under boost/process/v1.
    -DARROW_TESTING=OFF
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_INSTALL_PREFIX=${ARROW_PREFIX}/install
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DARROW_BUILD_STATIC=ON
    -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
    -DBOOST_ROOT=${BOOST_ROOT}
    -DBoost_NO_BOOST_CMAKE=TRUE
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    # Remove with Arrow upgrade to Arrow 20.
    -DARROW_CXXFLAGS=-Wno-documentation
    # Arrow inherits the CMake build type from Velox. So a Debug build turns
    # on additional warnings that can cause the build to fail
    # (e.g. deprecated-literal-operator in Apple Clang 21).
    # The SYSTEM version of Arrow is always a Release build.
    -DBUILD_WARNING_LEVEL=PRODUCTION
  )
  set(ARROW_LIBDIR ${ARROW_PREFIX}/install/lib)

  set(VELOX_ARROW_BUILD_VERSION 18.0.0)
  set(
    VELOX_ARROW_BUILD_SHA256_CHECKSUM
    9c473f2c9914c59ab571761c9497cf0e5cfd3ea335f7782ccc6121f5cb99ae9b
  )
  set(
    VELOX_ARROW_SOURCE_URL
    "https://github.com/apache/arrow/archive/refs/tags/apache-arrow-${VELOX_ARROW_BUILD_VERSION}.tar.gz"
  )

  velox_resolve_dependency_url(ARROW)

  ExternalProject_Add(
    arrow_ep
    PREFIX ${ARROW_PREFIX}
    URL ${VELOX_ARROW_SOURCE_URL}
    URL_HASH ${VELOX_ARROW_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR cpp
    CMAKE_ARGS ${ARROW_CMAKE_ARGS}
    # When ARROW_TESTING is re-enabled, restore the byproduct below:
    # BUILD_BYPRODUCTS ${ARROW_LIBDIR}/libarrow.a ${ARROW_LIBDIR}/libarrow_testing.a
    BUILD_BYPRODUCTS ${ARROW_LIBDIR}/libarrow.a
    PATCH_COMMAND
      patch -p1 -i ${CMAKE_CURRENT_LIST_DIR}/cmake-compatibility.patch && patch -p1 -i
      ${CMAKE_CURRENT_LIST_DIR}/arrow-testing-boost.patch
  )

  add_library(arrow STATIC IMPORTED GLOBAL)
#  add_library(arrow_testing STATIC IMPORTED GLOBAL)
  add_dependencies(arrow arrow_ep)
#  add_dependencies(arrow_testing arrow)
  file(MAKE_DIRECTORY ${ARROW_PREFIX}/install/include)
  set_target_properties(
      arrow
#      arrow_testing
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ARROW_PREFIX}/install/include
  )
  set_target_properties(arrow PROPERTIES IMPORTED_LOCATION ${ARROW_LIBDIR}/libarrow.a)
  set_property(TARGET arrow PROPERTY INTERFACE_LINK_LIBRARIES ${RE2})
  # Mirror the path-based interface dep from CMake/FindArrow.cmake — see
  # the SYSTEM-arrow comment there for the ODR-violation rationale that
  # forces us to use ${ARROW_LIBDIR}/libarrow.a (path) instead of the
  # `arrow` target name.
#  set_target_properties(
#    arrow_testing
#    PROPERTIES
#      IMPORTED_LOCATION ${ARROW_LIBDIR}/libarrow_testing.a
#      INTERFACE_LINK_LIBRARIES ${ARROW_LIBDIR}/libarrow.a
#  )
endif()
