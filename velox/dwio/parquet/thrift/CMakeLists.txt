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

include(FBThriftCppLibrary)

add_fbthrift_cpp_library(velox_dwio_parquet_thrift_raw parquet.thrift)
add_library(velox_dwio_parquet_thrift INTERFACE)
target_link_libraries(velox_dwio_parquet_thrift INTERFACE velox_dwio_parquet_thrift_raw)
if(WIN32)
  # Patch generated thrift header to undefine Windows OPTIONAL macro
  # which conflicts with parquet's FieldRepetitionType::OPTIONAL enum value
  set(PARQUET_TYPES_H "${CMAKE_CURRENT_BINARY_DIR}/gen-cpp2/parquet_types.h")
  set(PARQUET_PATCH_MARKER "${CMAKE_CURRENT_BINARY_DIR}/gen-cpp2/.optional_patched")
  add_custom_command(OUTPUT "${PARQUET_PATCH_MARKER}"
    COMMAND ${CMAKE_COMMAND}
      -D PATCH_FILE="${PARQUET_TYPES_H}"
      -D MARKER="${PARQUET_PATCH_MARKER}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/../../../../CMake/third-party/undef_optional.cmake"
    DEPENDS "${PARQUET_TYPES_H}"
    COMMENT "[Win32] Patching parquet_types.h to undefine OPTIONAL macro"
  )
  add_custom_target(velox_parquet_optional_patch ALL
    DEPENDS "${PARQUET_PATCH_MARKER}"
  )
  add_dependencies(velox_dwio_parquet_thrift_raw velox_parquet_optional_patch)
endif()
velox_add_test_headers(
  velox_dwio_parquet_thrift
  CompactV1ProtocolReaderWithRefill.h
  ParquetThrift.h
)
if(VELOX_MONO_LIBRARY)
  target_link_libraries(velox velox_dwio_parquet_thrift_raw)
endif()

velox_install_library_headers()
get_target_property(generated_headers velox_dwio_parquet_thrift_raw PUBLIC_HEADER)
get_target_property(header_install_dir velox_dwio_parquet_thrift_raw HEADER_INSTALL_DIR)
install(FILES ${generated_headers} DESTINATION ${header_install_dir})
