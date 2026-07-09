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
  # NOTE: Must run at build time (POST_BUILD), not configure time,
  # because the thrift compiler generates parquet_types.h at build time.
  set(PARQUET_TYPES_H "${CMAKE_CURRENT_BINARY_DIR}/gen-cpp2/parquet_types.h")
  set(PATCH_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/patch_parquet_optional.cmake")
  # Generate a cmake script that runs at build time via cmake -P.
  # Use bracket form + placeholder replacement to avoid CMake configure-time
  # expansion of ${_content} (must remain a runtime variable reference).
  set(__patch_content [[
file(READ "@PARQUET_TYPES_PATH@" _content)
string(REPLACE "// BEGIN declare_enums"
  "#ifdef OPTIONAL\n#undef OPTIONAL\n#endif\n// BEGIN declare_enums"
  _content "${_content}")
file(WRITE "@PARQUET_TYPES_PATH@" "${_content}")
]])
  string(REPLACE "@PARQUET_TYPES_PATH@" "${PARQUET_TYPES_H}" __patch_content "${__patch_content}")
  file(WRITE "${PATCH_SCRIPT}" "${__patch_content}")
  add_custom_command(TARGET velox_dwio_parquet_thrift_raw POST_BUILD
    COMMAND ${CMAKE_COMMAND} -P "${PATCH_SCRIPT}"
    COMMENT "[Win32] Patching parquet_types.h to undefine OPTIONAL macro"
  )
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
