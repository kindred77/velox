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
#
# - Try to find zstd
# Once done, this will define
#
# ZSTD_FOUND - system has Glog
# ZSTD_INCLUDE_DIRS - deprecated
# ZSTD_LIBRARIES -  deprecated
# zstd::zstd will be defined based on CMAKE_FIND_LIBRARY_SUFFIXES priority

include(FindPackageHandleStandardArgs)
include(SelectLibraryConfigurations)

find_library(ZSTD_LIBRARY_RELEASE zstd PATHS ${ZSTD_LIBRARYDIR})
find_library(ZSTD_LIBRARY_DEBUG zstdd PATHS ${ZSTD_LIBRARYDIR})

find_path(ZSTD_INCLUDE_DIR zstd.h PATHS ${ZSTD_INCLUDEDIR})

select_library_configurations(ZSTD)

find_package_handle_standard_args(zstd DEFAULT_MSG ZSTD_LIBRARY ZSTD_INCLUDE_DIR)

mark_as_advanced(ZSTD_LIBRARY ZSTD_INCLUDE_DIR)

#get_filename_component(libzstd_ext ${ZSTD_LIBRARY} EXT)
#if(libzstd_ext STREQUAL ".a")
#  set(libzstd_type STATIC)
#else()
#  set(libzstd_type SHARED)
#endif()

# would be "optimized;xxx/vcpkg/installed/x64-windows/lib/zstd.lib;debug;xxx/vcpkg/installed/x64-windows/debug/lib/zstdd.lib" in vcpkg on Win
# TODO: temporary workaround for vcpkg
set(ZSTD_LIBS_FORSEARCH ${ZSTD_LIBRARY})
list(FILTER ZSTD_LIBS_FORSEARCH INCLUDE REGEX ".*zstd\\.lib.*")
if(ZSTD_LIBS_FORSEARCH)
  message(STATUS "Using zstd.lib, may be in Windows: ${ZSTD_LIBS_FORSEARCH}")
  set(libzstd_type STATIC)
  set(ZSTD_LIBRARY ${ZSTD_LIBS_FORSEARCH})
  message(STATUS "zstd.lib in vcpkg has been set to: ${ZSTD_LIBRARY}")
else()
  get_filename_component(libzstd_ext ${ZSTD_LIBRARY} EXT)
  if(libzstd_ext STREQUAL ".a")
    set(libzstd_type STATIC)
  else()
    set(libzstd_type SHARED)
  endif()
endif()

if(NOT TARGET zstd::zstd)
  add_library(zstd::zstd ${libzstd_type} IMPORTED)
  set_target_properties(zstd::zstd PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}")
  #set_target_properties(
  #  zstd::zstd
  #  PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C" IMPORTED_LOCATION "${ZSTD_LIBRARIES}"
  #)


  # Set the locations for different configurations
    if(ZSTD_LIBRARY_RELEASE)
        set_target_properties(zstd::zstd PROPERTIES
            IMPORTED_LOCATION_RELEASE "${ZSTD_LIBRARY_RELEASE}"
        )
    endif()
    if(ZSTD_LIBRARY_DEBUG)
        set_target_properties(zstd::zstd PROPERTIES
            IMPORTED_LOCATION_DEBUG "${ZSTD_LIBRARY_DEBUG}"
        )
    endif()
    # Fallback for single-config generators (like Makefiles)
    if(ZSTD_LIBRARY_RELEASE AND NOT ZSTD_LIBRARY_DEBUG)
        set_target_properties(zstd::zstd PROPERTIES
            IMPORTED_LOCATION "${ZSTD_LIBRARY_RELEASE}"
        )
    endif()
endif()

# Set the ZSTD_LIBRARIES variable for compatibility
if(ZSTD_FOUND)
    set(ZSTD_LIBRARIES zstd::zstd)
endif()
