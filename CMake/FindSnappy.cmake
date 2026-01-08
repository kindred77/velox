# Copyright (c) Facebook, Inc. and its affiliates.
# - Try to find snappy
# Once done, this will define
#
# SNAPPY_FOUND - system has Glog
# SNAPPY_INCLUDE_DIRS - deprecated
# SNAPPY_LIBRARIES -  deprecated
# Snappy::snappy will be defined based on CMAKE_FIND_LIBRARY_SUFFIXES priority

include(FindPackageHandleStandardArgs)
include(SelectLibraryConfigurations)

find_library(SNAPPY_LIBRARY_RELEASE snappy PATHS ${SNAPPY_LIBRARYDIR})
find_library(SNAPPY_LIBRARY_DEBUG snappyd PATHS ${SNAPPY_LIBRARYDIR})

find_path(SNAPPY_INCLUDE_DIR snappy.h PATHS ${SNAPPY_INCLUDEDIR})

select_library_configurations(SNAPPY)

find_package_handle_standard_args(Snappy DEFAULT_MSG SNAPPY_LIBRARY SNAPPY_INCLUDE_DIR)

mark_as_advanced(SNAPPY_LIBRARY SNAPPY_INCLUDE_DIR)

#get_filename_component(libsnappy_ext ${SNAPPY_LIBRARY} EXT)
#if(libsnappy_ext STREQUAL ".a")
#  set(libsnappy_type STATIC)
#else()
#  set(libsnappy_type SHARED)
#endif()

# would be "optimized;xxx/vcpkg/installed/x64-windows/lib/snappy.lib;debug;xxx/vcpkg/installed/x64-windows/debug/lib/snappyd.lib" in vcpkg on Win
# TODO: temporary workaround for vcpkg
set(SNAPPY_LIBS_FORSEARCH ${SNAPPY_LIBRARY})
list(FILTER SNAPPY_LIBS_FORSEARCH INCLUDE REGEX ".*snappy\\.lib.*")
if(SNAPPY_LIBS_FORSEARCH)
  message(STATUS "Using snappy.lib, may be in Windows: ${SNAPPY_LIBS_FORSEARCH}")
  set(libsnappy_type STATIC)
  set(SNAPPY_LIBRARY ${SNAPPY_LIBS_FORSEARCH})
  message(STATUS "snappy.lib in vcpkg has been set to: ${SNAPPY_LIBRARY}")
else()
  get_filename_component(libsnappy_ext ${SNAPPY_LIBRARY} EXT)
  if(libsnappy_ext STREQUAL ".a")
    set(libsnappy_type STATIC)
  else()
    set(libsnappy_type SHARED)
  endif()
endif()

if(NOT TARGET Snappy::snappy)
  add_library(Snappy::snappy ${libsnappy_type} IMPORTED)
  set_target_properties(
    Snappy::snappy
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${SNAPPY_INCLUDE_DIR}"
  )
  #set_target_properties(
  #  Snappy::snappy
  #  PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C" IMPORTED_LOCATION "${SNAPPY_LIBRARIES}"
  #)

  # Set the locations for different configurations
    if(SNAPPY_LIBRARY_RELEASE)
        set_target_properties(Snappy::snappy PROPERTIES
            IMPORTED_LOCATION_RELEASE "${SNAPPY_LIBRARY_RELEASE}"
        )
    endif()
    if(SNAPPY_LIBRARY_DEBUG)
        set_target_properties(Snappy::snappy PROPERTIES
            IMPORTED_LOCATION_DEBUG "${SNAPPY_LIBRARY_DEBUG}"
        )
    endif()
    # Fallback for single-config generators (like Makefiles)
    if(SNAPPY_LIBRARY_RELEASE AND NOT SNAPPY_LIBRARY_DEBUG)
        set_target_properties(Snappy::snappy PROPERTIES
            IMPORTED_LOCATION "${SNAPPY_LIBRARY_RELEASE}"
        )
    endif()
endif()

# Set the SNAPPY_LIBRARIES variable for compatibility
if(SNAPPY_FOUND)
    set(SNAPPY_LIBRARIES Snappy::snappy)
endif()
