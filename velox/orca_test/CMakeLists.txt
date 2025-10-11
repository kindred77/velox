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

add_compile_options(-DUSE_CMAKE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")

include_directories(${XERCES_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third_libs/mygporca/libgpos/include)
include_directories(${PROJECT_SOURCE_DIR}/third_libs/mygporca/libgpopt/include)
include_directories(${PROJECT_SOURCE_DIR}/third_libs/mygporca/libmeta)
include_directories(${PROJECT_SOURCE_DIR}/third_libs/mygporca/libparser)
include_directories(${PROJECT_SOURCE_DIR}/third_libs/mygporca/pgquery/include)
include_directories(${PROJECT_SOURCE_DIR}/third_libs/mygporca/libtranslator)
include_directories(${PROJECT_SOURCE_DIR}/third_libs/mygporca/libnaucrates/include)
include_directories(${PROJECT_SOURCE_DIR}/third_libs/mygporca/libgpopt/include)

add_executable(OrcaTest MyGPOrcaTest.cpp)

target_link_libraries(OrcaTest
        gpopt
        naucrates
        gpos
        parser
        translator
        meta_provider
        mygporca_pg_query)
