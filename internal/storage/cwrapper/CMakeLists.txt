# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

cmake_minimum_required( VERSION 3.18 )
project(wrapper)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set( ARROW_OUTPUT "${ARROW_BUILD}/output" )
set( ARROW_TERGET_VERSION "5.0.0" )

macro( build_arrow )
    if( NOT DEFINED ARROW_VERSION )
        set( ARROW_VERSION ${ARROW_TERGET_VERSION} )
    endif()

    message( STATUS "Building ARROW-${ARROW_VERSION} from source" )

    if ( DEFINED ENV{MILVUS_ARROW_URL} )
        set( ARROW_SOURCE_URL "$ENV{MILVUS_ARROW_URL}" )
    else ()
        set( ARROW_SOURCE_URL "https://github.com/apache/arrow/archive/apache-arrow-${ARROW_VERSION}.tar.gz" )
    endif ()

    set( ARROW_CMAKE_ARGS
        "-DARROW_PARQUET=ON"
        "-DARROW_COMPUTE=ON"
        "-DARROW_BUILD_STATIC=ON"
        "-DARROW_BUILD_SHARED=OFF"
        "-DARROW_BUILD_TESTS=OFF"
        "-DARROW_TEST_MEMCHECK=OFF"
        "-DARROW_BUILD_BENCHMARKS=OFF"
        "-DARROW_CUDA=OFF"
        "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
        )

    ExternalProject_Add(
            arrow_ep
            PREFIX              ${ARROW_BUILD}
            DOWNLOAD_DIR        ${THIRDPARTY_DOWNLOAD_PATH}
            INSTALL_DIR         ${ARROW_OUTPUT}
            URL                 ${ARROW_SOURCE_URL}
            URL_MD5             "9caf5dbd36ef4972c3a591bcfeaf59c8"
            CMAKE_ARGS          ${ARROW_CMAKE_ARGS}
            SOURCE_SUBDIR       "cpp"
            BUILD_IN_SOURCE     1
            BUILD_COMMAND       ${MAKE}
            INSTALL_COMMAND     ${MAKE} install
    )

    ExternalProject_Get_Property( arrow_ep INSTALL_DIR )
    if ( NOT IS_DIRECTORY ${INSTALL_DIR}/include )
        file( MAKE_DIRECTORY "${INSTALL_DIR}/include" )
    endif ()

    add_library( arrow_dependencies STATIC IMPORTED )
    set_target_properties( arrow_dependencies
            PROPERTIES
                IMPORTED_GLOBAL                 TRUE
                IMPORTED_LOCATION               ${INSTALL_DIR}/lib/libarrow_bundled_dependencies.a
                INTERFACE_INCLUDE_DIRECTORIES   ${INSTALL_DIR}/include )
    add_dependencies( arrow_dependencies arrow_ep )

    add_library( arrow STATIC IMPORTED )
    set_target_properties( arrow
            PROPERTIES
                IMPORTED_GLOBAL                 TRUE
                IMPORTED_LOCATION               ${INSTALL_DIR}/lib/libarrow.a
                INTERFACE_INCLUDE_DIRECTORIES   ${INSTALL_DIR}/include )
    add_dependencies( arrow arrow_ep arrow_dependencies )

    add_library( parquet STATIC IMPORTED )
    set_target_properties( parquet
            PROPERTIES
            IMPORTED_GLOBAL                 TRUE
            IMPORTED_LOCATION               ${INSTALL_DIR}/lib/libparquet.a
            INTERFACE_INCLUDE_DIRECTORIES   ${INSTALL_DIR}/include )
    add_dependencies( parquet arrow-ep )
    target_link_libraries( parquet INTERFACE arrow )

    get_target_property( ARROW_DEPENDENCIES_LIB arrow_dependencies IMPORTED_LOCATION )
    get_target_property( ARROW_STATIC_LIB arrow IMPORTED_LOCATION )
    get_target_property( ARROW_INCLUDE_DIR arrow INTERFACE_INCLUDE_DIRECTORIES )
    message( STATUS ${ARROW_INCLUDE_DIR} )
    set_directory_properties( PROPERTY INCLUDE_DIRECTORIES ${ARROW_INCLUDE_DIR} )

    get_target_property( PARQUET_STATIC_LIB parquet IMPORTED_LOCATION )

endmacro()

find_package( Arrow PATHS ${ARROW_OUTPUT} )
find_package( Parquet PATHS ${ARROW_OUTPUT}/lib/cmake/arrow )

if ( NOT DEFINED ${Arrow_FOUND} )
    build_arrow()
else ()
    set( ARROW_DEPENDENCIES_LIB arrow_dependencies ${ARROW_OUTPUT}/lib/libarrow_bundled_dependencies.a )
    if ( NOT DEFINED ${PARQUET_STATIC_LIB} )
        set( PARQUET_STATIC_LIB ${ARROW_OUTPUT}/lib/parquet.a )
    endif ()
endif ()

add_library( wrapper STATIC )
target_sources( wrapper PUBLIC ParquetWrapper.cpp PayloadStream.cpp )
set_target_properties( wrapper
            PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR} )
target_include_directories( wrapper PRIVATE ${ARROW_INCLUDE_DIR} )
target_link_libraries( wrapper PUBLIC parquet pthread )

if(NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
endif()

install( FILES ${ARROW_STATIC_LIB}
               ${ARROW_DEPENDENCIES_LIB}
               ${PARQUET_STATIC_LIB}
        DESTINATION lib )

install( TARGETS wrapper DESTINATION ${CMAKE_INSTALL_PREFIX} )

if (BUILD_TESTING)
    add_subdirectory(test)
endif()
