#-------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------
set( ARROW_OUTPUT "${ARROW_BUILD}/output" )
set( ARROW_TERGET_VERSION "5.0.0" )

macro( build_arrow )
    message( STATUS "Building ARROW-${ARROW_VERSION} from source" )

    if( NOT DEFINED ARROW_VERSION )
        set( ARROW_VERSION ${ARROW_TERGET_VERSION} )
    endif()

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
    endif()

    add_library( arrow_bundled_dependencies STATIC IMPORTED )
    set_target_properties( arrow_bundled_dependencies 
            PROPERTIES
                IMPORTED_GLOBAL                 TRUE
                IMPORTED_LOCATION               ${INSTALL_DIR}/lib/libarrow_bundled_dependencies.a
                INTERFACE_INCLUDE_DIRECTORIES   ${INSTALL_DIR}/include )
    add_dependencies( arrow_bundled_dependencies arrow_ep )

    add_library( arrow STATIC IMPORTED )
    set_target_properties( arrow
            PROPERTIES
                IMPORTED_GLOBAL                 TRUE
                IMPORTED_LOCATION               ${INSTALL_DIR}/lib/libarrow.a
                INTERFACE_INCLUDE_DIRECTORIES   ${INSTALL_DIR}/include )
    add_dependencies( arrow arrow_ep )
    target_link_libraries( arrow INTERFACE arrow_bundled_dependencies )

endmacro()

set( Arrow_DIR ${ARROW_OUTPUT}/lib/cmake/arrow )
find_package( Arrow )

if ( ${Arrow_FOUND} )
    message( STATUS "Found ARROW-${ARROW_VERSION}" )
    add_library( arrow ALIAS arrow_static )
else ()
    build_arrow()
endif ()

get_target_property( var arrow INTERFACE_INCLUDE_DIRECTORIES )
message( STATUS ${var} )
set_directory_properties( PROPERTY INCLUDE_DIRECTORIES ${var} )
