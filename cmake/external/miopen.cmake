# Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
include(ExternalProject)

set(MIOPEN_PROJECT       extern_miopen)
set(MIOPEN_PREFIX_DIR    ${ANAKIN_TEMP_THIRD_PARTY_PATH}/miopen)
set(MIOPEN_INSTALL_ROOT  ${ANAKIN_THIRD_PARTY_PATH}/miopen)
set(MIOPEN_SOURCE_DIR    ${MIOPEN_PREFIX_DIR}/src/${MIOPEN_PROJECT})
set(MIOPEN_BINARY_DIR    ${MIOPEN_PREFIX_DIR}/src/${MIOPEN_PROJECT}-build)
set(MIOPEN_LIB           ${MIOPEN_INSTALL_ROOT}/lib/libMIOpen.so  CACHE FILEPATH "miopen library." FORCE)

if(NOT Boost_FOUND)
    set(BOOST_ROOT       ${BOOST_INSTALL_ROOT} CACHE FILEPATH "boost library/" FORCE)
endif()

message(STATUS "Scanning external modules ${Green}MIOPEN${ColourReset} ...")

ExternalProject_Add(
    ${MIOPEN_PROJECT}_customize
    GIT_REPOSITORY        "ssh://git@icode.baidu.com:8235/baidu/third-party/miopen"
    GIT_TAG               "cbd4e7dbad0599c7327cb43888476ab8d966f285"
    PREFIX                ${ANAKIN_TEMP_THIRD_PARTY_PATH}/miopen/customize_miopen_file
    SOURCE_DIR            ${ANAKIN_THIRD_PARTY_PATH}/miopen/customize_miopen_file
    CONFIGURE_COMMAND     ""
    BUILD_COMMAND         ""
    INSTALL_COMMAND       ""
)

ExternalProject_Add(
    ${MIOPEN_PROJECT}
    DEPENDS               ${MIOPEN_PROJECT}_customize
    GIT_REPOSITORY        "ssh://git@icode.baidu.com:8235/baidu/third-party/miopen"
    GIT_TAG               1.4.2
    PREFIX                ${MIOPEN_PREFIX_DIR}
    CMAKE_ARGS            -DMIOPEN_BACKEND=OpenCL -DCMAKE_INSTALL_PREFIX=${MIOPEN_INSTALL_ROOT} -DCMAKE_INSTALL_LIBDIR=lib -DBOOST_ROOT=${BOOST_ROOT}
    #LOG_DOWNLOAD          1
    LOG_BUILD             1
    
)

ExternalProject_Add_Step(
    ${MIOPEN_PROJECT} ${MIOPEN_PROJECT}_customize
    DEPENDEES         download
    DEPENDERS         build
    COMMAND           ${CMAKE_COMMAND} -E copy_directory ${ANAKIN_THIRD_PARTY_PATH}/miopen/customize_miopen_file ${MIOPEN_SOURCE_DIR}
    ALWAYS            1
    EXCLUDE_FORM_MAIN 1
    LOG               1
)
include_directories(${MIOPEN_INSTALL_ROOT}/include)
add_library(miopen SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET miopen PROPERTY IMPORTED_LOCATION ${MIOPEN_LIB})
add_dependencies(miopen ${MIOPEN_PROJECT})

list(APPEND ANAKIN_SABER_DEPENDENCIES miopen)
list(APPEND ANAKIN_LINKER_LIBS ${MIOPEN_LIB})
