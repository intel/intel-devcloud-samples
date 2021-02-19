# Copyright (c) 2017 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 2.8)

project(object-detection)

set (TARGET_NAME "object-detection-cpp")


if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

#Set up of Output directory to this format "*/build/intel64/(Release or Debug)"

set (ARCH "intel64")

set (BIN_FOLDER ${ARCH})


set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/${BIN_FOLDER})
set (LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/${BIN_FOLDER})
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/${BIN_FOLDER})
set (CMAKE_LIBRARY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build/${BIN_FOLDER})
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/${BIN_FOLDER})
set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/${BIN_FOLDER})
set (CMAKE_PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/${BIN_FOLDER})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/${BIN_FOLDER})
set (LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/${BIN_FOLDER})

#InferenceEngine_DIR needs to be the path where inferenceengineconfig.cmake file is.

set(InferenceEngine_DIR "${INTEL_OPENVINO_DIR}/inference_engine/share" CACHE STRING "InferenceEngine_DIR")
message(STATUS "InferenceEngine_DIR is set to ${InferenceEngine_DIR}")

find_package(InferenceEngine)

find_package(ngraph REQUIRED)

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/object_detection_sample_ssd.h
        )

# Create named folders for the sources within the .vcproj
source_group("src" FILES ${MAIN_SRC})

set(InferenceEngine_INCLUDE_DIRS "${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/include" CACHE STRING "InferenceEngine_INCLUDE_DIRS")


include_directories(${InferenceEngine_INCLUDE_DIRS})

set(common_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common" CACHE STRING "common_DIR")


include_directories(${common_DIR})
include_directories(${common_DIR}/format_reader)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
	set(IE_LIB_FOLDER "${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/Release" CACHE STRING "IE_LIB_FOLDER")
else()
	set(IE_LIB_FOLDER "${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/Debug" CACHE STRING "IE_LIB_FOLDER")
endif()


# Create library file from sources.

add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})


add_subdirectory(common/format_reader)


add_dependencies(${TARGET_NAME} format_reader)


set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")

#just added with linux
include_directories("opt/intel/openvino/opencv/include")
set(OpenCV_INCLUDE_DIRS "${INTEL_OPENVINO_DIR}/opencv/include" CACHE STRING "OpenCV_INCLUDE_DIRS")
include_directories( ${OpenCV_INCLUDE_DIRS} )

#Learning --> Just make sure that all O's in OpenCV are capitalized
set (OpenCV_DIR "${INTEL_OPENVINO_DIR}/opencv/cmake" CACHE STRING "OpenCV_DIR") 
find_package(OpenCV)

##set (OpenCV_LIBRARIES "${INTEL_OPENVINO_DIR}/opencv/lib"  CACHE STRING "OpenCV_LIBRARIES")

include(CMakeParseArguments)

#Link all libraries
##target_link_libraries(${TARGET_NAME} IE::ie_cpu_extension ${IE_LIB_FOLDER}/*   ${OpenCV_LIBRARIES}/* format_reader)
##target_link_libraries(${TARGET_NAME} IE::ie_cpu_extension ${IE_LIB_FOLDER}/*   ${OpenCV_LIBRARIES}/* format_reader)
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES} format_reader gflags ngraph::ngraph)


