cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(CPP_Squeezenet_Classification LANGUAGES CXX)

set (TARGET_NAME "run_squeezenet")

#Include ONNXRuntime C++ Headers
set(ONNXRT_INCLUDE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE STRING "ONNXRT_INCLUDE_HEADERS")
include_directories(${ONNXRT_INCLUDE_HEADERS})

#Find InferenceEngine component
find_package(InferenceEngine)

#Find OpenCV component
find_package(OpenCV REQUIRED core dnn imgcodecs imgproc)
message("Found OpenCV version: " ${OpenCV_VERSION})
include_directories(${OpenCV_INCLUDE_DIRS})

if(NOT InferenceEngine_FOUND)
        message("[ERROR] InferenceEngine component was not detected.")
endif()

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet_cpp_app.cpp
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime_cxx_api.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime_cxx_inline.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime_c_api.h
        )

# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

target_link_libraries(${TARGET_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/ort-Libraries/libonnxruntime_providers_openvino.so
    ${CMAKE_CURRENT_SOURCE_DIR}/ort-Libraries/libonnxruntime_providers_shared.so
    ${CMAKE_CURRENT_SOURCE_DIR}/ort-Libraries/libonnxruntime.so.1.11.0
    ${OpenCV_LIBS}
)
