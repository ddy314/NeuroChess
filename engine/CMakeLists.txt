cmake_minimum_required(VERSION 3.24)
project(NeuroChess LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

option(NEUROCHESS_NATIVE_ARCH "Compile the search with -march=native" ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Point TENSORRT_ROOT at a TensorRT 10+/11 SDK (include/ and lib/), or set
# TENSORRT_INCLUDE_DIR / TENSORRT_LIB_DIR individually.
find_path(TENSORRT_INCLUDE_DIR NvInferRuntime.h
  HINTS "${TENSORRT_ROOT}/include" REQUIRED)
find_library(TENSORRT_LIBRARY NAMES nvinfer libnvinfer.so.11 libnvinfer.so.10
  HINTS "${TENSORRT_LIB_DIR}" "${TENSORRT_ROOT}/lib" REQUIRED)
find_library(TENSORRT_PLUGIN_LIBRARY NAMES nvinfer_plugin libnvinfer_plugin.so.11 libnvinfer_plugin.so.10
  HINTS "${TENSORRT_LIB_DIR}" "${TENSORRT_ROOT}/lib" REQUIRED)

add_library(nc_inference STATIC src/trt_inference.cu)
target_include_directories(nc_inference SYSTEM PRIVATE "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(nc_inference PUBLIC "${TENSORRT_LIBRARY}" "${TENSORRT_PLUGIN_LIBRARY}" CUDA::cudart)
target_compile_options(nc_inference PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

add_executable(neurochess src/main.cpp)
target_include_directories(neurochess PRIVATE src third_party/chess-library)
target_link_libraries(neurochess PRIVATE nc_inference Threads::Threads)
if(NEUROCHESS_NATIVE_ARCH)
  target_compile_options(neurochess PRIVATE -march=native)
endif()

install(TARGETS neurochess RUNTIME DESTINATION bin)
