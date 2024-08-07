cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(dlprim_backend)

find_package(Torch 1.13 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
option(BUILD_CORE_ONLY "Core Build" ON)
add_subdirectory(dlprimitives)
include_directories("${OCL_PATH}")
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dlprimitives)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dlprimitives/include)
if(NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -Wall")
endif()

# Define our library target
add_library(pt_ocl SHARED 
        src/pointwise_ops.cpp
        src/tensor_ops.cpp
        src/vision_ops.cpp
        src/loss_ops.cpp
        src/random_ops.cpp
        src/norm_ops.cpp
        src/utils.cpp
        src/registeration.cpp
        src/CLTensor.cpp
        src/guard.cpp)
# Enable C++14
if(NOT WIN32)
	target_compile_features(pt_ocl PRIVATE cxx_std_14)
endif()	
# Link against LibTorch
target_link_libraries(pt_ocl "${TORCH_LIBRARIES}" ${OCL_LIB} dlprim_core)

