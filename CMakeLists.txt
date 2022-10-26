cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(dlprim_backend)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
option(BUILD_CORE_ONLY "Core Build" ON)
add_subdirectory(dlprimitives)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dlprimitives)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dlprimitives/include)

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
target_compile_features(pt_ocl PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(pt_ocl "${TORCH_LIBRARIES}" ${OCL_LIB} dlprim_core)

