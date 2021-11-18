cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(warp_perspective)

find_package(Torch REQUIRED)
find_path(DLPRIM_INC dlprim/context.hpp HINT /opt/dlprim/include)
find_library(DLPRIM_LIB dlprim_core HINT /opt/dlprim/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")
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
include_directories(${DLPRIM_INC})
target_link_libraries(pt_ocl "${TORCH_LIBRARIES}" OpenCL ${DLPRIM_LIB})
