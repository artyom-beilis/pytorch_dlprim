cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(warp_perspective)

find_package(Torch REQUIRED)
find_path(DLPRIM_INC dlprim/context.hpp HINT /opt/dlprim/include)
find_library(DLPRIM_LIB dlprim_core HINT /opt/dlprim/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")
# Define our library target
add_library(pt_ocl SHARED op.cpp)
# Enable C++14
target_compile_features(pt_ocl PRIVATE cxx_std_14)
# Link against LibTorch
include_directories(${DLPRIM_INC})
target_link_libraries(pt_ocl "${TORCH_LIBRARIES}" OpenCL ${DLPRIM_LIB})
