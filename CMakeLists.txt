cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(dlprim_backend)

include(cmake/CPM.cmake)

CPMAddPackage(
    NAME opencl_headers
    GITHUB_REPOSITORY KhronosGroup/OpenCL-Headers
    GIT_TAG v2024.10.24
)

CPMAddPackage(
    NAME opencl_clhpp
    GITHUB_REPOSITORY KhronosGroup/OpenCL-CLHPP
    GIT_TAG v2024.10.24
    OPTIONS "BUILD_EXAMPLES OFF" "BUILD_TESTING OFF"
)

set(OCL_PATH ${opencl_headers_SOURCE_DIR} ${opencl_clhpp_SOURCE_DIR}/include CACHE PATH "Path to OpenCL headers")
find_library(OCL_LIB OpenCL REQUIRED)

CPMAddPackage(
    NAME dlprimitives
    GITHUB_REPOSITORY artyom-beilis/dlprimitives
    GIT_TAG master
    OPTIONS
        "USE_ONNX OFF"
        "BUILD_CORE_ONLY ON"
        "BUILD_EXAMPLES OFF"
        "BUILD_TESTING OFF"
        "DLPRIM_STATIC_BUILD ON"
)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch 1.13 REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
option(BUILD_CORE_ONLY "Core Build" ON)
option(DLPRIM_STATIC_BUILD "Build static version only" ON)
option(BUILD_CPLUS_EXAMPLE "Build C++ examples" OFF)

message(" Pytorch libraries: ${TORCH_LIBRARIES}")
message(" Python libraries: ${Python3_LIBRARIES}")
message(" Python include: ${Python3_INCLUDE_DIRS}")

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
        src/guard.cpp
        src/python_iface.cpp
        )

target_compile_features(pt_ocl PRIVATE cxx_std_17)

# Link against LibTorch
target_link_libraries(pt_ocl ${TORCH_LIBRARIES} ${OCL_LIB} dlprim_core ${Python3_LIBRARIES})
target_include_directories(pt_ocl PRIVATE 
    "${OCL_PATH}"
    "${dlprimitives_SOURCE_DIR}/include"
    "${dlprimitives_BINARY_DIR}"
    "${Python3_INCLUDE_DIRS}"
    "${pybind11_INCLUDE_DIRS}"
)
if(WIN32)
    target_link_libraries(pt_ocl ${Python3_LIBRARIES})
endif()
set_target_properties(pt_ocl PROPERTIES PREFIX "" OUTPUT_NAME "pt_ocl")
if(NOT WIN32)
	set_target_properties(pt_ocl PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pytorch_ocl/")
else()
	set_target_properties(pt_ocl PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pytorch_ocl/"
		SUFFIX ".pyd"
	)
	 
endif()

install(TARGETS pt_ocl DESTINATION pytorch_ocl)

# Add C++ examples
if(BUILD_CPLUS_EXAMPLE)
    add_subdirectory(examples/cplus)
endif()