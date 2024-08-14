cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(dlprim_backend)
find_package(PythonLibs 3 REQUIRED)
find_package(Torch 1.13 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
option(BUILD_CORE_ONLY "Core Build" ON)
add_subdirectory(dlprimitives)
include_directories("${OCL_PATH}")
include_directories(${PYTHON_INCLUDE_DIR})
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
        src/guard.cpp
        src/python_iface.cpp
        )
# Enable C++14
if(NOT WIN32)
	target_compile_features(pt_ocl PRIVATE cxx_std_14)
endif()	
# Link against LibTorch
message("Torch Libs ${TORCH_LIBRARIES}")
target_link_libraries(pt_ocl "${TORCH_LIBRARIES}" /home/artik/Projects/venv/pt_cpu_2.4/lib/python3.10/site-packages/torch/lib/libtorch_python.so   ${OCL_LIB} dlprim_core)
#target_link_libraries(pt_ocl "${TORCH_LIBRARIES}"  ${OCL_LIB} dlprim_core)
set_target_properties(pt_ocl PROPERTIES PREFIX "" OUTPUT_NAME "pt_ocl")
set_target_properties(pt_ocl PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pytorch_ocl/")

foreach(PYNAME __init__.py)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/pytorch_ocl/${PYNAME} "${CMAKE_BINARY_DIR}/pytorch_ocl/${PYNAME}" COPYONLY)
endforeach()

