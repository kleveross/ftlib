cmake_minimum_required(VERSION 3.8.1)
project(fault_tolerant_lib LANGUAGES CUDA CXX)

add_subdirectory(pybind11)
pybind11_add_module(fault_tolerant_lib src/fault_tolerant_lib.cxx)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
target_link_libraries(fault_tolerant_lib PRIVATE ${CUDA_LIBRARIES})

find_package(NCCL REQUIRED)
target_link_libraries(fault_tolerant_lib PRIVATE ${NCCL_LIBRARIES})
include_directories(${NCCL_INCLUDE_DIRS})

