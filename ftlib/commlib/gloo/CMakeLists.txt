cmake_minimum_required(VERSION 3.8.1)

project(gloo_lib LANGUAGES CXX)

find_package(pybind11)
pybind11_add_module(gloo_lib src/gloo_lib.cxx)

# we are not building with nccl
#find_package(CUDA REQUIRED)
#include_directories("${CUDA_INCLUDE_DIRS}")
#target_link_libraries(fault_tolerant_lib PRIVATE ${CUDA_LIBRARIES})

# find_package(NCCL REQUIRED)
# target_link_libraries(fault_tolerant_lib PRIVATE ${NCCL_LIBRARIES})
# include_directories(${NCCL_INCLUDE_DIRS})

