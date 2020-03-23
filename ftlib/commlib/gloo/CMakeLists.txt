cmake_minimum_required(VERSION 3.8.1)

project(gloo_lib LANGUAGES CXX)

find_package(pybind11)
pybind11_add_module(gloo_lib src/gloo_lib.cxx)

