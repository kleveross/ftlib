#!/bin/bash

git clone https://github.com/pybind/pybind11
cd pybind11
python3 ./setup.py build
cd ..

mkdir build
cd build
cmake ..
make
cp ./*.so ../

cd ..
rm -rf ./pybind11
rm -rf ./build

