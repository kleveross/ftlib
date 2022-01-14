FROM nvidia/cuda:10.1-devel-ubuntu18.04

RUN apt update && apt -y install wget cmake unzip

RUN wget -O miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh && \
    bash miniconda.sh -b && rm miniconda.sh

ENV PATH=/root/miniconda3/bin:${PATH}

RUN conda init bash && echo "auto_activate_base: true" > ~/.condarc

RUN pip install --no-cache-dir pytest pybind11==2.4.3 numpy

RUN wget -O /usr/share/cmake-3.10/Modules/FindNCCL.cmake https://raw.githubusercontent.com/BVLC/caffe/master/cmake/Modules/FindNCCL.cmake

RUN wget -O pybind11.tar.gz https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz && \
    tar -xf ./pybind11.tar.gz && cd pybind11-2.4.3 && mkdir build && cd build && \
    cmake .. && make -j4 && make install && cd ../.. && rm -rf ./pybind11*

RUN wget -O gloo.zip https://github.com/facebookincubator/gloo/archive/master.zip && \
    unzip ./gloo.zip && cd gloo-master && mkdir build && cd build && \
    cmake .. && make -j4 && make install && cd ../.. && rm -rf ./gloo* 

COPY . /root/gloo/

RUN cd /root/gloo && ls && mkdir build && cd build && cmake .. && make -j4
