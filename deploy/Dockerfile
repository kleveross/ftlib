FROM nvidia/cuda:latest
RUN apt update && apt install -y git cmake python3 python3-pip python3-dev
ADD requirements.txt /root
COPY deploy/FindNCCL.cmake /usr/share/cmake-3.10/Modules
RUN pip3 install -r /root/requirements.txt

COPY ftlib /ftlib
RUN cd /ftlib/consensus/shared_storage/proto/ && bash gen_grpc.sh
RUN cd /ftlib/framework/dummy_nccl/ && bash build_with_pybind11.sh
