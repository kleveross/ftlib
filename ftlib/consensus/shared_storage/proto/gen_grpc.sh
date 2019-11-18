#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python3 -m grpc_tools.protoc -I${DIR} --python_out=${DIR} --grpc_python_out=${DIR} ${DIR}/communicate.proto

sed 's;import communicate_pb2;from . import communicate_pb2;' communicate_pb2_grpc.py > output.py
mv output.py communicate_pb2_grpc.py