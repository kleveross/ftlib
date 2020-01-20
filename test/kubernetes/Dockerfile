FROM golang:1.12.14
WORKDIR /root/
RUN git clone -b kubernetes-example https://github.com/zw0610/ftlib.git
RUN cd /root/ftlib/ftlib/consensus/gossip && bash ./gen_shared_lib.sh

FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime
WORKDIR /root/
COPY --from=0 /root/ftlib/ftlib /opt/conda/lib/python3.6/site-packages/ftlib
COPY --from=0 /root/ftlib/test/kubernetes /root/example
CMD ["/bin/bash"]
