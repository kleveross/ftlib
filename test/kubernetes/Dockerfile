FROM golang:1.12.14
WORKDIR /root/
COPY . /root/ftlib
RUN cd /root/ftlib/ftlib/consensus/gossip && bash ./gen_shared_lib.sh

FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
WORKDIR /root/
COPY --from=0 /root/ftlib/ftlib /opt/conda/lib/python3.6/site-packages/ftlib
COPY --from=0 /root/ftlib/test/kubernetes /root/example
CMD ["/bin/bash"]
