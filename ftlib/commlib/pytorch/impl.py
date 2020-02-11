__pytorch_version__ = "1.2.0"

import logging
from datetime import timedelta

import torch.distributed as dist

from ftlib.commlib.basic_commlib import BasicCommLib
from ftlib.commlib.commlib_status import CommLibStatus


class PyTorch(BasicCommLib):
    def __init__(
        self, grad_sync_timeout=10, max_try=30, port=12355, backend="gloo"
    ):
        self.type = "pytorch"
        self.grad_sync_timeout = grad_sync_timeout
        self._max_try = max_try
        self._port = port
        self._backend = backend
        self._timeout = timedelta(minutes=1)

    @BasicCommLib.register_api
    def grad_sync_done(self, *args, **kwargs):
        model = None
        if "model" in kwargs.keys():
            model = kwargs["model"]
        elif len(args) > 0:
            model = args[0]
        if model is None:
            return CommLibStatus.FAIL
        try:
            size = float(dist.get_world_size())
            for param in model.parameters():
                dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
                param.grad.data /= size
        except Exception as e:
            logging.error(str(e))
            return CommLibStatus.FAIL
        return CommLibStatus.SUCCESS

    @BasicCommLib.register_api
    def allreduce(self, data, op="MEAN"):
        # torch.distributed.ReduceOp has no option for 'MEAN'
        # so far, we only implemented 'MEAN'
        reduce_op = dist.reduce_op.SUM
        dist.all_reduce(data, op=reduce_op)
        if op == "MEAN":
            size = float(dist.get_world_size())
            data /= size

    @BasicCommLib.register_api
    def broadcast(self, data, root_rank):
        dist.broadcast(data, root_rank)

    @BasicCommLib.register_api
    def barrier(self):
        dist.barrier()

    def rebuild(self, rank, size, master_addr):
        if dist.is_initialized():
            logging.info("aborting communicator")
            self.abort_communicator()

        assert not dist.is_initialized()

        init_method = f"tcp://{master_addr}:{self._port}"

        logging.info(
            "initializing process group with "
            + f"backend={self._backend}, rank={rank}, world_size={size}, "
            + f"master_port={self._port}, master_addr={master_addr}"
        )
        dist.init_process_group(
            backend=self._backend,
            timeout=self._timeout,
            world_size=size,
            rank=rank,
            init_method=init_method,
        )

        return dist.is_initialized()

    def abort_communicator(self):
        if dist.is_initialized():
            dist.destroy_process_group()
