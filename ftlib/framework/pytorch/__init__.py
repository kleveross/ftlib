__version__ = "0.0.1"

__pytorch_version__ = "1.2.0"

import logging
import os

import torch.distributed as dist

from ..basic import BasicFramework
from ..framework_status import FrameworkStatus


class PyTorch(BasicFramework):

    def __init__(self, grad_sync_timeout=10, max_try=30, port=12355):
        self.type = "pytorch"
        self.grad_sync_timeout = grad_sync_timeout
        self._max_try = max_try
        self._port = port

    def grad_sync_done(self, *args, **kwargs):
        model = None
        if "model" in kwargs.keys():
            model = kwargs["model"]
        elif len(args) > 0:
            model = args[0]
        if model is None:
            return FrameworkStatus.FAIL
        try:
            size = float(dist.get_world_size())
            for param in model.parameters():
                dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
                param.grad.data /= size
        except Exception as e:
            logging.error(str(e))
            return FrameworkStatus.FAIL
        return FrameworkStatus.SUCCESS

    def allreduce(self, data, op="MEAN"):
        # torch.distributed.ReduceOp has no option for 'MEAN'
        # so far, we only implemented 'MEAN'
        reduce_op = dist.reduce_op.SUM
        dist.all_reduce(data, op=reduce_op)
        if op == "MEAN":
            size = float(dist.get_world_size())
            data /= size

    def broadcast(self, data, root_rank):
        dist.broadcast(data, root_rank)

    def barrier(self):
        dist.barrier()

    def rebuild(self, rank, size, master_addr, backend="gloo"):
        if rank == 0:
            os.environ["MASTER_ADDR"] = "localhost"
        else:
            os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(self._port)
        dist.init_process_group(backend=backend, rank=rank, world_size=size)

        return dist.is_initialized()

    def abort_communicator(self):
        dist.destroy_process_group()
