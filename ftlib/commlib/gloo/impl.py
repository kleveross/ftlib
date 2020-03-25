import os
import signal

from ftlib.commlib.basic_commlib import BasicCommLib
from ftlib.commlib.gloo import gloo_lib  # type: ignore


def handler(signum, frame):
    print("Signal handler called with signal", signum)
    raise Exception("end of time")


signal.signal(signal.SIGALRM, handler)


class GLOO(BasicCommLib):
    def __init__(self, shared_path=None, wait_time=5):
        self.type = "GLOO"
        if shared_path is None:
            self.shared_path = os.getenv("GLOO_FILE_STORE", "/crystal")
        else:
            self.shared_path = shared_path

        self._wait_time = wait_time
        self._default_timeout = wait_time

        self._gloo = None

    def rebuild(self, rank, size, master_addr):
        if self._gloo is not None:
            self.abort_communicator()

        prefix = master_addr.replace(".", "")

        self._gloo = gloo_lib.Gloo(self.shared_path, prefix, rank, size)

    @BasicCommLib.register_api
    def barrier(self, *args, **kwargs):
        self._gloo.barrier(self._default_timeout)

    @BasicCommLib.register_api
    def broadcast(self, data, root_rank, *args, **kwargs):
        return self._gloo.broadcast(data, root_rank, self._default_timeout)

    @BasicCommLib.register_api
    def allreduce(self, data, *args, **kwargs):
        return self._gloo.allreduce(data, self._default_timeout)

    def abort_communicator(self):
        if self._gloo is not None:
            self._gloo = None
