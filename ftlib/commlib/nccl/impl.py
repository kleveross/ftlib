import logging
import os
import signal
import time

import numpy as np

from ftlib.commlib.basic_commlib import BasicCommLib
from ftlib.commlib.nccl import fault_tolerant_lib  # type: ignore


# common utils section
# deprecated
def try_write_file(directory, filename, content):
    logging.info("writing: {}/{}".format(directory, filename))
    with open(os.path.join(directory, filename), "w") as f:
        try:
            f.write(content)
        except Exception as e:
            logging.warning("Error!" + str(e))
        else:
            return True
        return False


# handler for timeout
def handler(signum, frame):
    print("Signal handler called with signal", signum)
    raise Exception("end of time")


signal.signal(signal.SIGALRM, handler)


class NCCL(BasicCommLib):
    def __init__(
        self,
        grad_sync_timeout=10,
        shared_path=None,
        filename="nccl_id_file",
        max_try=30,
        wait_time=5,
    ):
        self.type = "NCCL"
        self.grad_sync_timeout = grad_sync_timeout
        if shared_path is None:
            self.shared_path = os.getenv("NCCL_ID_DIR", "/crystal")
        self.shared_path = shared_path
        self._nccl_context = fault_tolerant_lib.nccl_context()
        self._nccl_id_filename = filename
        self._max_try = max_try
        self._wait_time = wait_time
        self._default_timeout = wait_time

    @BasicCommLib.register_api
    def grad_sync_done(self):
        raise NotImplementedError

    @BasicCommLib.register_api
    def broadcast(self, data, root_rank, timeout=None):
        if timeout is None:
            timeout == self._default_timeout

        logging.debug("broadcasting: " + str(data))

        if not isinstance(data, np.ndarray):
            data = np.array(data).astype(np.float32)

        if data.dtype is not np.float32:
            raise NotImplementedError(
                "data types other fp32 are not implemented"
            )

        broadcast_call = self._nccl_context.broadcast(data, root_rank)
        signal.alarm(timeout)
        call_success = False
        while not call_success:
            call_success = broadcast_call.check_complete()
        signal.alarm(0)

        if not call_success:
            raise RuntimeError("broadcast failed")

        logging.debug("receiving: " + str(data))

        # the data is broadcast in-place
        return data

    @BasicCommLib.register_api
    def allreduce(self, data, op="SUM", timeout=None):
        if op != "SUM":
            raise ValueError(
                "Only SUM operation is currently allowed in NCCL allreduce"
            )
        if timeout is None:
            timeout = self._default_timeout
        logging.debug("averaging: " + str(data))

        if not isinstance(data, np.ndarray):
            data = np.array(data).astype(np.float32)

        if data.dtype is not np.float32:
            raise NotImplementedError(
                "data types other fp32 are not implemented"
            )

        allreduce_call = self._nccl_context.allreduce(data)
        signal.alarm(timeout)
        call_success = False
        while not call_success:
            call_success = allreduce_call.check_complete()
        signal.alarm(0)

        if not call_success:
            raise RuntimeError("allreduce failed")

        logging.debug("receiving: " + str(data))

        # the data is allreduced in-place
        return data

    def barrier(self):
        # maybe we can allreduce a byte
        pass

    def _rebuild_as_root(self, rank, size):
        self._nccl_context.generateNCCLID()
        nccl_id_array = self._nccl_context.getNCCLID()
        nccl_id_str = ",".join([str(x) for x in nccl_id_array])

        assert rank == 0
        try_write_file(self.shared_path, self._nccl_id_filename, nccl_id_str)

        logging.debug("communicator initializing rank")
        self._nccl_context.commInitRank(size, rank)
        return True

    def _rebuild_as_rest(self, rank, size):
        logging.debug("start _rebuild_with_rank_as_other")
        self._nccl_context.generateNCCLID()
        logging.debug("nccl id initialized")

        logging.debug("sleep for {}".format(self._wait_time))
        time.sleep(self._wait_time)

        full_path = os.path.join(self.shared_path, self._nccl_id_filename)
        nccl_id_str = None

        for _ in range(self._max_try):
            logging.debug("trying to retrieve nccl id")

            if os.path.isfile(full_path):
                with open(full_path, "r") as f:
                    nccl_id_str = f.read()

            if nccl_id_str is None:
                logging.warning("cannot get nccl id, wait ...")
                time.sleep(2)
            else:
                break
        logging.debug("nccl id got: {}".format(nccl_id_str))

        if nccl_id_str is None:
            return False

        nccl_id_array = np.array(
            [int(x) for x in nccl_id_str.split(",")], dtype=np.int32
        )
        self._nccl_context.setNCCLID(nccl_id_array)

        logging.info("other rank start communicator init rank")
        self._nccl_context.commInitRank(size, rank)
        return True

    def rebuild(self, rank, size, *args, **kwargs):
        if rank == 0:
            res = self._rebuild_as_root(rank, size)
        else:
            res = self._rebuild_as_rest(rank, size)
        return res

    def abort_communicator(self):
        self._nccl_context.commAbort()
