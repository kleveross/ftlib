import logging
import os
import signal
import time

import numpy as np

from ftlib.commlib.basic_commlib import BasicCommLib
from ftlib.commlib.commlib_status import CommLibStatus
from ftlib.commlib.nccl import fault_tolerant_lib  # type: ignore


# common utils section
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
        shared_path="/crystal",
        filename="nccl_id_file",
        max_try=30,
    ):
        self.type = "dummy_NCCL"
        self.grad_sync_timeout = grad_sync_timeout
        self.shared_path = shared_path
        self._nccl_context = fault_tolerant_lib.nccl_context()
        self._nccl_id_filename = filename
        self._max_try = max_try

    @BasicCommLib.register_api
    def grad_sync_done(self):
        try:
            self.allreduce(np.array(range(10)).astype(np.float))
        except Exception as e:
            logging.warning(str(e))
            return CommLibStatus.FAIL
        return CommLibStatus.SUCCESS

    @BasicCommLib.register_api
    def broadcast(self, data, root):
        logging.debug("broadcasting: " + str(data))

        success = self._nccl_context.setInput(data)
        if not success:
            raise Exception("set input failed")

        success = self._nccl_context.broadcastAsync(root)
        if not success:
            raise Exception("async broadcast operation failed")

        signal.alarm(10)

        success = False
        while not success:
            success = self._nccl_context.checkOpResult(1)
        signal.alarm(0)

        data = self._nccl_context.getOutput()
        logging.debug("receiving: " + str(data))

    @BasicCommLib.register_api
    def allreduce(self, data, op="SUM"):
        if op != "SUM":
            raise ValueError(
                "Only SUM operation is currently allowed in NCCL allreduce"
            )
        logging.debug("averaging: " + str(data))

        success = self._nccl_context.setInput(data)
        if not success:
            raise Exception("set input failed")

        success = self._nccl_context.allreduceAsync()
        if not success:
            raise Exception("asynchronizely allreduce failed")

        signal.alarm(10)

        success = False
        while not success:
            success = self._nccl_context.checkOpResult(1)
        signal.alarm(0)

        output = self._nccl_context.getOutput()
        logging.debug("receiving: " + str(output))

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

    def rebuild(self, rank, size):
        if rank == 0:
            res = self._rebuild_as_root(rank, size)
        else:
            res = self._rebuild_as_rest(rank, size)
        return res

    def abort_communicator(self):
        self._nccl_context.commAbort()
