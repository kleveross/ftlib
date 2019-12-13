__version__ = "0.0.1"

import logging
import threading

from .consensus.consensus_status import ConsensusStatus
from .framework.framework_status import FrameworkStatus
from .ftlib_status import FTAllReduceStatus, FTRebuildStatus

# in the ftlib package, user is able to initialize the package
# with specific consensus method and framework


def gen_constant_class(implemented_list):
    class SOMEList:
        IMPLEMENTED = implemented_list

        @staticmethod
        def contains(item):
            return item in SOMEList.IMPLEMENTED

    return SOMEList


FrameWorkList = gen_constant_class(["dummy_NCCL", "pytorch"])
ConsensusList = gen_constant_class(["gossip", "shared_storage"])


class BasicFTLib:
    lock_count = 0

    def __init__(self, passive_check=False):
        self._initialized = False
        self._skip_allreduce = False
        self._new_member_join = False
        self._passive_check = passive_check

        self.initialized = (
            self._initialized_passive_check
            if self._passive_check
            else self._initialized_active_check
        )
        self.skip_allreduce = (
            self._skip_allreduce_passive_check
            if self._passive_check
            else self._skip_allreduce_active_check
        )

        self.rank = None
        self.size = None
        self.member_list = None

        self._lock = threading.Lock()

        self.consensus = None
        self.framework = None

    def _initialized_passive_check(self):
        return self._initialized

    def _initialized_active_check(self):
        self.consensus.confirm()
        return self._initialized

    def _skip_allreduce_passive_check(self):
        return self._skip_allreduce

    def _skip_allreduce_active_check(self):
        self.consensus.confirm()
        return self._skip_allreduce

    def lock(self):
        logging.debug(
            "trying to lock with lock count: {}".format(self.lock_count)
        )
        self._lock.acquire()
        self.lock_count = self.lock_count + 1
        logging.debug("after look, lock count: {}".format(self.lock_count))

    def unlock(self):
        logging.debug(
            "trying to unlock with lock count: {}".format(self.lock_count)
        )
        self._lock.release()
        self.lock_count = self.lock_count - 1
        logging.debug("after unlock, lock count: {}".format(self.lock_count))

    def init(
        self,
        consensus,
        framework,
        consensus_init_kwargs=None,
        framework_init_kwargs=None,
    ):
        assert FrameWorkList.contains(framework)
        assert ConsensusList.contains(consensus)

        if consensus == "shared_storage":
            from .consensus.shared_storage import SharedStorage

            self.consensus = SharedStorage(self)
        elif consensus == "gossip":
            from .consensus.gossip import Gossip

            assert consensus_init_kwargs is not None
            self.consensus = Gossip(self, **consensus_init_kwargs)

        if framework == "dummy_NCCL":
            from .framework.dummy_nccl import DummyNCCL

            self.framework = DummyNCCL()
        elif framework == "pytorch":
            from .framework.pytorch import PyTorch

            self.framework = PyTorch()

    def _rebuild(self):
        try:
            consensus_result = self.consensus.confirm()
            if consensus_result == ConsensusStatus.SUCCESS:
                (
                    self.rank,
                    self.size,
                    master_addr,
                ) = self.consensus.get_rank_size(maddr=True)
            if consensus_result == ConsensusStatus.SKIP_ALLREDUCE:
                return consensus_result
            if consensus_result == ConsensusStatus.FAIL:
                raise Exception("consensus not built")
        except Exception as e:
            logging.warning(str(e))
            return FTRebuildStatus.ABORT

        try:
            if self.framework.type == "dummy_NCCL":
                if_success = self.framework.rebuild(self.rank, self.size)
            if self.framework.type == "pytorch":
                if_success = self.framework.rebuild(
                    self.rank, self.size, master_addr=master_addr
                )
        except Exception as e:
            logging.warning(str(e))
            return FTRebuildStatus.ABORT

        if if_success:
            logging.info("rebuild succeeded")
            self.lock()
            self._initialized = True
            self._new_member_join = False
            self._skip_allreduce = False
            self.unlock()
        else:
            logging.warning("rebuild failed")

        return if_success

    def wait_weights_ready(self, *args, **kwargs):
        return self.allreduce_average(*args, **kwargs)

    def allreduce_average(self, *args, **kwargs):
        # if skil_allreduce == True, then average_gradient shouldn't be called
        if self.skip_allreduce():
            return FTAllReduceStatus.NO_NEED

        # if the instance is not initialized, then start rebuild
        # TODO: put rebuild into a try loop?
        if not self.initialized():
            rebuild_result = self._rebuild()
            if rebuild_result == FTRebuildStatus.ABORT:
                return rebuild_result
            if rebuild_result == FTRebuildStatus.SKIP_ALLREDUCE:
                return rebuild_result

        # initialize the result flag for nccl allreduce operation
        result = FTAllReduceStatus.FAIL
        # try all reduce
        try:
            if self.rank == 0:
                logging.debug("I'm root and I'm averaging gradient")
                self.lock()
                logging.debug("I've locked control")
                if not self._new_member_join:
                    logging.debug("There is no new member joined")
                    result = self.framework.grad_sync_done(*args, **kwargs)
                    self.unlock()
                else:
                    logging.debug("Ah, there be new member(s) joined")
                    self.unlock()
                    raise Exception("new member joined")
            else:
                # TODO: maybe we can merge
                # grad_sync_done + locking with/without
                # locking regardless its rank
                result = self.framework.grad_sync_done(*args, **kwargs)

            if result == FrameworkStatus.SUCCESS:
                self.consensus.average_success()
            else:
                self.consensus.average_failure()
                raise Exception("nccl allreduce failed")

        except Exception as e:
            logging.exception(str(e))
            self._initialized = False
            if self.lock_count > 0:
                self.unlock()
            try:
                logging.info("aborting communicator")
                self.framework.abort_communicator()
            except Exception as e:
                logging.warning(str(e))
            return FTAllReduceStatus.ABORT

        return FTAllReduceStatus.SUCCESS
