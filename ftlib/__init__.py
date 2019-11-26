__version__ = '0.0.1'

import logging
import threading

# in the tflib package, user is able to initialize the package
# with specific consensus method adn framework
from .consensus.shared_storage import SharedStorage

from .consensus.consensus_status import ConsensusStatus
from .framework.framework_status import FrameworkStatus
from .ftlib_status import * 


class BasicFTLib:
    lock_count = 0
    def __init__(self):
        self._initialized = False
        self.skip_allreduce = False
        self._new_member_join = False

        self.rank = None
        self.size = None
        self.member_list = None

        self._lock = threading.Lock()

        self.consensus = None
        self.framework = None

    def lock(self):
        logging.debug('trying to lock with lock count: {}'.format(self.lock_count))
        self._lock.acquire()
        self.lock_count = self.lock_count + 1
        logging.debug('after look, lock count: {}'.format(self.lock_count))

    def unlock(self):
        logging.debug('trying to unlock with lock count: {}'.format(self.lock_count))
        self._lock.release()
        self.lock_count = self.lock_count - 1
        logging.debug('after unlock, lock count: {}'.format(self.lock_count))

    def initialized(self):
        return self._initialized

    def init(self, consensus, framework):
        assert framework in ['dummy_NCCL', 'pytorch']
        self.consensus = SharedStorage(self)
        if framework == 'dummy_NCCL':
            from .framework.dummy_nccl import DummyNCCL
            self.framework = DummyNCCL()
        if framework == 'pytorch':
            from .framework.pytorch import PyTroch
            self.framework = PyTroch()

    def _rebuild(self):
        try:
            consensus_result = self.consensus.confirm()
            if consensus_result == ConsensusStatus.SUCCESS:
                self.rank, self.size, master_addr = self.consensus.get_rank_size(maddr=True)
            if consensus_result == ConsensusStatus.SKIP_ALLREDUCE:
                return consensus_result
            if consensus_result == ConsensusStatus.FAIL:
                raise Exception('consensus not built')
        except Exception as e:
            logging.warning(str(e))
            return FTRebuildStatus.ABORT

        try:
            if self.framework.type == 'dummy_NCCL':
                if_success = self.framework.rebuild(self.rank, self.size)
            if self.framework.type == 'pytorch':
                if_success = self.framework.rebuild(self.rank, self.size, master_addr=master_addr)
        except Exception as e:
            logging.warning(str(e))
            return FTRebuildStatus.ABORT

        if if_success:
            logging.info('rebuild succeeded')
            self.lock()
            self._initialized = True
            self._new_member_join = False
            self.skip_allreduce = False
            self.unlock()
        else:
            logging.warning('rebuild failed')

        return if_success

    def wait_weights_ready(self, *args, **kwargs):
        return self.allreduce_average(*args, **kwargs)

    def allreduce_average(self, *args, **kwargs):
        # if skil_allreduce == True, then average_gradient shouldn't be called
        if self.skip_allreduce:
            return FTAllReduceStatus.NO_NEED

        # if the instance is not initialized, then start rebuild
        # TODO: put rebuild into a try loop?
        if not self._initialized:
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
