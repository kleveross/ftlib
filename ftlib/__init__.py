__version__ = '0.0.1'

import logging
import threading

# in the tflib package, user is able to initialize the package
# with specific consensus method adn framework
from .consensus.shared_storage import SharedStorage
from .framework.dummy_nccl import DummyNCCL


class BasicFTLib:
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
        self._lock.acquire()

    def unlock(self):
        self._lock.release()

    def initialized(self):
        return self._initialized

    def init(self, consensus, framework):
        self.consensus = SharedStorage(self)
        self.framework = DummyNCCL(self)

    def _rebuild(self):
        try:
            consensus_result = self.consensus.confirm()
            if consensus_result == 'success':
                self.rank, self.size = self.consensus.get_rank_size()
            if consensus_result == 'skip allreduce':
                return consensus_result
            if consensus_result == 'fail':
                raise Exception('consensus not built')
        except Exception as e:
            logging.warning(str(e))
            return 'abort'

        try:
            if_success = self.framework.rebuild(self.rank, self.size)
        except Exception as e:
            logging.warning(str(e))
            return 'abort'

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

    def wait_weights_ready(self, grads=None):
        return self.allreduce_average(grads)

    def allreduce_average(self, data):
        # if skil_allreduce == True, then average_gradient shouldn't be called
        if self.skip_allreduce:
            return 'no need'

        # if the instance is not initialized, then start rebuild
        # TODO: put rebuild into a try loop?
        if not self._initialized:
            rebuild_result = self._rebuild()
            if rebuild_result == 'abort':
                return rebuild_result
            if rebuild_result == 'skip allreduce':
                return rebuild_result

        # initialize the result flag for nccl allreduce operation
        result = 'failed'
        # try all reduce
        try:
            if self.rank == 0:
                logging.debug("I'm root and I'm averaging gradient")
                self._lock.acquire()
                logging.debug("I've locked control")
                if not self._new_member_join:
                    logging.debug("There is no new member joined")
                    result = self.framework.grad_sync_done()
                    self._lock.release()
                else:
                    logging.debug("Ah, there be new member(s) joined")
                    self._lock.release()
                    raise Exception("new member joined")
            else:
                result = self.framework.grad_sync_done()

            if result == 'success':
                self.consensus.average_success()
            else:
                self.consensus.average_fail()
                raise Exception("nccl allreduce failed")

        except Exception as e:
            logging.exception(str(e))
            self._initialized = False
            try:
                logging.info("aborting communicator")
                self.framework.abort_communicator()
            except Exception as e:
                logging.warning(str(e))
            return 'abort'

        return 'success'
