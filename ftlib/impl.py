import logging
import threading

from ftlib.commlib.commlib_status import CommLibStatus
from ftlib.consensus.consensus_status import ConsensusMode, ConsensusStatus
from ftlib.ftlib_status import FTAllReduceStatus, FTRebuildStatus
from ftlib.rank_assign_scheme import get_rank_size

# in the ftlib package, user is able to initialize the package
# with specific consensus method and communication library


def gen_constant_class(implemented_list):
    class SomeList:
        IMPLEMENTED = implemented_list

        @staticmethod
        def contains(item):
            return item in SomeList.IMPLEMENTED

    return SomeList


_commlib_list = gen_constant_class(["dummy_NCCL", "pytorch"])
_consensus_list = gen_constant_class(["gossip", "shared_storage"])


class BasicFTLib:
    _lock_count = 0
    _implemented_list = {
        "consensus": _consensus_list,
        "commlib": _commlib_list,
    }

    def __init__(
        self,
        consensus,
        commlib,
        consensus_init_kwargs=None,
        commlib_init_kwargs=None,
    ):
        self._is_initialized = False
        self._skip_allreduce = False
        self._new_member_join = False
        self._lock = threading.Lock()

        self.rank = None
        self.size = None
        self.member_list = None

        self.consensus = None
        self.commlib = None

        assert BasicFTLib._implemented_list["commlib"].contains(commlib)
        assert BasicFTLib._implemented_list["consensus"].contains(consensus)

        if consensus == "shared_storage":
            from ftlib.consensus.shared_storage import SharedStorage

            self.consensus = SharedStorage(self)
        elif consensus == "gossip":
            from ftlib.consensus.gossip import Gossip

            assert consensus_init_kwargs is not None
            self.consensus = Gossip(self, **consensus_init_kwargs)

        if commlib == "dummy_NCCL":
            from ftlib.commlib.dummy_nccl import DummyNCCL

            self.commlib = DummyNCCL()
        elif commlib == "pytorch":
            from ftlib.commlib.pytorch import PyTorch

            self.commlib = PyTorch()

        self._passive_or_active = self.consensus.passive_or_active()

        self.initialized = (
            self._initialized_passive_check
            if self._passive_or_active is ConsensusMode.PASSIVE
            else self._initialized_active_check
        )
        self.skip_allreduce = (
            self._skip_allreduce_passive_check
            if self._passive_or_active is ConsensusMode.PASSIVE
            else self._skip_allreduce_active_check
        )

        self._add_apis()

    def _initialized_passive_check(self):
        return self._is_initialized

    def _initialized_active_check(self):
        self.consensus.confirm()
        return self._is_initialized

    def _skip_allreduce_passive_check(self):
        return self._skip_allreduce

    def _skip_allreduce_active_check(self):
        res = self.consensus.confirm()
        if res == ConsensusStatus.SKIP_ALLREDUCE:
            self._skip_allreduce = True
        return self._skip_allreduce

    def lock(self):
        logging.debug(f"trying to lock with lock count: {self._lock_count}")
        self._lock.acquire()
        self._lock_count = self._lock_count + 1
        logging.debug(f"after look, lock count: {self._lock_count}")

    def unlock(self):
        logging.debug(f"trying to unlock with lock count: {self._lock_count}")
        self._lock.release()
        self._lock_count = self._lock_count - 1
        logging.debug(f"after unlock, lock count: {self._lock_count}")

    # TODO: execute still under development
    def execute(self, func, *args, **kwargs):
        new_func = self._wrap_api(None, func)
        return new_func(*args, **kwargs)

    def _rebuild(self):
        master_addr = None
        logging.info("trying to get consensus")
        try:
            consensus_result = self.consensus.confirm()
            if consensus_result == ConsensusStatus.SUCCESS:
                logging.debug("consensus got")
                member_list = self.consensus.get_memberlist()
                logging.debug("memberlist got: {}".format(member_list))
                (self.rank, self.size, master_addr,) = get_rank_size(
                    member_list, self.consensus.id()
                )
                logging.debug("rank, size, master_addr got")
            if consensus_result == ConsensusStatus.SKIP_ALLREDUCE:
                logging.debug("consensus is skip allreduce")
                return consensus_result
            if consensus_result == ConsensusStatus.FAIL:
                logging.debug("failed to get consensus")
                raise Exception("consensus not built")
        except Exception as e:
            logging.warning(
                "failed to get consensus because {}".format(str(e))
            )
            return FTRebuildStatus.ABORT
        logging.info("consensus built")
        logging.info(
            "total size = {size} with master worker: {master}".format(
                size=self.size, master=master_addr
            )
        )
        succeeded = None
        # TODO: generalize the `commlib.rebuild` so that there is
        # no need to change the following code to adopt a new
        # communication library
        try:
            if self.commlib.type == "dummy_NCCL":
                succeeded = self.commlib.rebuild(self.rank, self.size)
            elif self.commlib.type == "pytorch":
                if master_addr is None:
                    raise ValueError(
                        "PyTorch framework requires master address for rebuild"
                    )
                succeeded = self.commlib.rebuild(
                    self.rank, self.size, master_addr=master_addr
                )
        except Exception as e:
            logging.warning(str(e))
            return FTRebuildStatus.ABORT

        if succeeded:
            logging.info("rebuild succeeded")
            self.lock()
            self._is_initialized = True
            self._new_member_join = False
            self._skip_allreduce = False
            self.unlock()
        else:
            logging.warning("rebuild failed")

        return succeeded

    def wait_weights_ready(self, *args, **kwargs):
        return self._wrap_api(self.commlib, "grad_sync_done")(*args, **kwargs)

    def _wrap_api(self, cls_instance, api_name):
        def func(*argc, **kwargs):
            # if skil_allreduce == True, then average_gradient
            # shouldn't be called
            if self.skip_allreduce():
                return FTAllReduceStatus.NO_NEED

            # if the instance is not initialized, then start rebuild
            # TODO: put rebuild into a try loop?
            if not self.initialized():
                rebuild_result = self._rebuild()
                if rebuild_result == FTRebuildStatus.ABORT:
                    logging.warning("rebuild process returns abort")
                    return FTAllReduceStatus.ABORT
                if rebuild_result == FTRebuildStatus.SKIP_ALLREDUCE:
                    logging.warning("rebuild process returns skip allreduce")
                    return FTAllReduceStatus.ABORT

            try:
                self.lock()
                if self.rank == 0 and self._new_member_join:
                    logging.warning(
                        "there is new member join, aborting {api_name}".format(
                            api_name=api_name
                        )
                    )

                    raise Exception("new member joined")
                if cls_instance is None:
                    api_name(*argc, **kwargs)
                else:
                    result = getattr(cls_instance, api_name)(*argc, **kwargs)
                if result == CommLibStatus.SUCCESS:
                    self.consensus.average_success()
                else:
                    self.consensus.average_failure()
                    raise Exception("operation fails")
            except Exception as e:
                logging.exception(str(e))
                self._is_initialized = False
                return FTAllReduceStatus.ABORT
            finally:
                self.unlock()

            return FTAllReduceStatus.SUCCESS

        return func

    def _add_apis(self):
        api_list = self.commlib.get_registered()
        for api in api_list:
            new_api = self._wrap_api(self.commlib, api)
            self.__setattr__(api, new_api)
