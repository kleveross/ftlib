import logging
import threading
import time

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


_commlib_list = gen_constant_class(["NCCL", "pytorch"])
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

        # get member list after initialization succeeded

        if commlib == "NCCL":
            from ftlib.commlib.nccl import NCCL

            self.commlib = NCCL()
        elif commlib == "pytorch":
            from ftlib.commlib.pytorch import PyTorch

            self.commlib = PyTorch()

        self._passive_or_active = self.consensus.passive_or_active()
        logging.info(
            "FTLib is using {mode} mode.".format(
                mode="activate"
                if self._passive_or_active == ConsensusMode.ACTIVE
                else "passive"
            )
        )

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

        self.build()

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

    def set_initialized(self, is_initialized):
        self._is_initialized = is_initialized
        if not is_initialized:
            self.commlib.abort_communicator()
            self.consensus.average_failure()

    def manual_join(self, *args, **kwargs):
        return self.consensus.manual_join(*args, **kwargs)

    def consensus_joined(self):
        return self.consensus.joined

    # TODO: execute still under development
    def execute(self, func, *args, **kwargs):
        # Args:
        #     func, a function object to be executed
        #     *args, args passed to the function
        #     **kwargs, kwargs passed to the function
        # Returns:
        #     objects returned by `func`

        # Check whether if it there is only one worker, which means
        # the function should execute function without confirming
        # consensus being built
        if self.skip_allreduce():
            return func(*args, **kwargs)

        # Check whether the consensus has been built, if not, rebuild
        # if the rebuild succeeds, continue the following steps
        if not self.initialized():
            logging.info("FTLib not initialized, (re-)initializing...")
            # TODO: we should consider retrying rebuild process
            #  for multiple times
            rebuild_result = self._rebuild()
            if rebuild_result != FTAllReduceStatus.SUCCESS:
                if rebuild_result == FTRebuildStatus.FAIL:
                    raise Exception("rebuild process returns failed")
                elif rebuild_result == FTRebuildStatus.ABORT:
                    raise Exception("rebuild process returns abort")
                elif rebuild_result == FTRebuildStatus.SKIP_ALLREDUCE:
                    # there is no need to rebuild
                    logging.debug("concensus rebuild returns skip_allreduce")
                    return func(*args, **kwargs)

        # Execute the `func` after the consensus is built
        try:
            logging.debug("start to execute func")
            res = func(*args, **kwargs)
        except Exception as e:
            logging.exception(str(e))
            self.set_initialized(False)
            return
        else:
            return res

    def _confirm(self):
        # here we use self.consensus.confirm to try reaching a global consensus
        initial_count_left = 3
        initial_wait_time = 4.0  # seconds

        count_left = initial_count_left
        max_timeout = 25.0  # seconds

        wait_time = initial_wait_time

        result = self.consensus.confirm()
        start_time = time.time()
        while (
            start_time + max_timeout - time.time()
        ) > 0.0 and count_left > 0:
            time.sleep(wait_time)
            result = self.consensus.confirm()
            count_left = count_left - 1
            if self.consensus.ml_changed():
                wait_time = initial_wait_time
                count_left = initial_count_left
            else:
                wait_time = wait_time * 0.5

        return result

    def _rebuild(self):
        # In `_rebuild`, there is only one timeout setting, which is
        # embedded in `self.commlib.rebuild`. It is necessary to follow
        # this config:
        #     max(time_of_consensus_built) < self.commlib.rebuild.timeout
        # Meanwhile, the consensus built process does not rush to a
        # conclusion with single `self.consensus.confirm` call.
        #
        # Worker 1                        Worker 2
        #   |                                |
        #   ||<-consensus starts             |
        #   ||                               |
        #   ||                               |
        #   ||<-consensus reached            |
        #   |                                |
        #  ||<-(re-)init starts              |
        #  ||                                |
        #  ||                                ||<-consensus starts
        #  ||                                ||
        #  ||                                ||
        #  ||                                ||<-consensus reached
        #  ||                                |
        #  ||                               ||<-(re-)init starts
        #  ||<-(re-)init times out          ||<-(re-)init succeeds
        #   |                                |
        master_addr = None
        logging.info("trying to get consensus")

        member_list = None
        try:
            consensus_result = self._confirm()
            if consensus_result == ConsensusStatus.SUCCESS:
                logging.debug("consensus got")
                member_list = self.consensus.get_memberlist()
                logging.debug("memberlist got: {}".format(member_list))
                (self.rank, self.size, master_addr,) = get_rank_size(
                    member_list, self.consensus.id(), self.member_list
                )
                logging.debug("rank, size, master_addr got")
            if consensus_result == ConsensusStatus.SKIP_ALLREDUCE:
                logging.debug("consensus is skip allreduce")
                return FTRebuildStatus.SKIP_ALLREDUCE
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
        #  no need to change the following code to adopt a new
        #  communication library
        try:
            if self.commlib.type == "NCCL":
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
            self.member_list = member_list
            self.unlock()
        else:
            logging.warning("rebuild failed")

        return FTRebuildStatus.SUCCESS

    def wait_gradients_ready(self, *args, **kwargs):
        return self._wrap_api(self.commlib, "grad_sync_done")(*args, **kwargs)

    def build(self):
        assert self.rank is None or self.size is None
        result = self._rebuild()
        if result == FTRebuildStatus.ABORT:
            raise Exception("building consensus failed")
        if result == FTRebuildStatus.SKIP_ALLREDUCE:
            self.rank = 0
            self.size = 1

    def _wrap_api(self, cls_instance, api_name):
        def func(*argc, **kwargs):
            # Args:
            #     *args, args passed to the function
            #     **kwargs, kwargs passed to the function
            # Returns:
            #     if collective ops returns successfully, returns the value
            #     otherwise, returns FTAllReduceStatus
            #     TODO: rename FTAllReduceStatus to FTCollectiveStatus

            # the api here is
            ops = (
                api_name
                if cls_instance is None
                else getattr(cls_instance, api_name)
            )

            # if skil_allreduce == True, then any collective ops
            # shouldn't be called
            if self.skip_allreduce():
                return FTAllReduceStatus.NO_NEED

            # if the instance is not initialized, then start rebuild
            # TODO: put rebuild into a try loop?
            if not self.initialized():
                logging.info("FTLib not initialized. (re-)building...")
                # TODO: we should consider retrying rebuild process
                #  for multiple times
                rebuild_result = self._rebuild()
                if rebuild_result == FTRebuildStatus.ABORT:
                    logging.warning("rebuild process returns abort")
                    return FTAllReduceStatus.ABORT
                if rebuild_result == FTRebuildStatus.FAIL:
                    logging.warning("rebuild process returns fail")
                    return FTAllReduceStatus.ABORT
                if rebuild_result == FTRebuildStatus.SKIP_ALLREDUCE:
                    logging.warning("rebuild process returns skip allreduce")
                    return FTAllReduceStatus.NO_NEED

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
                    result = ops(*argc, **kwargs)
                else:
                    result = ops(*argc, **kwargs)
            except Exception as e:
                logging.exception(str(e))
                self.set_initialized(False)
                return FTAllReduceStatus.FAIL
            else:
                self.consensus.average_success()
                return result
            finally:
                self.unlock()

        return func

    def _add_apis(self):
        api_list = self.commlib.get_registered()
        for api in api_list:
            new_api = self._wrap_api(self.commlib, api)
            self.__setattr__(api, new_api)
