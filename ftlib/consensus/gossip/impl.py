import logging
import time
from ctypes import POINTER, Structure, c_char_p, c_int, c_longlong, cdll

from ..basic import BasicConsensus
from ..consensus_status import ConsensusStatus


#####################################################################
# In this Gossip Implementation, we use memberlist from hashicorp   #
# as a backend (https://github.com/hashicorp/memberlist)            #
#####################################################################
class Gossip(BasicConsensus):
    def __init__(
        self,
        ftlib,
        known_addr_list,
        so_path="memberlist.so",
        log_path="/tmp/memberlist.log",
    ):
        super(Gossip, self).__init__()

        self._ftlib = ftlib

        self._lib = cdll.LoadLibrary(so_path)
        self._lib.join.argtypes = [self._create_GoSlice(c_char_p)]
        self._lib.get_member_list.restype = self.member_list

        res = self._lib.init_memberlist(log_path)
        if res != 0:
            raise RuntimeError("failed to initialize memberlist")

        joined = self._join(known_addr_list=known_addr_list)
        if not joined:
            raise RuntimeError("failed to join the group")

        time.sleep(5)

        self._cache = self.get_member_list()

    def _create_GoSlice(self, c_data_type):
        class GoSlice(Structure):
            _fields_ = [
                ("data", POINTER(c_data_type)),
                ("len", c_longlong),
                ("cap", c_longlong),
            ]

        return GoSlice

    def get_member_list(self):
        return self._lib.get_member_list()

    def _join(self, known_addr_list, codec="utf-8"):
        assert type(known_addr_list) == list

        addr_list_len = len(known_addr_list)
        assert addr_list_len >= 1

        data_type = c_char_p
        content_tuple = tuple(
            [data_type(addr.encode(codec)) for addr in known_addr_list]
        )

        t = (
            (data_type * addr_list_len),
            content_tuple,
            addr_list_len,
            addr_list_len,
        )

        res = self._lib.join(t)

        return res > 0

    def confirm(self):
        try:
            self._ftlib.lock()
            new_ml = self.get_member_list()
            if new_ml.size > 1:
                self._ftlib._skip_allreduce = False
            for idx in range(new_ml.size):
                if new_ml[idx] not in self._cache:
                    self._ftlib._new_member_join = True
            self._cache = new_ml
        except Exception as e:
            logging.warning(str(e))
            return ConsensusStatus.FAIL
        else:
            if new_ml.size == 1:
                return ConsensusStatus.SKIP_ALLREDUCE
            if new_ml.size < 1:
                return ConsensusStatus.FAIL
            return ConsensusStatus.SUCCESS
        finally:
            self._ftlib.unlock()

    class member_list(Structure):
        _fields_ = [("addrs", (c_char_p * 1024)), ("size", c_int)]

        def __len__(self):
            return self.size

        def __getitem__(self, idx, codec="utf-8"):
            if idx >= self.__len__():
                raise IndexError(
                    "list index out of range: trying to get "
                    + "{idx} while only has {item_num}".format(
                        idx=idx, item_num=self.__len__()
                    )
                )
            return self.addrs[idx].decode(codec)

        def __eq__(self, val):
            val_set = set([val[i] for i in range(len(val))])
            for i in range(len(self)):
                if self[i] not in val_set:
                    return False
                val_set.remove(self[i])
            return len(val_set) == 0

        def __contains__(self, item, codec="utf-8"):
            for idx in range(self.size):
                if self.addrs[idx].decode(codec) == item:
                    return True
            return False
