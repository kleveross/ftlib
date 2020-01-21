import logging
import os
import socket
import time
from ctypes import POINTER, Structure, c_char_p, c_int, c_longlong, cdll

from ftlib.consensus.basic_consensus import BasicConsensus
from ftlib.consensus.consensus_status import ConsensusMode, ConsensusStatus


#####################################################################
# In this Gossip Implementation, we use memberlist from hashicorp   #
# as a backend (https://github.com/hashicorp/memberlist)            #
#####################################################################
class Gossip(BasicConsensus):
    def __init__(
        self, ftlib, known_addr_list, log_file="/tmp/memberlist.log",
    ):
        super(Gossip, self).__init__()

        self._ftlib = ftlib

        self._lib = cdll.LoadLibrary(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "memberlist.so"
            )
        )
        self._data_type = c_char_p
        self._goslice_type = self._create_GoSlice(self._data_type)
        self._lib.join.argtypes = [self._goslice_type]
        self._lib.get_memberlist.restype = self.member_list

        res = self._lib.init_memberlist(log_file.encode("utf-8"))
        if res != 0:
            raise RuntimeError("failed to initialize memberlist")

        if known_addr_list is not None and known_addr_list != []:
            joined = self._join(known_addr_list=known_addr_list)
            if not joined:
                raise RuntimeError("failed to join the group")

        time.sleep(5)

        self._cache = self.get_memberlist()
        logging.debug(self._cache)

    def _create_GoSlice(self, c_data_type):
        class GoSlice(Structure):
            _fields_ = [
                ("data", POINTER(c_data_type)),
                ("len", c_longlong),
                ("cap", c_longlong),
            ]

        return GoSlice

    def get_memberlist(self):
        raw_memberlist = self._lib.get_memberlist()
        logging.debug("got {} workers".format(len(raw_memberlist)))
        memberlist = {raw_memberlist[i] for i in range(len(raw_memberlist))}
        logging.debug("memberlist: {}".format(memberlist))
        return memberlist

    def passive_or_active(self):
        return ConsensusMode.ACTIVE

    def _join(self, known_addr_list, codec="utf-8"):
        assert type(known_addr_list) == list

        addr_list_len = len(known_addr_list)
        assert addr_list_len >= 1

        content_tuple = tuple(
            [self._data_type(addr.encode(codec)) for addr in known_addr_list]
        )

        t = self._goslice_type(
            (self._data_type * addr_list_len)(*content_tuple),
            addr_list_len,
            addr_list_len,
        )
        logging.info("Waiting 15 seconds before join")
        time.sleep(15)
        res = self._lib.join(t)

        return res > 0

    def id(self):
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)

    def confirm(self):
        try:
            self._ftlib.lock()
            new_ml = self.get_memberlist()

            if len(new_ml) == 1:
                self._ftlib._skip_allreduce = True
            else:
                self._ftlib._skip_allreduce = False

            logging.debug(f"old memberlist: {self._cache}")
            logging.debug(f"new memberlist: {new_ml}")
            if new_ml != self._cache:
                self._ftlib._is_initialized = False

            if len([m for m in new_ml if m not in self._cache]) > 0:
                self._ftlib._new_member_join = True

            self._cache = new_ml
        except Exception as e:
            logging.warning(str(e))
            return ConsensusStatus.FAIL
        else:
            if len(new_ml) == 1:
                return ConsensusStatus.SKIP_ALLREDUCE
            if len(new_ml) < 1:
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
