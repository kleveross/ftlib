__version__ = "0.0.1"
from typing import Set


class BasicCommLib:
    registered_apis: Set[str] = set()

    @classmethod
    def register_api(cls, api):
        cls.registered_apis.add(api.__name__)
        return api

    def __init__(self):
        pass

    def grad_sync_done(self, *args, **kwargs):
        raise NotImplementedError

    def abort_communicator(self):
        raise NotImplementedError

    def rebuild(self):
        pass
