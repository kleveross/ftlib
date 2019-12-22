from typing import Set


class BasicCommLib:
    registered_apis: Set[str] = set()

    @classmethod
    def register_api(cls, api):
        cls.registered_apis.add(api.__name__)
        return api

    def __init__(self):
        pass

    def get_registered(self):
        return BasicCommLib.registered_apis

    def grad_sync_done(self, *args, **kwargs):
        raise NotImplementedError

    def abort_communicator(self):
        raise NotImplementedError

    def rebuild(self):
        pass
