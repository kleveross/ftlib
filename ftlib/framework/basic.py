__version__ = "0.0.1"


class BasicFramework:
    def __init__(self):
        pass

    def grad_sync_done(self, *args, **kwargs):
        raise NotImplemented

    def abort_communicator(self):
        raise NotImplemented

    def rebuid(self):
        pass
