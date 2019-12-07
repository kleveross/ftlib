__version__ = "0.0.1"


class BasicFramework:
    def __init__(self):
        pass

    def grad_sync_done(self, *args, **kwargs):
        raise NotImplementedError

    def abort_communicator(self):
        raise NotImplementedError

    def rebuid(self):
        pass
