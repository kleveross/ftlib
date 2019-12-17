__version__ = "0.0.1"


class BasicConsensus:
    def __init__(self):
        pass

    def confirm(self, *args, **kwargs):
        # should return if there is a consensus
        pass

    def get_member_list(self):
        raise NotImplementedError

    def average_success(self):
        pass

    def average_failure(self):
        pass
