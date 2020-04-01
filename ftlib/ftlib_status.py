from enum import Enum


class FTCollectiveStatus(Enum):
    NO_NEED = -1
    ABORT = 2
    FAIL = 1
    SUCCESS = 0


class FTRebuildStatus(Enum):
    SUCCESS = 0
    FAIL = 1
    SKIP_ALLREDUCE = -1
    ABORT = 2
