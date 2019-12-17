from enum import Enum


class ConsensusStatus(Enum):
    SUCCESS = 0
    FAIL = 1
    SKIP_ALLREDUCE = -1


class ConsensusMode(Enum):
    PASSIVE = 0
    ACTIVE = 1
