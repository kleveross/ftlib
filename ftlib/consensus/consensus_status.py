from enum import Enum

class ConsensusStatus(Enum):
    SUCCESS = 0
    FAIL = 1
    SKIP_ALLREDUCE = -1