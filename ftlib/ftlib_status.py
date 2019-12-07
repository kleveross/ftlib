from enum import Enum

from .consensus.consensus_status import ConsensusStatus


class FTAllReduceStatus(Enum):
    NO_NEED = -1
    ABORT = 2
    FAIL = 1
    SUCCESS = 0


class FTRebuildStatus(ConsensusStatus):
    ABORT = 2
