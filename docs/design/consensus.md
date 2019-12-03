## Consensus

The consensus protocol in FTLib acts as a shadow precondition for any collective communication operations. Any changes from the consensus protocol will reset the initialization flag of FTLib to `False`, deterring any communication operations after the rebuild procedure returns success.

A member list is maintained by the implementation of consensus protocol.

The rank-assign scheme in FTLib can extract worker identification from the member list, such like address of each worker. This unique identification helps the rank-assign scheme to designate individual rank number to each worker, which most communication libraries require when initializing.

When FTLib start to `rebuild`, it uses the `confirm` API of consensus protocol to check the member list is agreed by all existing workers.

Not exposed to FTLib though, a `report_join` API in consensus protocol will be called inside the `confirm` function if the worker is freshly launched and has not successfully reported before. During the whole lifetime of a worker, the `report_join` API will not be called for a second time after a successfully trial.

Every time FTLib succeeds or fails to perform collective operations, it will call the corresponding functions to inform the consensus protocol whether any actions need to be taken. However, these two functions can be ignored if no actions needed by the specific consensus protocol.

## Consensus API Introduction

### 1. BasicConsensus.confirm()

### 2. BasicConsensus.get_member_list()

### 4. BasicConsensus.average_failure()

### 5. BasicConsensus.average_success()

