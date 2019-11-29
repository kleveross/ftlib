## Consensus

The consensus protocol in FTLib acts as a shadow precondition for any collective communication operations. Any changes from the consensus protocol will reset the initialization flag of FTLib to `False`, deferring any collective communication operations, and lead to the rebuild procedure.

A member list is maintained by the implementation of consensus protocol.

The rank-assign scheme in FTLib can extract worker identification from the member list, such like address of each workers. Such unique identification helps the rank-assign scheme to designate unique rank number to each worker, which most collective communication libraries require during initialization.

When FTLib start to `rebuild`, it uses the `confirm` API of consensus protocol to check the consensus of member list is agreed by all existing workers.

Not exposed to FTLib though, a `report_join` function in consensus protocol will be called inside the `confirm` function if the worker is just launched and has not reported before. For a process in a worker's lifetime, the `report_join` will be called only once.

Every time FTLib succeeds or fails to perform collective operations, it will call the corresponding consensus function to inform the consensus protocol of whether any actions need to be taken. However, these two functions do not have to act meaningfully.

## Consensus API Introduction

### 1. BasicConsensus.confirm()

### 2. BasicConsensus.get_member_list()

### 3. ? BasicConsensus.get_rank_size()

### 4. BasicConsensus.average_failure()

### 5. BasicConsensus.average_success()

