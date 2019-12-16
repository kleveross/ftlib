## Consensus

The consensus protocol in FTLib acts as a shadow precondition for any collective communication operations. Any changes from the consensus protocol will reset the `_initialized` flag of FTLib to `False`, deterring any communication operations after the rebuild procedure returns success. Depending on the choice of passive or active mode, such change takes place immediately or eventually (when FTLib check the consensus next time).

A member list is maintained by an implementation of consensus protocol.

The rank-assign scheme in FTLib can extract worker identification from the member list, such like address of each worker, which helps the rank-assign scheme to designate individual rank number to each worker. We map the identification of each worker with a hash function and sort the workers with respect to hash value. The worker with minimal hash value will be considered as the *master* worker. While the *master* worker acts identically as other workers in the training phase, it may have additional components or operations during initialization or other stages.

As described in [`FTLib`](./ftlib.md), if active mode is on, the `confirm` API will be called for in `initialized()` and `skip_allreduce()`, checking if member list has changed before communication API calls. 

Not exposed to FTLib though, a `report_join` API in consensus protocol will be called inside the `confirm` function if the worker is freshly launched and has not successfully reported before. During the whole lifetime of a worker, the `report_join` API will not be called for a second time after a successfully trial.

Every time FTLib succeeds or fails to perform collective operations, it will call the corresponding functions to inform the consensus whether any actions need to be taken. However, these two functions can be ignored if no actions are needed by the specific consensus protocol.

## Consensus API Introduction (TODO: move to dev-guide)

### 1. BasicConsensus.confirm()

### 2. BasicConsensus.report_join()

### 3. BasicConsensus.get_member_list()

### 4. BasicConsensus.passive_or_active()

### 5. BasicConsensus.new_member_join()

### 6. optional: BasicConsensus.average_failure()

### 7. optional: BasicConsensus.average_success()

