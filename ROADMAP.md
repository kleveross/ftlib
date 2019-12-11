# Road Map for Fault-Tolerant Library

Here we lay out the plan for a short-term period as well as long-term goal.

**The ultimate goal for `FTLib` is to help any data-parallel distributed trainings continue after worker lost/join.**

## Short-Term

### Short-Term Goal

For Consensus part, we aimed to provide another choice besides the shared-storage-based. A gossip-based consensus protocol will be introduced for cluster without shared-storage.

For FTLib part, we will refactor again after a refresh of design docs.

We may consider adding another framework/communciation library if time permits.

### Steps

1. setup ci []
2. add gossip-based consensus []
2. refresh design docs and provide developer guide []
3. refactor code according to the most recent design docs []
5. setup python package structure []
6. optional: add another communciation library support []

### Presumed Time

We consider the short-term period ends by the end of this December. It may be subject to changes due to our workloads from other projects.
