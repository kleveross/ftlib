## Communication Library

We believe there is no need to re-implement a communication library compatible with framework-specific data structures. Therefore, FTLib will import a user-specific communication library during initialization. 

There are also deep learning frameworks that require highly customized usage of the collective communication APIs. In this case, we also expose several low-level APIs to facilitate that.

Since we are working with ElasticDL, some low-level APIS are exposed.

Users are also able to combine operations from the communication library as well as the deep learning frameworks to customize a function and throw it to `FTLib.execute()` to perform all-in-one action. In this way, we can save the synchronizations between actions and overlap the communication and computation actions wrapped in `FTLib.execute()`.

Given most communication libraries bound with deep learning frameworks require an initialization step, we ask the `Framework` to provide another two APIs: `rebuild` and `abort_communicator`. The `rebuild` API does the initialization stuff while the `abort_communicator` covers all de-initialization things, which includes `ncclCommAbort` if `NCCL` is the backend or other API calls necessary before re-initialization.

## Communication API Introduction (TODO: move to Dev-guide)

### 1. communication.rebuild()

### 2. communication.abort_communicator()

### 3. communication.allreduce()

### 4. communication.broadcast()

### 5. communication.barrier()