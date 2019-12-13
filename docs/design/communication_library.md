## Communication Library

We believe there is no need to re-implement a communication library compatible with data structures differing from this dl framework to another. Therefore, FTLib will import a user-specific communication library during initialization. 

Some deep learning frameworks wrap communication libraries, such like NCCL or Gloo, and make it compatible with its own data structure. Such wrapped communication libraries can be adopted with less effort to be compatible with deep learning frameworks.

Since we are working with ElasticDL, some low-level APIS are exposed.

Users are also able to combine operations from the communication library as well as the deep learning frameworks to customize a function and throw it to `FTLib.execute()` to perform all-in-one action.

Given most communication libraries bound with dl frameworks require a initialization step, we ask the `Framework` to provide another two APIs: `rebuild` and `abort_communicator`. The `rebuild` API does the initialization stuff while the `abort_communicator` covers all de-initialization things.

## Communication API Introduction (TODO: move to Dev-guide)

### 1. communication.rebuild()

### 2. communication.abort_communicator()

### 3. communication.allreduce()

### 4. communication.broadcast()

### 5. communication.barrier()