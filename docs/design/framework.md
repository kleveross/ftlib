## Framework

Although it is named as `Framework`, it is defacto a wrapper for a communication library. We believe there is no need to re-implement a communication library compatible with data structures differing from this dl framework to another

On the top of different dl frameworks, the `Framework` in FTLib provides basic collective operation APIs, including `allreduce`, `broadcast` and `barrier`. Each implementation of `Framework` is expected to work with the data structure in a dl framework.

Given most communication libraries bound with dl frameworks require a initialization step, we ask the `Framework` to provide another two APIs: `rebuild` and `abort_communicator`. The `rebuild` API does the initialization stuff while the `abort_communicator` covers all de-initialization things.

## Framework API Introduction

### 1. Framework.rebuild

### 2. Framework.abort_communicator

### 3. Framework.allreduce

### 4. Framework.broadcast

### 5. Framework.barrier