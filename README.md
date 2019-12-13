# FTLib

[![Build Status](https://travis-ci.org/caicloud/ftlib.svg?branch=master)](https://travis-ci.org/caicloud/ftlib)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

FTLib (Fault-Tolerant Library) is a framework to keep data-parallel distributed training continue regardless worker loss or join. It exposes collective communication APIs with fault-tolerance support by gluing a `consensus` to a `communication library`, both of which can be user-specific. A distributed training using FTLib is able to continue as long as at least one single worker is alive and when new workers join the training.

## Status

Prototyping

## Design

* [Design docs](https://github.com/caicloud/ftlib/tree/master/docs/design)

## Develop Guide

**TODO**
Please refer to the [design docs](https://github.com/caicloud/ftlib/tree/master/docs/design).

## See also

* [ElasticDL](https://github.com/sql-machine-learning/elasticdl/)

## Getting started

### Where to use FTLib

- Less reliable infrastructure/script

Distributed training jobs running on less reliable infrastructure risks more as any worker or communication failure will leads to the termination of the entire job.

- Dynamic workload system

A system may reduce the total workload of distributed training jobs to release resources so that resource can be squeezed out for jobs with higher priority. Without such jobs with higher-priority, the system can increase the workload to avoid resource idling.

### Requirements

The requirements for using `FTLib` differs with choices of consensus and communication library. Please refer the `requirements.txt` under each consensus and communication library(*Not available, still in todo list*).

### Usage

Please refer [`test`](./test) for details on how to use `FTLib` in distributed training.

### Layout

```
.
├── CHANGELOG.md
├── deploy
├── docs
│   ├── design
│   └── imgs
├── ftlib
│   ├── consensus
│   ├── framework
│   ├── ftlib_status.py
│   ├── __init__.py
│   └── rank_assign_scheme.py
├── LICENSE
├── OWNERS
├── README.md
├── requirements.txt
├── ROADMAP
├── scripts
└── test
```

## License

FTLib is [Apache license](LICENSE). Implementations of consensus and communication library may come with different licenses.