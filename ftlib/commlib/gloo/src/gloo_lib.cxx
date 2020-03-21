#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <assert.h>
#include <time.h>
#include <pthread.h>
#include <time.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "nccl.h"

namespace py = pybind11;

