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

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


// ------------ section for NCCL API python binding with timeout -------------//


struct nccl_context {
    nccl_context() {
        send_buff = NULL;
        recv_buff = NULL;
        cudaEventCreate(&finish);
        max_duration = 10;
        len = 0;

        int deviceCount = 0;
        CUDACHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
           printf("Cannot find any CUDA device.\n");
        }
        if (deviceCount > 1) {
           printf("Found %d CUDA device(s).\n", deviceCount);
        }
        CUDACHECK(cudaStreamCreate(&s));
    }

    void generateNCCLID() {
        ncclGetUniqueId(&id);
    }

    py::array_t<int> getNCCLID() {
        auto result = py::array_t<int>(NCCL_UNIQUE_ID_BYTES);
        py::buffer_info buff = result.request();
        int *ptr = (int*) buff.ptr;

        for (size_t i = 0; i < NCCL_UNIQUE_ID_BYTES; i++) {
            ptr[i] = int(id.internal[i]);
        }
        return result;
    }

    bool setInput(py::array_t<float> input_data) {
        len = input_data.size();
        const unsigned int bytes = len * sizeof(float);

        if (send_buff != NULL) {
            cudaFree(send_buff);
        }
        if (recv_buff != NULL) {
            cudaFree(recv_buff);
        }
        CUDACHECK(cudaMalloc(&send_buff, bytes));
        CUDACHECK(cudaMalloc(&recv_buff, bytes));

        auto input_data_info = input_data.request();
        auto res = cudaMemcpy(send_buff, input_data_info.ptr, bytes, cudaMemcpyHostToDevice);

        return res == cudaSuccess;
    }

    py::array_t<float> getOutput() {
	const unsigned int bytes = len * sizeof(float);
        auto output_data = py::array_t<float>(len);
        auto output_data_info = output_data.request();
        CUDACHECK(cudaMemcpy(output_data_info.ptr, recv_buff, bytes, cudaMemcpyDeviceToHost));

        return output_data;
    }

    bool allreduceAsync() {
        NCCLCHECK(ncclAllReduce(
                                (const void*)send_buff,
                                (void*)recv_buff,
                                len,
                                ncclFloat,
                                ncclSum,
                                comm,
                                s
                                )
        );
        auto res = cudaEventRecord(finish);
        return res == cudaSuccess;
    }

    bool checkAllreduce(int x_sec) {
        auto res = cudaEventQuery(finish);
        if (res != cudaSuccess)
            sleep(x_sec);
        return res == cudaSuccess;
    }

    bool commAbort() {
        return ncclCommAbort(comm) == ncclSuccess;
    }

    void setNCCLID(py::array_t<int> rr_id) {
        py::buffer_info buff = rr_id.request();
        int *ptr = (int*) buff.ptr;
        if (id.internal != nullptr) {
            for (size_t i = 0; i < NCCL_UNIQUE_ID_BYTES; i++) {
                id.internal[i] = char(ptr[i]);
            }
        } 
        
    }

    void commInitRank(const int size, const int rank) {
        NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
    }

    void destroyComm() {
        NCCLCHECK(ncclCommDestroy(comm));
    }

    ncclUniqueId id;
    cudaEvent_t finish;
    ncclComm_t comm;
    cudaStream_t s;
    float *send_buff, *recv_buff;
    size_t len;
    int max_duration;
};

PYBIND11_MODULE(fault_tolerant_lib, m) {
    m.doc() = "pybind11 fault_tolerant_lib plugin";

    py::class_<nccl_context>(m, "nccl_context")
        .def(py::init<>())
        .def("generateNCCLID", &nccl_context::generateNCCLID)
        .def("getNCCLID", &nccl_context::getNCCLID)
        .def("setNCCLID", &nccl_context::setNCCLID)
        .def("setInput", &nccl_context::setInput)
        .def("getOutput", &nccl_context::getOutput)
        .def("commAbort", &nccl_context::commAbort)
        .def("allreduceAsync", &nccl_context::allreduceAsync)
        .def("checkAllreduce", &nccl_context::checkAllreduce)
        .def("destroyComm", &nccl_context::destroyComm)
        .def("commInitRank", &nccl_context::commInitRank);
}
