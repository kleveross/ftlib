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

template<typename T>
void move_to_device(T* device_buff, T* host_buff, size_t bytes) {
    CUDACHECK(cudaMemcpy(
                        device_buff,
                        host_buff,
                        bytes,
                        cudaMemcpyHostToDevice
                        )
    );
}

template<typename T>
void move_to_host(T* host_buff, T* device_buff, size_t bytes) {
    CUDACHECK(cudaMemcpy(
                        host_buff,
                        device_buff,
                        bytes,
                        cudaMemcpyDeviceToHost
                        )
    );
}

template<typename T>
class nccl_call {
    cudaEvent_t completed;
public:
    int len;
    size_t bytes;

    T* host_buff;
    T* device_buff;

    nccl_call(const py::array_t<T>& data) {
        auto data_buffer_info = data.request();
        host_buff = static_cast<T*>(data_buffer_info.ptr);

        len = data.size();
        bytes = len * sizeof(T);
        CUDACHECK(cudaMalloc(&device_buff, bytes));

        cudaEventCreate(&completed);
    }

    ~nccl_call() {
        device_buff = nullptr;
    }

    void to_device() {
        move_to_device<T>(device_buff, host_buff, bytes);
    }

    void to_host() {
        move_to_host<T>(host_buff, device_buff, bytes);
        if (device_buff != nullptr) cudaFree(device_buff);
    }

    // void run_async() {} Not implemented in this version
    // let inherited class will implemente this virtual function

    bool check_complete() {
        auto res = cudaEventQuery(completed);

        if (res==cudaSuccess) to_host();

        return res;
    }

    void record(cudaStream_t* s_ptr) {
        CUDACHECK(cudaEventRecord(completed, *s_ptr));
    }
};

// explicit instantiation
template class nccl_call<float>;
typedef nccl_call<float> nccl_call_fp32;

struct nccl_context {
    ncclUniqueId id;
    ncclComm_t comm;
    cudaStream_t s;

    nccl_context() {
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
        int *ptr = static_cast<int*>(buff.ptr);

        for (size_t i = 0; i < NCCL_UNIQUE_ID_BYTES; i++) {
            ptr[i] = int(id.internal[i]);
        }
        return result;
    }

    void setNCCLID(py::array_t<int>& rr_id) {
        py::buffer_info buff = rr_id.request();
        int *ptr = static_cast<int*>(buff.ptr);
        if (id.internal != nullptr) {
            for (size_t i = 0; i < NCCL_UNIQUE_ID_BYTES; i++) {
                id.internal[i] = char(ptr[i]);
            }
        }

    }

    bool commAbort() {
        return ncclCommAbort(comm) == ncclSuccess;
    }

    void commInitRank(const int size, const int rank, py::array_t<int> rr_id) {
        setNCCLID(rr_id);

        NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
    }

    template<typename T>
    std::unique_ptr< nccl_call<T> > broadcast(py::array_t<T> data, int root_rank) {
        std::unique_ptr< nccl_call<T> > call(new nccl_call<T>(data));

        call->to_device();

        // change dtype to others when new data type is introduced
        // dtype shall be assigned based on T
        ncclDataType_t dtype = ncclFloat;
        // launch broadcast
        NCCLCHECK(ncclBroadcast(
                                (const void*)call->device_buff,
                                (void*)call->device_buff,
                                call->len,
                                dtype,
                                root_rank,
                                comm,
                                s
                                )
        );

        call->record(&s);

        return call;
    }

    template<typename T>
    std::unique_ptr< nccl_call<T> > allreduce(py::array_t<T> data) {
        std::unique_ptr< nccl_call<T> > call(new nccl_call<T>(data));

        call->to_device();

        // change dtype to others when new data type is introduced
        // dtype shall be assigned based on T
        ncclDataType_t dtype = ncclFloat;
        ncclRedOp_t reduce_ops = ncclSum;
        // launch allreduce
        NCCLCHECK(ncclAllReduce(
                                (const void*)call->device_buff,
                                (void*)call->device_buff,
                                call->len,
                                dtype,
                                reduce_ops,
                                comm,
                                s
                                )
        );

        call->record(&s);

        return call;
    }
};

template std::unique_ptr< nccl_call<float> > nccl_context::broadcast(py::array_t<float>, int);
template std::unique_ptr< nccl_call<float> > nccl_context::allreduce(py::array_t<float>);

PYBIND11_MODULE(fault_tolerant_lib, m) {
    m.doc() = "pybind11 fault_tolerant_lib plugin";

    py::class_<nccl_context>(m, "nccl_context")
        .def(py::init<>())
        .def("generateNCCLID", &nccl_context::generateNCCLID)
        .def("getNCCLID", &nccl_context::getNCCLID)
        .def("commAbort", &nccl_context::commAbort)
        .def("commInitRank", &nccl_context::commInitRank)
        .def("allreducefp32", &nccl_context::allreduce<float>)
        .def("broadcastfp32", &nccl_context::broadcast<float>);

    py::class_<nccl_call_fp32>(m, "nccl_call_fp32")
        .def(py::init<const py::array_t<float>&>())
        .def("check_complete", &nccl_call_fp32::check_complete);
}
