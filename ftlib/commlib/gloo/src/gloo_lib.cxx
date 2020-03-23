#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <assert.h>
#include <stdlib.h>

#include "gloo/transport/tcp/device.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/file_store.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/allreduce.h"
#include "gloo/broadcast.h"
#include "gloo/barrier.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<float>);

struct Gloo {
    gloo::transport::tcp::attr attr;
    std::shared_ptr<::gloo::transport::Device> device;
    std::shared_ptr<gloo::rendezvous::Context> context;

    Gloo(
        std::string &store_path,
        std::string &prefix,
        int rank, int size) {
        attr.iface = "lo";
        attr.ai_family = AF_UNSPEC;
        device = gloo::transport::tcp::CreateDevice(attr);

        auto fileStore = gloo::rendezvous::FileStore(store_path);
        auto prefixStore = gloo::rendezvous::PrefixStore(prefix, fileStore);

        context = std::make_shared<gloo::rendezvous::Context>(rank, size);
        context->connectFullMesh(prefixStore, device);
    }

    template<typename T>
    py::array_t<T> broadcast(py::array_t<T> data, int root_rank, int timeout) {
        py::buffer_info buff = data.request();
        const int count = data.size();
        
        gloo::BroadcastOptions opts(context);
        opts.setRoot(root_rank);
        
        auto t = std::chrono::duration<int, std::milli>(timeout*1000);
        opts.setTimeout(t);

        opts.setInput(const_cast<T*>((const T*) buff.ptr), count);
        opts.setOutput((T*) buff.ptr, count);

        gloo::broadcast(opts);
        return data;
    }

    template<typename T>
    py::array_t<T> allreduce(py::array_t<T> data, int timeout) {
        py::buffer_info buff = data.request();
        const int count = data.size();

        gloo::AllreduceOptions opts(context);

        auto t = std::chrono::duration<int, std::milli>(timeout*1000);
        opts.setTimeout(t);

        opts.setInput(const_cast<T*>((const T*) buff.ptr), count);
        opts.setOutput((T*) buff.ptr, count);

        gloo::allreduce(opts);
        return data;
    }

    void barrier(int timeout) {
        gloo::BarrierOptions opts(context);

        auto t = std::chrono::duration<int, std::milli>(timeout*1000);
        opts.setTimeout(t);

        gloo::barrier(opts);
    }

};

PYBIND11_MODULE(gloo_lib, m) {
    m.doc() = "pybind11 gloo plugin";

    py::class_<Gloo>(m, "Gloo")
    .def(py::init<std::string&, std::string&, int, int>())
    .def("barrier", &Gloo::barrier)
    .def("broadcast", &Gloo::broadcast<float>)
    .def("allreduce", &Gloo::allreduce<float>);
}