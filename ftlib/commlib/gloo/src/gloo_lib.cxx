#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>
#include <string>

#include <assert.h>
#include <stdlib.h>

#include "gloo/transport/tcp/device.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/file_store.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/allreduce_ring.h"
#include "gloo/broadcast_one_to_all.h"
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
    py::array_t<T> broadcast(py::array_t<T> data, int root_rank) {
        py::buffer_info buff = data.request();
        const int count = data.size();
        std::vector<T*> ptrs = std::vector<T*>(count);
        for (size_t i = 0; i < count; i++)
        {
            ptrs[i] = static_cast<T*>(buff.ptr) + i;
        }
        
        auto op = std::make_shared<gloo::BroadcastOneToAll<T>>(context, ptrs, count, root_rank);
        op->run();
        return data;
    }

    template<typename T>
    py::array_t<T> allreduce(py::array_t<T> data) {
        py::buffer_info buff = data.request();
        const int count = data.size();
        std::vector<T*> ptrs = std::vector<T*>(count);
        for (size_t i = 0; i < count; i++)
        {
            ptrs[i] = static_cast<T*>(buff.ptr) + i;
        }

        auto op = std::make_shared<gloo::AllreduceRing<T>>(context, ptrs, count);
        op->run();
        return data;
    }

    void barrier(void) {
        gloo::BarrierOptions opts(context);
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