/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#include "sycl_common.h"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>


// forward declarations of functions defined in cudnn_att.cpp
void create_dnn();
void destroy_dnn();
void attention_forward_dnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             int B, int T, int NH, int C);

void attention_backward_dnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, float* stats, // inputs
                              int B, int T, int NH, int C);
  
 
//============dnn utils========================= 
#define UNUSED(x) ((void)(x))
 
struct sycl_deletor {
    sycl_deletor() = delete;
    ::sycl::context ctx_;
    sycl_deletor(const ::sycl::context &ctx) : ctx_(ctx) {}
    void operator()(void *ptr) {
        if (ptr) ::sycl::free(ptr, ctx_);
    }
};
                            
void *sycl_malloc_wrapper(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    return malloc_shared(size, *static_cast<const ::sycl::device *>(dev),
            *static_cast<const ::sycl::context *>(ctx));
}

void sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event) {
    // Device is not used in this example, but it may be useful for some users
    // application.
    UNUSED(device);
    // immediate synchronization here is for test purpose. For performance,
    // users may need to store the ptr and event and handle them separately
    if (event) {
        auto sycl_deps_ptr = static_cast<::sycl::event *>(event);
        sycl_deps_ptr->wait();
    }
    free(ptr, *static_cast<const ::sycl::context *>(context));
}


void allocate_sycl_graph_mem(std::vector<dnnl::graph::tensor> &tensors,
        const std::vector<dnnl::graph::logical_tensor> &lts,
        std::vector<std::shared_ptr<void>> &data_buffer, sycl::queue &q,
        const dnnl::engine &eng) {
    tensors.reserve(lts.size());
    for (const auto &lt : lts) {
        const auto mem_size = lt.get_mem_size();

        // memory allocation
        data_buffer.push_back({});
        data_buffer.back().reset(::sycl::malloc_shared(mem_size, q.get_device(),
                                         q.get_context()),
                sycl_deletor {q.get_context()});

        dnnl::graph::tensor new_ts {lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);
    }
}

void allocate_sycl_graph_mem(std::vector<dnnl::graph::tensor> &tensors,
        const std::vector<dnnl::graph::logical_tensor> &lts,
        std::vector<std::shared_ptr<void>> &data_buffer,
        std::unordered_map<size_t, dnnl::graph::tensor> &global_outputs_ts_map,
        sycl::queue &q, const dnnl::engine &eng, bool is_input) {
    tensors.reserve(lts.size());
    for (const auto &lt : lts) {
        const auto lt_id = lt.get_id();
        const auto mem_size = lt.get_mem_size();

        // check if the input is an output of another partition
        if (is_input) {
            auto pos = global_outputs_ts_map.find(lt_id);
            if (pos != global_outputs_ts_map.end()) {
                tensors.push_back(pos->second);
                continue;
            }
        }

        // memory allocation
        data_buffer.push_back({});
        data_buffer.back().reset(::sycl::malloc_shared(mem_size, q.get_device(),
                                         q.get_context()),
                sycl_deletor {q.get_context()});

        dnnl::graph::tensor new_ts {lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);

        // record the connection relationship between partitions
        if (!is_input) global_outputs_ts_map[lt_id] = tensors.back();
    }
}

#endif // CUDNN_ATT_H