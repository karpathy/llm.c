// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#define NOMINMAX
#include "dnn_att.h"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include "sycl_utils.h"
using namespace dnnl::graph;




// Specific configurations based on the enabled precision
#if defined(ENABLE_FP32)
static_assert(false, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#elif defined(ENABLE_FP16)
#define CUDNN_16BIT fe::DataType_t::HALF
#else // Default to bfloat16
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#endif



using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;


enum UIDs {
    Q_UID,
    K_UID,
    V_UID,
    Attn_scale_UID,
    O_UID,
    Stats_UID,
    dO_UID,
    dQ_UID,
    dK_UID,
    dV_UID
};

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast

// Crude DNN implementation of fwd
graph lookup_cache_or_build_graph_fwd(int B,int H,int T,int HS, int is_inference_only, sycl::queue &q, dnnl::engine::kind &ekind) {

    //static cache_type_fwd user_maintained_cache_fwd;


    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    
    
    dims Q_UID_dim{B, H, T, HS};
    dims K_UID_dim{B, H, T, HS};
    dims Q_K_UID_dim{B, H, T, T};
    dims V_UID_dim{B, H, T, HS};
    dims attn_scale_dim{1, 1, 1, 1};
    
    logical_tensor Q_UID_desc{0, data_type::f32, Q_UID_dim, layout_type::strided};
    logical_tensor K_UID_desc{1, data_type::f32, K_UID_dim, layout_type::strided};
    logical_tensor Q_K_UID_desc{2, data_type::f32, Q_K_UID_dim, layout_type::strided};
    
    op matmul_qk {0, op::kind::MatMul, {Q_UID_desc, K_UID_desc},
            {Q_K_UID_desc}, "matmul_qk"};
    matmul_qk.set_attr<bool>(op::attr::transpose_b, true);
    
    logical_tensor attn_scale_desc {3, data_type::f32, attn_scale_dim,
            layout_type::strided, logical_tensor::property_type::constant};
    logical_tensor scaled_Q_K_UID_desc {
            4, data_type::f32, Q_K_UID_dim, layout_type::strided};
    op scale_div {1, op::kind::Divide, {Q_K_UID_desc, attn_scale_desc},
            {scaled_Q_K_UID_desc}, "scale_div"};
    
    
    logical_tensor softmax_Q_K_UID_desc {
            5, data_type::f32, Q_K_UID_dim, layout_type::strided};
    op softmax {
            2, op::kind::SoftMax, {scaled_Q_K_UID_desc}, {softmax_Q_K_UID_desc}, "softmax"};
    softmax.set_attr<int64_t>(op::attr::axis, -1);

    logical_tensor V {
            6, data_type::f32, V_UID_dim, layout_type::strided};
    logical_tensor matmul_Q_K_V_UID_desc {
            7, data_type::f32, Q_K_UID_dim, layout_type::strided};
    op matmul_v {3, op::kind::MatMul, {softmax_Q_K_UID_desc, V},
            {matmul_Q_K_V_UID_desc}, "matmul_v"};
            
    graph g(ekind);
    g.add_op(matmul_qk);
    g.add_op(scale_div);
    g.add_op(softmax);
    g.add_op(matmul_v);
    g.finalize();
    

}

void lookup_cache_or_build_graph_bwd(int B, int NH, int T, int HS) {
    
    //to-do
    return ;
}


void attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             int B, int T, int NH, int C,
                             sycl::queue &q, engine::kind &ekind) {

    allocator alloc = sycl_interop::make_allocator(sycl_malloc_wrapper, sycl_free_wrapper);
    engine eng = sycl_interop::make_engine_with_allocator(
            q.get_device(), q.get_context(), alloc);
    stream strm = dnnl::sycl_interop::make_stream(eng, q);
    

    int HS = C / NH; // number of features per head
    bool is_inference_only = (stats == nullptr);
    

    // Get graph and tensors from cache (or generate it on first use)
    graph g = lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only, q, ekind);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = inp;
    void* devPtrK = (inp + C);
    void* devPtrV = (inp + 2 * C);
    float attn_scale_cpu = 1.0 / sqrtf(HS);
    void* devPtrO = out;

    // Build variant pack
    std::unordered_map<int64_t , void*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    if (is_inference_only == false) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    //(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    //cudaCheck(cudaGetLastError());
    
    
    std::vector<partition> partitions = g.get_partitions();
    
    std::vector<logical_tensor> inputs = partitions[0].get_input_ports();
    std::vector<logical_tensor> outputs = partitions[0].get_output_ports();
    compiled_partition sdpa_cpartition
            = partitions[0].compile(inputs, outputs, eng);

    std::vector<tensor> inputs_ts, outputs_ts;
    std::vector<std::shared_ptr<void>> data_buffer;
    std::unordered_map<size_t, tensor> global_outputs_ts_map;
    // This is helper function 
    allocate_sycl_graph_mem(
            inputs_ts, inputs, data_buffer, global_outputs_ts_map, q, eng, true);
    allocate_sycl_graph_mem(outputs_ts, outputs, data_buffer,
            global_outputs_ts_map, q, eng, false);

    sdpa_cpartition.execute(strm, inputs_ts, outputs_ts);
    strm.wait();

}

void attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, float* stats, // inputs
                              int B, int T, int NH, int C) {
    // to-do
    return;
}


int main(){

  auto ekind = engine::kind::gpu;
        
  sycl::queue q = (ekind == engine::kind::gpu)
            ? sycl::queue(
                    sycl::gpu_selector_v, sycl::property::queue::in_order {})
            : sycl::queue(
                    sycl::cpu_selector_v, sycl::property::queue::in_order {});

  
    
    int B = 32; // batch size
    int T = 128; // sequence length
    int C = 768; // embedding size
    int NH = 1;

    
    // create host memory of random numbers
    float* out = (float*)malloc(B * T * NH * C * sizeof(float));
    float* stats = (float*)malloc(B * T * NH * C * sizeof(float));
    float* inp = make_random_float(B * T * NH * C * sizeof(float));

    // Device memory allocation
    float* d_out = sycl::malloc_device<float>(B * T * NH * C, q);
    float* d_stats = sycl::malloc_device<float>(B * T *NH * C, q);
    float* d_inp = sycl::malloc_device<float>(B * T * NH * C, q);

    // Copy data to device
    q.memcpy(d_inp, inp, B * T * NH * C * sizeof(float)).wait();
    q.memcpy(d_stats, stats, B * T * NH * C * sizeof(float)).wait();
    q.memcpy(d_out, out, B * T * NH * C * sizeof(float)).wait();
     
    attention_forward_cudnn(d_out, d_stats, d_inp, B, T, NH, C, q, ekind);
    

  return 0;
}