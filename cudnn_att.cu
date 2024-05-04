// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#include <cudnn_frontend.h>
#include <cuda_bf16.h>
#include <nvtx3/nvToolsExt.h>

// Specific configurations based on the enabled precision
#if defined(ENABLE_FP32)
typedef float floatX;

// use fp16 (note: this may require gradient scaler, currently not implemented!)
#elif defined(ENABLE_FP16)
typedef half floatX;
#define CUBLAS_LOWP CUDA_R_16F

#else // Default to bfloat16
typedef __nv_bfloat16 floatX;
#endif

// CUDA error checking
static void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// Profiler utils
namespace {
    class NvtxRange {
    public:
        NvtxRange(const char* s) { nvtxRangePush(s); }

        NvtxRange(const std::string& base_str, int number) {
            std::string range_string = base_str + " " + std::to_string(number);
            nvtxRangePush(range_string.c_str());
        }

        ~NvtxRange() { nvtxRangePop(); }
    };
}
#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

namespace fe = cudnn_frontend;
#if CUBLAS_LOWP == CUDA_R_16BF
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#else
#define CUDNN_16BIT fe::DataType_t::HALF
#endif

static cudnnHandle_t cudnn_handle;
static size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256MiB!)
static void* cudnn_workspace = NULL;
#define checkCudnnErr(err) assert((int)err == 0);

static void checkCudnnFE(fe::error_object e, const char *file, int line) {
    if(!e.is_good()) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, e.err_msg.c_str());
        exit(EXIT_FAILURE);
    }
}
#define checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

using graph_tensors_fwd = std::tuple<std::shared_ptr<fe::graph::Graph>,
    std::shared_ptr<fe::graph::Tensor_attributes>,  // Q,
    std::shared_ptr<fe::graph::Tensor_attributes>,  // K,
    std::shared_ptr<fe::graph::Tensor_attributes>,  // V,
    std::shared_ptr<fe::graph::Tensor_attributes>,  // Attn_scale,
    std::shared_ptr<fe::graph::Tensor_attributes>,  // O
    std::shared_ptr<fe::graph::Tensor_attributes> // Stats
>;

using graph_tensors_bwd = std::tuple<std::shared_ptr<fe::graph::Graph>,
    std::shared_ptr<fe::graph::Tensor_attributes>,  // Q,
    std::shared_ptr<fe::graph::Tensor_attributes>,  // K,
    std::shared_ptr<fe::graph::Tensor_attributes>,  // V,
    std::shared_ptr<fe::graph::Tensor_attributes>,  // O
    std::shared_ptr<fe::graph::Tensor_attributes>,  // dO
    std::shared_ptr<fe::graph::Tensor_attributes>,  // Stats
    std::shared_ptr<fe::graph::Tensor_attributes>,  // Attn_scale,
    std::shared_ptr<fe::graph::Tensor_attributes>,  // dQ,
    std::shared_ptr<fe::graph::Tensor_attributes>,  // dK,
    std::shared_ptr<fe::graph::Tensor_attributes> // dV
>;

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast
using cache_type_fwd = std::unordered_map<std::size_t, graph_tensors_fwd>;
using cache_type_bwd = std::unordered_map<std::size_t, graph_tensors_bwd>;

// Loosely based on cuDNN frontend samples functions and massively simplified
template <typename... Args>
auto lookup_cache_or_build_graph_fwd(Args... args) {
    static cache_type_fwd user_maintained_cache_fwd;
    auto [B, H, T, HS, is_inference_only] = std::make_tuple(args...);

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("attn_scale")
                                        .set_dim({1, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_is_pass_by_value(true)
                                        .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(true);

    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->set_output(true).set_dim({B, H, T, HS}).set_stride({H * HS * T, HS, H * HS, 1});

    assert(stats == nullptr || is_inference_only == false);
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
            .set_dim({B, H, T, 1})
            .set_stride({H * T, T, 1, 1});
    }

    checkCudnnFE(graph->validate());
    auto key = graph->key();
    auto it = user_maintained_cache_fwd.find(key);
    if (it != user_maintained_cache_fwd.end()) {
        return it->second;
    }

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    auto tuple = std::make_tuple(graph, Q, K, V, attn_scale, O, stats);
    user_maintained_cache_fwd.insert({key, tuple});
    return tuple;
}

template <typename... Args>
auto lookup_cache_or_build_graph_bwd(Args... args) {
    static cache_type_bwd user_maintained_cache_bwd;
    auto [B, NH, T, HS] = std::make_tuple(args...);

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_dim({B, NH, T, HS})
                               .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_dim({B, NH, T, HS})
                               .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_dim({B, NH, T, HS})
                               .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto O = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("O")
                               .set_dim({B, NH, T, HS})
                               .set_stride({NH * HS * T, HS, NH * HS, 1}));
    auto dO = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("dO")
                                .set_dim({B, NH, T, HS})
                                .set_stride({NH * HS * T, HS, NH * HS, 1}));

    auto stats = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("stats")
                                   .set_dim({B, NH, T, 1})
                                   .set_stride({NH * T, T, 1, 1})
                                   .set_data_type(fe::DataType_t::FLOAT));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("attn_scale")
                                        .set_dim({1, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_is_pass_by_value(true)
                                        .set_data_type(fe::DataType_t::FLOAT));
    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes()
        .set_name("flash_attention_backward")
        .set_causal_mask(true)
        .set_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1});
    dK->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1});
    dV->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1});

    checkCudnnFE(graph->validate());
    auto key = graph->key();
    auto it = user_maintained_cache_bwd.find(key);
    if (it != user_maintained_cache_bwd.end()) {
        return it->second;
    }

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    auto tuple = std::make_tuple(graph, Q, K, V, O, dO, stats, attn_scale, dQ, dK, dV);
    user_maintained_cache_bwd.insert({key, tuple});
    return tuple;
}

void attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             int B, int T, int NH, int C) {
    NVTX_RANGE_FN();
    int HS = C / NH; // number of features per head
    bool is_inference_only = (stats == nullptr);

    // Get graph and tensors from cache (or generate it on first use)
    auto [graph, Q, K, V, attn_scale, O, softmax_stats] =
        lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = inp;
    void* devPtrK = (inp + C);
    void* devPtrV = (inp + 2 * C);
    float attn_scale_cpu = 1.0 / sqrtf(HS);
    void* devPtrO = out;

    // Build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ}, {K, devPtrK}, {V, devPtrV}, {attn_scale, &attn_scale_cpu}, {O, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    if (is_inference_only == false) {
        variant_pack[softmax_stats] = stats;
    }

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    if (graph->get_workspace_size() > cudnn_workspace_size) {
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    // Execute graph
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

void attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, float* stats, // inputs
                              int B, int T, int NH, int C) {
    NVTX_RANGE_FN();
    int HS = C / NH; // number of features per head

    // Get graph and tensors from cache (or generate it on first use)
    auto [graph, Q, K, V, O, dO, Stats, attn_scale, dQ, dK, dV] =
        lookup_cache_or_build_graph_bwd(B, NH, T, HS);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = qkvr;
    void* devPtrK = (qkvr + NH * HS);
    void* devPtrV = (qkvr + 2 * NH * HS);
    void* devPtrO = o;
    void* devPtrdO = dout;
    void* devPtrStats = stats;
    float attn_scale_cpu = 1.0 / sqrtf(HS);

    void* devPtrdQ = dqkvr;
    void* devPtrdK = (dqkvr + NH * HS);
    void* devPtrdV = (dqkvr + 2 * NH * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ}, {K, devPtrK}, {V, devPtrV}, {O, devPtrO}, {dO, devPtrdO}, {Stats, devPtrStats},
        {dQ, devPtrdQ}, {dK, devPtrdK}, {dV, devPtrdV},
        {attn_scale, &attn_scale_cpu}};

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    if (graph->get_workspace_size() > cudnn_workspace_size) {
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    // Execute graph
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

void create_cudnn() {
    checkCudnnErr(cudnnCreate(&cudnn_handle));
}

void destroy_cudnn() {
    if (cudnn_workspace != NULL) { cudaCheck(cudaFree(cudnn_workspace)); }
    checkCudnnErr(cudnnDestroy(cudnn_handle));
}