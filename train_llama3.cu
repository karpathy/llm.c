/*
GPT-2 Transformer Neural Net training loop. See README.md for usage.
*/
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
// ----------- CPU utilities -----------
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
// defines: create_dir_if_not_exists, find_max_step, ends_with_bin
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
// defines: evalloader_init, evalloader_reset, evalloader_next_batch, evalloader_free
#include "llmc/dataloader.h"
// defines: manual_seed, normal_ (same as torch.manual_seed and torch.normal)
#include "llmc/rand.h"
// defines: lr_scheduler_init, get_learning_rate
#include "llmc/schedulers.h"
// defines: sample_softmax, random_f32
#include "llmc/sampler.h"
// defines: logger_init, logger_log_eval, logger_log_val, logger_log_train
#include "llmc/logger.h"
// defines: get_flops_promised
#include "llmc/mfu.h"
// defines: OutlierDetector, init_detector, update_detector
#include "llmc/outlier_detector.h"
// ----------- GPU utilities -----------
// defines:
// WARP_SIZE, MAX_1024_THREADS_BLOCKS, CEIL_DIV, cudaCheck, PRECISION_MODE
// NVTX_RANGE_FN
#include "llmc/cuda_common.h"
// defines:
// Packed128, f128, x128
// warpReduceSum, warpReduceMax, blockReduce, copy_and_cast_kernel, cudaMallocConditionallyManaged
#include "llmc/cuda_utils.cuh"
// defines: CUBLAS_LOWP, cublasCheck, cublaslt_workspace_size, cublaslt_workspace
// defines: cublas_compute, cublaslt_handle, cublas_handle
#include "llmc/cublas_common.h"
// ----------- Layer implementations in CUDA -----------
// defines: encoder_forward, encoder_backward
#include "llmc/encoder.cuh"
// defines: layernorm_forward, residual_forward, fused_residual_forward5, layernorm_backward
#include "llmc/layernorm.cuh"
// defines: rmsnorm_forward, fused_residual_rmsnorm_forward5, rmsnorm_backward
#include "llmc/rmsnorm.cuh"
// defines: matmul_cublaslt, matmul_forward, matmul_backward, gelu_forward, gelu_backward_inplace
#include "llmc/matmul.cuh"
#ifdef ENABLE_CUDNN
// defines: create_cudnn, destroy_cudnn, attention_forward_cudnn, attention_backward_cudnn
#include "llmc/cudnn_att.h"
#else
// defines: attention_forward, attention_backward
#include "llmc/attention.cuh"
#endif
// defines: fused_classifier
#include "llmc/fused_classifier.cuh"
// defines: adamw_kernel3
#include "llmc/adamw.cuh"
// defines: global_norm_squared
#include "llmc/global_norm.cuh"
// defines: repkv_forward, repkv_backward
#include "llmc/repkv.cuh"
// defines: precompute_freqs_cis, rope_forward, rope_backward_inplace
#include "llmc/rope.cuh"
// defines: swiglu_forward, swiglu_backward
#include "llmc/swiglu.cuh"
// ----------- Multi-GPU support -----------
// defines: ncclFloatX, ncclCheck, MultiGpuConfig, ShardInfo
// defines: printf0, multi_gpu_config
// defines: multi_gpu_config_init, multi_gpu_config_free
// defines: set_zero_configs, multi_gpu_cpu_float_sum, multi_gpu_barrier
// defines: multi_gpu_get_shard_offset, multi_gpu_async_reduce_gradient
#include "llmc/zero.cuh"

// ----------------------------------------------------------------------------
// global vars for I/O
char filename_buffer[512];

// ----------------------------------------------------------------------------
// global vars containing information about the GPU this process is running on
cudaDeviceProp deviceProp; // fills in common_start()
cudaStream_t main_stream;
// buffer size to use for device <-> disk io
constexpr const size_t IO_BUF_SIZE = 32 * 1024 * 1024;

// ----------------------------------------------------------------------------
// LLama-3 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 128256
    int padded_vocab_size; // padded to e.g. %128==0, 128256
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of query heads in attention, e.g. 32
    int num_kv_heads; // number of key and value heads in attention, e.g. 8 (<-- new in Llama 3)
    int channels; // number of channels, e.g. 2048
    int multiple_of; // used in feedforward layer sizing, e.g. 1024 (<-- new in Llama 3)
    int use_scaled_rope; // whether to use scaled rope
    float ffn_dim_multiplier; // multiplier used in feedforward layer, e.g. 1.3 (<-- new in Llama 3)
    float norm_eps; // epsilon used in layernorm, e.g. 1e-5
    float rope_theta; // theta used in ROPE attention, e.g. 500000.0 (<-- new in Llama 3)
    bool use_biases;  // we always allocate memory for biases; to match llama3 they are not used
    bool tied_weights; // untied for large models (3.1 8B/70B/405B), tied for small (3.2 1B/3B)
} LLama3Config;

// the parameters of the model
constexpr const int NUM_PARAMETER_TENSORS = 16;
typedef struct {
    floatX* wte; // (V, C)
    floatX* wlmhead; // (V, C)
    floatX* ln1w; // (L, C)
    floatX* ln1b; // (L, C)
    floatX* qkvw; // (L, 3*C, C)
    floatX* qkvb; // (L, 3*C)
    floatX* attprojw; // (L, C, C)
    floatX* attprojb; // (L, C)
    floatX* ln2w; // (L, C)
    floatX* ln2b; // (L, C)
    floatX* fcw; // (L, 4*C, C)
    floatX* fcb; // (L, 4*C)
    floatX* fcprojw; // (L, C, 4*C)
    floatX* fcprojb; // (L, C)
    floatX* lnfw; // (C)
    floatX* lnfb; // (C)
} ParameterTensors;
static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");

void fill_in_parameter_sizes(size_t* param_sizes, size_t* param_sizeof, LLama3Config config) {
    // see train_llama3.py write_tensors() function for detailed docs of some of the trickery here
    // trick 1: all biases are still present but set to zero
    // trick 2: the SwiGLU weights are "packed" into one, concatenated
    // trick 3: the positional embedding is replaced with the final classifier layer weights
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t L = config.num_layers;
    // calculation following the .py code inside CausalSelfAttention
    // we have to calculate the number of channels in the QKV projection
    size_t n_head = config.num_heads;
    size_t n_kv_head = config.num_kv_heads;
    size_t hd = C / n_head; // head dimension
    size_t qkv_channels = (n_head + 2*n_kv_head) * hd; // Q, K, V channels
    // calculation following the .py code inside MLP
    // we have to calculate the number of channels in the SwiGLU projections c_fc + c_fc2
    size_t hidden_dim = 4 * C;
    hidden_dim = (2 * hidden_dim) / 3;
    hidden_dim = config.ffn_dim_multiplier * hidden_dim;
    hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) / config.multiple_of);
    size_t ffn_channels = hidden_dim * 2; // c_fc + c_fc2 concatenated
    // now populate the parameter sizes
    param_sizes[0] = Vp * C; // wte
    if(config.tied_weights) {
        param_sizes[1] = 0; // no lm_head with tied weights
    } else {
        param_sizes[1] = Vp * C; // (3) lm_head (final classifier layer weights)
    }

    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b; (1) all biases are zero it's ok
    param_sizes[4] = L * (qkv_channels) * C; // qkvw
    param_sizes[5] = L * (qkv_channels); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * ffn_channels * C; // fcw; (2) this is twice the size
    param_sizes[11] = L * ffn_channels; // fcb
    param_sizes[12] = L * C * hidden_dim; // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
    // populate the parameter sizes in bytes (all the same for now, keeping for future use)
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        param_sizeof[i] = sizeof(floatX);
    }
}

// allocate memory for the parameters and point the individual tensors to the right places
void* malloc_and_point_parameters(ParameterTensors* params, size_t* param_elements, size_t *param_sizeof) {
    // calculate the total number of parameters and bytes across all tensors
    size_t num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters_bytes += param_elements[i] * param_sizeof[i];
    }
    // malloc all parameters all at once on the device
    void* params_memory;
    cudaCheck(cudaMalloc((void**)&params_memory, num_parameters_bytes));
    // assign all the tensors their place in the array
    floatX** ptrs[] = {
        &params->wte, &params->wlmhead, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    char* params_memory_iterator = (char*)params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX*)params_memory_iterator;
        params_memory_iterator += param_elements[i] * param_sizeof[i];
    }
    // tied weights?
    if(param_elements[1] == 0) {
        params->wlmhead = nullptr;
    }
    return params_memory;
}

constexpr int NUM_ACTIVATION_TENSORS = 22;
typedef struct {
    floatX* encoded; // (B, T, C)
    floatX* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    floatX* atty; // (L, B, T, C)
    // cuDNN saves only some statistics information
#if ENABLE_CUDNN
    float* att;  // (L, B, NH, T)
#else
    floatX* att; // (L, B, NH, T, T)
#endif

    floatX* residual2; // (L, B, T, C)
    floatX* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    floatX* fch; // (L, B, T, 4*C)
    floatX* fch_gelu; // (L, B, T, 4*C)
    floatX* residual3; // (L, B, T, C)
    floatX* lnf; // (B, T, C);   if LN recomputation is enabled (-r 2 and above), will be used for _all_ layernorms
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* losses; // (B, T), will be accumulated in micro-steps
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    floatX* qkvr; // (L, B, T, 3*C)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    floatX* output;

    // some additional scratch buffers
    floatX* scratch_bt4c;   // (B, T, 4*C)
    floatX* scratch_btc;    // (B, T, C)
    floatX* scratch_bt4c2;  // (B, T, 4*C), for simplicify use this one for backward pass too, probably not needed
} ActivationTensors;


struct TensorSpec {
    void** ptr;
    size_t size;
    DType type;
};


#define TENSOR_SPEC(pointer, size) TensorSpec{(void**)(&pointer), (size), dtype_of(pointer)};

void fill_in_activation_sizes(const ActivationTensors* data, TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS], size_t B, size_t T, LLama3Config config, int recompute) {
    const size_t Vp = config.padded_vocab_size;
    const size_t L = config.num_layers;
    const size_t NH = config.num_heads;
    const size_t C = config.channels;
    const size_t n_head = config.num_heads; // num query heads
    const size_t n_kv_head = config.num_kv_heads; // num key and value heads
    const size_t hd = C / n_head; // the size of each head
    const size_t qkv_channels = (n_head + 2*n_kv_head) * hd; // Q, K, V channels
    // SwiGLU-related calculation to determine the number of channels here
    size_t hidden_dim = 4 * C;
    hidden_dim = (2 * hidden_dim) / 3;
    hidden_dim = config.ffn_dim_multiplier * hidden_dim;
    hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) / config.multiple_of);
    size_t ffn_channels = hidden_dim * 2; // c_fc + c_fc2 concatenated
    size_t ffn_channels_post_gelu = hidden_dim; // swiglu will halve the channels

    tensors[0] = TENSOR_SPEC(data->encoded, B * T * C);
    // if recompute >= 1 then we will recompute the layernorm forward activation during backward pass
    tensors[1] = TENSOR_SPEC(data->ln1,  (recompute < 2) ? L * B * T * C : 0);
    tensors[2] = TENSOR_SPEC(data->ln1_mean, 0); // Llama 3 does not use this activation
    tensors[3] = TENSOR_SPEC(data->ln1_rstd, L * B * T);
    tensors[4] = TENSOR_SPEC(data->atty, L * B * T * C);
    #ifdef ENABLE_CUDNN
    // FP32 stats tensor for cuDNN to be passed to backward pass
    tensors[5] = TENSOR_SPEC(data->att, L * B * NH * T);
    #else
    tensors[5] = TENSOR_SPEC(data->att, L * B * NH * T * T);
    #endif
    tensors[6] = TENSOR_SPEC(data->residual2, L * B * T * C);
    // if recompute >= 1 then we will recompute the layernorm forward activation during backward pass
    tensors[7] = TENSOR_SPEC(data->ln2, (recompute < 2) ? L * B * T * C : 0);
    tensors[8] = TENSOR_SPEC(data->ln2_mean, L * B * T);
    tensors[9] = TENSOR_SPEC(data->ln2_rstd, L * B * T);
    tensors[10] = TENSOR_SPEC(data->fch, L * B * T * ffn_channels);
    // if recompute >= 1 then we will recompute gelu_forward during backward and use this as scratch buffer
    tensors[11] = TENSOR_SPEC(data->fch_gelu, (recompute < 1) ? L * B * T * ffn_channels_post_gelu : B * T * ffn_channels_post_gelu);
    tensors[12] = TENSOR_SPEC(data->residual3, L * B * T * C);
    tensors[13] = TENSOR_SPEC(data->lnf, B * T * C);
    tensors[14] = TENSOR_SPEC(data->lnf_mean, B * T);
    tensors[15] = TENSOR_SPEC(data->lnf_rstd, B * T);
    tensors[16] = TENSOR_SPEC(data->losses, B * T);
    tensors[17] = TENSOR_SPEC(data->qkvr, L * B * T * 3*C); // 3*C is correct - this is QKV after replication of KV
    tensors[18] = TENSOR_SPEC(data->output, B * T * max(qkv_channels, max(ffn_channels, max(NH*T, Vp))));
    tensors[19] = TENSOR_SPEC(data->scratch_bt4c, B * T * ffn_channels);
    tensors[20] = TENSOR_SPEC(data->scratch_btc, B * T * C);
    tensors[21] = TENSOR_SPEC(data->scratch_bt4c2, B * T * ffn_channels);
}

void* malloc_and_point_activations(TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS]) {
    size_t bytes = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes += tensors[i].size * sizeof_dtype(tensors[i].type);
    }
    printf0("allocating %d MiB for activations\n", (int)round(bytes / (1024 * 1024)));
    void* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, bytes));
    // cudaMalloc does not guarantee initial memory values so we memset the allocation here
    // this matters because e.g. non-cuDNN attention assumes the attention buffer is zeroed
    // todo - up to ~100ms on slow GPUs, could theoretically be more selective, but this is safer
    cudaCheck(cudaMemset(acts_memory, 0, bytes));
    char* acts_memory_iterator = (char*)acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        // extra protection so we don't accidentally use an empty buffer
        if(tensors[i].size == 0) {
            *(tensors[i].ptr) = NULL;
        }else {
            *(tensors[i].ptr) = acts_memory_iterator;
            acts_memory_iterator += tensors[i].size * sizeof_dtype(tensors[i].type);
        }
    }
    return acts_memory;
}

typedef struct {
    LLama3Config config;
    // the weights of the model, and their sizes
    ParameterTensors params;
    size_t param_elements[NUM_PARAMETER_TENSORS];
    size_t param_sizeof[NUM_PARAMETER_TENSORS];
    void* params_memory;
    size_t num_parameters;
    size_t num_parameters_bytes;
    // gradients of the weights
    ParameterTensors grads;
    void* grads_memory;
    // buffers for the AdamW optimizer
    floatOpt* m_memory;
    floatOpt* v_memory;
    float* master_weights;     // is NULL unless fp32 weights is enabled.
    // the activations of the model, and their sizes
    ActivationTensors acts;
    TensorSpec acts_specs[NUM_ACTIVATION_TENSORS];
    void* acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after the last backward micro-batch, will be populated with mean loss across all GPUs and micro-steps
    float* accumulated_mean_loss; // GPU buffer used to accumulate loss across micro-steps
    float* cpu_losses; // CPU buffer to copy the losses to, allocated with cudaMallocHost
    unsigned long long rng_state; // the RNG state for seeding stochastic rounding etc.
    unsigned long long rng_state_last_update; // RNG before last llama3_update() to re-round identically from master weights
    int use_master_weights; // keep master weights copy in float for optim update? 0|1
    bool init_state;   // set to true if master weights need to be initialized
    int gelu_fusion; // fuse gelu via cuBLASLt (0=none, 1=forward, 2=forward+backward)
    int recompute; // recompute gelu | layernorm forward during model backward? 0|1|2
    // todo - if other functions need cpu scratch buffers in the future, reuse as generic scratch?
    int* workload_indices; // encoder_backward, B*T*num_c_groups (int)
    int4* bucket_info;     // encoder_backward, B*T*num_c_groups (int4) - size for worst case
    floatX* freqs_cis; // (T, hd) for RoPE
} LLama3;

void llama3_init_common(LLama3 *model) {
    // common inits outside of the model weights
    // memory lazily initialized in forward()
    model->acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->accumulated_mean_loss = NULL;
    model->cpu_losses = NULL;
    // the B,T params are determined and set, fixed on first batch in forward()
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f designates no loss, set at end of forward()
    model->params_memory = NULL;
    // memory lazily initialized in backward()
    model->grads_memory = NULL;
    model->workload_indices = NULL; // on cpu, for encoder_backward
    model->bucket_info = NULL; // on cpu, for encoder_backward
    // memory lazily initialized in update()
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->master_weights = NULL;
    // other default settings
    model->rng_state = 13371337 + multi_gpu_config.process_rank; // used in stochastic rounding
    model->use_master_weights = 1; // safe default: do keep master weights in fp32
    model->init_state = true;
    model->recompute = 1; // good default: recompute gelu but not layernorm
    model->gelu_fusion = 0; //deviceProp.major >= 9 ? 2 : 0; // default: off for now (default must match main())
    model->freqs_cis = NULL;
}

void llama3_allocate_weights(LLama3 *model) {
    // fill in all the parameter tensor dimensions and types
    fill_in_parameter_sizes(model->param_elements, model->param_sizeof, model->config);
    model->num_parameters = 0;
    model->num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_elements[i];
        model->num_parameters_bytes += model->param_elements[i] * model->param_sizeof[i];
    }

    // TODO TAKE OUT ----------------------------------------------------------
    // DEBUGGING: print out the sizes of the parameters
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        printf("param_elements[%d] = %zu\n", i, model->param_elements[i]);
    }
    printf("num_parameters = %zu\n", model->num_parameters);
    printf("num_parameters_bytes = %zu\n", model->num_parameters_bytes);
    // ------------------------------------------------------------------------

    // create memory for model parameters on the device
    assert(model->params_memory == nullptr);
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_elements, model->param_sizeof);
}

void llama3_allocate_state(LLama3 *model, int B, int T) {
    printf0("allocating %d MiB for parameter gradients\n", (int)round(model->num_parameters * sizeof(floatX) / (1024 * 1024)));
    assert(model->grads_memory == nullptr);
    model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_elements, model->param_sizeof);

    // record the current B,T as well
    model->batch_size = B;
    model->seq_len = T;

    // allocate the space
    fill_in_activation_sizes(&model->acts, model->acts_specs, B, T, model->config, model->recompute);
    model->acts_memory = malloc_and_point_activations(model->acts_specs);
    // also create memory for caching inputs and targets
    cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&model->targets, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(((void**)&model->accumulated_mean_loss), sizeof(float)));
    cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));

    // initialise cpu scratch buffers for encoder backward
    size_t num_c_groups = CEIL_DIV(model->config.channels, (WARP_SIZE * x128::size));
    assert((size_t)(model->batch_size * model->seq_len) * num_c_groups < (1ULL<<31ULL)); // todo - maybe an issue for llama3-400B(?)
    model->workload_indices = (int*)mallocCheck(sizeof(int) * model->batch_size * model->seq_len * num_c_groups);
    model->bucket_info = (int4*)mallocCheck(sizeof(int4) * model->batch_size * model->seq_len * num_c_groups);

    // precompute freqs_cis for RoPE
    int hd = model->config.channels / model->config.num_heads;
    printf("calculating and allocating %zu KiB for freqs_cis\n", (T * hd * sizeof(floatX)) >> 10);
    floatX* freqs_cis_cpu = (floatX*)mallocCheck(T * hd * sizeof(floatX));
    precompute_freqs_cis(freqs_cis_cpu, hd, T, model->config.rope_theta, model->config.use_scaled_rope);
    cudaCheck(cudaMalloc((void**)&model->freqs_cis, T * hd * sizeof(floatX)));
    cudaCheck(cudaMemcpy(model->freqs_cis, freqs_cis_cpu, T * hd * sizeof(floatX), cudaMemcpyHostToDevice));
    free(freqs_cis_cpu);

    // cudaMallocConditionallyManaged can fall back to cudaMallocManaged if not enough memory on device
    // and returns a status code of 1 if it had to fall back, in that case we want to print warning.
    int memory_status = 0;

    // we will now init the optimizer states and master weights
    // this is usually a substantial amount of memory allocation right here.
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters; // num parameters we are responsible for
    printf0("allocating %zu MiB for AdamW optimizer state m\n", (shard_num_parameters * sizeof(floatOpt)) >> 20);
    printf0("allocating %zu MiB for AdamW optimizer state v\n", (shard_num_parameters * sizeof(floatOpt)) >> 20);
    assert(model->m_memory == nullptr);
    assert(model->v_memory == nullptr);
    memory_status |= cudaMallocConditionallyManaged((void**)&model->m_memory, shard_num_parameters * sizeof(floatOpt));
    memory_status |= cudaMallocConditionallyManaged((void**)&model->v_memory, shard_num_parameters * sizeof(floatOpt));

    if (model->use_master_weights == 1) {
        assert(model->master_weights == nullptr);
        printf0("allocating %zu MiB for master copy of params\n", (shard_num_parameters * sizeof(float)) >> 20);
        memory_status |= cudaMallocConditionallyManaged((void**) &model->master_weights, shard_num_parameters * sizeof(float));
    }

    // report on mixed memory allocation status (re-using our float reduce function, bit awk ok)
    int reduced_memory_status = (int) multi_gpu_cpu_float_sum((float)memory_status, &multi_gpu_config);
    if (reduced_memory_status >= 1) {
        printf0("WARNING: Fell back to cudaMallocManaged when initializing m,v,master_weights on %d GPUs\n", reduced_memory_status);
        printf0("         Prevents an OOM, but code may run much slower due to device <-> host memory movement\n");
    }
    // report on device memory usage
    size_t free, total;
    cudaCheck(cudaMemGetInfo(&free, &total));
    printf0("device memory usage: %zd MiB / %zd MiB\n", (total-free) / 1024 / 1024, total / 1024 / 1024);
    // give an estimate of the maximum batch size
    size_t bytes_per_sequence = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes_per_sequence += model->acts_specs[i].size * sizeof_dtype(model->acts_specs[i].type) / B;
    }
    printf0("memory per sequence: %zu MiB\n", bytes_per_sequence / 1024 / 1024);
    printf0(" -> estimated maximum batch size: %zu\n", B + free / bytes_per_sequence);
}

void llama3_write_to_checkpoint(LLama3 *model, const char* checkpoint_path) {
    // write the model to a checkpoint file
    printf0("Writing model to %s\n", checkpoint_path);
    FILE *model_file = fopenCheck(checkpoint_path, "wb");
    // write the header first
    int model_header[256];
    memset(model_header, 0, sizeof(model_header));
    model_header[0] = 20240803; // magic number
    assert(PRECISION_MODE == PRECISION_FP32 || PRECISION_MODE == PRECISION_BF16);
    model_header[1] = PRECISION_MODE == PRECISION_FP32 ? 3 : 5; // version
    model_header[2] = model->config.max_seq_len;
    model_header[3] = model->config.vocab_size;
    model_header[4] = model->config.num_layers;
    model_header[5] = model->config.num_heads;
    model_header[6] = model->config.num_kv_heads;
    model_header[7] = model->config.channels;
    model_header[8] = model->config.multiple_of;
    model_header[9] = model->config.use_scaled_rope;
    model_header[10] = model->config.tied_weights;
    model_header[11] = 3;
    model_header[12] = model->config.tied_weights ? 2 : 1;
    fwriteCheck(model_header, sizeof(int), 256, model_file);
    float float_header[256];
    float_header[0] = model->config.ffn_dim_multiplier;
    float_header[1] = model->config.norm_eps;
    float_header[2] = model->config.rope_theta;
    fwriteCheck(float_header, sizeof(float), 256, model_file);
    // write the parameters
    device_to_file(model_file, model->params_memory, model->num_parameters_bytes,
                   IO_BUF_SIZE, main_stream);
    // close file, we're done
    fcloseCheck(model_file);
}

void llama3_build_from_checkpoint(LLama3 *model, const char* checkpoint_path, bool weight_init=true) {
    // If weight_init is true, we will load the weights from this checkpoint .bin file
    // We sometimes want this to be false, if we are going to initialize these weights from
    // the master weights that are instead stored in the state .bin file.
    // In that case, this function mostly loads the model hyperparameters from the header.

    if (PRECISION_MODE == PRECISION_FP16) {
        // TODO for later perhaps, would require us dynamically converting the
        // model weights from fp32 to fp16 online, here in this function, or writing
        // the fp16 weights directly from Python, which we only do for fp32/bf16 atm.
        fprintf(stderr, "build_from_checkpoint() does not support fp16 right now.\n");
        exit(EXIT_FAILURE);
    }

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int header_int[256]; // int section of the header
    freadCheck(header_int, sizeof(int), 256, model_file);
    assert(sizeof(int) == 4); // i think the python export code currently assumes this is int32
    float header_float[256]; // float section of the header
    freadCheck(header_float, sizeof(float), 256, model_file);
    assert(sizeof(float) == 4); // i think the python export code currently assumes this is float32
    if (header_int[0] != 20240803) { printf("Bad magic model file\n"); exit(EXIT_FAILURE); }
    int version = header_int[1];
    if (!(version == 3 || version == 5)) {
        // 3 = fp32, padded vocab
        // 5 = bf16, padded vocab, layernorms also in bf16
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_llama3.py`\n");
        exit(EXIT_FAILURE);
    }

    // check if the precision mode of the checkpoing matches the model precision
    if (weight_init) {
        if (PRECISION_MODE == PRECISION_BF16 && version != 5) {
            fprintf(stderr, "Precision is configured as BF16 but model at %s is not.\n", checkpoint_path);
            fprintf(stderr, "---> HINT: are you sure you're loading a _bf16.bin file?\n");
            exit(EXIT_FAILURE);
        }
        if (PRECISION_MODE == PRECISION_FP32 && version != 3) {
            fprintf(stderr, "Precision is configured as FP32 but model at %s is not.\n", checkpoint_path);
            fprintf(stderr, "---> HINT: to turn on FP32 you have to compile like: `make train_llama3cu PRECISION=FP32`\n");
            fprintf(stderr, "---> HINT: are you sure you're loading a .bin file without any _bf16 in the name?\n");
            exit(EXIT_FAILURE);
        }
    }

    // read in hyperparameters from the header
    // first the integer section
    model->config.max_seq_len = header_int[2];
    model->config.vocab_size = header_int[3];
    model->config.padded_vocab_size = model->config.vocab_size; // in Llama 3 there is no need for padding
    model->config.num_layers = header_int[4];
    model->config.num_heads = header_int[5];
    model->config.num_kv_heads = header_int[6];
    model->config.channels = header_int[7];
    model->config.multiple_of = header_int[8];
    model->config.use_scaled_rope = header_int[9];
    model->config.use_biases = false;
    model->config.tied_weights = header_int[10];
    int major_version = header_int[11]; // currently unused, e.g. 3
    int minor_version = header_int[12]; // 1 or 2
    // now the float section
    model->config.ffn_dim_multiplier = header_float[0];
    model->config.norm_eps = header_float[1];
    model->config.rope_theta = header_float[2];

    // ------------------------------------------------------------------------
    // TODO TAKE OUT ----------------------------------------------------------
    // Debugging: print all of the values above to check visually and EXIT
    printf("CHECK:\n");
    printf("max_seq_len: %d\n", model->config.max_seq_len);
    printf("vocab_size: %d\n", model->config.vocab_size);
    printf("padded_vocab_size: %d\n", model->config.padded_vocab_size);
    printf("num_layers: %d\n", model->config.num_layers);
    printf("num_heads: %d\n", model->config.num_heads);
    printf("num_kv_heads: %d\n", model->config.num_kv_heads);
    printf("channels: %d\n", model->config.channels);
    printf("multiple_of: %d\n", model->config.multiple_of);
    printf("use_scaled_rope: %d\n", model->config.use_scaled_rope);
    printf("major version: %d\n", major_version);
    printf("minor version: %d\n", minor_version);
    printf("ffn_dim_multiplier: %f\n", model->config.ffn_dim_multiplier);
    printf("norm_eps: %f\n", model->config.norm_eps);
    printf("rope_theta: %f\n", model->config.rope_theta);
    // ------------------------------------------------------------------------

    // allocate memory for the model parameters
    llama3_allocate_weights(model);

    // read in the parameters if weight_init is true
    if (weight_init) {
        assert(model->params_memory != NULL);
        file_to_device(model->params_memory, model_file, model->num_parameters_bytes, IO_BUF_SIZE, main_stream);
    }
    fcloseCheck(model_file);

    // only return from this function once we are certain the params are ready on the GPU
    cudaCheck(cudaDeviceSynchronize());
}

// propagate inputs through the network to produce logits.
// right now, this function is fully synchronous with the host
void llama3_forward(LLama3 *model, const int* inputs, size_t B, size_t T) {
    NVTX_RANGE_FN();
    // we must be careful and use size_t instead of int, otherwise
    // we could overflow int. E.g. l * B * NH * T * T overflows int at B 16.

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // convenience parameters
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;
    const size_t n_head = model->config.num_heads;
    const size_t n_kv_head = model->config.num_kv_heads;
    const size_t hd = C / n_head; // head dimension
    const size_t qkv_channels = (n_head + 2*n_kv_head) * hd; // Q, K, V channels
    size_t hidden_dim = 4 * C;
    hidden_dim = (2 * hidden_dim) / 3;
    hidden_dim = model->config.ffn_dim_multiplier * hidden_dim;
    hidden_dim = model->config.multiple_of * ((hidden_dim + model->config.multiple_of - 1) / model->config.multiple_of);
    size_t ffn_channels = hidden_dim * 2; // c_fc + c_fc2 concatenated
    size_t ffn_channels_post_gelu = hidden_dim; // swiglu halves the channels

    // validate B,T are not larger than the values used at initialisation
    // (smaller B,T are okay for inference only)
    if (B > model->batch_size || T > model->seq_len) {
        printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
        exit(EXIT_FAILURE);
    }

    // copy inputs/targets to the model
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    // validate inputs, all indices must be in the range [0, V)
    // we can do this while the copies are already underway
    tokenCheck(inputs, B*T, V);

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    encoder_forward(acts.encoded, model->inputs, params.wte, NULL, B, T, C, main_stream); // encoding goes into residual[0]
    // first rmsnorm isn't fused
    rmsnorm_forward((model->recompute < 2) ? acts.ln1 : acts.lnf, acts.ln1_rstd, acts.encoded, params.ln1w, B, T, C, main_stream);

    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);

        floatX* residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        floatX* l_qkvw = params.qkvw + l * qkv_channels * C;
        floatX* l_qkvb = model->config.use_biases ? params.qkvb + l * qkv_channels: nullptr;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_attprojb = model->config.use_biases ? params.attprojb + l * C : nullptr;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_fcw = params.fcw + l * ffn_channels * C;
        floatX* l_fcb = model->config.use_biases ? params.fcb + l * ffn_channels : nullptr;
        floatX* l_fcprojw = params.fcprojw + l * C * ffn_channels_post_gelu;
        floatX* l_fcprojb = model->config.use_biases ? params.fcprojb + l * C : nullptr;

        // get the pointers of the activations for this layer
        floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch = acts.fch + l * B * T * ffn_channels;
        // reuse the same activation buffer at each layer, as we'll re-compute the gelu during backward
        // very useful because we dramatically reduce VRAM usage, and may be able to fit larger batch size
        floatX* l_fch_swiglu = (model->recompute < 1) ? acts.fch_gelu + l * B * T * ffn_channels_post_gelu : acts.fch_gelu;
        floatX* l_residual3 = acts.residual3 + l * B * T * C;
        floatX* scratch = (floatX*)acts.output; // used for non-cudnn attention, fcproj, attproj, etc.
        floatX* qkv_rep_scratch = (floatX*)acts.scratch_bt4c; // we can use the BT4C scratch for qkv replication

        // Attention block
        // The input l_ln1 now holds the (already layernormed) input
        #ifdef ENABLE_CUDNN
            printf("cuDNN path TODO\n"); exit(0);
            matmul_forward_cublaslt(l_qkvr, l_ln1, l_qkvw, l_qkvb, B, T, C, qkv_channels, main_stream);
            float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
            attention_forward_cudnn(l_atty, (float*)l_att, l_qkvr, B, T, NH, C, main_stream);
        #else
            // unused parts of attention buffer must be zeroed (T-dependent)
            floatX* l_att = acts.att + l * B * NH * T * T;
            if (T != model->seq_len) { cudaCheck(cudaMemset(l_att, 0, B * NH * T * T * sizeof(floatX))); }
            // 1) projection to QKV vectors (note k,v may be fewer heads than q)
            matmul_forward_cublaslt(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, qkv_channels, main_stream);
            // 2) replicate k,v so that all of q,k,v have the same number of heads. done for simplicity, for now
            repkv_forward(qkv_rep_scratch, scratch, B, T, n_head, n_kv_head, hd, main_stream);
            // 3) apply RoPE to q,k in place
            rope_forward(qkv_rep_scratch, qkv_rep_scratch, model->freqs_cis, B, T, n_head, hd, main_stream);
            // 4) attention: att <- softmax(qk^T)v
            attention_forward(l_atty, l_qkvr, l_att, qkv_rep_scratch, B, T, C, NH, main_stream);
        #endif

        matmul_forward_cublaslt(scratch, l_atty, l_attprojw, l_attprojb, B, T, C, C, main_stream);
        fused_residual_rmsnorm_forward5(l_residual2, l_ln2, l_ln2_rstd, residual, scratch, l_ln2w, B*T, C, main_stream);
        matmul_forward_cublaslt(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, ffn_channels, main_stream);
        swiglu_forward(l_fch_swiglu, l_fch, B, T, ffn_channels_post_gelu, main_stream);
        matmul_forward_cublaslt(scratch, l_fch_swiglu, l_fcprojw, l_fcprojb, B, T, ffn_channels_post_gelu, C, main_stream);

        // OK, fusion across blocks.
        if(l+1 != L) {
            floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + (l + 1) * B * T * C : acts.lnf;
            float* l_ln1_rstd = acts.ln1_rstd + (l + 1) * B * T;
            const floatX* l_ln1w = params.ln1w + (l + 1) * C;
            fused_residual_rmsnorm_forward5(l_residual3, l_ln1, l_ln1_rstd, l_residual2, scratch, l_ln1w, B * T, C, main_stream);
        } else {
            fused_residual_rmsnorm_forward5(l_residual3, acts.lnf, acts.lnf_rstd, l_residual2, scratch, params.lnfw, B * T, C, main_stream);
        }
    }

    floatX* lm_head = model->config.tied_weights ? params.wte : params.wlmhead;
    matmul_forward_cublaslt(acts.output, acts.lnf, lm_head, NULL, B, T, C, Vp, main_stream);

    cudaCheck(cudaDeviceSynchronize());
}


// Forwards both the model and the loss and is used for validation splits and evals.
// In particular it populates cpu_losses with loss at each token.
// Some of the evals (e.g. HellaSwag) require the per-token losses, which are produced here.
float llama3_validate(LLama3 *model, const int* inputs, const int* targets, size_t B, size_t T) {
    assert(targets != NULL);
    // forward the model itself
    llama3_forward(model, inputs, B, T);
    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;

    NvtxRange classifier_and_loss_range("classifier_and_loss");
    ActivationTensors acts = model->acts;
    float mean_loss = 0.0f;
    // fused classifier: does the forward pass and first part of the backward pass
    const float dloss = 1.0f / (B * T); // results in the uniform average loss over all elements
    // note: we don't need to generate dlogits here
    cudaCheck(cudaMemset(acts.losses, 0, B*T*sizeof(float)));
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B*T, V); // while the memcpy is underway, validate the targets
    fused_classifier(acts.output, acts.losses, dloss, model->targets, B, T, V, Vp, False, main_stream);
    cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < B*T; i++) {
        mean_loss += model->cpu_losses[i];
    }
    mean_loss /= B*T;
    cudaCheck(cudaDeviceSynchronize());
    return mean_loss;
}

void llama3_backward_and_reduce(LLama3 *model, int* inputs, const int* targets, int grad_accum_steps, int micro_step) {
    if(model->grads_memory == nullptr) {
        fprintf(stderr, "Need to allocate gradients before backward");
        exit(EXIT_FAILURE);
    }
    NVTX_RANGE_FN();
    bool last_step = micro_step == grad_accum_steps - 1;
    // on the first micro-step zero the gradients, as we're about to += accumulate into them
    if (micro_step == 0) {
        // there are currently two state vars during the gradient accumulation inner loop:
        // 1) the losses accumulate += into acts.losses, reset here
        // 2) the gradients accumulate += into grads_memory, reset here
        cudaCheck(cudaMemsetAsync(model->acts.losses, 0, model->batch_size * model->seq_len * sizeof(float), main_stream));
        cudaCheck(cudaMemsetAsync(model->grads_memory, 0, model->num_parameters * sizeof(floatX), main_stream));
    }

    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    const size_t B = model->batch_size;
    const size_t T = model->seq_len;
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;
    const size_t n_head = model->config.num_heads;
    const size_t n_kv_head = model->config.num_kv_heads;
    const size_t hd = C / n_head; // head dimension
    const size_t qkv_channels = (n_head + 2*n_kv_head) * hd; // Q, K, V channels
    size_t hidden_dim = 4 * C;
    hidden_dim = (2 * hidden_dim) / 3;
    hidden_dim = model->config.ffn_dim_multiplier * hidden_dim;
    hidden_dim = model->config.multiple_of * ((hidden_dim + model->config.multiple_of - 1) / model->config.multiple_of);
    size_t ffn_channels = hidden_dim * 2; // c_fc + c_fc2 concatenated
    size_t ffn_channels_post_gelu = hidden_dim; // swiglu halves the channels

    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;

    // accumulate the losses inside acts.losses, and kick off the backward pass inside the fused classifier
    NvtxRange classifier_and_loss_range("classifier_and_loss");
    const float dloss = 1.0f / (float)(B * T * grad_accum_steps); // results in the uniform average loss over all elements
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B*T, V);
    fused_classifier(acts.output, acts.losses, dloss, model->targets, B, T, V, Vp, True, main_stream);
    // ------------------------------------------------------------------------
    // backward pass: go in the reverse order of the forward pass, and call backward() functions

    // reset residual stream gradients (put here to work with gradient accumulation)
    floatX* dresidual = (floatX*)model->acts.scratch_btc; // the main buffer holding the gradient in the backward pass
    cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));

    // re-use the output buffer of the forward pass as a scratchpad during backward pass
    float*  scratchF = (float*)acts.output;
    floatX* scratchX = (floatX*)acts.output;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // this was done in the fused classifier kernel as last step of forward pass
    // technically that is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    // next: backward the classifier matmul
    floatX* w_lm_head = model->config.tied_weights ? params.wte : params.wlmhead;
    floatX* g_lm_head = model->config.tied_weights ? grads.wte : grads.wlmhead;

    matmul_backward(model->acts.scratch_bt4c, g_lm_head, NULL, acts.output, acts.lnf, w_lm_head, NULL, B, T, C, Vp, main_stream);
    // backward the final layernorm
    floatX* residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    rmsnorm_backward(dresidual, grads.lnfw, scratchF, model->acts.scratch_bt4c, residual, params.lnfw, acts.lnf_rstd, B, T, C, main_stream);
    // from this point on, we no longer need the values stored in the last residual, so we can reuse that memory as generic
    // scratch for backward computations
    floatX* dl_btc = residual;

    // now backward all the layers
    for (int l = L-1; l >= 0; l--) {
        NvtxRange layer_range("Layer", l);

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        floatX* l_ln1w = params.ln1w + l * C;
        floatX* l_qkvw = params.qkvw + l * qkv_channels * C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_fcw = params.fcw + l * ffn_channels * C;
        floatX* l_fcprojw = params.fcprojw + l * C * ffn_channels_post_gelu;
        // get the pointers of the gradients of the weights for this layer
        floatX* dl_ln1w = grads.ln1w + l * C;
        floatX* dl_ln1b = grads.ln1b + l * C;
        floatX* dl_qkvw = grads.qkvw + l * qkv_channels * C;
        floatX* dl_qkvb = model->config.use_biases ? grads.qkvb + l * qkv_channels : nullptr;
        floatX* dl_attprojw = grads.attprojw + l * C * C;
        floatX* dl_attprojb = model->config.use_biases ? grads.attprojb + l * C : nullptr;
        floatX* dl_ln2w = grads.ln2w + l * C;
        floatX* dl_ln2b = model->config.use_biases ? grads.ln2b + l * C : nullptr;
        floatX* dl_fcw = grads.fcw + l * ffn_channels * C;
        floatX* dl_fcb = model->config.use_biases ? grads.fcb + l * ffn_channels : nullptr;
        floatX* dl_fcprojw = grads.fcprojw + l * C * ffn_channels_post_gelu;
        floatX* dl_fcprojb = model->config.use_biases ? grads.fcprojb + l * C : nullptr;
        // get the pointers of the activations for this layer
        floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch_pre_gelu = acts.fch + l * B * T * ffn_channels;
        floatX* l_fch_swiglu = (model->recompute < 1) ? acts.fch_gelu + l * B * T * ffn_channels_post_gelu : acts.fch_gelu;
        // get the pointers of the gradients of the activations for this layer
        // notice that there is no l *, because we just have a single copy, and keep
        // re-using this memory in every Transformer block as we calculate backward pass

        floatX* dl_bt4c = (floatX*)model->acts.scratch_bt4c;
        floatX* dl_bt4c2 = (floatX*)model->acts.scratch_bt4c2; // same size as dl_bt4c, just a second buffer

        // start the backward pass for this layer
        if(model->recompute >= 1) {
            // recompute >= 1 means we recompute gelu. in this case,
            // l_fch_swiglu is just a buffer, so re-compute the gelu from l_fch here
            swiglu_forward(l_fch_swiglu, l_fch_pre_gelu, B, T, ffn_channels_post_gelu, main_stream);
        }
        // backward the 2nd matmul of MLP
        matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_swiglu, l_fcprojw, scratchF, B, T, ffn_channels_post_gelu, C, main_stream);
        // backward the swiglu here, use scratchX to hold the grad because SwiGLU can't be inplace
        swiglu_backward(dl_bt4c2, dl_bt4c, l_fch_pre_gelu, B, T, ffn_channels_post_gelu, main_stream);
        // backward the 1st matmul of MLP
        if(model->recompute >= 2) {
            // same as gelu above, l_ln1 and l_ln2 are just buffers if recompute >= 2, recompute them here on demand
            rmsnorm_forward(l_ln2, l_ln2_rstd, l_residual2, l_ln2w, B, T, C, main_stream);
        }
        matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c2, l_ln2, l_fcw, scratchF, B, T, C, ffn_channels, main_stream);
        // rmsnorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        rmsnorm_backward(dresidual, dl_ln2w, scratchF, dl_btc, l_residual2, l_ln2w, l_ln2_rstd, B, T, C, main_stream);
        matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, scratchF, B, T, C, C, main_stream);

        // <--- gradient here matches OK

        #ifdef ENABLE_CUDNN
        printf("cuDNN path TODO\n"); exit(0);
        float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
        attention_backward_cudnn(dl_bt4c, dl_btc, l_qkvr, l_atty, (float*)l_att, B, T, NH, C, main_stream);
        #else
        floatX* l_att = acts.att + l * B * NH * T * T;
        // we need B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
        floatX* buffer_a = l_atty;
        floatX* buffer_b = l_fch_pre_gelu;        // this is B x T x 4C, so even larger than what we need
        attention_backward(dl_bt4c, buffer_b, scratchX, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH, main_stream);
        #endif
        // backward rope (this can be done in-place)
        rope_backward_inplace(dl_bt4c, dl_bt4c, model->freqs_cis, B, T, NH, hd, main_stream);
        // backward repkv (use scratchX as gradient buffer here)
        repkv_backward(dl_bt4c2, dl_bt4c, B, T, NH, n_kv_head, hd, main_stream);
        // backward QKV projection
        if(model->recompute >= 2) {
            rmsnorm_forward(l_ln1, l_ln1_rstd, residual, l_ln1w, B, T, C, main_stream);
        }
        matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c2, l_ln1, l_qkvw, scratchF, B, T, C, qkv_channels, main_stream);
        rmsnorm_backward(dresidual, dl_ln1w, scratchF, dl_btc, residual, l_ln1w, l_ln1_rstd, B, T, C, main_stream);

        // Accumulate gradients from this layer in a background stream.
        if(last_step) {
            floatX* const pointers[] = {
                dl_ln1w, dl_ln1b,
                dl_qkvw, dl_qkvb,
                dl_attprojw, dl_attprojb,
                dl_ln2w, dl_ln2b,
                dl_fcw, dl_fcb,
                dl_fcprojw, dl_fcprojb
            };
            const size_t nelem[] = {
                C, C,
                qkv_channels * C, qkv_channels,
                C * C, C,
                C, C,
                ffn_channels * C, ffn_channels,
                C * ffn_channels_post_gelu, C
            };
            multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
        }
    }

    encoder_backward(grads.wte, NULL, scratchX, model->workload_indices, model->bucket_info,
                     dresidual, model->inputs, inputs, B, T, C, random_u32(&model->rng_state), main_stream);

    // Aggregate all gradients that are not part of the transformer blocks
    if(last_step) {
        // reduce all the losses within the current GPU (across all microsteps)
        global_sum_deterministic(model->accumulated_mean_loss, acts.losses, B*T, main_stream);
        // reduce loss across GPUs to a single, final float across all microsteps and GPUs
        #if MULTI_GPU
        ncclCheck(ncclAllReduce(model->accumulated_mean_loss, model->accumulated_mean_loss, sizeof(float), ncclFloat, ncclAvg, multi_gpu_config.nccl_comm, main_stream));
        #endif
        cudaCheck(cudaMemcpyAsync(&model->mean_loss, model->accumulated_mean_loss, sizeof(float), cudaMemcpyDeviceToHost, main_stream));
        // reduce the gradients for non-transformer block parameters
        floatX* const pointers[] = {grads.wte, grads.wlmhead, grads.lnfw, grads.lnfb};
        const size_t nelem[] = {Vp * C, Vp * C, C, C};
        multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
    }

    cudaCheck(cudaDeviceSynchronize());
    if(last_step) {
        model->mean_loss /= B*T*grad_accum_steps;
    } else {
        model->mean_loss = -1.f; // no loss available yet
    }
}

// Gets the offset of a specific tensor for a specific layer in the LLama3 model
// layer_id is ignored for weights that are not part of a transformer block
ShardInfo llama3_get_tensor_at_layer(const LLama3 *model, int layer_id, int param_tensor_id) {
    // first offset our way to the parameter tensor start
    ptrdiff_t offset = 0;
    for (int i = 0; i < param_tensor_id; i++) {
        offset += (ptrdiff_t)model->param_elements[i];
    }
    size_t size = model->param_elements[param_tensor_id] ;
    // if we are in the transformer block, we need to additionally offset by the layer id
    if(2 <= param_tensor_id && param_tensor_id <= 13) {
        size /= model->config.num_layers;
        offset += (ptrdiff_t)(layer_id * size);
    }
    return {offset, size};
}

float llama3_calculate_grad_norm(LLama3 *model, MultiGpuConfig* multi_gpu_config) {
    NVTX_RANGE_FN();
    floatX* grads_memory = (floatX*)model->grads_memory;

    // repurposing this buffer (which isn't needed now) to write grad norm into it
    float* grad_norm_squared = (float*)model->acts.output;
    float grad_norm_squared_cpu = 0.0f;

    int num_slices[2] = {1, model->config.num_layers};
    int max_num_block_sums = get_max_num_block_sums(num_slices, 2);
    if (multi_gpu_config->zero_stage == 1) {
        // because of the ncclReduceScatter() in backward,
        // grads_memory only contains the averaged gradients at the local shards,
        // so we only calculate the grad norm at the grads_memory belonging to the local shards
        for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            ShardInfo tensor = llama3_get_tensor_at_layer(model, 0, i);
            ShardInfo shard = multi_gpu_get_shard_offset(tensor.size, multi_gpu_config, 1);
            ptrdiff_t offset = tensor.offset + shard.offset;
            bool is_first_pass = (i == 0);
            if((i < 2 || i > 13)) {
                global_norm_squared(grad_norm_squared, grads_memory + offset, shard.size, 0, 1,
                                    max_num_block_sums, is_first_pass, main_stream);
            } else {
                global_norm_squared(grad_norm_squared, grads_memory + offset, shard.size, tensor.size, model->config.num_layers,
                                    max_num_block_sums, is_first_pass, main_stream);
            }
        }
        global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, main_stream);
#if MULTI_GPU
        // further sum the (partial) squared norm across all GPUs
        ncclCheck(ncclAllReduce(grad_norm_squared, grad_norm_squared, sizeof(float), ncclFloat, ncclSum, multi_gpu_config->nccl_comm, main_stream));
#endif
    } else {
        // in regular DDP, backward has averaged the gradients across all GPUs
        // so each GPU can compute the squared norm over the whole grad vector, with no added comms needed
        global_norm_squared(grad_norm_squared, grads_memory, model->num_parameters, 0, 1, max_num_block_sums, true, main_stream);
        global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, main_stream);
    }
    cudaCheck(cudaMemcpy(&grad_norm_squared_cpu, grad_norm_squared, sizeof(float), cudaMemcpyDeviceToHost));
    float grad_norm_cpu = sqrtf(grad_norm_squared_cpu);
    return grad_norm_cpu;
}

void llama3_update(LLama3 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, float grad_scale, int t,
                   MultiGpuConfig* multi_gpu_config, bool init_from_master_only=false) {
    // update the model parameters using the AdamW optimizer
    // keep in mind that optimizer sharding (ZeRO-1) assigns different parameters to different GPUs
    // so we may not be responsible for the entire parameter tensor
    // also, this function was very simple a while back but become very complex, only because we want to
    // selectively weight decay some, but not all tensors :(
    // TODO: revisit and probably refactor this entire function
    NVTX_RANGE_FN();
    if(model->grads_memory == nullptr || model->m_memory == nullptr || model->v_memory == nullptr) {
        fprintf(stderr, "Need to allocate optimizer state before update");
        exit(EXIT_FAILURE);
    }

    bool init_state = model->init_state;
    if(init_state) {
        model->init_state = false;
        NvtxRange rng("InitOpt");
        cudaCheck(cudaMemset(model->m_memory, 0, multi_gpu_config->shard_num_parameters * sizeof(floatOpt)));
        cudaCheck(cudaMemset(model->v_memory, 0, multi_gpu_config->shard_num_parameters * sizeof(floatOpt)));
    }

    // save RNG state at this point so we can round from master weights identically when restoring from a checkpoint
    model->rng_state_last_update = model->rng_state;

    // AdamW update
    // handle adamw for all the transformer blocks
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        // generate a unique seed for each tensor
        unsigned int seed = random_u32(&model->rng_state);

        int num_layers = model->config.num_layers;
        if((i < 2 || i > 13)) {
            num_layers = 1;
        }

        ShardInfo tensor = llama3_get_tensor_at_layer(model, 0, i);
        if(tensor.size == 0)
            continue;
        ShardInfo shard = multi_gpu_get_shard_offset(tensor.size, multi_gpu_config, 1);
        ptrdiff_t local_offset_full = tensor.offset + shard.offset;
        ptrdiff_t local_offset_partial = tensor.offset / multi_gpu_config->num_processes;

        // we only want to weight decay the 2D tensors and leave all 1D tensors alone
        // in particular this also decays the embedding weights, but this is ok:
        // - the token embeddings are weight shared and participate in the final projection to logits
        // - the position embeddings actively participate at every forward/backward pass
        float wd = (i == 0 || i == 1 || i == 4 || i == 6 || i == 10 || i == 12) ? weight_decay : 0.0f;
        floatX* param_ptr = (floatX*)model->params_memory + local_offset_full;
        floatX* grad_ptr = (floatX*)model->grads_memory + local_offset_full;

        ptrdiff_t opt_state_offset = multi_gpu_config->zero_stage < 1 ?  local_offset_full : local_offset_partial;
        floatOpt* m_ptr = model->m_memory + opt_state_offset;
        floatOpt* v_ptr = model->v_memory + opt_state_offset;
        float* master_ptr = nullptr;
        if (model->master_weights != nullptr) { master_ptr = model->master_weights + opt_state_offset; }
        if(init_state && model->master_weights != nullptr ) {
            size_t grid_size = CEIL_DIV(shard.size, 512);
            copy_and_cast_kernel<<<dim3(grid_size, num_layers), 512, 0, main_stream>>>(master_ptr, param_ptr, shard.size,
                                                                     shard.size, tensor.size);
            cudaCheck(cudaGetLastError());
        }

        if (init_from_master_only) {
            // when resuming training from a checkpoint with master weights (allows changing precision)
            init_from_master(param_ptr, master_ptr, shard.size, tensor.size, shard.size, num_layers, seed, main_stream);
        } else {
            // ok finally call the kernel to update the weights with AdamW
            adamw_update(param_ptr, master_ptr, grad_ptr,
                        m_ptr, v_ptr,
                        shard.size, tensor.size, tensor.size, shard.size, num_layers,
                        learning_rate,
                        beta1, beta2, t, eps, wd, grad_scale, seed, main_stream);
        }

        if (multi_gpu_config->zero_stage == 1) {
#if MULTI_GPU
            ncclCheck(ncclGroupStart());
            for(int l = 0; l < num_layers; ++l) {
                // gather updated shards of model->params_memory from each process
                ncclCheck(ncclAllGather(param_ptr + l * tensor.size,
                                        (floatX*) model->params_memory + tensor.offset + l * tensor.size,
                                        shard.size, ncclFloatX,
                                        multi_gpu_config->nccl_comm, multi_gpu_config->nccl_stream));
            }
            ncclCheck(ncclGroupEnd());
#endif
        }
    }

    cudaCheck(cudaDeviceSynchronize());
}

float llama3_estimate_mfu(LLama3 *model, int num_tokens, float dt) {
    /*
    Estimate model flops utilization (MFU)
    ref: Section 2.1 of https://arxiv.org/pdf/2001.08361
    Note: Ideally, the N here would be only the parameters that actually
    participate in matrix multiplications. In this N, we are over-estimating by
    including LayerNorm params, biases, and the position embedding weights,
    but these are very small terms. Also keep in mind that we would want to exclude
    the token embedding weights, but in GPT-2 these are weight shared, so they
    participate in the classifier matmul, so they are correct to be included in N.
    Note 2: The first term (6 * N) in flops_per_token is all weight matmuls, the
    second is the attention matmul, which is also usually a small contribution.
    */
    size_t N = model->num_parameters;
    if(!model->config.tied_weights) {
        N -= model->param_elements[0];      // remove embedding parameters, which can be significant at 128k vocab
    }

    int L = model->config.num_layers;
    int C = model->config.channels;
    int T = model->seq_len;
    size_t flops_per_token = 6 * N + (size_t)6 * L * C * T;
    size_t flops_per_step = flops_per_token * num_tokens;
    // express our flops throughput as ratio of A100 bfloat16 peak flops
    float flops_achieved = (float)flops_per_step * (1.0f / dt); // per second
    float flops_promised = get_flops_promised(deviceProp.name, PRECISION_MODE) * 1e12f;
    if(flops_promised < 0) {
        return -1.f;   // don't know
    }
    float mfu = flops_achieved / flops_promised;
    return mfu;
}

void llama3_free(LLama3 *model) {
    cudaFreeCheck(&model->params_memory);
    cudaFreeCheck(&model->grads_memory);
    cudaFreeCheck(&model->m_memory);
    cudaFreeCheck(&model->v_memory);
    cudaFreeCheck(&model->master_weights);
    cudaFreeCheck(&model->acts_memory);
    cudaFreeCheck(&model->inputs);
    cudaFreeCheck(&model->targets);
    cudaFreeCheck(&model->accumulated_mean_loss);
    cudaCheck(cudaFreeHost(model->cpu_losses));
    free(model->workload_indices);
    free(model->bucket_info);
}

// ----------------------------------------------------------------------------
// common init & free code for all of train/test/profile

void common_start(bool override_enable_tf32 = true, bool print_device_info = true) {

    // get CUDA device infos
    cudaCheck(cudaGetDeviceProperties(&deviceProp, multi_gpu_config.local_device_idx));
    if (print_device_info) {
        printf("[System]\n");
        printf("Device %d: %s\n", multi_gpu_config.local_device_idx, deviceProp.name);
    }

    // set up the cuda streams. atm everything is on the single main stream
    cudaCheck(cudaStreamCreate(&main_stream));
    nvtxNameCudaStreamA(main_stream, "main stream");

    // set up cuBLAS and cuBLASLt
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    bool enable_tf32 = PRECISION_MODE == PRECISION_FP32 && deviceProp.major >= 8 && override_enable_tf32;
    cublas_compute = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

    #ifdef ENABLE_CUDNN
    create_cudnn();
    #endif
}

void common_free(LLama3 &model) {
    cudaCheck(cudaStreamDestroy(main_stream));
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    #ifdef ENABLE_CUDNN
    destroy_cudnn();
    #endif
}


void save_state(const char* filename, int step, LLama3* model, DataLoader* loader) {
    printf("Writing state to %s\n", filename);
    FILE *state_file = fopenCheck(filename, "wb");
    int state_header[256];
    memset(state_header, 0, sizeof(state_header));
    // basic identifying information
    state_header[0] = 20240527; // magic number
    state_header[1] = 1; // version number
    state_header[2] = multi_gpu_config.num_processes; // number of processes
    state_header[3] = multi_gpu_config.process_rank; // rank of this process
    state_header[4] = model->use_master_weights;  // whether we're using fp32 master weights
    state_header[5] = loader->should_shuffle; // shuffle state of the dataloader
    // int main state, start at 10 to leave some padding
    state_header[10] = step; // step of the optimization
    // model rng state, start at 20 to leave some padding
    *((unsigned long long*)&state_header[20]) = model->rng_state; // random number generator state
    *((unsigned long long*)&state_header[22]) = model->rng_state_last_update; // last llama3_update
    // dataloader state, start at 30 to leave some padding
    *((size_t*)&state_header[30]) = loader->current_shard_idx; // shard of the dataset
    *((size_t*)&state_header[32]) = loader->current_sample_idx; // position in shard
    fwriteCheck(state_header, sizeof(int), 256, state_file);

    // write AdamW m, v, and master_weights here (they are all float, unless OPTIMIZER_LOW_PRECISION is defined)
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    device_to_file(state_file, model->m_memory, shard_num_parameters * sizeof(floatOpt), IO_BUF_SIZE, main_stream);
    device_to_file(state_file, model->v_memory, shard_num_parameters * sizeof(floatOpt), IO_BUF_SIZE, main_stream);
    if(model->use_master_weights) {
        device_to_file(state_file, model->master_weights, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    }

    // write dataloader state if we are using the Permuted version of it
    if (loader->should_shuffle) {
        fwriteCheck(&loader->glob_result.gl_pathc, sizeof(size_t), 1, state_file);  // number of shards
        fwriteCheck(loader->shard_indices, sizeof(int), loader->glob_result.gl_pathc, state_file);
        fwriteCheck(&loader->shard_num_samples, sizeof(size_t), 1, state_file);
        fwriteCheck(loader->intra_shard_indices, sizeof(int), loader->shard_num_samples, state_file);
        fwriteCheck(&loader->shuffle_rng, sizeof(mt19937_state), 1, state_file);
    }
    fcloseCheck(state_file);
}

void load_state(int* step, LLama3* model, DataLoader* loader, const char* filename) {
    FILE *state_file = fopenCheck(filename, "rb");
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    assert(state_header[0] == 20240527); // magic number
    assert(state_header[1] == 1); // version number
    assert(state_header[2] == multi_gpu_config.num_processes); // number of processes
    assert(state_header[3] == multi_gpu_config.process_rank); // rank of this process
    int use_master_weights = state_header[4];  // whether we're using fp32 master weights
    int should_shuffle = state_header[5]; // shuffle state of the dataloader
    *step = state_header[10]; // step of the optimization
    model->rng_state = *((unsigned long long*)&state_header[20]); // random number generator state
    model->rng_state_last_update = *((unsigned long long*)&state_header[22]); // last llama3_update
    size_t current_shard_idx = *((size_t*)&state_header[30]); // shard index
    size_t current_sample_idx = *((size_t*)&state_header[32]); // position in shard

    // read AdamW m, v, master_weights (they are all float)
    // allocate all the needed memory as necessary
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    if(use_master_weights == 1 && !model->use_master_weights) {
        printf0("Warning: Master weights are present in state, but not enabled for current run.");
    } else if (use_master_weights == 0 && model->use_master_weights) {
        printf0("Error: Master weights requested, but not present in state file.");
        exit(EXIT_FAILURE);
    }

    model->init_state = false;      // we just got the state from file, no need to do first-touch init
    assert(model->m_memory != nullptr);
    assert(model->v_memory != nullptr);
    file_to_device(model->m_memory, state_file, shard_num_parameters * sizeof(floatOpt), IO_BUF_SIZE, main_stream);
    file_to_device(model->v_memory, state_file, shard_num_parameters * sizeof(floatOpt), IO_BUF_SIZE, main_stream);
    if(model->use_master_weights) {
        assert(model->master_weights != nullptr);
        file_to_device(model->master_weights, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
        // restore weights from the master weights using the RNG state before last weight update
        model->rng_state = model->rng_state_last_update;
        llama3_update(model, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, &multi_gpu_config, /* init_from_master_only*/ true);
        model->rng_state = *((unsigned long long*)&state_header[20]); // use final RNG state from checkpoint after this
    }

    // revive the DataLoader object and its state
    loader->should_shuffle = should_shuffle;
    if (should_shuffle == 1) {
        // ensure the number of shards matches
        size_t glob_result_gl_pathc;
        freadCheck(&glob_result_gl_pathc, sizeof(size_t), 1, state_file);
        assert(glob_result_gl_pathc == loader->glob_result.gl_pathc);
        // read the shard indices
        loader->shard_indices = (int*)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
        freadCheck(loader->shard_indices, sizeof(int), loader->glob_result.gl_pathc, state_file);
        // ensure the number of samples matches
        size_t shard_num_samples;
        freadCheck(&shard_num_samples, sizeof(size_t), 1, state_file);
        assert(shard_num_samples == loader->shard_num_samples);
        // read the intra-shard indices
        loader->intra_shard_indices = (int*)mallocCheck(loader->shard_num_samples * sizeof(int));
        freadCheck(loader->intra_shard_indices, sizeof(int), loader->shard_num_samples, state_file);
        // read the shuffle rng state
        freadCheck(&loader->shuffle_rng, sizeof(mt19937_state), 1, state_file);
    }
    dataloader_resume(loader, current_shard_idx, current_sample_idx);

    // all done, close state file
    fcloseCheck(state_file);
}

void write_checkpoint(const char* output_log_dir, int step, LLama3* model, DataLoader* train_loader, MultiGpuConfig* multi_gpu_config) {
    // a checkpoint contains: model weights, optimizer/dataloader state, and a DONE file
    printf0("Writing checkpoint at step %d\n", step);
    int rank = multi_gpu_config->process_rank;
    // only rank 0 writes the model file because it is the same across all ranks
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, step);
        llama3_write_to_checkpoint(model, filename_buffer);
    }
    // all ranks write their state file
    snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, step, rank);
    save_state(filename_buffer, step, model, train_loader);
    // DONE file is a signal that this checkpoint as a whole is complete
    multi_gpu_barrier(multi_gpu_config);
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/DONE_%08d", output_log_dir, step);
        FILE* done_file = fopenCheck(filename_buffer, "w");
        fcloseCheck(done_file);
    }
}

void delete_checkpoint(const char* output_log_dir, int step, MultiGpuConfig* multi_gpu_config) {
    // mirrors write_checkpoint function, cleans up checkpoint from disk
    printf0("Deleting checkpoint at step %d\n", step);
    int rank = multi_gpu_config->process_rank;
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, step);
        remove(filename_buffer);
    }
    snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, step, rank);
    remove(filename_buffer);
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/DONE_%08d", output_log_dir, step);
        remove(filename_buffer);
    }
}

#ifndef TESTING
// if we are TESTING (see test_llama3.cu), we'll skip everything below this point

// ----------------------------------------------------------------------------
// training resumption logic, very useful when jobs crash once in a while
// the goal is that we can resume optimization from any checkpoint, bit-perfect
// note that "state" refers to things not already saved in the model checkpoint file

// ----------------------------------------------------------------------------
// CLI, poor man's argparse
// (all single letters have been claimed now)

void error_usage() {
    fprintf(stderr, "Usage:   ./train_llama3cu [options]\n");
    fprintf(stderr, "Options:\n");
    // file system input / output
    fprintf(stderr, "  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
    fprintf(stderr, "  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
    fprintf(stderr, "  -e <string> input .bin filename or descriptor, see code comments as docs. (default = llama3.2_1B_bf16.bin)\n");
    fprintf(stderr, "  -o <string> output log dir (default = NULL, no logging)\n");
    fprintf(stderr, "  -lg <int>   log gpu info every x steps (default = -1; disabled)\n");
    fprintf(stderr, "  -n <int>    write optimization checkpoints every how many steps? (default 0, don't)\n");
    fprintf(stderr, "  -nk <int>   max number of checkpoints to keep in the directory, removing old ones (0 = disable, default)\n");
    fprintf(stderr, "  -nm <int>   every how many step checkpoints are considered major? major checkpoints never get deleted.\n");
    fprintf(stderr, "  -y <int>    resume optimization found inside output log dir? (0=restart/overwrite, 1=resume/append)\n");
    // token layout for each step of the optimization
    fprintf(stderr, "  -b <int>    (per-GPU, micro) batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
    fprintf(stderr, "  -d <int>    total desired batch size (default = B * T * num_processes, i.e. no grad accumulation\n");
    // workload (number of steps)
    fprintf(stderr, "  -x <int>    max_steps of optimization to run (-1 (default) = disable, run 1 epoch)\n");
    // optimization
    fprintf(stderr, "  -k <string> learning rate scheduler (default = cosine)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 3e-4f)\n");
    fprintf(stderr, "  -u <int>    learning rate warmup iterations (default = 0, no warmup)\n");
    fprintf(stderr, "  -q <float>  learning rate decay: final fraction, at end of training (default = 1.0 (no decay))\n");
    fprintf(stderr, "  -c <float>  weight decay (default = 0.0f)\n");
    fprintf(stderr, "  -sl <float> outlier stability: skip update if loss goes above this in zscore (0.0f=off)\n");
    fprintf(stderr, "  -sg <float> outlier stability: skip update if grad_norm goes above this in zscore (0.0f=off)\n");
    // evaluation
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    fprintf(stderr, "  -h <int>    hellaswag eval run? (default = 0)\n");
    // debugging
    fprintf(stderr, "  -a <int>    overfit a single batch? 0/1. useful for debugging\n");
    // numerics
    fprintf(stderr, "  -f <int>    enable_tf32 override (default: 1, set to 0 to disable tf32)\n");
    fprintf(stderr, "  -w <int>    keep f32 copy of weights for the optimizer? (default: 1)\n");
    fprintf(stderr, "  -ge <int>   gelu fusion: 0=none, 1=forward, 2=forward+backward (default: 2 for >=SM90, 0 for older GPUs)\n");
    // memory management
    fprintf(stderr, "  -z <int>    zero_stage, Zero Optimization Stage, 0,1,2,3 (default = 0)\n");
    fprintf(stderr, "  -r <int>    recompute: less memory but less speed. (default = 1), 0|1|2 = none,gelu,gelu+ln\n");
    // multi-node settings
    fprintf(stderr, "  -pn <int>    num_processes (default = 1)\n");
    fprintf(stderr, "  -pr <int>    process_rank (default = 0)\n");
    fprintf(stderr, "  -pg <int>    gpus_per_node (default = 8)\n");
    fprintf(stderr, "  -pm <string> nccl_init_method: tcp,fs,mpi (default = mpi)\n");
    fprintf(stderr, "  -ps <string> server_ip - used only when nccl_init_method is tcp (default = -1)\n");
    fprintf(stderr, "  -pp <string> fs_path - used only when nccl_init_method is fs (default = /tmp)\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// main training loop
int main(int argc, char *argv[]) {
    // read in the (optional) command line arguments
    const char* train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* load_filename = "llama3.2_1B_bf16.bin"; // bf16 weights of the Llama 3.2 1B model
    const char* lr_scheduler_type = "cosine";
    const char* output_log_dir = NULL;
    int checkpoint_every = 0; // write checkpoints every how many steps?
    int checkpoints_keep = 0; // how long checkpoint history do we keep? (in units of checkpoints)
    int major_checkpoint_every = 0; // major checkpoints never get deleted when maintaining history
    int resume = 0; // resume the optimization, if one is found inside output_log_dir?
    int B = 4; // batch size
    int T = 64; // sequence length max
    int total_batch_size = -1; // will be calculated down below later, if not provided
    float learning_rate = 1e-5f;
    int log_gpu_every = -1;
    int warmup_iterations = 0;
    float final_learning_rate_frac = 1.0f; // final fraction of learning rate, at end of training
    float weight_decay = 0.0f;
    float skip_update_lossz = 0.0f; // skip update if loss goes above this in zscore
    float skip_update_gradz = 0.0f; // skip update if grad_norm goes above this in zscore
    int val_loss_every = 20; // every how many steps do we eval validation loss?
    int val_max_steps = 20; // how many batches max do we eval for validation loss?
    int sample_every = 20; // every how many steps to do inference?
    int genT = 64; // number of steps of inference we will do
    int overfit_single_batch = 1; // useful for debugging, 1 = only load a single data batch once
    int max_steps = 10;
    int override_enable_tf32 = 1;
    int use_master_weights = 1;
    int gelu_fusion = -1; // 0 = none, 1 = forward, 2 = forward+backward (-1 => per-GPU default)
    int recompute = 1; // recompute during backward setting, 0 = none, 1 = recompute gelu
    int zero_stage = 0; // Zero Optimization Stage for Multi-GPU training
    int hellaswag_eval = 0;
    // multi-node settings
    int num_processes = 1;  // this should be set by the slurm environment
    int process_rank = 0;  // this should be set by the slurm environment
    int gpus_per_node = 8;  // this should be set by the slurm environment
    char nccl_init_method[256] = "mpi";  // "tcp" or "fs" or "mpi"
    char server_ip[256] = "";  // used if init_method set to "tcp" -> set to your server ip address
    char fs_path[256] = "";  // used if init_method set to "fs" -> set to a shared filesystem path
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (!(strlen(argv[i]) == 2 || strlen(argv[i]) == 3)) { error_usage(); } // must be -x[y] (one dash, one or two letters)
        // read in the args
        if (argv[i][1] == 'i') { train_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'j') { val_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'e') { load_filename = argv[i+1]; }
        else if (argv[i][1] == 'o') { output_log_dir = argv[i+1]; }
        else if (argv[i][1] == 'n' && argv[i][2] == '\0') { checkpoint_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'y') { resume = atoi(argv[i+1]); }
        else if (argv[i][1] == 'b') { B = atoi(argv[i+1]); } // Per-GPU (micro) batch size
        else if (argv[i][1] == 't') { T = atoi(argv[i+1]); }
        else if (argv[i][1] == 'd') { total_batch_size = atoi(argv[i+1]); }
        else if (argv[i][1] == 'l' && argv[i][2] == '\0') { learning_rate = atof(argv[i+1]); }
        else if (argv[i][1] == 'l' && argv[i][2] == 'g') { log_gpu_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'u') { warmup_iterations = atoi(argv[i+1]); }
        else if (argv[i][1] == 'q') { final_learning_rate_frac = atof(argv[i+1]); }
        else if (argv[i][1] == 'c') { weight_decay = atof(argv[i+1]); }
        else if (argv[i][1] == 'x') { max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'm') { val_max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == '\0') { sample_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g' && argv[i][2] == 'e') { gelu_fusion = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
        else if (argv[i][1] == 'a') { overfit_single_batch = atoi(argv[i+1]); }
        else if (argv[i][1] == 'f') { override_enable_tf32 = atoi(argv[i+1]); }
        else if (argv[i][1] == 'w') { use_master_weights = atoi(argv[i+1]); }
        else if (argv[i][1] == 'z') { zero_stage = atoi(argv[i+1]); }
        else if (argv[i][1] == 'r') { recompute = atoi(argv[i+1]); }
        else if (argv[i][1] == 'h') { hellaswag_eval = atoi(argv[i+1]); }
        else if (argv[i][1] == 'k') { lr_scheduler_type = argv[i+1]; }
        else if (argv[i][1] == 'p' && argv[i][2] == 'i') { strcpy(nccl_init_method, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'f') { strcpy(fs_path, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 's') { strcpy(server_ip, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'n') { num_processes = atoi(argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'r') { process_rank = atoi(argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'g') { gpus_per_node = atoi(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == 'l') { skip_update_lossz = atof(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == 'g') { skip_update_gradz = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'k') { checkpoints_keep = atoi(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'm') { major_checkpoint_every = atoi(argv[i+1]); }
        else { error_usage(); }
    }

    multi_gpu_config = multi_gpu_config_init(num_processes, process_rank, gpus_per_node, server_ip, fs_path, nccl_init_method);
    common_start(override_enable_tf32, false); // common init code for train/test/profile

    // should do a bit more error checking here
    assert(warmup_iterations >= 0);
    if (output_log_dir != NULL) {
        assert(strlen(output_log_dir) < 400); // careful bunch of hardcoded snprintf around this
    }
    int tokens_per_fwdbwd = B * T * multi_gpu_config.num_processes; // one micro-batch processes this many tokens
    // calculate sensible default for total batch size as assuming no gradient accumulation
    if (total_batch_size == -1) { total_batch_size = tokens_per_fwdbwd; }
    // in the future, we might want to set gelu fusion to 2 for SM90+ and 0 for other GPUs
    if (gelu_fusion == -1) { gelu_fusion = 0; } // (deviceProp.major >= 9) ? 2 : 0; } // in llama3_init_common for test_llama3cu...
    // calculate the number of gradient accumulation steps from the desired total batch size
    assert(total_batch_size % tokens_per_fwdbwd == 0);
    int grad_accum_steps = total_batch_size / tokens_per_fwdbwd;
    // if we're only overfitting a single batch for debugging, let's overfit the first batch
    // from val instead of train split, because val is smaller and faster. (train_llama3.py does the same)
    if (overfit_single_batch == 1) { train_data_pattern = val_data_pattern; }
    printf0("+-----------------------+----------------------------------------------------+\n");
    printf0("| Parameter             | Value                                              |\n");
    printf0("+-----------------------+----------------------------------------------------+\n");
    printf0("| train data pattern    | %-50s |\n", train_data_pattern);
    printf0("| val data pattern      | %-50s |\n", val_data_pattern);
    printf0("| output log dir        | %-50s |\n", output_log_dir == NULL ? "NULL" : output_log_dir);
    printf0("| checkpoint_every      | %-50d |\n", checkpoint_every);
    printf0("| resume                | %-50d |\n", resume);
    printf0("| micro batch size B    | %-50d |\n", B);
    printf0("| sequence length T     | %-50d |\n", T);
    printf0("| total batch size      | %-50d |\n", total_batch_size);
    printf0("| LR scheduler          | %-50s |\n", lr_scheduler_type);
    printf0("| learning rate (LR)    | %-50e |\n", learning_rate);
    printf0("| warmup iterations     | %-50d |\n", warmup_iterations);
    printf0("| final LR fraction     | %-50e |\n", final_learning_rate_frac);
    printf0("| weight decay          | %-50e |\n", weight_decay);
    printf0("| skip update lossz     | %-50f |\n", skip_update_lossz);
    printf0("| skip update gradz     | %-50f |\n", skip_update_gradz);
    printf0("| max_steps             | %-50d |\n", max_steps);
    printf0("| val_loss_every        | %-50d |\n", val_loss_every);
    printf0("| val_max_steps         | %-50d |\n", val_max_steps);
    printf0("| sample_every          | %-50d |\n", sample_every);
    printf0("| genT                  | %-50d |\n", genT);
    printf0("| overfit_single_batch  | %-50d |\n", overfit_single_batch);
    printf0("| use_master_weights    | %-50s |\n", use_master_weights ? "enabled" : "disabled");
    printf0("| gelu_fusion           | %-50d |\n", gelu_fusion);
    printf0("| recompute             | %-50d |\n", recompute);
    printf0("+-----------------------+----------------------------------------------------+\n");
    const char* precision_str = (PRECISION_MODE == PRECISION_FP32)
                              ? (cublas_compute == CUBLAS_COMPUTE_32F_FAST_TF32 ? "TF32" : "FP32")
                              : (PRECISION_MODE == PRECISION_FP16 ? "FP16" : "BF16");
    printf0("| device                | %-50s |\n", deviceProp.name);
    printf0("| peak TFlops           | %-50.1f |\n", get_flops_promised(deviceProp.name, PRECISION_MODE));
    printf0("| precision             | %-50s |\n", precision_str);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // figure out if we are going to be resuming the optimization
    int resuming = 0;
    // find the DONE file with the highest step count
    int resume_max_step = find_max_step(output_log_dir);
    if (resume == 1) { // is -y 1 resume flag set?
        assert(output_log_dir != NULL);
        if (resume_max_step != -1) {
            resuming = 1; // -y 1 is set, and we found a checkpoint we can resume from
            snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, resume_max_step);
        }
    }

    // build the GPT-2 model
    LLama3 model;
    llama3_init_common(&model);
    if (resuming == 1) {
        // if `-y 1` was set, then we are resuming from the latest checkpoint
        // if we are using master weights, we'll init them later inside load_state()
        bool weight_init = !use_master_weights;
        llama3_build_from_checkpoint(&model, filename_buffer, weight_init);
    } else if (ends_with_bin(load_filename)) {
        // otherwise, if this is a .bin file, we assume it's a model, let's init from it
        llama3_build_from_checkpoint(&model, load_filename);
    } else {
        // For Llama 3.1 we currently demand a .bin file to load the model from, and
        // initializing from scratch is currently not supported (but can be added later)
        printf0("Error: Llama 3 cannot be initialized from scratch right now\n");
        exit(EXIT_FAILURE);
    }

    model.use_master_weights = use_master_weights;
    model.gelu_fusion = gelu_fusion;
    model.recompute = recompute;
    printf0("| weight init method    | %-50s |\n", resuming == 1 ? "intermediate checkpoint" : load_filename);
    printf0("| max_sequence_length T | %-50d |\n", model.config.max_seq_len);
    printf0("| vocab_size V          | %-50d |\n", model.config.vocab_size);
    printf0("| padded_vocab_size Vp  | %-50d |\n", model.config.padded_vocab_size);
    printf0("| num_layers L          | %-50d |\n", model.config.num_layers);
    printf0("| num_heads NH          | %-50d |\n", model.config.num_heads);
    printf0("| channels C            | %-50d |\n", model.config.channels);
    printf0("| num_parameters        | %-50zu |\n", model.num_parameters);
    printf0("+-----------------------+----------------------------------------------------+\n");
    assert(T <= model.config.max_seq_len);

    // build DataLoaders for both train and val
    int permute_train_loader = (overfit_single_batch == 1) ? 0 : 1;
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_data_pattern, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes, permute_train_loader);
    dataloader_init(&val_loader, val_data_pattern, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes, 0);
    // figure out the number of training steps we will run for
    int train_num_batches = max_steps; // passed in from command line
    if (train_num_batches == -1) {
        // sensible default is to train for exactly one epoch
        size_t ntok = train_loader.num_tokens;
        // the number of (outer loop) steps each process should take for us to reach one epoch
        train_num_batches = ntok / total_batch_size;
    }
    // figure out the number of validation steps to run for
    int val_num_batches = val_max_steps; // passed in from command line
    if (val_num_batches == -1) {
        // sensible default is to evaluate the full validation split
        size_t ntok = val_loader.num_tokens;
        // note that unlike the training loop, there is no gradient accumulation inner loop here
        val_num_batches = ntok / tokens_per_fwdbwd;
    }
    printf0("| train_num_batches     | %-50d |\n", train_num_batches);
    printf0("| val_num_batches       | %-50d |\n", val_num_batches);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // build an EvalLoader for HellaSwag
    EvalLoader eval_loader;
    const char* hellaswag_path = "dev/data/hellaswag/hellaswag_val.bin";
    const bool hellaswag_available = access(hellaswag_path, F_OK) == 0;
    const bool run_hellaswag = hellaswag_eval && hellaswag_available;
    if (run_hellaswag) {
        evalloader_init(&eval_loader, hellaswag_path, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes);
    }
    printf0("| run hellaswag         | %-50s |\n", run_hellaswag ? "yes" : "no");
    printf0("+-----------------------+----------------------------------------------------+\n");

    // pretty print in a table the multi-gpu configuration as well
    set_zero_configs(&multi_gpu_config, zero_stage, model.num_parameters);
    printf0("| num_processes         | %-50d |\n", multi_gpu_config.num_processes);
    printf0("| zero_stage            | %-50d |\n", multi_gpu_config.zero_stage);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // prints outside of pretty table to here and below
    if (!hellaswag_available) {
        printf0("HellaSwag eval not found at %s, skipping its evaluation\n", hellaswag_path);
        printf0("You can run `python dev/data/hellaswag.py` to export and use it with `-h 1`.\n");
    }
    // more prints related to allocations from llama3_build_from_checkpoint down here to not mess up our table above
    printf0("num_parameters: %zu => bytes: %zu\n", model.num_parameters, model.num_parameters_bytes);
    printf0("allocated %d MiB for model parameters\n", (int)round(model.num_parameters_bytes / (1024 * 1024)));
    // few more prints for gradient accumulation math up above
    printf0("batch_size B=%d * seq_len T=%d * num_processes=%d and total_batch_size=%d\n",
            B, T, multi_gpu_config.num_processes, total_batch_size);
    printf0("=> setting grad_accum_steps=%d\n", grad_accum_steps);

    // set up logging
    if (multi_gpu_config.process_rank == 0) { create_dir_if_not_exists(output_log_dir); }
    Logger logger;
    logger_init(&logger, output_log_dir, multi_gpu_config.process_rank, resume);

    // set up the Tokenizer
    Tokenizer tokenizer;
    // tokenizer_init(&tokenizer, "llama3_tokenizer.bin"); // TODO: port tokenizer later from GPT2 -> Llama 3

    // set up learning rate scheduler
    LearningRateScheduler lr_scheduler;
    lr_scheduler_init(&lr_scheduler, lr_scheduler_type, learning_rate,
                      warmup_iterations, train_num_batches, final_learning_rate_frac);

    // some memory for generating samples from the model
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    floatX* cpu_logits_raw = (floatX*)mallocCheck(model.config.vocab_size * sizeof(floatX));
    float*  cpu_logits = (float*)mallocCheck(model.config.vocab_size * sizeof(float));

    // if we found a checkpoint to resume from, load the optimization state
    int step = 0;
    llama3_allocate_state(&model, B, T);
    if (resuming == 1) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, resume_max_step, multi_gpu_config.process_rank);
        load_state(&step, &model, &train_loader, filename_buffer);
    }

    // init an OutlierDetector the training loss
    OutlierDetector loss_outlier_detector, grad_norm_outlier_detector;
    init_detector(&loss_outlier_detector);
    init_detector(&grad_norm_outlier_detector);

    // train
    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaProfilerStart());
    double total_sum_iteration_time_s = 0.0;
    float ema_tokens_per_second = 0.0f;
    for (; step <= train_num_batches; step++) {
        NvtxRange step_range("Train step", step);

        int last_step = step == train_num_batches;

        if(0) { // TODO DELETE; START: IGNORE ALL THIS BLOCK WHILE GETTING STUFF TO WORK

        // once in a while estimate the validation loss (all processes collaborate)
        if (step % val_loss_every == 0 || last_step) {
            NvtxRange validation_range("validation");
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                val_loss += llama3_validate(&model, val_loader.inputs, val_loader.targets, B, T);
            }
            val_loss /= val_num_batches;
            val_loss = multi_gpu_cpu_float_sum(val_loss, &multi_gpu_config) / multi_gpu_config.num_processes;
            printf0("val loss %f\n", val_loss);
            logger_log_val(&logger, step, val_loss);
        }

        // once in a while estimate HellaSwag accuracy (all processes collaborate)
        if (run_hellaswag &&
           ((step > 0 && step % val_loss_every == 0) || last_step)) {
            NvtxRange evaluation_range("evaluation");
            float eval_acc_norm = 0.0f;
            evalloader_reset(&eval_loader);
            for (int i = 0; i < eval_loader.num_batches; i++) {
                if (i % 10 == 0) { printf("evaluating HellaSwag: %d/%d\r", i, eval_loader.num_batches); }
                evalloader_next_batch(&eval_loader);
                llama3_validate(&model, eval_loader.inputs, eval_loader.targets, B, T);
                int correct = evalloader_stat_losses(&eval_loader, model.cpu_losses);
                eval_acc_norm += (float)correct;
            }
            // careful because not all ranks may have the exact same allocation of number of examples
            eval_acc_norm = multi_gpu_cpu_float_sum(eval_acc_norm, &multi_gpu_config);
            printf0("HellaSwag: %d/%d = %f\n", (int)eval_acc_norm, eval_loader.num_examples, eval_acc_norm / eval_loader.num_examples);
            logger_log_eval(&logger, step, eval_acc_norm / eval_loader.num_examples);
        }

        // once in a while do model inference to print generated text (only rank 0)
        if (multi_gpu_config.process_rank == 0 && sample_every > 0 &&
           (step > 0 && (step % sample_every) == 0 || last_step)) {
            NvtxRange generation_range("generation");
            unsigned long long sample_rng_state = 1337;
            // fill up gen_tokens with the <|endoftext|> token, which kicks off the generation
            int eot_token = tokenizer.eot_token;
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = eot_token;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                NvtxRange generation_range("Generation step", t);
                // we try not to be too wasteful for inference by not calculating all of B,T
                // Using a smaller B is always bit-for-bit identical, but T is more tricky
                // for non-CUDNN, we need to make sure the attention buffer is memset to 0
                // for cuDNN, it might suddenly decide to use a slightly different algorithm...
                // on cuDNN 9.2.1 with cuDNN FrontEnd 1.5.2, T >= 256 seems bit-for-bit identical
                // (but even if it wasn't fully identical that's probably not the end of the world)
                // note this is still somewhat wasteful because we don't have a KV cache!
                llama3_forward(&model, gen_tokens, 1, CEIL_DIV(t, min(T,256)) * min(T,256));
                // get the V-dimensional vector probs[0, t-1, :]
                floatX* logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
                // move probs back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
                cudaCheck(cudaMemcpy(cpu_logits_raw, logits, model.config.vocab_size * sizeof(floatX), cudaMemcpyDeviceToHost));
                // convert to FP32 into cpu_logits (this does nothing useful if floatX == float)
                for (int i = 0; i < model.config.vocab_size; i++) {
                    cpu_logits[i] = (float)cpu_logits_raw[i];
                }
                // sample the next token
                float coin = random_f32(&sample_rng_state);
                int next_token = sample_softmax(cpu_logits, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // once in a while checkpoint the optimization state (all ranks)
        if ((checkpoint_every > 0 && output_log_dir != NULL && resuming == 0) &&
            ((step > 0 && step % checkpoint_every == 0) || last_step)) {
            // writes model .bin file, state .bin files, and DONE file for step
            write_checkpoint(output_log_dir, step, &model, &train_loader, &multi_gpu_config);
            // we only keep checkpoints_keep checkpoints on disk to save space
            // so now that we wrote a new checkpoint, delete one old one (unless it is a "major" checkpoint)
            // we only do this is checkpoint keeping is turned on (checkpoints_keep > 0)
            int step_delete = step - checkpoints_keep * checkpoint_every;
            if (checkpoints_keep > 0 && step_delete > 0 &&
               (major_checkpoint_every == 0 || step_delete % major_checkpoint_every != 0)
                ) {
                delete_checkpoint(output_log_dir, step_delete, &multi_gpu_config);
            }
        }
        resuming = 0;
        } // TODO DELETE; END: IGNORE ALL THIS BLOCK WHILE GETTING STUFF TO WORK

        // bit confusing: we want to make sure to eval and sample on 0th iteration
        // but also after the very last iteration. so we loop for step <= train_num_batches
        // instead of just < train_num_batches (one extra due to <=), only to do
        // the validation/sampling one last time, and then we break right here as we're done.
        if (last_step) { break; }

        // --------------- TRAINING SECTION BEGIN -----------------
        if (overfit_single_batch == 1) {
            // if we are trying to overfit a single batch, we reset the loader here
            dataloader_reset(&train_loader);
        }
        // do one training step, doing forward/backward/update on total_batch_size tokens
        cudaCheck(cudaEventRecord(start));
        // gradient and loss accumulation loop over micro-batches
        for (int micro_step = 0; micro_step < grad_accum_steps; micro_step++) {
            // fetch the next data batch
            dataloader_next_batch(&train_loader);
            // forward pass. note that we pass in grad_accum_steps, which scales down the loss
            llama3_forward(&model, train_loader.inputs, B, T);
            // backward pass. all model params accumulate gradients with += inside this inner loop
            llama3_backward_and_reduce(&model, train_loader.inputs, train_loader.targets, grad_accum_steps, micro_step);
        }
        float zloss = (float)(update_detector(&loss_outlier_detector, (double)model.mean_loss)); // loss z-score
        // fetch the next learning rate
        float step_learning_rate = get_learning_rate(&lr_scheduler, step);
        // calculate the gradient norm and how much we wish to scale the gradient
        float grad_norm = llama3_calculate_grad_norm(&model, &multi_gpu_config);
        float zgrad = (float)(update_detector(&grad_norm_outlier_detector, (double)grad_norm)); // grad z-score
        // update the model parameters
        if (isfinite(zloss) && skip_update_lossz != 0.0f && zloss > skip_update_lossz) {
            printf0("skipping update due to loss z-score of %f\n", zloss);
        } else if (isfinite(zgrad) && skip_update_gradz != 0.0f && zgrad > skip_update_gradz) {
            printf0("skipping update due to grad z-score of %f\n", zgrad);
        } else {
            // clip the gradient norm to a maximum value
            float grad_clip = 1.0f;
            float grad_scale = (grad_norm > grad_clip) ? grad_clip / grad_norm : 1.0f;
            llama3_update(&model, step_learning_rate, 0.9f, 0.95f, 1e-8f, weight_decay, grad_scale, step+1, &multi_gpu_config);
        }
        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(end)); // wait for the end event to finish to get correct timings
        // --------------- TRAINING SECTION END -------------------
        // everything that follows now is just diagnostics, prints, logging, etc.

        // todo - move or double-buffer all of this timing logic to avoid idling the GPU at this point!
        float time_elapsed_ms;
        cudaCheck(cudaEventElapsedTime(&time_elapsed_ms, start, end));
        size_t tokens_processed = (size_t)multi_gpu_config.num_processes * B * T * grad_accum_steps;
        float tokens_per_second = tokens_processed / time_elapsed_ms * 1000.0f;
        float bias_corrected_ema_tokens_per_second = tokens_per_second; // by default set to non-ema version
        if (step > 0) { // consider the first batch to be a warmup (e.g. cuBLAS/cuDNN initialisation)
            total_sum_iteration_time_s += time_elapsed_ms / 1000.0f;
            // smooth out the tok/s with an exponential moving average, and bias correct just like in AdamW
            ema_tokens_per_second = 0.95f * ema_tokens_per_second + 0.05f * tokens_per_second;
            bias_corrected_ema_tokens_per_second = ema_tokens_per_second / (1.0f - powf(0.95f, step));
        }
        float mfu = llama3_estimate_mfu(&model, B * T * grad_accum_steps, time_elapsed_ms / 1000.0f);
        printf0("step %4d/%d | loss %7.6f (%+.2fz)| norm %6.4f (%+.2fz)| lr %.2e | %.2f ms | %.1f%% bf16 MFU | %.0f tok/s\n",
                step + 1, train_num_batches, model.mean_loss, zloss, grad_norm, zgrad, step_learning_rate,
                time_elapsed_ms, 100*mfu, bias_corrected_ema_tokens_per_second);
        if(log_gpu_every > 0 && (step + 1) % log_gpu_every == 0) {
            GPUUtilInfo gpu_info = get_gpu_utilization_info();
            printf0("                  compute %2.1f%% | memory: %2.1f%% | fan: %2d%% | %4d MHz / %4d MHz | %3d W / %3d W | %d°C / %d°C | %s\n",
                    gpu_info.gpu_utilization, gpu_info.mem_utilization, gpu_info.fan, gpu_info.clock, gpu_info.max_clock, gpu_info.power / 1000, gpu_info.power_limit / 1000,
                    gpu_info.temperature, gpu_info.temp_slowdown, gpu_info.throttle_reason);
        }
        logger_log_train(&logger, step, model.mean_loss, step_learning_rate, grad_norm);

        // disable the profiler after 3 steps of optimization
        if (step == 3) { cudaProfilerStop(); }
    }
    // add a total average, for optimizations that are only mild improvements (excluding 1st batch as warmup)
    printf0("total average iteration time: %f ms\n", total_sum_iteration_time_s / (train_num_batches-1) * 1000);

    // free and destroy everything
    cudaCheck(cudaEventDestroy(end));
    cudaCheck(cudaEventDestroy(start));
    if (run_hellaswag) { evalloader_free(&eval_loader); }
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    free(cpu_logits_raw);
    free(cpu_logits);
    free(gen_tokens);
    multi_gpu_config_free(&multi_gpu_config);
    llama3_free(&model);
    common_free(model);
    return 0;
}
#endif
