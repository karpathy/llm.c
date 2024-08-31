/*
GPT-2 Transformer Neural Net training loop. See README.md for usage.
*/
bool UNIQUE_TENSOR_MEMORY = false;

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
// warpReduceSum, warpReduceMax, blockReduce, copy_and_cast_kernel
#include "llmc/cuda_utils.cuh"
// defines: CUBLAS_LOWP, cublasCheck, cublaslt_workspace_size, cublaslt_workspace
// defines: cublas_compute, cublaslt_handle, cublas_handle
#include "llmc/cublas_common.h"
// ----------- Layer implementations in CUDA -----------
// defines: encoder_forward, encoder_backward
#include "llmc/encoder.cuh"
// defines: layernorm_forward, residual_forward, fused_residual_forward5, layernorm_backward
#include "llmc/layernorm.cuh"
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
// GPT-2 model definition

enum TT : uint8_t {
    PARAMETER=0, PARAMETER_GRADIENT, PARAMETER_MASTER, PARAMETER_OPT_M, PARAMETER_OPT_V,
    MULTIUSE, ACTIVATION, ACTIVATION_GRADIENT,
    DEFAULT, COUNT=DEFAULT
};

typedef struct {
    int wte, wpe, lnfw, lnfb; // not per layer
    int ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b, fcw, fcb, fcprojw, fcprojb; // per layer
} ParameterTensors;

typedef struct {
    int encoded, lnf, lnf_mean, lnf_rstd, losses, output; // not per layer
    int ln1, ln1_mean, ln1_rstd, atty, att, attproj, residual2, ln2, ln2_mean, ln2_rstd, fch, fch_gelu, fcproj, residual3, qkvr; // per layer
} ActivationTensors;

typedef struct {
    int bt4c;   // (B, T, 4*C)
    int btc;    // (B, T, C)
} MultiuseTensors;

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

typedef struct {
    GPT2Config config;
    ParameterTensors params;
    ParameterTensors params_grads;
    ParameterTensors params_master;
    ParameterTensors params_opt_m;
    ParameterTensors params_opt_v;
    ActivationTensors acts;
    ActivationTensors acts_grads;
    MultiuseTensors multiuse;

    size_t num_parameters;
    size_t num_parameters_bytes;

    char* gpu_mem;
    void* params_memory;
    void* grads_memory;
    float* m_memory;
    float* v_memory;
    float* master_weights;

    // other run state configuration
    int batch_size = 0; // the batch size (B) of current forward pass
    int seq_len = 0; // the sequence length (T) of current forward pass
    int* inputs = NULL; // the input tokens for the current forward pass
    int* targets = NULL; // the target tokens for the current forward pass
    float mean_loss = -1.0f; // after the last backward micro-batch, will be populated with mean loss across all GPUs and micro-steps
    float* accumulated_mean_loss = NULL; // GPU buffer used to accumulate loss across micro-steps
    float* cpu_losses = NULL; // CPU buffer to copy the losses to, allocated with cudaMallocHost
    bool init_state = true;   // set to true if master weights need to be initialized
    int use_master_weights = 1; // keep master weights copy in float for optim update? 0|1
    int gelu_fusion = 0; // fuse gelu via cuBLASLt (0=none, 1=forward, 2=forward+backward)
    int recompute = 0; // recompute gelu | layernorm forward during model backward? 0|1|2
    // todo - if other functions need cpu scratch buffers in the future, reuse as generic scratch?
    int* workload_indices = NULL; // encoder_backward, B*T*num_c_groups (int)
    int4* bucket_info = NULL;     // encoder_backward, B*T*num_c_groups (int4) - size for worst case

    unsigned long long rng_state; // the RNG state for seeding stochastic rounding etc.
    unsigned long long rng_state_last_update; // RNG before last gpt2_update() to re-round identically from master weights
} GPT2;

typedef struct {
    char name[16];
    size_t offset; // into base pointer
    size_t num_elements; // per shard
    size_t num_shards;
    DType data_type;
    TT tensor_type;
} TensorSpec;

TensorSpec tensor_specs[16*1024];
size_t num_tensor_specs = 0;
size_t current_tensor_offset = 0;
TT current_tensor_type = TT::PARAMETER;
size_t tensors_start[TT::COUNT] = {0};
size_t tensors_bytes[TT::COUNT] = {0};
size_t tensors_elements[TT::COUNT] = {0};

int add_tensor_spec(const char* name, size_t num_elements, size_t num_shards, DType data_type, int copy_offset_from=-1, TT tensor_type=TT::DEFAULT) {
    assert(num_tensor_specs < 16*1024);
    assert((num_elements % num_shards) == 0);
    TensorSpec* spec = &tensor_specs[num_tensor_specs++];

    strncpy(spec->name, name, 16);
    spec->num_elements = num_elements / num_shards;
    spec->num_shards = num_shards;
    spec->data_type = data_type;
    spec->tensor_type = (tensor_type == TT::DEFAULT) ? current_tensor_type : tensor_type;
    tensors_elements[spec->tensor_type] += spec->num_elements;


    if (copy_offset_from >= 0) {
        spec->offset = tensor_specs[copy_offset_from].offset;
        size_t original_tensor_bytes = tensor_specs[copy_offset_from].num_elements * sizeof_dtype(tensor_specs[copy_offset_from].data_type);
        size_t new_tensor_bytes = spec->num_elements * sizeof_dtype(data_type);
        assert(new_tensor_bytes <= original_tensor_bytes);
    } else {
        spec->offset = current_tensor_offset;
        current_tensor_offset += spec->num_elements * sizeof_dtype(data_type);

        tensors_bytes[spec->tensor_type] += spec->num_elements * sizeof_dtype(data_type);
        if (tensors_start[spec->tensor_type] == 0 && spec->tensor_type != 0) {
            tensors_start[spec->tensor_type] = spec->offset;
        }
    }
    return num_tensor_specs - 1;
}

int add_layer_specs(int num_layers, const char* name, size_t num_elements, size_t num_shards, DType data_type, int copy_offset_from=-1, bool copy_per_layer=false, TT tensor_type=TT::DEFAULT) {
    int first_tensor_id = num_tensor_specs;
    for (int l = 0; l < num_layers; l++) {
        char layer_name[16];
        assert(snprintf(layer_name, 16, "%s_%d", name, l) >= 0);
        add_tensor_spec(num_layers > 1 ? layer_name : name, num_elements, num_shards, data_type, copy_offset_from, tensor_type);
        if (copy_per_layer) {
            copy_offset_from++;
        }
    }
    return first_tensor_id;
}

#define TENSOR_SPECS(name, dim1, dim2) spec->name = add_layer_specs(dim1, #name, dim2, shards, dtype)
#define TENSOR_SPECS_LOWP(name, dim1, dim2) spec->name = add_layer_specs(dim1, #name, dim2, shards, dtype_lowp)
#define TENSOR_SPECS_FP32(name, dim1, dim2) spec->name = add_layer_specs(dim1, #name, dim2, shards, DType::FP32) // todo - won't work loading model

void gpt2_allocate(GPT2 *model) {
    size_t Vp = model->config.padded_vocab_size;
    size_t C = model->config.channels;
    size_t maxT = model->config.max_seq_len;
    size_t L = model->config.num_layers;
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t NH = model->config.num_heads;
    size_t output_size = B*T * max(4*C, max(NH*T, Vp));
    size_t BTC = B*T*C;

    size_t shards = 1;
    int num_gpu = multi_gpu_config.num_processes;
    int shards_opt = (multi_gpu_config.zero_stage >= 1) ? num_gpu : 1;
    int shards_grad = (multi_gpu_config.zero_stage >= 2) ? num_gpu : 1;

    // 1) parameters & optimizer state
    for (int t = TT::PARAMETER; t <= TT::PARAMETER_OPT_V; t++) {
        DType dtype = (t <= TT::PARAMETER_GRADIENT) ? DTYPE_FLOATX : DType::FP32;
        DType dtype_lowp = (t <= TT::PARAMETER_GRADIENT) ? DTYPE_FLOATX : DType::FP32; // FP8 in the future

        current_tensor_type = (TT)t;
        ParameterTensors* spec;
        switch (t) {
            case TT::PARAMETER: spec = &model->params; shards = 1; break;
            case TT::PARAMETER_GRADIENT: spec = &model->params_grads; shards = shards_grad; break;
            case TT::PARAMETER_MASTER: spec = &model->params_master; shards = shards_opt; break;
            case TT::PARAMETER_OPT_M: spec = &model->params_opt_m; shards = shards_opt; break;
            case TT::PARAMETER_OPT_V: spec = &model->params_opt_v; shards = shards_opt; break;
        }
        if (t == PARAMETER_MASTER && !model->use_master_weights) {
            continue;
        }

        TENSOR_SPECS     (wte,        1, Vp * C);
        TENSOR_SPECS     (wpe,        1, maxT * C);
        TENSOR_SPECS     (ln1w,       L, C);
        TENSOR_SPECS     (ln1b,       L, C);
        TENSOR_SPECS_LOWP(qkvw,       L, 3 * C * C);
        TENSOR_SPECS     (qkvb,       L, 3 * C);
        TENSOR_SPECS_LOWP(attprojw,   L, C * C);
        TENSOR_SPECS     (attprojb,   L, C);
        TENSOR_SPECS     (ln2w,       L, C);
        TENSOR_SPECS     (ln2b,       L, C);
        TENSOR_SPECS_LOWP(fcw,        L, 4 * C * C);
        TENSOR_SPECS_LOWP(fcb,        L, 4 * C);
        TENSOR_SPECS_LOWP(fcprojw,    L, 4 * C * C);
        TENSOR_SPECS     (fcprojb,    L, C);
        TENSOR_SPECS     (lnfw,       1, C);
        TENSOR_SPECS     (lnfb,       1, C);
    }

    // 2) multiuse & scratch tensors
    if (UNIQUE_TENSOR_MEMORY) {
        model->multiuse.bt4c = -1;
        model->multiuse.btc = -1;
    } else {
        model->multiuse.bt4c = add_tensor_spec("multiuse_bt4c", 4 * BTC, 1, DTYPE_FLOATX, -1, TT::MULTIUSE);
        model->multiuse.btc = add_tensor_spec("multiuse_btc", BTC, 1, DTYPE_FLOATX, -1, TT::MULTIUSE);
    }

    /*
    tensors[0] = TENSOR_SPEC(data->encoded, B * T * C);
    // if recompute >= 1 then we will recompute the layernorm forward activation during backward pass
    tensors[1] = TENSOR_SPEC(data->ln1,  (recompute < 2) ? L * B * T * C : 0);
    tensors[2] = TENSOR_SPEC(data->ln1_mean, L * B * T);
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
    tensors[10] = TENSOR_SPEC(data->fch, L * B * T * 4*C);
    // if recompute >= 1 then we will recompute gelu_forward during backward and use this as scratch buffer
    tensors[11] = TENSOR_SPEC(data->fch_gelu, (recompute < 1) ? L * B * T * 4*C : B * T * 4*C);
    tensors[12] = TENSOR_SPEC(data->residual3, L * B * T * C);
    tensors[13] = TENSOR_SPEC(data->lnf, B * T * C);
    tensors[14] = TENSOR_SPEC(data->lnf_mean, B * T);
    tensors[15] = TENSOR_SPEC(data->lnf_rstd, B * T);
    tensors[16] = TENSOR_SPEC(data->losses, B * T);
    tensors[17] = TENSOR_SPEC(data->qkvr, L * B * T * 3*C);
    tensors[18] = TENSOR_SPEC(data->output, B * T * max(3*C, max(NH*T, Vp)));

    tensors[19] = TENSOR_SPEC(data->scratch_bt4c, B * T * 4 * C);
    tensors[20] = TENSOR_SPEC(data->scratch_btc, B * T * C);
    */


    // 3) activations
    current_tensor_type = TT::ACTIVATION;
    ActivationTensors* spec = &model->acts;
    DType dtype = DTYPE_FLOATX;
    DType dtype_lowp = DTYPE_FLOATX; // todo FP8
    shards = 1;

    TENSOR_SPECS     (encoded,    1, BTC);
    TENSOR_SPECS     (lnf,        1, BTC);
    TENSOR_SPECS_FP32(lnf_mean,   1, B*T);
    TENSOR_SPECS_FP32(lnf_rstd,   1, B*T);
    TENSOR_SPECS_FP32(losses,     1, B*T);
    TENSOR_SPECS     (output,     1, output_size);
    TENSOR_SPECS_FP32(ln1_mean,   L, B*T);
    TENSOR_SPECS_FP32(ln1_rstd,   L, B*T);
    TENSOR_SPECS     (atty,       L, BTC);
    TENSOR_SPECS     (residual2,  L, BTC);
    TENSOR_SPECS_FP32(ln2_mean,   L, B*T);
    TENSOR_SPECS_FP32(ln2_rstd,   L, B*T);
    TENSOR_SPECS     (residual3,  L, BTC);
    TENSOR_SPECS_LOWP(fch,        L, 4 * BTC);
    TENSOR_SPECS     (qkvr,       L, 3 * BTC);
    #ifdef ENABLE_CUDNN
    TENSOR_SPECS_FP32(att,        L, NH * B * T);
    #else
    TENSOR_SPECS     (att,        L, NH * B * T * T);
    #endif

    if (UNIQUE_TENSOR_MEMORY) {
        TENSOR_SPECS_LOWP(fcproj,   L, BTC);
        TENSOR_SPECS_LOWP(attproj,  L, BTC);
    } else {
        spec->fcproj = add_layer_specs(L, "fcproj", BTC, L, dtype_lowp, model->multiuse.btc);
        spec->attproj = add_layer_specs(L, "attproj", BTC, L, dtype_lowp, model->multiuse.btc);
    }

    if (model->recompute < 1 || UNIQUE_TENSOR_MEMORY) {
        TENSOR_SPECS(ln1,           L, BTC);
        TENSOR_SPECS(ln2,           L, BTC);
        TENSOR_SPECS(fch_gelu,      L, 4 * BTC);
    } else if (model->recompute < 2) {
        TENSOR_SPECS(ln1,           L, BTC);
        TENSOR_SPECS(ln2,           L, BTC);
        spec->fch_gelu = add_layer_specs(L, "fch_gelu", 4 * BTC, L, dtype_lowp, model->acts.output);
    } else {
        spec->ln1 = add_layer_specs(L, "ln1", BTC, L, dtype, model->acts.output);
        spec->ln2 = add_layer_specs(L, "ln2", BTC, L, dtype, model->acts.output);
        spec->fch_gelu = add_layer_specs(L, "fch_gelu", 4 * BTC, L, dtype_lowp, model->acts.output);
    }

    // 4) activation gradients
    current_tensor_type = TT::ACTIVATION_GRADIENT;
    dtype_lowp = DTYPE_FLOATX; // todo FP8
    shards = 1;

    if (UNIQUE_TENSOR_MEMORY) {
        TENSOR_SPECS(output,     1, B*T * max(3*C, max(NH*T, Vp)));
        TENSOR_SPECS(lnf,        1, BTC);
        TENSOR_SPECS(ln1,        L, BTC);
        TENSOR_SPECS(atty,       L, BTC);
        TENSOR_SPECS(residual2,  L, BTC);
        TENSOR_SPECS(ln2,        L, BTC);
        TENSOR_SPECS(fch,        L, 4 * BTC);
        TENSOR_SPECS(fch_gelu,   L, 4 * BTC);
        TENSOR_SPECS(residual3,  L, BTC);
        TENSOR_SPECS(qkvr,       L, 3 * BTC);
    } else {
        spec->output = add_layer_specs(1, "output", output_size, 1, dtype, model->acts.output);
        spec->lnf = add_layer_specs(1, "lnf", BTC, 1, dtype, model->multiuse.btc);
        spec->ln1 = add_layer_specs(L, "ln1", BTC, 1, dtype, model->multiuse.btc);
        spec->atty = add_layer_specs(L, "atty", BTC, 1, dtype, model->multiuse.btc);
        spec->residual2 = add_layer_specs(L, "residual2", BTC, 1, dtype, model->multiuse.btc);
        spec->ln2 = add_layer_specs(L, "ln2", BTC, 1, dtype, model->multiuse.btc);
        spec->fch = add_layer_specs(L, "fch", 4 * BTC, 1, dtype, model->multiuse.bt4c);
        spec->fch_gelu = add_layer_specs(L, "fch_gelu", 4 * BTC, 1, dtype, model->multiuse.bt4c);
        spec->residual3 = add_layer_specs(L, "residual3", BTC, 1, dtype, model->multiuse.btc);
        spec->qkvr = add_layer_specs(L, "qkvr", 3 * BTC, 1, dtype, model->multiuse.bt4c);
    }

    // allocate a single huge GPU buffer for all the tensors
    printf("Current tensor offset in MiB: %zu", current_tensor_offset / (1024*1024));
    cudaCheck(cudaMalloc(&model->gpu_mem, current_tensor_offset));

    //initialise helper variables
    model->num_parameters = tensors_elements[TT::PARAMETER];
    model->num_parameters_bytes = tensors_bytes[TT::PARAMETER];

    model->params_memory = (void*)(model->gpu_mem + tensors_start[TT::PARAMETER]);
    model->grads_memory = (void*)(model->gpu_mem + tensors_start[TT::PARAMETER_GRADIENT]);
    model->m_memory = (float*)(model->gpu_mem + tensors_start[TT::PARAMETER_OPT_M]);
    model->v_memory = (float*)(model->gpu_mem + tensors_start[TT::PARAMETER_OPT_V]);
    if (model->use_master_weights) {
        model->master_weights = (float*)(model->gpu_mem + tensors_start[TT::PARAMETER_MASTER]);
    }

    // printf gpu_mem and params_memory
    printf("gpu_mem: %p\n", model->gpu_mem);
    printf("params_memory: %p\n", model->params_memory);
    printf("number of parameters: %zu\n", model->num_parameters);
    printf("number of parameters bytes: %zu\n", model->num_parameters_bytes);

    // parameter gradient bytes
    size_t param_grad_bytes = tensors_bytes[TT::PARAMETER_GRADIENT];
    printf("number of parameter gradient bytes: %zu\n", param_grad_bytes / (1024*1024));
    // number of master weight bytes
    size_t master_weight_bytes = tensors_bytes[TT::PARAMETER_MASTER];
    printf("number of master weight bytes: %zu\n", master_weight_bytes / (1024*1024));
    // opt state m
    size_t m_bytes = tensors_bytes[TT::PARAMETER_OPT_M];
    printf("number of m bytes: %zu\n", m_bytes / (1024*1024));
    // opt state v
    size_t v_bytes = tensors_bytes[TT::PARAMETER_OPT_V];
    printf("number of v bytes: %zu\n", v_bytes / (1024*1024));
    // number of act bytes
    size_t act_bytes = tensors_bytes[TT::ACTIVATION];
    printf("number of act bytes: %zu\n", act_bytes / (1024*1024));
    // number of act gradient bytes
    size_t act_grad_bytes = tensors_bytes[TT::ACTIVATION_GRADIENT];
    printf("number of act grad bytes: %zu\n", act_grad_bytes / (1024*1024));
    // number of multiuse bytes
    size_t multiuse_bytes = tensors_bytes[TT::MULTIUSE];
    printf("number of multiuse bytes: %zu\n", multiuse_bytes / (1024*1024));

    printf("number of act+actgrad+multiuse bytes: %zu\n", (multiuse_bytes + act_bytes + act_grad_bytes) / (1024*1024));

    // =======================
    // allocate_state stuff
    // =======================
    // allocate the space
    cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&model->targets, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(((void**)&model->accumulated_mean_loss), sizeof(float)));
    cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));

    // initialise cpu scratch buffers for encoder backward
    size_t num_c_groups = CEIL_DIV(model->config.channels, (WARP_SIZE * x128::size));
    assert((size_t)(model->batch_size * model->seq_len) * num_c_groups < (1ULL<<31ULL)); // todo - maybe an issue for llama3-400B(?)
    model->workload_indices = (int*)mallocCheck(sizeof(int) * model->batch_size * model->seq_len * num_c_groups);
    model->bucket_info = (int4*)mallocCheck(sizeof(int4) * model->batch_size * model->seq_len * num_c_groups);

    size_t free, total;
    cudaCheck(cudaMemGetInfo(&free, &total));
    printf0("device memory usage: %zd MiB / %zd MiB\n", (total-free) / 1024 / 1024, total / 1024 / 1024);

    // give an estimate of the maximum batch size
    size_t bytes_per_sequence = tensors_bytes[TT::ACTIVATION] / B;
    bytes_per_sequence += tensors_bytes[TT::ACTIVATION_GRADIENT] / B;
    bytes_per_sequence += tensors_bytes[TT::MULTIUSE] / B; // pessimistic?
    printf0("memory per sequence: %zu MiB\n", bytes_per_sequence / 1024 / 1024);
    printf0(" -> estimated maximum batch size: %zu\n", B + free / bytes_per_sequence);
}

void gpt2_init_common(GPT2 *model) {
    // other default settings
    model->rng_state = 13371337 + multi_gpu_config.process_rank; // used in stochastic rounding
}

void gpt2_write_to_checkpoint(GPT2 *model, const char* checkpoint_path) {
    // write the model to a checkpoint file
    printf0("Writing model to %s\n", checkpoint_path);
    FILE *model_file = fopenCheck(checkpoint_path, "wb");
    // write the header first
    int model_header[256];
    memset(model_header, 0, sizeof(model_header));
    model_header[0] = 20240326; // magic number
    assert(PRECISION_MODE == PRECISION_FP32 || PRECISION_MODE == PRECISION_BF16);
    model_header[1] = PRECISION_MODE == PRECISION_FP32 ? 3 : 5; // version
    model_header[2] = model->config.max_seq_len;
    model_header[3] = model->config.vocab_size;
    model_header[4] = model->config.num_layers;
    model_header[5] = model->config.num_heads;
    model_header[6] = model->config.channels;
    model_header[7] = model->config.padded_vocab_size;
    fwriteCheck(model_header, sizeof(int), 256, model_file);
    // write the parameters
    device_to_file(model_file, model->params_memory, model->num_parameters_bytes,  IO_BUF_SIZE, main_stream);
    // close file, we're done
    fcloseCheck(model_file);
}

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path, bool weight_init=true) {
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
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(EXIT_FAILURE); }
    int version = model_header[1];
    if (!(version == 3 || version == 5)) {
        // 3 = fp32, padded vocab
        // 5 = bf16, padded vocab, layernorms also in bf16
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
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
            fprintf(stderr, "---> HINT: to turn on FP32 you have to compile like: `make train_gpt2cu PRECISION=FP32`\n");
            fprintf(stderr, "---> HINT: are you sure you're loading a .bin file without any _bf16 in the name?\n");
            exit(EXIT_FAILURE);
        }
    }

    // read in hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    gpt2_allocate(model);

    // read in the parameters if weight_init is true
    if (weight_init) {
        file_to_device(model->params_memory, model_file, model->num_parameters_bytes, IO_BUF_SIZE, main_stream);
    }
    fcloseCheck(model_file);

    // only return from this function once we are certain the params are ready on the GPU
    cudaCheck(cudaDeviceSynchronize());
}

void gpt2_set_hyperparameters(GPT2Config* config, const char* depth_str) {
    int depth = atoi(depth_str);
    assert(depth > 0); // atoi returns 0 if not a number
    int channels, num_heads;
    if      (depth == 6)  { channels = 384; num_heads = 6; }   // (unofficial) gpt2-tiny (30M)
    else if (depth == 12) { channels = 768; num_heads = 12; }  // gpt2 (124M)
    else if (depth == 24) { channels = 1024; num_heads = 16; } // gpt2-medium (350M)
    else if (depth == 36) { channels = 1280; num_heads = 20; } // gpt2-large (774M)
    else if (depth == 48) { channels = 1600; num_heads = 25; } // gpt2-xl (1558M)
    else if (depth == 60) { channels = 1920; num_heads = 30; } // (unofficial) 2.7B
    else if (depth == 72) { channels = 2880; num_heads = 30; } // (unofficial) 7.3B
    else if (depth == 84) { channels = 3456; num_heads = 36; } // (unofficial) 12.2B
    else { fprintf(stderr, "Unsupported GPT-2 depth: %d\n", depth); exit(EXIT_FAILURE); }
    config->num_layers = depth;
    config->channels = channels;
    config->num_heads = num_heads;
    config->max_seq_len = 1024;
}

void gpt3_set_hyperparameters(GPT2Config* config, const char* channels_str) {
    // we use channels instead of depth for GPT-3 because GPT-3 model depths are not one-to-one
    // note that our models are not necessarily identical to GPT-3 because
    // we use dense attention, not the alternating dense/banded attention of GPT-3
    int channels = atoi(channels_str);
    assert(channels > 0); // atoi returns 0 if not a number
    int depth, head_size;
    if      (channels == 384)   { depth = 6;  head_size = 64; }  // (unofficial) gpt3-tiny (31M)
    else if (channels == 768)   { depth = 12; head_size = 64; }  // gpt3-small (125M)
    else if (channels == 1024)  { depth = 24; head_size = 64; }  // gpt3-medium (350M)
    else if (channels == 1536)  { depth = 24; head_size = 96; }  // gpt3-large (760M)
    else if (channels == 2048)  { depth = 24; head_size = 128; } // gpt3-xl (1.3B) [heads fixed]
    else if (channels == 2560)  { depth = 32; head_size = 80; }  // gpt3-2.7B
    else if (channels == 4096)  { depth = 32; head_size = 128; } // gpt3-6.7B
    else if (channels == 5140)  { depth = 40; head_size = 128; } // gpt3-13B
    else if (channels == 12288) { depth = 96; head_size = 128; } // gpt3 (175B)
    else { fprintf(stderr, "Unsupported GPT-3 channels: %d\n", channels); exit(EXIT_FAILURE); }
    assert(channels % head_size == 0);
    config->num_layers = depth;
    config->channels = channels;
    config->num_heads = channels / head_size;
    config->max_seq_len = 2048; // NOTE: GPT-3 uses context length of 2048 tokens, up from 1024 in GPT-2
}

void gpt_build_from_descriptor(GPT2 *model, const char* descriptor) {
    // The model descriptor can be:
    // - legacy format "dX", where X is number, e.g. "d12". This creates GPT-2 model with 12 layers.
    // - new explicit format "gpt2:dX", same as above, e.g. "gpt2:d48" for GPT-2 with 48 layers.
    // - "gpt3:cX", where X is now the channel count, e.g. "gpt3:c768" is the smallest GPT-3 model.

    // check the valid prexies and dispatch to the right setup function
    assert(descriptor != NULL);
    size_t len = strlen(descriptor);
    if (len > 1 && descriptor[0] == 'd') {
        gpt2_set_hyperparameters(&model->config, descriptor + 1); // pass along the depth str without the 'd'
    } else if (len > 6 && strncmp(descriptor, "gpt2:d", 6) == 0) {
        gpt2_set_hyperparameters(&model->config, descriptor + 6); // pass along the depth str without the 'gpt2:d'
    } else if (len > 6 && strncmp(descriptor, "gpt3:c", 6) == 0) {
        gpt3_set_hyperparameters(&model->config, descriptor + 6); // pass along the channels str without the 'gpt3:c'
    } else {
        fprintf(stderr, "Unsupported model descriptor: %s\n", descriptor); exit(EXIT_FAILURE);
    }

    // both GPT-2 and GPT-3 use the same tokenizer with 50257 tokens
    model->config.vocab_size = 50257;
    model->config.padded_vocab_size = 50304; // padded to 128 for CUDA kernel efficiency

    gpt2_allocate(model);

    // allocate and random init the memory for all the parameters with GPT-2 schema
    // weights ~N(0, 0.02), biases 0, c_proj weights ~N(0, 0.02/(2*L)**0.5)
    // NOTE: assuming all parameters are of the type floatX, could be relaxed later
    mt19937_state init_rng;
    manual_seed(&init_rng, 42);
    floatX* params_memory_cpu = (floatX*)mallocCheck(model->num_parameters_bytes);
    memset(params_memory_cpu, 0, model->num_parameters_bytes);
    // fill in all the weights with random values
    float residual_scale = 1.0f / sqrtf(2.0f * model->config.num_layers);
    // we have to init all these tensors exactly in the order that PyTorch initializes them
    // so that we can match them up and get correctness and exactly the same initial conditions
    /*
    size_t L = model->config.num_layers;
    size_t offset = 0;
    for (int l = 0; l < L; l++) {
        offset = 0;
        for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            // the layernorm parameters are all initialized to 1
            if (l == 0 && (i == 2 || i == 8 || i == 14)) { // only at l = 0 to init these just once
                for (size_t j = 0; j < model->param_elements[i]; j++) {
                    params_memory_cpu[offset + j] = 1.0f;
                }
            }
            // weights tensors are handled here
            if ((l == 0 && (i == 0 || i == 1)) // only at l = 0, init the wte and wpe tensors
              || i == 4 || i == 6 || i == 10 || i == 12) {
                size_t n = model->param_elements[i];
                size_t layer_offset = 0;
                if (i == 0) {
                    // for wte tensor (padded vocab) override to init V instead of Vp rows
                    n = model->config.vocab_size * model->config.channels;
                }
                if (i == 4 || i == 6 || i == 10 || i == 12) {
                    // weight tensors, we are only initializing layer l
                    assert(n % L == 0);
                    n = n / L;
                    layer_offset = l * n;
                }
                // in GPT-2, the projections back into the residual stream are additionally
                // scaled by 1/sqrt(2*L) for training stability
                float scale = (i == 6 || i == 12) ? 0.02f * residual_scale : 0.02f;
                // okay let's draw the random numbers and write them
                float *fp32_buffer = (float*)mallocCheck(n * sizeof(float));
                normal_(fp32_buffer, n, 0.0f, scale, &init_rng);
                for (size_t j = 0; j < n; j++) {
                    params_memory_cpu[offset + layer_offset + j] = (floatX)fp32_buffer[j];
                }
                free(fp32_buffer);
            }
            offset += model->param_elements[i];
        }
    }
    */
    // copy them to GPU
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, model->num_parameters_bytes, cudaMemcpyHostToDevice));
    free(params_memory_cpu);
}

#define GPU_X(x) (floatX*)((char*)model->gpu_mem + tensor_specs[x].offset)
#define GPU_F32(x) (float*)((char*)model->gpu_mem + tensor_specs[x].offset)

// propagate inputs through the network to produce logits.
// right now, this function is fully synchronous with the host
void gpt2_forward(GPT2 *model, const int* inputs, size_t B, size_t T) {
    NVTX_RANGE_FN();
    // we must be careful and use size_t instead of int, otherwise
    // we could overflow int. E.g. l * B * NH * T * T overflows int at B 16.

    // convenience parameters
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;

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
    encoder_forward(GPU_X(acts.encoded), model->inputs, GPU_X(params.wte), GPU_X(params.wpe), B, T, C, main_stream); // encoding goes into residual[0]

    // first layernorm isn't fused
    layernorm_forward(GPU_X((model->recompute < 2) ? acts.ln1 : acts.lnf), GPU_F32(acts.ln1_mean), GPU_F32(acts.ln1_rstd), GPU_X(acts.encoded), GPU_X(params.ln1w), GPU_X(params.ln1b), B, T, C, main_stream);

    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);

        floatX* residual = GPU_X(l == 0 ? acts.encoded : (acts.residual3 + l - 1));

        // get the pointers of the weights for this layer
        floatX* l_qkvw = GPU_X(params.qkvw + l);
        floatX* l_qkvb = GPU_X(params.qkvb + l);
        floatX* l_attprojw = GPU_X(params.attprojw);
        floatX* l_attprojb = GPU_X(params.attprojb);
        floatX* l_ln2w = GPU_X(params.ln2w);
        floatX* l_ln2b = GPU_X(params.ln2b);
        floatX* l_fcw = GPU_X(params.fcw);
        floatX* l_fcb = GPU_X(params.fcb);
        floatX* l_fcprojw = GPU_X(params.fcprojw);
        floatX* l_fcprojb = GPU_X(params.fcprojb);

        // get the pointers of the activations for this layer
        floatX* l_ln1 = GPU_X((model->recompute < 2) ? acts.ln1 + l : acts.lnf);
        floatX* l_qkvr = GPU_X(acts.qkvr + l);
        floatX* l_atty = GPU_X(acts.atty + l);
        floatX* l_residual2 = GPU_X(acts.residual2 + l);
        floatX* l_ln2 = GPU_X((model->recompute < 2) ? acts.ln2 + l : acts.lnf);
        float* l_ln2_mean = GPU_F32(acts.ln2_mean + l);
        float* l_ln2_rstd = GPU_F32(acts.ln2_rstd + l);
        floatX* l_fch = GPU_X(acts.fch + l);
        // reuse the same activation buffer at each layer, as we'll re-compute the gelu during backward
        // very useful because we dramatically reduce VRAM usage, and may be able to fit larger batch size
        floatX* l_fch_gelu = GPU_X((model->recompute < 1) ? acts.fch_gelu + l : acts.fch_gelu);
        floatX* l_residual3 = GPU_X(acts.residual3 + l);
        floatX* scratch = GPU_X(acts.output); // used for non-cudnn attention, fcproj, attproj, etc.

        // now do the forward pass
        #ifdef ENABLE_CUDNN
        float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
        matmul_forward_cublaslt(l_qkvr, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        attention_forward_cudnn(l_atty, (float*)l_att, l_qkvr, B, T, NH, C, main_stream);
        #else
        floatX* l_att = GPU_X(acts.att + l);
        if (T != model->seq_len) { // unused parts of attention buffer must be zeroed (T-dependent)
            cudaCheck(cudaMemset(l_att, 0, B * NH * T * T * sizeof(floatX)));
        }
        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        matmul_forward_cublaslt(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH, main_stream);
        #endif

        matmul_forward_cublaslt(scratch, l_atty, l_attprojw, l_attprojb, B, T, C, C, main_stream);
        fused_residual_forward5(l_residual2, l_ln2, l_ln2_mean, l_ln2_rstd, residual, scratch, l_ln2w, l_ln2b, B*T, C, main_stream);
        matmul_forward_cublaslt(l_fch_gelu, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, main_stream, l_fch, model->gelu_fusion);
        matmul_forward_cublaslt(scratch, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, main_stream);
        // OK, fusion across blocks.
        if(l+1 != L) {
            floatX* l_ln1 = GPU_X((model->recompute < 2) ? acts.ln1 + (l + 1) : acts.lnf);
            float* l_ln1_mean = GPU_F32(acts.ln1_mean + (l + 1));
            float* l_ln1_rstd = GPU_F32(acts.ln1_rstd + (l + 1));
            const floatX* l_ln1w = GPU_X(params.ln1w + (l + 1));
            const floatX* l_ln1b = GPU_X(params.ln1b + (l + 1));
            fused_residual_forward5(l_residual3, l_ln1, l_ln1_mean, l_ln1_rstd, l_residual2, scratch, l_ln1w, l_ln1b,
                                    B * T, C, main_stream);
        } else {
            fused_residual_forward5(l_residual3, GPU_X(acts.lnf), GPU_F32(acts.lnf_mean), GPU_F32(acts.lnf_rstd), l_residual2, scratch,
                                    GPU_X(params.lnfw), GPU_X(params.lnfb),
                                    B * T, C, main_stream);
        }
    }

    matmul_forward_cublaslt(GPU_X(acts.output), GPU_X(acts.lnf), GPU_X(params.wte), NULL, B, T, C, Vp, main_stream);
    cudaCheck(cudaDeviceSynchronize());
}


// Forwards both the model and the loss and is used for validation splits and evals.
// In particular it populates cpu_losses with loss at each token.
// Some of the evals (e.g. HellaSwag) require the per-token losses, which are produced here.
float gpt2_validate(GPT2 *model, const int* inputs, const int* targets, size_t B, size_t T) {
    assert(targets != NULL);
    // forward the model itself
    gpt2_forward(model, inputs, B, T);
    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;

    NvtxRange classifier_and_loss_range("classifier_and_loss");
    ActivationTensors acts = model->acts;
    float mean_loss = 0.0f;
    // fused classifier: does the forward pass and first part of the backward pass
    const float dloss = 1.0f / (B * T); // results in the uniform average loss over all elements
    // note: we don't need to generate dlogits here
    cudaCheck(cudaMemset(GPU_F32(acts.losses), 0, B*T*sizeof(float)));
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B*T, V); // while the memcpy is underway, validate the targets
    fused_classifier(GPU_X(acts.output), GPU_F32(acts.losses), dloss, model->targets, B, T, V, Vp, False, main_stream);
    cudaCheck(cudaMemcpy(model->cpu_losses, GPU_F32(acts.losses), B * T * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < B*T; i++) {
        mean_loss += model->cpu_losses[i];
    }
    mean_loss /= B*T;
    cudaCheck(cudaDeviceSynchronize());
    return mean_loss;
}

void gpt2_backward_and_reduce(GPT2 *model, int* inputs, const int* targets, int grad_accum_steps, int micro_step) {
    return; // debugging forward only first...

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
        cudaCheck(cudaMemsetAsync(GPU_X(model->acts.losses), 0, model->batch_size * model->seq_len * sizeof(float), main_stream));
        cudaCheck(cudaMemsetAsync(model->grads_memory, 0, tensors_bytes[TT::PARAMETER_GRADIENT], main_stream));
    }

    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    const size_t B = model->batch_size;
    const size_t T = model->seq_len;
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;

    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->params_grads;
    ActivationTensors acts = model->acts;

    // accumulate the losses inside acts.losses, and kick off the backward pass inside the fused classifier
    NvtxRange classifier_and_loss_range("classifier_and_loss");
    const float dloss = 1.0f / (float)(B * T * grad_accum_steps); // results in the uniform average loss over all elements
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B*T, V);
    fused_classifier(GPU_X(acts.output), GPU_F32(acts.losses), dloss, model->targets, B, T, V, Vp, True, main_stream);

    // backward pass: go in the reverse order of the forward pass, and call backward() functions

    // reset residual stream gradients (put here to work with gradient accumulation)
    floatX* dresidual = GPU_X(model->multiuse.btc); // the main buffer holding the gradient in the backward pass
    cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));

    // re-use the output buffer of the forward pass as a scratchpad during backward pass
    float*  scratchF = GPU_F32(acts.output);
    floatX* scratchX = GPU_X(acts.output);

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // this was done in the fused classifier kernel as last step of forward pass
    // technically that is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    // next: backward the classifier matmul
    matmul_backward(GPU_X(model->multiuse.bt4c), GPU_X(grads.wte), NULL, GPU_X(acts.output), GPU_X(acts.lnf), GPU_X(params.wte), NULL, B, T, C, Vp, main_stream);
    // backward the final layernorm
    floatX* residual = GPU_X(acts.residual3 + (L-1)); // last residual is in residual3
    layernorm_backward(dresidual, GPU_X(grads.lnfw), GPU_X(grads.lnfb), scratchF, GPU_X(model->multiuse.bt4c), residual, GPU_X(params.lnfw), GPU_F32(acts.lnf_mean), GPU_F32(acts.lnf_rstd), B, T, C, main_stream);

    // from this point on, we no longer need the values stored in the last residual, so we can reuse that memory as generic
    // scratch for backward computations
    floatX* dl_btc = residual;

    // now backward all the layers
    for (int l = L-1; l >= 0; l--) {
        NvtxRange layer_range("Layer", l);

        residual = GPU_X(l == 0 ? acts.encoded : acts.residual3 + (l-1));

        // get the pointers of the weights for this layer
        floatX* l_ln1w = GPU_X(params.ln1w + l);
        floatX* l_ln1b = GPU_X(params.ln1b + l);
        floatX* l_qkvw = GPU_X(params.qkvw + l);
        floatX* l_attprojw = GPU_X(params.attprojw + l);
        floatX* l_ln2w = GPU_X(params.ln2w + l);
        floatX* l_ln2b = GPU_X(params.ln2b + l);
        floatX* l_fcw = GPU_X(params.fcw + l);
        floatX* l_fcprojw = GPU_X(params.fcprojw + l);
        // get the pointers of the gradients of the weights for this layer
        floatX* dl_ln1w = GPU_X(grads.ln1w + l);
        floatX* dl_ln1b = GPU_X(grads.ln1b + l);
        floatX* dl_qkvw = GPU_X(grads.qkvw + l);
        floatX* dl_qkvb = GPU_X(grads.qkvb + l);
        floatX* dl_attprojw = GPU_X(grads.attprojw + l);
        floatX* dl_attprojb = GPU_X(grads.attprojb + l);
        floatX* dl_ln2w = GPU_X(grads.ln2w + l);
        floatX* dl_ln2b = GPU_X(grads.ln2b + l);
        floatX* dl_fcw = GPU_X(grads.fcw + l);
        floatX* dl_fcb = GPU_X(grads.fcb + l);
        floatX* dl_fcprojw = GPU_X(grads.fcprojw + l);
        floatX* dl_fcprojb = GPU_X(grads.fcprojb + l);
        // get the pointers of the activations for this layer
        floatX* l_ln1 = GPU_X((model->recompute < 2) ? acts.ln1 + l : acts.lnf);
        float* l_ln1_mean = GPU_F32(acts.ln1_mean + l);
        float* l_ln1_rstd = GPU_F32(acts.ln1_rstd + l);
        floatX* l_qkvr = GPU_X(acts.qkvr + l * B * T * 3*C);
        floatX* l_atty = GPU_X(acts.atty + l * B * T * C);
        floatX* l_residual2 = GPU_X(acts.residual2 + l);
        floatX* l_ln2 = GPU_X((model->recompute < 2) ? acts.ln2 + l : acts.lnf);
        float* l_ln2_mean = GPU_F32(acts.ln2_mean + l);
        float* l_ln2_rstd = GPU_F32(acts.ln2_rstd + l);
        floatX* l_fch_pre_gelu = GPU_X(acts.fch + l);
        floatX* l_fch_gelu = GPU_X((model->recompute < 1) ? acts.fch_gelu + l : acts.fch_gelu);
        // get the pointers of the gradients of the activations for this layer
        // notice that there is no l *, because we just have a single copy, and keep
        // re-using this memory in every Transformer block as we calculate backward pass

        floatX* dl_bt4c = (floatX*)GPU_X(model->multiuse.bt4c);

        // start the backward pass for this layer
        if(model->recompute >= 1) {
            // recompute >= 1 means we recompute gelu. in this case,
            // l_fch_gelu is just a buffer, so re-compute the gelu from l_fch here
            gelu_forward(l_fch_gelu, l_fch_pre_gelu, B*T*4*C, main_stream);
        }
        matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, scratchF, B, T, 4*C, C, main_stream, l_fch_pre_gelu, model->gelu_fusion);
        if(model->recompute >= 2) {
            // same as gelu above, l_ln1 and l_ln2 are just buffers if recompute >= 2, recompute them here on demand
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C, main_stream);
        }
        matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, scratchF, B, T, C, 4 * C, main_stream);
        // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        layernorm_backward(dresidual, dl_ln2w, dl_ln2b, scratchF, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C, main_stream);
        matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, scratchF, B, T, C, C, main_stream);

        #ifdef ENABLE_CUDNN
        float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
        attention_backward_cudnn(dl_bt4c, dl_btc, l_qkvr, l_atty, (float*)l_att, B, T, NH, C, main_stream);
        #else
        floatX* l_att = GPU_X(acts.att + l);
        // we need B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
        floatX* buffer_a = l_atty;
        floatX* buffer_b = l_fch_pre_gelu;        // this is B x T x 4C, so even larger than what we need
        attention_backward(dl_bt4c, buffer_b, scratchX, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH, main_stream);
        #endif
        if(model->recompute >= 2) {
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C, main_stream);
        }
        // QKV parameter gradients
        matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, scratchF, B, T, C, 3 * C, main_stream);
        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, scratchF, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C, main_stream);

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
                3 * C * C, 3 * C,
                C * C, C,
                C, C,
                4 * C * C, 4 * C,
                C * 4 * C, C
            };
            multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
        }
    }
    encoder_backward(GPU_X(grads.wte), GPU_X(grads.wpe), scratchX, model->workload_indices, model->bucket_info,
                     dresidual, model->inputs, inputs, B, T, C, random_u32(&model->rng_state), main_stream);

    // Aggregate all gradients that are not part of the transformer blocks
    if(last_step) {
        // reduce all the losses within the current GPU (across all microsteps)
        global_sum_deterministic(model->accumulated_mean_loss, GPU_X(acts.losses), B*T, main_stream);
        // reduce loss across GPUs to a single, final float across all microsteps and GPUs
        #if MULTI_GPU
        ncclCheck(ncclAllReduce(model->accumulated_mean_loss, model->accumulated_mean_loss, sizeof(float), ncclFloat, ncclAvg, multi_gpu_config.nccl_comm, main_stream));
        #endif
        cudaCheck(cudaMemcpyAsync(&model->mean_loss, model->accumulated_mean_loss, sizeof(float), cudaMemcpyDeviceToHost, main_stream));
        // reduce the gradients for non-transformer block parameters
        floatX* const pointers[] = {GPU_X(grads.wte), GPU_X(grads.wpe), GPU_X(grads.lnfw), GPU_X(grads.lnfb)};
        const size_t nelem[] = {Vp * C, T * C, C, C};
        multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
    }

    cudaCheck(cudaDeviceSynchronize());
    if(last_step) {
        model->mean_loss /= B*T*grad_accum_steps;
    } else {
        model->mean_loss = -1.f; // no loss available yet
    }
}

// Gets the offset of a specific tensor for a specific layer in the GPT2 model
// layer_id is ignored for weights that are not part of a transformer block
/*
ShardInfo gpt2_get_tensor_at_layer(const GPT2 *model, int layer_id, int param_tensor_id) {
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
*/

float gpt2_calculate_grad_norm(GPT2 *model, MultiGpuConfig* multi_gpu_config) {
    NVTX_RANGE_FN();
    floatX* grads_memory = (floatX*)model->grads_memory;

    // repurposing this buffer (which isn't needed now) to write grad norm into it
    float* grad_norm_squared = GPU_F32(model->acts.output);
    float grad_norm_squared_cpu = 0.0f;

    /*

    int num_slices[2] = {1, model->config.num_layers};
    int max_num_block_sums = get_max_num_block_sums(num_slices, 2);
    if (multi_gpu_config->zero_stage == 1) {
        // because of the ncclReduceScatter() in backward,
        // grads_memory only contains the averaged gradients at the local shards,
        // so we only calculate the grad norm at the grads_memory belonging to the local shards
        for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            ShardInfo tensor = gpt2_get_tensor_at_layer(model, 0, i);
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
    */

    float grad_norm_cpu = sqrtf(grad_norm_squared_cpu);
    return grad_norm_cpu;
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, float grad_scale, int t,
                 MultiGpuConfig* multi_gpu_config, bool init_from_master_only=false) {
    return; // debugging forward only for now

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
        cudaCheck(cudaMemset(model->m_memory, 0, multi_gpu_config->shard_num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->v_memory, 0, multi_gpu_config->shard_num_parameters * sizeof(float)));
    }

    // save RNG state at this point so we can round from master weights identically when restoring from a checkpoint
    model->rng_state_last_update = model->rng_state;

    // AdamW update
    // handle adamw for all the transformer blocks
    /*
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        // generate a unique seed for each tensor
        unsigned int seed = random_u32(&model->rng_state);

        int num_layers = model->config.num_layers;
        if((i < 2 || i > 13)) {
            num_layers = 1;
        }

        ShardInfo tensor = gpt2_get_tensor_at_layer(model, 0, i);
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
        float* m_ptr = model->m_memory + opt_state_offset;
        float* v_ptr = model->v_memory + opt_state_offset;
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
    */

    cudaCheck(cudaDeviceSynchronize());
}

float gpt2_estimate_mfu(GPT2 *model, int num_tokens, float dt) {
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

void gpt2_free(GPT2 *model) {
    cudaFreeCheck(&model->gpu_mem);
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

void common_free(GPT2 &model) {
    cudaCheck(cudaStreamDestroy(main_stream));
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    #ifdef ENABLE_CUDNN
    destroy_cudnn();
    #endif
}


void save_state(const char* filename, int step, GPT2* model, DataLoader* loader) {
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
    *((unsigned long long*)&state_header[22]) = model->rng_state_last_update; // last gpt2_update
    // dataloader state, start at 30 to leave some padding
    *((size_t*)&state_header[30]) = loader->current_shard_idx; // shard of the dataset
    *((size_t*)&state_header[32]) = loader->current_sample_idx; // position in shard
    fwriteCheck(state_header, sizeof(int), 256, state_file);

    // write AdamW m, v, and master_weights here (they are all float)
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    device_to_file(state_file, model->m_memory, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    device_to_file(state_file, model->v_memory, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
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

void load_state(int* step, GPT2* model, DataLoader* loader, const char* filename) {
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
    model->rng_state_last_update = *((unsigned long long*)&state_header[22]); // last gpt2_update
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
    file_to_device(model->m_memory, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    file_to_device(model->v_memory, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    if(model->use_master_weights) {
        assert(model->master_weights != nullptr);
        file_to_device(model->master_weights, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
        // restore weights from the master weights using the RNG state before last weight update
        model->rng_state = model->rng_state_last_update;
        gpt2_update(model, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, &multi_gpu_config, /* init_from_master_only*/ true);
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

void write_checkpoint(const char* output_log_dir, int step, GPT2* model, DataLoader* train_loader, MultiGpuConfig* multi_gpu_config) {
    // a checkpoint contains: model weights, optimizer/dataloader state, and a DONE file
    printf0("Writing checkpoint at step %d\n", step);
    int rank = multi_gpu_config->process_rank;
    // only rank 0 writes the model file because it is the same across all ranks
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, step);
        gpt2_write_to_checkpoint(model, filename_buffer);
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
// if we are TESTING (see test_gpt2.cu), we'll skip everything below this point

// ----------------------------------------------------------------------------
// training resumption logic, very useful when jobs crash once in a while
// the goal is that we can resume optimization from any checkpoint, bit-perfect
// note that "state" refers to things not already saved in the model checkpoint file

// ----------------------------------------------------------------------------
// CLI, poor man's argparse
// (all single letters have been claimed now)

void error_usage() {
    fprintf(stderr, "Usage:   ./train_gpt2cu [options]\n");
    fprintf(stderr, "Options:\n");
    // file system input / output
    fprintf(stderr, "  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
    fprintf(stderr, "  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
    fprintf(stderr, "  -e <string> input .bin filename or descriptor, see code comments as docs. (default = gpt2_124M_bf16.bin)\n");
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
    const char* load_filename = "gpt2_124M_bf16.bin"; // bf16 weights of the model
    const char* lr_scheduler_type = "cosine";
    const char* output_log_dir = NULL;
    int checkpoint_every = 0; // write checkpoints every how many steps?
    int checkpoints_keep = 0; // how long checkpoint history do we keep? (in units of checkpoints)
    int major_checkpoint_every = 0; // major checkpoints never get deleted when maintaining history
    int resume = 0; // resume the optimization, if one is found inside output_log_dir?
    int B = 4; // batch size
    int T = 1024; // sequence length max
    int total_batch_size = -1; // will be calculated down below later, if not provided
    float learning_rate = 3e-4f;
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
    int overfit_single_batch = 0; // useful for debugging, 1 = only load a single data batch once
    int max_steps = -1;
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
    if (gelu_fusion == -1) { gelu_fusion = 0; } // (deviceProp.major >= 9) ? 2 : 0; } // in gpt2_init_common for test_gpt2cu...
    // calculate the number of gradient accumulation steps from the desired total batch size
    assert(total_batch_size % tokens_per_fwdbwd == 0);
    int grad_accum_steps = total_batch_size / tokens_per_fwdbwd;
    // if we're only overfitting a single batch for debugging, let's overfit the first batch
    // from val instead of train split, because val is smaller and faster. (train_gpt2.py does the same)
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
    GPT2 model;
    gpt2_init_common(&model);
    model.use_master_weights = use_master_weights;
    model.gelu_fusion = gelu_fusion;
    model.recompute = recompute;
    model.batch_size = B;
    model.seq_len = T;

    if (resuming == 1) {
        // if `-y 1` was set, then we are resuming from the latest checkpoint
        // if we are using master weights, we'll init them later inside load_state()
        bool weight_init = !use_master_weights;
        gpt2_build_from_checkpoint(&model, filename_buffer, weight_init);
    } else if (ends_with_bin(load_filename)) {
        // otherwise, if this is a .bin file, we assume it's a model, let's init from it
        gpt2_build_from_checkpoint(&model, load_filename);
    } else {
        // if it's not .bin, it could be a "special descriptor". This descriptor is used to
        // construct GPT-2 / GPT-3 models in a convenient format. See the function for docs.
        gpt_build_from_descriptor(&model, load_filename);
    }

    printf0("| weight init method    | %-50s |\n", resuming == 1 ? "intermediate checkpoint" : load_filename);
    printf0("| max_sequence_length T | %-50d |\n", model.config.max_seq_len);
    printf0("| vocab_size V          | %-50d |\n", model.config.vocab_size);
    printf0("| padded_vocab_size Vp  | %-50d |\n", model.config.padded_vocab_size);
    printf0("| num_layers L          | %-50d |\n", model.config.num_layers);
    printf0("| num_heads NH          | %-50d |\n", model.config.num_heads);
    printf0("| channels C            | %-50d |\n", model.config.channels);
    printf0("| num_parameters        | %-50zu |\n", model.num_parameters);
    printf0("+-----------------------+----------------------------------------------------+\n");

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
    // more prints related to allocations from gpt2_build_from_checkpoint down here to not mess up our table above
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
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

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
    if (resuming == 1) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, resume_max_step, multi_gpu_config.process_rank);
        load_state(&step, &model, &train_loader, filename_buffer);
    }

    // init an OutlierDetector the training loss
    OutlierDetector loss_outlier_detector, grad_norm_outlier_detector;
    init_detector(&loss_outlier_detector);
    init_detector(&grad_norm_outlier_detector);

    // do some checks here before we kick off training
    // cross-check the desired sequence length T with the model's max sequence length
    if (T < model.config.max_seq_len) {
        printf0("!!!!!!!!\n");
        printf0("WARNING:\n");
        printf0("- The training sequence length is: T=%d (set with -t)\n", T);
        printf0("- The model's max sequence length is: max_seq_len=%d\n", model.config.max_seq_len);
        printf0("You are attempting to train with a sequence length shorter than the model's max.\n");
        printf0("This will lead to unused parameters in the wpe position embedding weights.\n");
        printf0("If you know what you're doing you can ignore this warning.\n");
        printf0("If you're like ???, you are most likely misconfiguring your training run.\n");
        printf0("---> HINT: If you're training GPT-2 use -t 1024. If GPT-3, use -t 2048.\n");
        printf0("!!!!!!!!\n");
    }
    // in any case, this must be true or we'd index beyond the model's wpe (position embedding table)
    assert(T <= model.config.max_seq_len);

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

        // once in a while estimate the validation loss (all processes collaborate)
        if (step % val_loss_every == 0 || last_step) {
            NvtxRange validation_range("validation");
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                val_loss += gpt2_validate(&model, val_loader.inputs, val_loader.targets, B, T);
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
                gpt2_validate(&model, eval_loader.inputs, eval_loader.targets, B, T);
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
                gpt2_forward(&model, gen_tokens, 1, CEIL_DIV(t, min(T,256)) * min(T,256));
                // get the V-dimensional vector probs[0, t-1, :]
                floatX* logits = (floatX*)(model.gpu_mem + tensors_start[model.acts.output] + (t - 1) * model.config.padded_vocab_size);
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
            gpt2_forward(&model, train_loader.inputs, B, T);
            // backward pass. all model params accumulate gradients with += inside this inner loop
            gpt2_backward_and_reduce(&model, train_loader.inputs, train_loader.targets, grad_accum_steps, micro_step);
        }
        float zloss = (float)(update_detector(&loss_outlier_detector, (double)model.mean_loss)); // loss z-score
        // fetch the next learning rate
        float step_learning_rate = get_learning_rate(&lr_scheduler, step);
        // calculate the gradient norm and how much we wish to scale the gradient
        float grad_norm = gpt2_calculate_grad_norm(&model, &multi_gpu_config);
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
            gpt2_update(&model, step_learning_rate, 0.9f, 0.95f, 1e-8f, weight_decay, grad_scale, step+1, &multi_gpu_config);
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
        float mfu = gpt2_estimate_mfu(&model, B * T * grad_accum_steps, time_elapsed_ms / 1000.0f);
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
    gpt2_free(&model);
    common_free(model);
    return 0;
}
#endif
