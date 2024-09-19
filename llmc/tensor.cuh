#ifndef TENSOR_CUH
#define TENSOR_CUH

// ...
//#define FAKE_FP8
#define UNIQUE_TENSOR_MEMORY false
#define LAYERS_PER_ACTIVATION_CHECKPOINT 0 // 0 = disabled
// ...

#include "cuda_common.h"
#include "cuda_utils.cuh"
#include <assert.h>

// ----------------------------------------------------------------------------

enum TT : uint8_t {
    PARAMETER=0, PARAMETER_GRAD, PARAMETER_OPT_M, PARAMETER_OPT_V, PARAMETER_MASTER, // 1 allocation each
    MULTIUSE, // single allocation shared for activations, activation gradients, and scratch
    DEFAULT, COUNT=DEFAULT, NUM_TYPES_PARAM=PARAMETER_MASTER+1
};

enum TFlags : uint8_t {
    NONE=0,
    GRADIENT=1,
    REUSED_MEMORY=2,
    TENSOR_2D=4, // used for matmul weights and activation outputs only (not inputs or gradients)
    BIAS=8,
    LAYERNORM=16,
    RESIDUAL=32,
    EMBEDDING=64,
    STATS=128
};

// ----------------------------------------------------------------------------
// forward declarations & extern variables defined in the training file
struct TensorSpec;
constexpr size_t MAX_TENSORS = 32768; // only increases CPU memory usage if unused
constexpr size_t MAX_ABSMAX_HISTORY = 32; // todo - command line option

extern TensorSpec tensor_specs[MAX_TENSORS];
extern TensorSpec* tensor_specs_gpu;
extern size_t tensors_start[TT::COUNT];
extern size_t tensors_bytes[TT::COUNT];
extern size_t tensors_elements[TT::COUNT];
extern int num_tensor_specs;

extern TT current_tensor_type; // todo - avoid having this somehow?
extern int absmax_history_index; // todo - move into model struct?
extern float* gpu_scale_memory;
extern unsigned int* gpu_absmax_memory;
// end element of each tensor to optimise iterating through them in kernels
extern size_t* gpu_tensor_end_element;

__device__ __constant__ TensorSpec* tensor_specs_ptr;
__device__ __constant__ float* gpu_scale_memory_ptr;
__device__ __constant__ unsigned int* gpu_absmax_memory_ptr;
__device__ __constant__ size_t* tensor_end_element_ptr;

// ----------------------------------------------------------------------------
// Helper macros for accessing tensors in the training loop
#define TENSOR(x,layer)  get_tensor(x, DEFAULT, layer)
#define ACT_L(x,layer)   get_tensor(model->acts.x, MULTIUSE, layer)
#define MULTI_L(x,layer) get_tensor(model->multiuse.x, MULTIUSE, layer)
#define AGRAD_L(x,layer) get_tensor(model->acts_grads.x, MULTIUSE, layer)
#define PARAM_L(x,layer) get_tensor(model->params[PARAMETER].x, PARAMETER, layer)
#define PGRAD_L(x,layer) get_tensor(model->params[PARAMETER_GRAD].x, PARAMETER_GRAD, layer)
#define ACT(x)     ACT_L(x,l)
#define MULTI(x)   MULTI_L(x,l)
#define AGRAD(x)   AGRAD_L(x,l)
#define PARAM(x)   PARAM_L(x,l)
#define PGRAD(x)   PGRAD_L(x,l)
#define ACT_0(x)   ACT_L(x,0)
#define MULTI_0(x) MULTI_L(x,0)

// ----------------------------------------------------------------------------

template<typename ElementType=float>
struct TensorGPU {
    int id = -1; // TensorSpec index in tensor_specs[] array
    ElementType* data_ptr = NULL;
    float* scale_descale_ptr = NULL;
    unsigned int* absmax_ptr = NULL;
    size_t num_elements = 0;

    static __device__ __host__ TensorGPU from(ElementType* ptr=nullptr) {
        TensorGPU tmp;
        tmp.data_ptr = ptr;
        return tmp;
    }
    template<typename T>
    __device__ __host__ T* as() {
        return reinterpret_cast<T*>(data_ptr);
    }
    __device__ __host__  operator ElementType*() const {
        return data_ptr;
    }
    __device__ __host__ ElementType& operator[](size_t index) {
        return data_ptr[index];
    }
    __device__ __host__ const ElementType& operator[](size_t index) const {
        return data_ptr[index];
    }
    __device__ __host__ int num_per_128() const {
        return sizeof(int4) / sizeof(ElementType);
    }
    __device__ __host__ bool is_null() const {
        return (data_ptr == NULL);
    }
    __device__ __host__ bool enabled() const {
        return (data_ptr != NULL);
    }

    static constexpr bool no_scaling = (sizeof(ElementType) != 1); // todo - this prevents scaling FP16

    __device__ __host__ float get_scalar(size_t index, bool disable_scaling=no_scaling) const {
        #ifdef FAKE_FP8
        disable_scaling = true;
        #endif
        ElementType* __restrict__ data_ptr_restricted = data_ptr;
        float* __restrict__ scale_ptr_restricted = scale_descale_ptr;

        float value = (float)data_ptr_restricted[index];
        float descale = (scale_descale_ptr && !disable_scaling) ? scale_ptr_restricted[1] : 1.0f;
        return value * descale; // [1] = descale
    }

    __device__ __host__ ElementType set_scalar(size_t index, float value, bool disable_scaling=no_scaling) {
        #ifdef FAKE_FP8
        disable_scaling = true;
        #endif
        ElementType* __restrict__ data_ptr_restricted = data_ptr;
        float* __restrict__ scale_ptr_restricted = scale_descale_ptr;

        float scale = (scale_descale_ptr && !disable_scaling) ? scale_ptr_restricted[0] : 1.0f;
        ElementType output = (ElementType)(value * scale);
        data_ptr_restricted[index] = output;
        return output;
    }
};

typedef TensorGPU<floatX> tensorX;
typedef TensorGPU<float> tensor32;
typedef TensorGPU<half> tensorFP16;
typedef TensorGPU<nv_bfloat16> tensorBF16;
#ifdef ENABLE_FP8
typedef TensorGPU<__nv_fp8_e4m3> tensor8;
typedef TensorGPU<__nv_fp8_e5m2> tensor8e5;
#else
typedef TensorGPU<floatX> tensor8;
typedef TensorGPU<floatX> tensor8e5;
#endif
extern TensorGPU<floatX> null_tensorX;

// ----------------------------------------------------------------------------

// this is the "foundation" of the other tensor classes (TensorGPU and tensor128)
// they all implicitly refer to this (in tensor_specs[] and tensor_specs_gpu[] for now) with the id
// and these other classes are created by converting from this one (sometimes implicitly)
struct TensorSpec {
    int id;
    char* ptr; // = model->tensor_memory[tensor_type] + offset
    char name[16];
    TT tensor_type;
    DType data_type;
    int flags;

    size_t offset; // into tensor type's base pointer
    size_t start_element; // on this shard
    size_t num_elements; // per shard
    short num_shards;
    short remaining_layers;

    template <typename T>
    __host__ __device__ operator T*() const {
        // todo - sanity check DType matches T
        return reinterpret_cast<T*>(ptr);
    }

    template <typename T>
    __device__ __host__ operator TensorGPU<T>() const {
        TensorGPU<T> tensor;
        tensor.num_elements = num_elements;
        tensor.data_ptr = this->operator T*();
        tensor.id = id;

        #ifdef __CUDA_ARCH__
        tensor.scale_descale_ptr = gpu_scale_memory_ptr + 2*id;
        tensor.absmax_ptr = gpu_absmax_memory_ptr + id;
        #else
        tensor.scale_descale_ptr = gpu_scale_memory + 2*id;
        tensor.absmax_ptr = gpu_absmax_memory + id;
        #endif

        return tensor;
    }
};

// ----------------------------------------------------------------------------

// debug helper function (enable in get_tensor() for extreme logging)
void print_tensor_elements(int tensor_id) {
    TensorSpec spec = tensor_specs[tensor_id];
    size_t num_elements = spec.num_elements;
    const char* tensor_name = spec.name;
    TT tensor_type = spec.tensor_type;
    DType dtype = spec.data_type;
    size_t element_size = sizeof_dtype(dtype);

    void* gpu_tensor = spec.ptr;
    void* cpu_tensor = malloc(num_elements * element_size);

    // Get scale from GPU
    float scale, descale, absmax;
    cudaMemcpy(&scale, &gpu_scale_memory[spec.id * 2], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&descale, &gpu_scale_memory[spec.id * 2 + 1], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&absmax, &gpu_absmax_memory[spec.id], sizeof(float), cudaMemcpyDeviceToHost);

    printf("Printing tensor %s (tensor_type: %d, data_type: %d, flags: %d)\n", tensor_name, (int)tensor_type, (int)dtype, spec.flags);
    printf("GPU memory: %p\n", gpu_tensor);
    printf("CPU memory: %p\n", cpu_tensor);
    printf("Num elements: %zu\n", num_elements);
    printf("Element size: %zu\n", element_size);
    printf("Offset: %zu\n", spec.offset);
    printf("Scale: %f, Descale: %f, Absmax: %f\n", scale, descale, absmax);

    cudaCheck(cudaMemcpy(cpu_tensor, gpu_tensor, num_elements * element_size, cudaMemcpyDeviceToHost));

    printf("First 4 & Last 4 of %s:\n", tensor_name);
    for (int i = 0; i < 8; i++) {
        int idx = (i < 4) ? i : num_elements - 8 + i;
        switch (dtype) {
            case DType::FP32: printf("%.16f ", ((float*)cpu_tensor)[idx]); break;
            case DType::FP16: printf("%.16f ", (float)((__nv_half*)cpu_tensor)[idx]); break;
            case DType::BF16: printf("%.16f ", (float)((__nv_bfloat16*)cpu_tensor)[idx]); break;
            case DType::FP8E4M3: printf("%.16f ", (float)((__nv_fp8_e4m3*)cpu_tensor)[idx]); break;
            case DType::FP8E5M2: printf("%.16f ", (float)((__nv_fp8_e5m2*)cpu_tensor)[idx]); break;
        }
        if (i == 3) printf("\n");
    }
    printf("\n\n");

    free(cpu_tensor);
}

// ----------------------------------------------------------------------------

TensorSpec get_tensor(int spec_index, TT tensor_type, int layer) {
    TensorSpec spec = tensor_specs[spec_index];
    if (layer > 0 && spec.remaining_layers >= layer) {
        spec = tensor_specs[spec_index + layer];
    } else if (layer > 0 && spec.remaining_layers > 0) {
        printf("ERROR: get_tensor() for %s layer %d but only %d layers remaining\n", spec.name, layer, spec.remaining_layers);
        assert(false);
    }
    assert(spec.tensor_type == tensor_type || tensor_type == DEFAULT);
    //print_tensor_elements(spec.id); // enable for extreme debugging
    return spec;
}

// this can only be called at initialisation time, once tensor_specs has been uploaded to the GPU, it is fixed in stone
int add_tensor_spec(const char* name, size_t total_elements, size_t num_shards, DType data_type, int copy_offset_from=-1, int flags=TFlags::NONE, TT tensor_type=TT::DEFAULT) {
    assert(num_tensor_specs < MAX_TENSORS);
    assert((total_elements % num_shards) == 0);
    TensorSpec* spec = &tensor_specs[num_tensor_specs];

    spec->id = num_tensor_specs;
    strncpy(spec->name, name, 15);
    spec->name[15] = 0;
    spec->tensor_type = (tensor_type == TT::DEFAULT) ? current_tensor_type : tensor_type;
    spec->data_type = data_type;
    spec->flags = flags;

    // parameter tensors must fit in a 32-bit unsigned integer (used as an optimisation in e.g. global_norm_tensors_loop)
    // todo - either 1) 32-bit everywhere (with a DEFINE?), 2) 64-bit everywhere despite the small performance impact, 3) ?
    assert(total_elements < 4UL*1024*1024*1024 || spec->tensor_type == TT::MULTIUSE);

    spec->start_element = tensors_elements[spec->tensor_type];
    spec->num_elements = total_elements / num_shards;
    spec->num_shards = num_shards;
    spec->remaining_layers = 0;

    if (copy_offset_from >= 0) {
        TensorSpec base_spec = tensor_specs[copy_offset_from];
        base_spec.flags |= (flags & REUSED_MEMORY);
        spec->offset = base_spec.offset;

        size_t original_tensor_bytes = base_spec.num_elements * sizeof_dtype(base_spec.data_type);
        size_t new_tensor_bytes = spec->num_elements * sizeof_dtype(data_type);
        assert(new_tensor_bytes <= original_tensor_bytes);
        assert(spec->tensor_type == base_spec.tensor_type);
    } else {
        spec->offset = tensors_bytes[spec->tensor_type];
        tensors_bytes[spec->tensor_type] += spec->num_elements * sizeof_dtype(data_type);
        if (tensors_start[spec->tensor_type] == 0 && spec->tensor_type != 0) {
            tensors_start[spec->tensor_type] = num_tensor_specs;
        }
    }

    tensors_elements[spec->tensor_type] += spec->num_elements;
    return num_tensor_specs++;
}

int add_layer_specs(int num_layers, const char* name, size_t total_elements, size_t num_shards, DType data_type,
                    int copy_offset_from=-1, int flags=TFlags::NONE, int reuse_every_n_layers=0,
                    TT tensor_type=TT::DEFAULT) {
    int first_tensor_id = num_tensor_specs;
    if (reuse_every_n_layers > 0 && num_layers > 1) {
        flags |= REUSED_MEMORY;
    }
    for (int l = 0; l < num_layers; l++) {
        char layer_name[16];
        assert(snprintf(layer_name, 15, "%s_%d", name, l) >= 0);
        if (reuse_every_n_layers > 0 && l >= reuse_every_n_layers) {
            copy_offset_from = first_tensor_id + (l % reuse_every_n_layers);
        }
        int spec = add_tensor_spec(num_layers > 1 ? layer_name : name, total_elements, num_shards, data_type, copy_offset_from, flags, tensor_type);
        tensor_specs[spec].remaining_layers = num_layers - (l + 1);
    }
    return first_tensor_id;
}

// ----------------------------------------------------------------------------

// the 1st num_tensor_specs values are the absmax of the current/last step
// the next [MAX_ABSMAX_HISTORY * num_tensor_specs] values are the history from previous steps
__global__ void update_scale_descale_kernel(int num_tensor_specs, int absmax_history_index) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tensor_specs) return;

    // copy current absmax to history then clear it
    gpu_absmax_memory_ptr[tid + (absmax_history_index * num_tensor_specs)] = gpu_absmax_memory_ptr[tid];
    gpu_absmax_memory_ptr[tid] = 0;
    float absmax = 0.0f;

    // get the maximum absmax from the history (todo - do we want to mitigate outliers here?)
    #pragma unroll
    for (int i = 1; i <= MAX_ABSMAX_HISTORY; i++) {
        absmax = max(absmax, __uint_as_float(gpu_absmax_memory_ptr[tid + (i * num_tensor_specs)]));
    }

    // calculate scale based on the maximum absmax
    float scale = (absmax != 0.0f) ? (1.0f / absmax) : 1.0f;

    // FP8 e4m3 vs e5m2 (the latter is currently only used for activation gradients)
    bool use_e5m2 = (tensor_specs_ptr[tid].data_type == DType::FP8E5M2);
    #ifdef FAKE_FP8
    if (tensor_specs_ptr[tid].flags & TFlags::GRADIENT && tensor_specs_ptr[tid].tensor_type == TT::MULTIUSE) {
        use_e5m2 = true;
    }
    #endif

    if  (use_e5m2) {
        if (absmax != 0.0f) {
            scale *= 32768.0f;
        } else {
            // hacky default to avoid extreme gradient underflow on the 1st step
            scale = 4096.0f;
        }
    } else if (tensor_specs_ptr[tid].data_type == DType::FP8E4M3) {
        // todo - power benefit of making sure top bit of exponent is (nearly always) zero?
        // this can be done simply by *not* multiplying here, so that the "maximum" is 1.0f
        // we probably want some threshold for badly behaved parameters to use the full range
        //if (tensor_specs_ptr[tid].tensor_type != TT::PARAMETER || absmax >= 4.0f) {
        if (absmax != 0.0f) {
            scale *= 256.0f;
        }
    }

    // update scale and descale memory
    // descale must be delayed by one step for parameters (see comment in gpt2_update).
    gpu_scale_memory_ptr[tid * 2] = scale;

    if (tensor_specs_ptr[tid].tensor_type == TT::PARAMETER) {
        float old_scale = gpu_scale_memory_ptr[tid * 2];
        gpu_scale_memory_ptr[tid * 2 + 1] = 1.0f / old_scale;
    } else {
        gpu_scale_memory_ptr[tid * 2 + 1] = 1.0f / scale;
    }
}

void update_scales_from_absmax() {
    int block_size = 256;
    int num_blocks = CEIL_DIV(num_tensor_specs, block_size);

    update_scale_descale_kernel<<<num_blocks, block_size>>>(num_tensor_specs, absmax_history_index + 1);
    absmax_history_index = (absmax_history_index + 1) % MAX_ABSMAX_HISTORY;
}

// ----------------------------------------------------------------------------

template<typename ElementType=float>
struct tensor128 {
private:
    Packed128<ElementType> data128;
    ElementType* data_ptr;
    unsigned int *absmax_ptr = nullptr;
    float scale = 1.0f;
    float descale = 1.0f;
    float new_absmax = 0.0f;
    bool wrote_data = false;
    bool wrote_absmax = false;
    int id = -1;

    // fake fp8 mode (ignored without FAKE_FP8 define)
    bool faking_fp8 = false;
    bool mode_e5 = false;

public:
    bool scaling = (sizeof(ElementType) == 1);
    static constexpr const size_t elements = sizeof(int4) / sizeof(ElementType);
    __device__ tensor128() { scaling = false; }

    __device__ tensor128(TensorGPU<ElementType> tensor, bool disable_scaling=false) {
        data_ptr = tensor.data_ptr;
        id = tensor.id;

#ifdef FAKE_FP8
        // fake FP8 only applies to specific tensors to test expected training performance
        // todo - expand this to support more unusual formats and test things like blockwise scaling(?)
        if (!disable_scaling && id >= 0 && sizeof(ElementType) == 2 && tensor_specs_ptr[id].tensor_type != TT::PARAMETER_GRAD) {
            if ((tensor_specs_ptr[id].flags & (TFlags::RESIDUAL | TFlags::EMBEDDING | TFlags::BIAS)) == 0) {
                faking_fp8 = true;
                if  ((tensor_specs_ptr[id].flags & TFlags::GRADIENT) && (tensor_specs_ptr[id].tensor_type == TT::MULTIUSE)) {
                    mode_e5 = true;
                }
            }
        }
        scaling = false; // only do "fake" scaling
#endif

        scaling = scaling && !disable_scaling;
        if (scaling) {
            // using __restrict__ here should allow the compiler to cache/reuse this in loops etc.
            const float* __restrict__ ptr_restricted = tensor.scale_descale_ptr;
            scale = ptr_restricted[0];
            descale = ptr_restricted[1];
        }
        absmax_ptr = tensor.absmax_ptr;
    }

    __device__ void load(size_t offset, bool cache_streaming=false) {
        ElementType* addr = data_ptr + offset;
        data128 = cache_streaming ? load128cs(addr) : load128(addr);
    }

    __device__ void store(size_t offset, bool cache_streaming=false) {
        if (cache_streaming) store128cs(data_ptr + offset, data128);
        else store128(data_ptr + offset, data128);
        wrote_data = true;
    }

    template <typename OriginalType>
    __device__ void store_same_length(size_t offset, bool cache_streaming=false) {
        if (cache_streaming) store128_same_length_cs<OriginalType, ElementType>(data_ptr + offset, data128);
        else store128_same_length<OriginalType, ElementType>(data_ptr + offset, data128);
        wrote_data = true;
    }

    __device__ const Packed128<ElementType>& get128() const { return data128; }
    __device__ Packed128<ElementType>& get128() { return data128; }

    // call this manually if e.g. you use set_scalar() to update the tensor
    // todo - in the future, this could support more than just absmax
    __device__ void add_value_stats(float value, ElementType output=(ElementType)0.0f) {
        new_absmax = max(new_absmax, fabsf(value));
    }

    // get and set automatically apply scaling/descaling for FP8 values
    __device__ float get(int index) {
        float value = (float)data128[index] * (scaling ? descale : 1.0f);
        value = fake_fp8(faking_fp8, value, scale, descale, mode_e5); // ignored without FAKE_FP8
        return value;
    }

    __device__ void set(int index, float value) {
        float output = value * (scaling ? scale : 1.0f);
        output = fake_fp8(faking_fp8, output, scale, descale, mode_e5);
        data128[index] = (ElementType)(output);
        add_value_stats(value, data128[index]);
    }

    __device__ void set_stochastic(int index, float value, unsigned int random_number,
                                   int rotate_by_index=10, bool non_deterministic_rng=false) {
        float scaled_value = value * (scaling ? scale : 1.0f);

        // rotate the random number by the index so we can cheaply reuse the same RNG
        // obviously less good than having true per-index RNG, but should be good enough
        // when rounding FP32 to FP8, most of the bits make extremely little difference anyway...
        // x10 is used so that it never repeats for indices [0;15] with a minimum difference of 2 etc.
        if (rotate_by_index) {
            assert(index < 16); // >=16 would repeat and be extremely bad RNG
            random_number = __funnelshift_l(random_number, random_number, index * rotate_by_index);
        }

        // RNG without a seed from the host for quick testing, but obviously not deterministic
        // can be forced to get slightly different runs from which you can calculate an average
        #ifdef FORCE_NON_DETERMINISM
        non_deterministic_rng = true;
        #endif
        if (non_deterministic_rng) {
            unsigned int clock, laneid;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(clock));
            asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
            random_number = get_random_noise(clock, laneid, blockIdx.x * blockDim.x);
        }

        stochastic_rounding(scaled_value, data128[index], random_number);
        add_value_stats(value, data128[index]);
    }

    // return value: if true, we can skip __syncthreads() in the calling function as we have just done one
    __device__ bool update_absmax(int thread_id, int num_threads, bool exit=false, bool forced=false) {
        #ifdef FAKE_FP8
        if (absmax_ptr == NULL || !faking_fp8) {
            return false;
        }
        forced = true;
        #endif

        if (!forced && !scaling) {
            return false;
        }
        wrote_absmax = true;

        // lane_id must be obtained directly from the special register
        // otherwise, the compiler does silly things related to the redux/atomicMax
        unsigned int lane_id ;
        asm volatile("mov.u32 %0, %laneid;" : "=r"(lane_id));
        unsigned int num_warps = num_threads / WARP_SIZE;
        unsigned int warp_id = thread_id / WARP_SIZE;

        // use native integer reductions as much as possible (supported on all GPUs with FP8)
        // this might treat NaN/INF slightly differently but that is the least of our problems
        __shared__ unsigned int shared[32];
        unsigned int absmax_uint = *(unsigned int*)&new_absmax;
        asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));

        // with this condition instead of lane_id == 0, we have shared[lane_id] both here and below
        // this reduces the number of instructions for addressing
        if (lane_id == warp_id) {
            shared[lane_id] = absmax_uint;
        }

        // sync can be after exit (dead threads don't count) but must be before return
        // if this is the end of the kernel, the compiler puts a conditional EXIT right after BAR
        // but this way the EXIT is right before the barrier which frees the warps slightly quicker
        bool done = (warp_id != 0);
        if (done && exit) asm volatile("exit;"); // todo - does this help enough to be worth it?
        __syncthreads();
        if (done && !exit) return true;

        // one more warp reduction then global memory atomic
        // we want as few global atomics as possible (i.e. 1 per threadblock)
        absmax_uint = shared[lane_id];
        if (lane_id >= num_warps) {
            absmax_uint = 0;
        }

        asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
        if (lane_id == 0) {
            atomicMax(absmax_ptr, absmax_uint);
        }
        return true;
    }

    // helper function to avoid having to specify threadIdx/blockDim manually
    __device__ bool update_absmax(int block_dimensions, bool exit=false) {
        if (block_dimensions == 1) {
            return update_absmax(threadIdx.x, blockDim.x, exit);
        } else if (block_dimensions == 2) {
            return update_absmax(threadIdx.x + threadIdx.y * blockDim.x, blockDim.x * blockDim.y, exit);
        } else if (block_dimensions == 3) {
            return update_absmax(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y,
                                 blockDim.x * blockDim.y * blockDim.z, exit);
        }
        assert(false);
        return false;
    }

    __device__ void skip_absmax() {
        wrote_absmax = true;
    }

    __device__ ~tensor128() {
        // this should ~always be optimised away by the compiler
        if (!wrote_absmax && scaling && wrote_data) {
            //printf("id: %d\n", id);
            assert(false);
        }
    }
};

template <bool init=true, typename T>
__device__ tensor128<T> new_tensor128(TensorGPU<T> tensor, bool disable_scaling=false) {
    if constexpr (init) {
        return tensor128<T>(tensor, disable_scaling);
    } else {
        return tensor128<T>();
    }
}

template <typename T>
__device__ tensor128<T> load_tensor128(TensorGPU<T> tensor, size_t offset,
                                       bool cache_streaming = false, bool disable_scaling=false) {
    tensor128<T> t128(tensor, disable_scaling);
    t128.load(offset, cache_streaming);
    return t128;
}

#endif // TENSOR_CUH
