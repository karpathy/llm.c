/*
LayerNorm CUDA kernel, and also Residual, because sometimes they are fused

Note in llm.c we try to be clever in the backward pass to conserve memory.
All parameters use a += in the backward pass, so we can do gradient accumulation.
But all activations have = instead of += because these are faster (just read, no write).
This is okay for all activations except for those in the residual stream, where the
gradients have to add. We make sure that we do a += as necessary.
E.g., the layernorms are connected to the residuals so we += in layernorm backward.
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// CUDA kernels

void layernorm_forward_kernel3(floatX* __restrict__ out, floatX* __restrict__ mean, floatX* __restrict__ rstd,
                                    const floatX*  __restrict__ inp, const floatX*  __restrict__ weight,
                                    const floatX* __restrict__ bias, int N, int C,
                                    const sycl::nd_item<3> &item_ct1) {
    int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
    int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    int num_warps = item_ct1.get_local_range(2) / WARP_SIZE;

    int idx = item_ct1.get_group(2) * num_warps + warp_id;
    if(idx >= N) { return; } // guard

    // the row of input that this group of threads is responsible for
    const floatX* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        sum += (float)x[i];
    }
    sum = warpReduceSum(sum, item_ct1);
    float m = sum / C;
    if(lane_id == 0 && mean != nullptr) {
        
        
        *(mean + idx) = (floatX)m;
    }

    // rstd
    sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        float diff = (float)x[i] - m;
        sum += diff * diff;
    }
    sum = warpReduceSum(sum, item_ct1);
    float s = sycl::rsqrt(sum / C + 1e-5f);
    if(lane_id == 0 && rstd != nullptr) {
        
        *(rstd + idx) = (floatX)s;
    }

    // final normalization and scaling by weight/bias
    floatX* o = out + idx * C;
    for (int c = lane_id; c < C; c += WARP_SIZE) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        
        float n = s * ((float)*(x + c) - m);
        
        *(o + c) = (floatX)(n * (float)weight[c] + (float)bias[c]);
    }
}


SYCL_EXTERNAL void fused_residual_forward_kernel5(
    floatX *residual, floatX *normed, floatX *mean, floatX *rstd,
    const floatX *inp1, const floatX *inp2, const floatX *weight,
    const floatX *bias, int N, int C, const sycl::nd_item<3> &item_ct1,
    uint8_t *dpct_local) {
    assert(0);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    auto params = (char **)dpct_local;
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128 *s_res = reinterpret_cast<x128 *>(params) +
                  ((2 + item_ct1.get_local_id(1)) * C / x128::size);

    int sidx =
        (item_ct1.get_local_id(2) + WARP_SIZE * item_ct1.get_local_id(1)) *
        x128::size;
    for (int i = sidx; i < C;
         i += item_ct1.get_local_range(1) * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    
    item_ct1.barrier(sycl::access::fence_space::local_space);

    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
              item_ct1.get_local_id(1);
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    const float eps = 1e-5f;
    float sum = 0.0f;
    for (int c = item_ct1.get_local_id(2) * x128::size; c < C;
         c += WARP_SIZE * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (float)in1[k] + (float)in2[k];
            sum += (float)out[k];
        }
        store128cs(residual + c, out);
        s_res[c / x128::size] = out;
    }

    sum = warpReduceSum(sum, item_ct1);
    float m = sum / C;
    float v = 0.f;

    for (int c = item_ct1.get_local_id(2) * x128::size; c < C;
         c += WARP_SIZE * x128::size) {
        const x128 res = s_res[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)res[k] - m) * ((float)res[k] - m);
        }
    }

    v = warpReduceSum(v, item_ct1) / C;
    float s = sycl::rsqrt(v + eps);

    for (int c = item_ct1.get_local_id(2) * x128::size; c < C;
         c += WARP_SIZE * x128::size) {
        const x128 res = s_res[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        const x128 b = s_bias[c / x128::size];
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out[k] = o;
        }

        store128cs(normed + c, out);
    }
    // cache the mean and rstd for the backward pass later
    if (item_ct1.get_local_id(2) == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}

void residual_forward_kernel(floatX* out, const floatX* inp1, const floatX* inp2,
                             const sycl::nd_item<3> &item_ct1) {
    int idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2)) *
              x128::size;

    x128 packed_out;
    x128 packed_inp1 = load128cs(inp1 + idx);
    x128 packed_inp2 = load128cs(inp2 + idx);
    for (int k = 0; k < packed_inp1.size; k++) {
        packed_out[k] = (floatX)((float)packed_inp1[k] + (float)packed_inp2[k]);
    }
    store128(out + idx, packed_out);
}


SYCL_EXTERNAL void // todo - any warnings on Turing with only 1024 threads?
layernorm_backward_kernel10(floatX *dinp, floatX *dweight, floatX *dbias,
                            float *scratch, const floatX *dout,
                            const floatX *inp, const floatX *weight,
                            const floatX *mean, const floatX *rstd, int B,
                            int T, int C, const sycl::nd_item<3> &item_ct1,
                            uint8_t *dpct_local) {
    int BLOCK_SIZE = item_ct1.get_local_range(2);
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE; //number of warps in block
    auto shared = (float *)dpct_local;

    int warpId =
        item_ct1.get_local_id(2) / WARP_SIZE; // warp index within a block
    int baseIdx = item_ct1.get_group(2) * warpsInBlock + warpId;
    int warpThreadIdx =
        item_ct1.get_local_id(2) % WARP_SIZE; // Thread index within the warp
    int warpsInGrid = item_ct1.get_group_range(2) * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C = CEIL_DIV(C, C_per_iteration); // + 2;

    // the first half of shared memory is bias, second is weight
    size_t rounded_C = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    float* dbias_shared = shared;
    float* dweight_shared = shared + rounded_C;
    // warp zero doesn't actually write to the _tmp_shared memory locations, so we don't need to reserve memory
    // the obvious solution is to change the addressing below to use (threadId.x-32) as offset, but that causes
    // register spills, so instead we mess with the base pointer here, which doesn't increase register usage.
    float* dbias_tmp_shared = shared + 2 * rounded_C - WARP_SIZE * f128::size;
    float* dweight_tmp_shared = shared + 2 * rounded_C + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;

    // init shared memory to zero
    for (int i = item_ct1.get_local_id(2) * f128::size; i < rounded_C;
         i += BLOCK_SIZE * f128::size) {
        store128(dbias_shared + i, f128::zeros());
        store128(dweight_shared + i, f128::zeros());
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int bt = baseIdx; bt < B * T; bt += warpsInGrid) {
        const floatX* dout_bt = dout + bt * C;
        const floatX* inp_bt = inp +bt * C;
        floatX* dinp_bt = dinp + bt * C;

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt  + i);
            x128 weight128_i = load128(weight  + i);
            for (int k = 0; k < x128::size; k++) {
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * (float)inp128_i[k];
            }
        }

        const float mean_bt = (float)mean[bt];
        const float rstd_bt = (float)rstd[bt];
        dnorm_mean = warpReduceSum(dnorm_mean, item_ct1) / C;
        dnorm_norm_mean =
            warpReduceSum(dnorm_norm_mean, item_ct1) / C * rstd_bt -
            dnorm_mean * mean_bt * rstd_bt;

        for (int c = 0; c < iterations_C; c++) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);

            x128 dout128   = x128::zeros();
            x128 inp128    = x128::zeros();
            x128 dinp128   = x128::zeros();
            x128 weight128 = x128::zeros();

            if(global_index < C) {
                dout128 = load128cs(dout_bt + global_index);
                inp128 = load128cs(inp_bt + global_index);
                dinp128 = load128(dinp_bt + global_index);
                weight128 = load128(weight + global_index);
            }

            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 dbias_f;
                f128 dweight_f;
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    float dout_i = (float)dout128[x];
                    float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                    dbias_f[i] = dout_i;
                    dweight_f[i] = norm_bti * dout_i;

                    float dval = 0.0f;
                    dval += (float) weight128[x] * (float)dout128[x]; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    dinp128[x] = (floatX) ((float) dinp128[x] + dval);
                }

                if (warpId != 0) {
                    store128(dbias_tmp_shared +
                                 item_ct1.get_local_id(2) * f128::size,
                             dbias_f);
                    // this seems to generate a 64-bit store, instead of 128-bit.
                    // however, forcing 128-bit (e.g., using inline ptx), results in register
                    // spilling and much worse performance, so we'll keep it like this for now
                    // but ideally, we could reduce the register pressure a little.
                    store128(dweight_tmp_shared +
                                 item_ct1.get_local_id(2) * f128::size,
                             dweight_f);
                }
               
                item_ct1.barrier(sycl::access::fence_space::local_space);
                if (warpId == 0) {
                    for (int j = 1; j < warpsInBlock; j++) {
                        f128 dbias_tmp =
                            load128(dbias_tmp_shared +
                                    f128::size * (item_ct1.get_local_id(2) +
                                                  j * WARP_SIZE));
                        f128 dweight_tmp =
                            load128(dweight_tmp_shared +
                                    f128::size * (item_ct1.get_local_id(2) +
                                                  j * WARP_SIZE));
                        for(int i = 0; i < f128::size; ++i) {
                            dbias_f[i] += dbias_tmp[i];
                            dweight_f[i] += dweight_tmp[i];
                        }
                    }
                }
                
                item_ct1.barrier(sycl::access::fence_space::local_space);
                if (warpId == 0) {
                    f128 db_old = load128(dbias_shared + global_index + f128::size * o);
                    f128 dw_old = load128(dweight_shared + global_index + f128::size * o);
                    for(int i = 0; i < f128::size; ++i) {
                        dbias_f[i] += db_old[i];
                        dweight_f[i] += dw_old[i];
                    }
                    store128(dbias_shared + global_index + f128::size * o, dbias_f);
                    store128(dweight_shared + global_index + f128::size * o, dweight_f);
                }
            }
            if(global_index < C) {
                // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
                store128cg(dinp_bt + global_index, dinp128);
            }
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    // This is done by atomically incrementing a flag (cleared to 0 before launching the kernel)
    unsigned int* scratchFlag = (unsigned int*)(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    scratch += 32;
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    for (int i = item_ct1.get_local_id(2) * f128::size; i < C;
         i += BLOCK_SIZE * f128::size) {
        // Write to global memory in the same "shared memory banking friendly" order
        store128(scratch_dbias + i + 2 * C * item_ct1.get_group(2),
                 load128(dbias_shared + i));
        store128(scratch_dweight + i + 2 * C * item_ct1.get_group(2),
                 load128(dweight_shared + i));
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // that portion of shared memory is no longer used, so we can repurpose it for the scratch flag.
    unsigned int *tmp_flag = (unsigned int*)(shared + 2*rounded_C);
    if (item_ct1.get_local_id(2) == 0) {
        
        *tmp_flag = dpct::atomic_fetch_compare_inc<
            sycl::access::address_space::generic_space>(
            scratchFlag, item_ct1.get_group_range(2));
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (*tmp_flag == item_ct1.get_group_range(2) - 1) {
        // Reduction of the partial sums by the final block
        // todo - there isn't enough parallelism even inside that single SM...
        // ==> so could maybe split into another kernel with YET ANOTHER level of reduction?!
        for (int i = item_ct1.get_local_id(2) * f128::size; i < C;
             i += BLOCK_SIZE * f128::size) {
            f128 dbias_accum = f128::zeros();
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0;
                 read_block_idx < item_ct1.get_group_range(2);
                 read_block_idx++) {
                int offset = i + 2*C*read_block_idx;
                f128 dbias128 = load128(scratch_dbias + offset);
                f128 dweight128 = load128(scratch_dweight + offset);
                for(int k = 0; k < f128::size; k++) {
                    dbias_accum[k] += dbias128[k];
                    dweight_accum[k] += dweight128[k];
                }
            }
            store128(dbias_shared + i, dbias_accum);
            store128(dweight_shared + i, dweight_accum);
        }
       
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // convert from float/FP32 to floatX/BF16 for the final write
        // this is separate because it cannot use as many warps as the above (f128 vs x128)
        // todo - if we split this code into another kernel, we could maybe do it at the same time?
        for (int c = warpId; c < iterations_C; c += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 s_db = load128(dbias_shared + global_index + o * f128::size);
                f128 s_dw = load128(dweight_shared + global_index + o * f128::size);
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    dbias128[x] = (floatX)(s_db[i] + (float)dbias128[x]);
                    dweight128[x] = (floatX)(s_dw[i] + (float)dweight128[x]);
                }
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void layernorm_forward(floatX* out, floatX* mean, floatX* rstd,
                       floatX* inp, const floatX* weight, const floatX* bias,
                       int B, int T, int C, sycl::queue &q_ct1) {
 try{   
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N * WARP_SIZE, block_size);
    
    q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            layernorm_forward_kernel3(out, mean, rstd, inp, weight, bias, N, C,
                                      item_ct1);
        });
      });
    
 }
  catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
  }
}

void residual_forward(floatX* out, const floatX* inp1, const floatX* inp2, int N, sycl::queue &q_ct1) {
    
try {
    const int block_size = 256;
    assert(N % block_size == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            residual_forward_kernel(out, inp1, inp2, item_ct1);
        });
      });
    
  }
  catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
  }
}

void fused_residual_forward5(floatX *residual, floatX *normed, floatX *mean,
                             floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C, sycl::queue &q_ct1) 
try {
 
    const int block_size = 256;
    int block_y = block_size / WARP_SIZE;
    const int grid_size = CEIL_DIV(N, block_y);
    
    size_t smem = (2 + block_y) * C * sizeof(floatX);

    // reject status -> to do opencl fallback
    auto status = 0;
    
    if (status == 0) {
        
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(smem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                      sycl::range<3>(1, block_y, WARP_SIZE),
                                  sycl::range<3>(1, block_y, WARP_SIZE)),
                [=](sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                        fused_residual_forward_kernel5(
                            residual, normed, mean, rstd, inp1, inp2, weight,
                            bias, N, C, item_ct1,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    } else {
        residual_forward(residual, inp1, inp2, N*C, q_ct1);
        layernorm_forward(normed, mean, rstd, residual, weight, bias, N, 1, C, q_ct1);}
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
  }

    

void layernorm_backward(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                        const floatX* dout, const floatX* inp, const floatX* weight, const floatX* mean, const floatX* rstd,
                        int B, int T, int C, sycl::queue &q_ct1) {
try{
    dpct::device_info deviceProp;
    const int block_size = 512;
    const int blocks_per_sm = 2; // supported on every architecture and less cache thrashing than 3
    const int grid_size = blocks_per_sm * deviceProp.get_max_compute_units();
    size_t rounded_C = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    
    size_t shared_mem_size =
        (2 * rounded_C + 2 * (block_size - 32) * f128::size) * sizeof(float);

    std::memset(scratch, 0, 1 * sizeof(float));
                             
    
    q_ct1.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(shared_mem_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                layernorm_backward_kernel10(
                    dinp, dweight, dbias, scratch, dout, inp, weight, mean,
                    rstd, B, T, C, item_ct1,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}

int main(int argc, char** argv) {
    srand(0);

    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q = dev_ct1.in_order_queue();
    sycl::context ctx = q.get_context();
    
    int B = 32; // batch size
    int T = 128; // sequence length
    int C = 768; // embedding size
    int N = B * T;

    
    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // Device memory allocation
    float* d_out = sycl::malloc_device<float>(B * T * C, q);
    float* d_mean = sycl::malloc_device<float>(B * T, q);
    float* d_rstd = sycl::malloc_device<float>(B * T, q);
    float* d_inp = sycl::malloc_device<float>(B * T * C, q);
    float* d_weight = sycl::malloc_device<float>(C, q);
    float* d_bias = sycl::malloc_device<float>(C, q);

    // Copy data to device
    q.memcpy(d_inp, inp, B * T * C * sizeof(float)).wait();
    q.memcpy(d_weight, weight, C * sizeof(float)).wait();
    q.memcpy(d_bias, bias, C * sizeof(float)).wait();

    // read kernel_num from command line

    layernorm_forward( d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, q);

    validate_result(d_out, out, "out", B * T * C, 1e-5f);
    validate_result(d_mean, mean, "mean", B * T, 1e-5f);
    validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);


    std::cout << "All results match. Starting benchmarks.\n\n";

   
    // free memory
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);

    sycl::free(d_out, q);
    sycl::free(d_mean, q);
    sycl::free(d_rstd, q);
    sycl::free(d_inp, q);
    sycl::free(d_weight, q);
    sycl::free(d_bias, q);

    return 0;
}