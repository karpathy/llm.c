/*
Kernels for softmax forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt softmax_forward.cu -o softmax_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./softmax_forward 1

version 2 is a fused kernel that parallelizes over all of B,T,C
./softmax_forward 2

version 3 uses intra-warp reductions for maxval and sumval, must use block_size=32
./softmax_forward 3

version 4 uses both intra-warp reductions and shared memory for inter-warp reductions
so it can tolerate any block_size % 32 == 0. this is hopefully the most efficient version
./softmax_forward 4

version 5 is naive port from CPU code (softmax_online) to kernel: parallelizes over B,T, loops over C
./softmax_forward 5

version 6 is softmax_online that parallelizes over all of B,T,C
./softmax_forward 6

version 7 is softmax optimized for very large C.
./softmax_forward 7
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "common.h"
#include <cmath>

// ----------------------------------------------------------------------------
// CPU code reference

void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        // Note: since we want to ensure that the CUDA-kernels are accurate,
        // we do this accumulation in higher precision, so we can be assured
        // that our ground-truth is of high quality.
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        float norm = 1.f / (float)sum;
        for (int j = 0; j < C; j++) {
            out_row[j] *= norm;
        }
    }
}


// online version of softmax on CPU from the paper "Online normalizer calculation for softmax"
void softmax_forward_online_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        float sum = 0.0f;
		for (int j = 0; j < C; j++) {
			float maxval_prev = maxval;
			if (inp_row[j] > maxval) {
				maxval = inp_row[j];
				sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
			} else {
				sum += expf(inp_row[j] - maxval);
			}
		}

        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval) / sum;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

void softmax_forward_kernel1(float* out, const float* inp, int N, int C,
                             const sycl::nd_item<3> &item_ct1) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i < N) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = sycl::native::exp(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= (float)sum;
        }
    }
}

void softmax_forward_kernel2(float* out, const float* inp, int N, int C,
                             const sycl::nd_item<3> &item_ct1,
                             uint8_t *dpct_local) {
    // inp is (N, C)
    // in each row of C elements, first calculates maxval, then returns expf(val - maxval)
    auto shared = (float *)dpct_local;
    int idx = item_ct1.get_group(2);    // ranges [0, N)
    int tid = item_ct1.get_local_id(2); // ranges [0, block_size)
    int block_size = item_ct1.get_local_range(2);
    const float* x = inp + idx * C; // idx-th row of inp
    // thread coarsening
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += block_size) {
        maxval = sycl::fmax(maxval, (float)(x[i]));
    }
    shared[tid] = maxval;
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        /*
        DPCT1118:153: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (tid < stride) {
            shared[tid] = sycl::fmax(shared[tid], shared[tid + stride]);
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    float offset = shared[0];
    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = sycl::native::exp(x[i] - offset);
    }
    /*
    DPCT1065:499: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // thread coarsening again, for the sum
    x = out + idx * C; // idx-th row of out
    float sumval = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sumval += x[i];
    }
    shared[tid] = sumval;
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        /*
        DPCT1118:154: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:501: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // broadcast the sum to all threads in the block
    /*
    DPCT1065:500: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float sum = shared[0];
    // divide the input values by the sum
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = x[i] / sum;
    }
}

// warp-level reduction for finding the maximum value
SYCL_EXTERNAL float warpReduceMax(float val, const sycl::nd_item<3> &item_ct1) {
    for (int offset = 16; offset > 0; offset /= 2) {
        /*
        DPCT1121:155: Make sure that the "val" which is used in the SYCL group
        function/algorithm is initialized.
        */
        /*
        DPCT1096:549: The right-most dimension of the work-group used in the
        SYCL kernel that calls this function may be less than "32". The function
        "dpct::shift_sub_group_left" may return an unexpected result on the CPU
        device. Modify the size of the work-group to ensure that the value of
        the right-most dimension is a multiple of "32".
        */
        val = sycl::fmax(val, dpct::shift_sub_group_left(
                                  item_ct1.get_sub_group(), val, offset));
    }
    return val;
}

void softmax_forward_kernel3(float* out, const float* inp, int N, int C,
                             const sycl::nd_item<3> &item_ct1,
                             uint8_t *dpct_local) {
    // kernel must use block size of 32
    auto shared = (float *)dpct_local;
    int idx = item_ct1.get_group(2);
    int tid = item_ct1.get_local_id(2);
    const float* x = inp + idx * C;

    // Thread coarsening and within-warp reduction for maxval
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += item_ct1.get_local_range(2)) {
        maxval = sycl::fmax(maxval, (float)(x[i]));
    }
    maxval = warpReduceMax(maxval, item_ct1);

    // Broadcast maxval within the warp
    /*
    DPCT1096:564: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::select_from_sub_group" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    float offset =
        dpct::select_from_sub_group(item_ct1.get_sub_group(), maxval, 0);

    // Compute expf and write the result to global memory
    for (int i = tid; i < C; i += item_ct1.get_local_range(2)) {
        out[idx * C + i] = sycl::native::exp(x[i] - offset);
    }

    // Thread coarsening and within-warp reduction for sumval
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += item_ct1.get_local_range(2)) {
        sumval += x[i];
    }
    // No need to broadcast sumval since all threads in the warp will have the same value
    // (due to the fact that we're using __shfl_xor_sync)
    sumval = warpReduceSum(sumval, item_ct1);

    // Divide the input values by the sum
    for (int i = tid; i < C; i += item_ct1.get_local_range(2)) {
        out[idx * C + i] = x[i] / sumval;
    }
}

SYCL_EXTERNAL void softmax_forward_kernel4(float *out, const float *inp, int N,
                                           int C,
                                           const sycl::nd_item<3> &item_ct1,
                                           uint8_t *dpct_local) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    auto shared = (float *)dpct_local;
    int idx = item_ct1.get_group(2);
    int tid = item_ct1.get_local_id(2);
    int warpId = item_ct1.get_local_id(2) / 32; // warp index within a block
    int laneId = item_ct1.get_local_id(2) % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = item_ct1.get_local_range(2) / 32;

    // shared[] must be allocated to have warpsPerBlock elements
    // those will be used for max and sum values
    float* max_or_sum_storage = shared;

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += item_ct1.get_local_range(2)) {
        maxval = sycl::fmax(maxval, (float)(x[i]));
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval, item_ct1);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) max_or_sum_storage[warpId] = maxval;
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // now the 0th thread of the block reduces the max values in shared memory, i.e. across warps
    if (tid == 0) {
        float val = max_or_sum_storage[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = sycl::fmax(val, max_or_sum_storage[i]);
        }
        // store the final max in the first position
        max_or_sum_storage[0] = val;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // broadcast the max to all threads
    float offset = max_or_sum_storage[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += item_ct1.get_local_range(2)) {
        out[idx * C + i] = sycl::native::exp(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += item_ct1.get_local_range(2)) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval, item_ct1);

    // write sumval to shared memory
    if (laneId == 0) max_or_sum_storage[warpId] = sumval;
    /*
    DPCT1065:502: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = max_or_sum_storage[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += max_or_sum_storage[i];
        }
        max_or_sum_storage[0] = val;
    }
    /*
    DPCT1065:503: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // broadcast the sum to all threads
    float sum = max_or_sum_storage[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += item_ct1.get_local_range(2)) {
        out[idx * C + i] = x[i] / sum;
    }
}

void softmax_forward_online_kernel1(float* out, const float* inp, int N, int C,
                                    const sycl::nd_item<3> &item_ct1) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i < N) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            float maxval_prev = maxval;
            float current_val = inp_row[j];
			if (current_val > maxval) {
				maxval = current_val;
                                sum = sum * sycl::native::exp(maxval_prev -
                                                              maxval) +
                                      sycl::native::exp(current_val - maxval);
                        }
			else {
                                sum += sycl::native::exp(current_val - maxval);
                        }
		}

        for (int j = 0; j < C; j++) {
            out_row[j] = sycl::native::exp(inp_row[j] - maxval) / sum;
        }
    }
}

// struct for the reduction operation, guarantees 8-byte alignment
struct __dpct_align__(8) SumMax
{
    float maxval;
    float sum;
};

// forceinline helps avoid function call overhead
__dpct_inline__ SumMax reduce_sum_max_op(SumMax a, SumMax b) {
    bool a_bigger = (a.maxval > b.maxval);
    SumMax bigger_m = a_bigger ? a : b;
    SumMax smaller_m = a_bigger ? b : a;
    SumMax res;
    res.maxval = bigger_m.maxval;
    res.sum =
        bigger_m.sum +
        smaller_m.sum * sycl::native::exp(smaller_m.maxval - bigger_m.maxval);
    return res;
}

void softmax_forward_online_kernel2(float* out, const float* inp, int N, int C,
                                    const sycl::nd_item<3> &item_ct1) {

        sycl::group<3> block = item_ct1.get_group();
        sycl::sub_group warp = item_ct1.get_sub_group();
        /*
        DPCT1007:504: Migration of
        cooperative_groups::thread_block_tile::meta_group_size is not supported.
        */
        int idx = item_ct1.get_group(2) * warp.meta_group_size() +
                  item_ct1.get_sub_group().get_group_linear_id();
        if (idx >= N) {
		return;
	}

	// one row of inp, i.e. inp[idx, :] of shape (C,)
	const float* x = inp + idx * C;

    // base case for the reduction
    SumMax sm_partial;
	sm_partial.maxval = -INFINITY;
	sm_partial.sum = 0.0f;

	// first, thread coarsening by directly accessing global memory in series
        for (int i = item_ct1.get_sub_group().get_local_linear_id(); i < C;
             i += item_ct1.get_sub_group().get_local_linear_range()) {
                sm_partial = reduce_sum_max_op(sm_partial, { x[i], 1.0f });
	}

    // second, the reduction
        SumMax sm_total = sycl::reduce_over_group(
            item_ct1.get_sub_group(), sm_partial, reduce_sum_max_op);

        // divide the whole row by the sum
        for (int i = item_ct1.get_sub_group().get_local_linear_id(); i < C;
             i += item_ct1.get_sub_group().get_local_linear_range()) {
        // the below is equivalent to
        // out[idx * C + i] = expf(x[i] - sm_total.maxval) / sm_total.sum;
        // but uses special instruction that bypasses the cache
        __stcs(out + idx * C + i,
               sycl::native::exp(x[i] - sm_total.maxval) / sm_total.sum);
        }
}

/*
DPCT1110:156: The total declared local variable size in device function
softmax_forward_kernel7 exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void softmax_forward_kernel7(float *out, const float *inp, int N, int C,
                             const sycl::nd_item<3> &item_ct1,
                             uint8_t *dpct_local) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel4, but optimised for very large Cs with advanced unrolling

    // The trick is to read into a register array (all indices known at compile time)
    // and always read UNROLL_FACTOR values to maximise memory level parallelism
    // even if we would be out of bounds, we set the index to min(C-1, idx)
    // so we just do some unnecessary reads (obviously bad for small C)
    // the writes are in a separate loop with a conditional check for out of bounds
    // making it separate is necessary to convince the compiler to do the right thing
    const int UNROLL_FACTOR = 8;
    const int warpsPerBlock = item_ct1.get_local_range(2) / 32;

    auto shared = (float *)dpct_local;
    int idx = item_ct1.get_group(2);
    int tid = item_ct1.get_local_id(2);
    int warpId = item_ct1.get_local_id(2) / 32; // warp index within a block
    int laneId = item_ct1.get_local_id(2) % 32; // thread index within a warp

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    if (tid >= C) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }

    const float* x = inp + idx * C; // input
    float* y = out + idx * C; // output

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += item_ct1.get_local_range(2) * UNROLL_FACTOR) {
#pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            /*
            DPCT1064:509: Migrated min call is used in a macro/template
            definition and may not be valid for all macro/template uses. Adjust
            the code.
            */
            maxval = sycl::fmax(
                maxval,
                (float)(x[dpct::min(
                    C - 1,
                    (unsigned int)(i + u * item_ct1.get_local_range(2)))]));
        }
    }

    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval, item_ct1);
    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    /*
    DPCT1065:505: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; i++) {
            val = sycl::fmax(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    /*
    DPCT1065:506: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    // + thread coarsening for sum
    float sumval = 0.0f;
    for (int i = tid; i < C; i += item_ct1.get_local_range(2) * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = __ldcs(&x[dpct::min(
                C - 1, (unsigned int)(i + u * item_ct1.get_local_range(2)))]);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u * item_ct1.get_local_range(2) < C) {
                float output = sycl::native::exp(reg_array[u] - offset);
                y[dpct::min(
                    C - 1,
                    (unsigned int)(i + u * item_ct1.get_local_range(2)))] =
                    output;       // compiler likes redundant min()?!
                sumval += output; // combined into the same loop unlike kernel3
            }
        }
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval, item_ct1);
    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    /*
    DPCT1065:507: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    /*
    DPCT1065:508: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += item_ct1.get_local_range(2) * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = y[dpct::min(
                C - 1, (unsigned int)(i + u * item_ct1.get_local_range(2)))];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u * item_ct1.get_local_range(2) < C) {
                y[i + u * item_ct1.get_local_range(2)] = reg_array[u] / sum;
            }
        }
    }
}

void softmax_forward_online_kernel8(float* out, const float* inp, int N, int C,
                                    const sycl::nd_item<3> &item_ct1) {
    // online softmax paper: http://arxiv.org/abs/1805.02867
    // online softmax reduces loops from 3 to 2
    // which is done by calculating sumval and maxval in one loop
    const int warpsPerBlock = item_ct1.get_local_range(2) /
                              item_ct1.get_sub_group().get_local_range().get(0);
    int tid = item_ct1.get_local_id(2);

    if (tid >= C) {
        return;
    }

    int warpId = tid / item_ct1.get_sub_group().get_local_range().get(0);
    int laneId = tid % item_ct1.get_sub_group().get_local_range().get(0);
    // one warp one row
    int row = item_ct1.get_group(2) * warpsPerBlock + warpId;

    if (row >= N) {
        return;
    }

    const float* x = inp + row * C;
    float* const y = out + row * C;

    // merge calculating maxval and sumval in one loop
    // which is an arithmetic improvment from online softmax over normal softmax
    float maxval = -INFINITY, sumval = 0.0f, bigger;
    for (int i = laneId; i < C;
         i += item_ct1.get_sub_group().get_local_range().get(0)) {
        // when updating the maxval, dynamically updates the previous sumval by
        // multiplying e^{previous_maxval - current_maxval}
        bigger = sycl::fmax(maxval, (float)(x[i]));
        sumval = sumval * sycl::native::exp(maxval - bigger) +
                 sycl::native::exp(x[i] - bigger);
        maxval = bigger;
    }

    // use warp functions instead of cooperative groups for better readibility
    // calculate the warp wised maxval and sumval
    float offsetMaxval, offsetSumval;
    for (int offset = item_ct1.get_sub_group().get_local_range().get(0) / 2;
         offset > 0; offset >>= 1) {
        sycl::group_barrier(item_ct1.get_sub_group());
        /*
        DPCT1096:565: The right-most dimension of the work-group used in the
        SYCL kernel that calls this function may be less than "32". The function
        "dpct::shift_sub_group_left" may return an unexpected result on the CPU
        device. Modify the size of the work-group to ensure that the value of
        the right-most dimension is a multiple of "32".
        */
        offsetMaxval = dpct::shift_sub_group_left(item_ct1.get_sub_group(),
                                                  maxval, offset);
        /*
        DPCT1096:566: The right-most dimension of the work-group used in the
        SYCL kernel that calls this function may be less than "32". The function
        "dpct::shift_sub_group_left" may return an unexpected result on the CPU
        device. Modify the size of the work-group to ensure that the value of
        the right-most dimension is a multiple of "32".
        */
        offsetSumval = dpct::shift_sub_group_left(item_ct1.get_sub_group(),
                                                  sumval, offset);
        if (offsetMaxval > maxval) {
            sumval *= sycl::native::exp(maxval - offsetMaxval);
            maxval = offsetMaxval;
        } else {
            offsetSumval *= sycl::native::exp(offsetMaxval - maxval);
        }
        sumval += offsetSumval;
    }

    // sync the warp wised maxval and sumval
    // which are also the maxval and sumval of one row in C
    /*
    DPCT1096:567: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::select_from_sub_group" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    maxval = dpct::select_from_sub_group(item_ct1.get_sub_group(), maxval, 0);
    /*
    DPCT1096:568: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::select_from_sub_group" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sumval = dpct::select_from_sub_group(item_ct1.get_sub_group(), sumval, 0);

    for (int i = laneId; i < C;
         i += item_ct1.get_sub_group().get_local_range().get(0)) {
        y[i] = sycl::native::exp(x[i] - maxval) / sumval;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void softmax_forward1(float* out, const float* inp, int N, int C, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    /*
    DPCT1049:157: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                     {sycl::aspect::fp64});

        dpct::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                softmax_forward_kernel1(out, inp, N, C, item_ct1);
            });
    }
    /*
    DPCT1010:510: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void softmax_forward2(float* out, const float* inp, int N, int C, const int block_size) {
    int grid_size = N;
    /*
    DPCT1083:159: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t shared_mem_size = block_size * sizeof(float);
    /*
    DPCT1049:158: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(shared_mem_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                softmax_forward_kernel2(
                    out, inp, N, C, item_ct1,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
}

void softmax_forward3(float* out, const float* inp, int N, int C, int block_size) {
    block_size = 32; // awkward but ok. this one only works with block size 32
    int grid_size = N;
    /*
    DPCT1083:161: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t shared_mem_size = block_size * sizeof(float);
    /*
    DPCT1049:160: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(shared_mem_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                softmax_forward_kernel3(
                    out, inp, N, C, item_ct1,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
}

void softmax_forward4(float* out, const float* inp, int N, int C, int block_size) {
    int grid_size = N;
    // for each warp in the block we need a float that will be used for both maxval and sumval
    /*
    DPCT1083:163: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t shared_mem_size = block_size / 32 * sizeof(float);
    /*
    DPCT1049:162: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(shared_mem_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                softmax_forward_kernel4(
                    out, inp, N, C, item_ct1,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
}

void softmax_forward_online1(float* out, const float* inp, int N, int C, int block_size) {
    const int grid_size = ceil_div(N, block_size);
    /*
    DPCT1049:164: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                     {sycl::aspect::fp64});

        dpct::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                softmax_forward_online_kernel1(out, inp, N, C, item_ct1);
            });
    }
    /*
    DPCT1010:511: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void softmax_forward_online2(float* out, const float* inp, int N, int C, int block_size) {
    const int grid_size = ceil_div(N * 32, block_size);
    /*
    DPCT1049:165: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            softmax_forward_online_kernel2(out, inp, N, C, item_ct1);
        });
    /*
    DPCT1010:512: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void softmax_forward7(float* out, const float* inp, int N, int C, int block_size) {
    int grid_size = N;
    /*
    DPCT1083:167: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
    /*
    DPCT1049:166: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(shared_mem_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                softmax_forward_kernel7(
                    out, inp, N, C, item_ct1,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
}

void softmax_forward_online8(float* out, const float* inp, int N, int C, int block_size) {
    const int grid_size = ceil_div(N * 32, block_size);
    /*
    DPCT1049:168: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            softmax_forward_online_kernel8(out, inp, N, C, item_ct1);
        });
    /*
    DPCT1010:513: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

// kernel version dispatch
void softmax_forward(int kernel_num, float* out, const float* inp, int N, int C, const int block_size) {
    switch (kernel_num) {
        case 1:
            softmax_forward1(out, inp, N, C, block_size);
            break;
        case 2:
            softmax_forward2(out, inp, N, C, block_size);
            break;
        case 3:
            softmax_forward3(out, inp, N, C, block_size);
            break;
        case 4:
            softmax_forward4(out, inp, N, C, block_size);
            break;
        case 5:
            softmax_forward_online1(out, inp, N, C, block_size);
            break;
        case 6:
            softmax_forward_online2(out, inp, N, C, block_size);
            break;
        case 7:
            softmax_forward7(out, inp, N, C, block_size);
            break;
        case 8:
            softmax_forward_online8(out, inp, N, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int V = 50257;

    int deviceIdx = 0;
    /*
    DPCT1093:514: The "deviceIdx" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    cudaCheck(DPCT_CHECK_ERROR(dpct::select_device(deviceIdx)));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * V * sizeof(float));
    float* inp = make_random_float(B * T * V);

    // make the input less uniformly random: Otherwise, all probabilities will be basically zero,
    // and the tests are not actually meaningful.
    const int* outliers = make_random_int(B * T * 3, V);
    for(int k = 0; k < 3; ++k) {
        for(int j = 0; j < B * T; ++j) {
            inp[j * V + outliers[j*3 + k]] *= 20;
        }
    }

    // move to GPU
    float* d_out;
    float* d_inp;
    cudaCheck(DPCT_CHECK_ERROR(d_out = sycl::malloc_device<float>(
                                   B * T * V, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_inp = sycl::malloc_device<float>(
                                   B * T * V, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_inp, inp, B * T * V * sizeof(float))
                             .wait()));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    softmax_forward_cpu(out, inp, B * T, V);
    {
        float max_el = -INFINITY;
        for(int i = 0; i <  B * T * V; ++i) {
            max_el = std::max(max_el, out[i]);
        }
        assert(max_el > 1e-4);
        printf("Largest output is: %f\n", max_el);
    }

    // first check the correctness of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        softmax_forward(kernel_num, d_out, d_inp, B * T, V, block_size);
        validate_result(d_out, out, "out", B * T * V, 1e-4f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, softmax_forward,
                                              kernel_num, d_out, d_inp, B * T, V, block_size
                                              );

        printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1'000 / (B*T));
    }

    // free memory
    free(out);
    free(inp);
    free((void*)outliers);
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_out, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_inp, dpct::get_in_order_queue())));

    return 0;
}