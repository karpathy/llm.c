/*
The GPT-2 Encoder, which combines two encodings: token and position
In the forward pass, both encodings are added together
In the backward pass, the gradients flow to both, handled by different kernels
*/
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <stdint.h>
#include <utility>              // std::pair
#include <vector>
#include <algorithm>
#include <unordered_map>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// SYCL kernels

SYCL_EXTERNAL void encoder_forward_kernel3(floatX *out, const int *inp,
                                           const floatX *wte, const floatX *wpe,
                                           int B, int T, int C,
                                           const sycl::nd_item<3> &item_ct1) {
    int idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2)) *
              x128::size;
    int N = B * T * C;
    if (idx >= N) { return; }

    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;

    int ix = inp[b * T + t];

    floatX* out_btc = out + b * T * C + t * C + c;
    const floatX* wte_ix = wte + ix * C + c;
    const floatX* wpe_tc = wpe + t * C + c;

    x128 packed_out;
    x128 wte128 = load128cs(wte_ix);
    x128 wpe128 = load128cs(wpe_tc);
    for (int k = 0; k < x128::size; k++) {
        packed_out[k] = (floatX)((float)wte128[k] + (float)wpe128[k]);
    }
    store128(out_btc, packed_out);
}

template <int BLOCK_SIZE = 256>

void wte_backward_kernel(floatX *dwte, const sycl::int4 *bucket_info,
                         const int *workload_indices, const floatX *dout,
                         const int *inp, unsigned int seed, int B, int T, int C,
                         const sycl::nd_item<3> &item_ct1,
                         float *accum_shared) {
    // In order to be deterministic, we preprocess the inputs on the cpu into "buckets"
    // Each bucket corresponds to (WARP_SIZE * x128::size) channels for a single vocabulary token
    // Each thread handles x128::size channels, e.g. 256 per warp for BF16
    // Each block handles (BLOCK_SIZE / WARP_SIZE) elements in a single bucket in parallel
    // If a bucket has less than 8 elements, some warps will return immediately
    // If a bucket has more than 8 elements, we will loop over all of them
    // The buckets are sorted on the CPU so the largest buckets start 1st
    int bucket = item_ct1.get_group(2);
    int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
    int c_per_warp = WARP_SIZE * x128::size;

    int bucket_start_idx = bucket_info[bucket].x();
    int bucket_size = bucket_info[bucket].y();
    int bucket_ix = bucket_info[bucket].z();
    int c = bucket_info[bucket].w() * c_per_warp + (lane_id * x128::size);

    // Each thread handles "x128::size" channels, so at fp8, each warp would handle 512 channels
    // If C is not a multiple of this (e.g. 768), some buckets/c_groups cannot use the entire warp
    if (c >= C) { return; }
    // Exit early if this is a small bucket and this warp doesn't have any items to process
    if (warp_id >= bucket_size) { return; }

    float accum[x128::size] = {0.0f};

    for(int item = warp_id; item < bucket_size; item += BLOCK_SIZE/WARP_SIZE) {
        int bt = workload_indices[bucket_start_idx + item];

        const floatX* dout_btc = dout + bt * C + c;
        x128 packed_inp1 = load128cs(dout_btc);
        for (int k = 0; k < packed_inp1.size; k++) {
            accum[k] += (float)packed_inp1[k];
        }
    }

    if (warp_id != 0) {
        // we accumulate into warp 0, so only the other warps need to write to shared memory
        for (int k = 0; k < x128::size; k++) {
            accum_shared[item_ct1.get_local_id(2) + k * BLOCK_SIZE] = accum[k];
        }
        return; // only warp 0 is needed after writing to shared memory
    }

    // Read dwte for warp 0 even if other warps are not finished yet to maximise latency tolerance
    floatX* dwte_ix = dwte + bucket_ix * C + c;
    x128 packed_in_out = load128(dwte_ix);

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Accumulate into warp 0's registers by reading the values of the other warps in shared memory
    for (int i = item_ct1.get_local_id(2) + WARP_SIZE;
         i < dpct::min(BLOCK_SIZE, bucket_size * WARP_SIZE); i += WARP_SIZE) {
        for (int k = 0; k < x128::size; k++) {
            accum[k] += accum_shared[i + k * BLOCK_SIZE];
        }
    }

    // Add the result to dwte and write back to global memory (read-modify-write)
    for (unsigned int k = 0; k < x128::size; k++) {
        // We use stochastic rounding to go from FP32 to BF16 but the seed should be deterministic
        stochastic_rounding(accum[k] + (float)packed_in_out[k],
                            &packed_in_out[k], seed + k, item_ct1);
    }
    store128(dwte_ix, packed_in_out);
}

void wpe_backward_kernel(floatX* dwpe,
                                    const floatX* dout, const int* inp,
                                    int B, int T, int C, unsigned int seed,
                                    const sycl::nd_item<3> &item_ct1) {
    // Each thread handles x128::size "channel positions", e.g. 256 per warp for BF16
    // For gpt2-124M BF16, C=768 and T=1024, so 3 warps per channel and 3072 warps in total
    // For each "channel position" we sum the gradients for every batch at that C/T element
    // This way each dwte element is only updated once, and the kernel is fully deterministic!
    // The previous kernel was not deterministic, as batches were aggregated with atomicAdd
    int idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2)) *
              x128::size;
    if (idx >= T * C) { return; }

    // if C is not a multiple of WARP_SIZE*x128::size, it's OK for some warps to handle multiple t
    int t = idx / C;
    int c = idx % C;
    float accum[x128::size] = {0.0f};

    for (int b = 0; b < B; b++) {
        x128 packed_dout = load128cs(dout + (b * T * C) + (t * C) + c); // will never be read again
        for (int k = 0; k < x128::size; k++) {
            accum[k] += (float)packed_dout[k];
        }
    }

    floatX* dwpe_tc = dwpe + (t * C) + c;
    x128 packed_dwpe = load128(dwpe_tc);
    for (unsigned int k = 0; k < x128::size; k++) {
        // We use stochastic rounding to go from FP32 to BF16 but the seed should be deterministic
        stochastic_rounding(accum[k] + (float)packed_dwpe[k], &packed_dwpe[k],
                            seed + k, item_ct1);
    }
    store128(dwpe_tc, packed_dwpe);
}

// ----------------------------------------------------------------------------
// kernel launchers

void encoder_forward(floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C, sycl::queue &q) {
try{    
    const int block_size = 256;
    const int N = B * T * C;
    const int grid_size = CEIL_DIV(N, (int)(block_size * x128::size));
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            encoder_forward_kernel3(out, inp, wte, wpe, B, T, C, item_ct1);
        });
    });
   
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
}

// Fully deterministic (see comments in wte_backward_kernel and wpe_backward_kernel for more details)
void encoder_backward(floatX *dwte, floatX *dwpe,
                      floatX *scratch, // gpu outputs & scratch
                      int *workload_indices,
                      sycl::int4 *bucket_info, // cpu scratch buffers
                      const floatX *dout, const int *inp,
                      const int *inputs_cpu, // cpu/gpu inputs
                      int B, int T, int C, unsigned int seed,
                      sycl::queue &q) {
try{ 

    // Launch wpe kernel first (so it runs on the GPU in parallel with the CPU pre-processing for wte)
    const int block_size = 256;
    const int N = T * C / x128::size;
    const int grid_size = CEIL_DIV(N, block_size);
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                     {sycl::aspect::fp16});

        q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                wpe_backward_kernel(dwpe, dout, inp, B, T, C, seed, item_ct1);
            });
        });
    }
    

    // check the GPU scratch buffer is large enough to hold the bucket info and workload indices
    // todo - this is trivially true given hardcoded scratch buffer size here, is this useful?
    int num_c_groups = CEIL_DIV(C, x128::size * WARP_SIZE);
    assert(B * T * num_c_groups * (sizeof(sycl::int4) + sizeof(int)) <=
           B * T * 3 * C * sizeof(floatX));

    // Step 1: Sort inputs into buckets
    int total_items = 0;
    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    for (uint64_t bt = 0; bt < B * T; bt++) {
        for (uint64_t c_group = 0; c_group < num_c_groups; c_group++) {
            // todo - passing c_group/inputs_cpu[bt] in data to avoid a second hash lookup is a bit hacky
            uint64_t data = bt + (c_group<<32ULL) + ((uint64_t)inputs_cpu[bt]<<42ULL);
            buckets[c_group + num_c_groups * inputs_cpu[bt]].push_back(data);
            total_items++;
        }
    }

    // Step 2: Sort buckets by size in descending order
    // this is so the largest buckets are processed first by the GPU
    // otherwise, if they started late, they would still be running with the rest of the GPU idle
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sortedBuckets(buckets.begin(), buckets.end());
    std::sort(sortedBuckets.begin(), sortedBuckets.end(), // ugly because we don't have a typedef for the std::pair
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a, const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();
              });

    int num_buckets = buckets.size();
    int bucket_index = 0;
    int workload_index = 0;
    for (const auto& bucket : sortedBuckets) {
        bucket_info[bucket_index].x() = workload_index;       // bucket start
        bucket_info[bucket_index].y() = bucket.second.size(); // bucket size
        bucket_info[bucket_index].z() =
            (bucket.second[0] >> 42ULL) & ((1ULL << 20ULL) - 1); // bucket ix
        bucket_info[bucket_index].w() =
            (bucket.second[0] >> 32ULL) & ((1ULL << 10ULL) - 1); // bucket c

        for (uint64_t idx : bucket.second) {
            workload_indices[workload_index++] = (int)(idx & ((1ULL<<31ULL)-1ULL));
        }
        bucket_index++;
    }

    // Step 3: Copy data from host to device (async until the last one to avoid synchronising CPU/GPU twice)
    // todo - could use CUDA events (even without streams) to avoid CPU/GPU synchronisation completely
    sycl::int4 *d_bucket_info = (sycl::int4 *)scratch;
    int *d_workload_indices =
        (int *)(scratch + B * T * num_c_groups * sizeof(sycl::int4));
    q.memcpy(d_bucket_info, bucket_info, num_buckets * sizeof(sycl::int4));
    q.memcpy(d_workload_indices, workload_indices, total_items * sizeof(int)).wait();

    // Launch wte kernel
    // todo - profile block sizes on more content (depends on number of buckets and on GPU?)
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                     {sycl::aspect::fp16});

        q.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> accum_shared_acc_ct1(
                sycl::range<1>(x128::size * 256), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_buckets) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    wte_backward_kernel<256>(
                        dwte, d_bucket_info, d_workload_indices, dout, inp,
                        seed, B, T, C, item_ct1,
                        accum_shared_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
    
}

int main(){


 
    int B = 8;
    int T = 1024;
    int C = 768;
    int V = 50257;

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    int* inp = make_random_int(B * T, V);
    float* wte = make_random_float(V * C);
    float* wpe = make_random_float(T * C);

    // select device and create queue
    sycl::queue q(sycl::default_selector_v);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // move to GPU
    float* d_out = sycl::malloc_device<float>(B * T * C, q);
    int* d_inp = sycl::malloc_device<int>(B * T, q);
    float* d_wte = sycl::malloc_device<float>(V * C, q);
    float* d_wpe = sycl::malloc_device<float>(T * C, q);

    q.memcpy(d_inp, inp, B * T * sizeof(int)).wait();
    q.memcpy(d_wte, wte, V * C * sizeof(float)).wait();
    q.memcpy(d_wpe, wpe, T * C * sizeof(float)).wait();
    q.memcpy(d_out, out, B * T * C * sizeof(float)).wait();
    
    encoder_forward(d_out, d_inp, d_wte, d_wpe, B, T, C, q);
    
    validate_result(d_out, out, "out", B * T * C, 1e-5f);
    //validate_result(d_inp, inp, "inp", B * T, 1e-5f);
    validate_result(d_wpe, wpe, "wpe", T * C, 1e-5f);
    validate_result(d_wte, wte, "wte", V * C, 1e-5f);
    
    


return 0;

}
