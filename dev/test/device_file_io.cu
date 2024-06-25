/*
Tests device <-> file IO functions

compile and run as (from dev/test directory)
nvcc -o device_file_io device_file_io.cu && ./device_file_io
*/


#include "../../llmc/cuda_common.h"
#include <vector>
#include <random>
#include <cstdio>
#include <algorithm>

void test(size_t nelem, size_t wt_buf_size, size_t rd_buf_size) {

    float* data;
    cudaCheck(cudaMalloc(&data, nelem*sizeof(float)));

    // generate random array
    std::vector<float> random_data(nelem);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.f, 100.f);
    std::generate(random_data.begin(), random_data.end(), [&](){ return dist(rng); });

    cudaCheck(cudaMemcpy(data, random_data.data(), random_data.size()*sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    FILE* tmp = fopenCheck("tmp.bin", "w");
    device_to_file(tmp, data, nelem * sizeof(float), wt_buf_size, stream);
    fcloseCheck(tmp);


    float* reload;
    cudaCheck(cudaMalloc(&reload, nelem*sizeof(float)));

    tmp  = fopenCheck("tmp.bin", "r");
    file_to_device(reload, tmp, nelem * sizeof(float), rd_buf_size, stream);
    fcloseCheck(tmp);

    std::vector<float> cmp(nelem);
    cudaCheck(cudaMemcpy(cmp.data(), reload, nelem * sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < nelem; ++i) {
        if(random_data[i] != cmp[i])  {
            fprintf(stderr, "FAIL: Mismatch at position %d: %f vs %f\n", i, random_data[i], cmp[i]);
            remove("tmp.bin");
            exit(EXIT_FAILURE);
        }
    }

    cudaCheck(cudaFree(reload));
    cudaCheck(cudaFree(data));
    remove("tmp.bin");
}

int main() {
    test(1025, 10000, 10000);           // buffers larger than data
    test(1025, 1024, 513);              // different and smaller
    test(500, 500*sizeof(float),
         500*sizeof(float));            // exact match
    test(125'000, 10000, 10000);        // large array
}