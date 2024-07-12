#ifndef ABSMAX_HISTORY_CUH
#define ABSMAX_HISTORY_CUH

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <stdexcept>
#include <typeindex>
#include "cuda_common.h"
#include "cuda_utils.cuh"

constexpr uint8_t ABSMAX_HISTORY_SIZE = 16; // todo - need to tune this
#define ALWAYS_UPDATE_ABSMAX false // slow, for sanity checking only
#define SKIP_UPDATE_ZERO_ABSMAX true // do not update the scale/descale if max absmax is 0

// Offsets for calculated values
enum CalculatedValueOffset {
    ABSMAX_OFFSET = 0,
    SCALE_OFFSET = 1,
    DESCALE_OFFSET = 2,
    SCALING_FACTOR_OFFSET = 3,
    ABSMAX_VALUES_COUNT = 4
};

class TensorAbsMaxTracker {
public:
    static constexpr size_t INITIAL_TENSOR_COUNT = 1024;
    static constexpr float  DEFAULT_FUDGE_FACTOR = 1.0f;

    TensorAbsMaxTracker()
        : current_tensor_count(0), max_tensor_count(0), d_storage(nullptr), d_current_indices(nullptr) {
        grow_storage_if_needed(INITIAL_TENSOR_COUNT);
    }

    ~TensorAbsMaxTracker() {
        if (d_storage) cudaCheck(cudaFree(d_storage));
        if (d_current_indices) cudaCheck(cudaFree(d_current_indices));
    }

    template<typename T>
    float* get_absmax_data(const T* tensor_address, size_t size,
                           const void* associated_tensor = nullptr,
                           float scale_factor = 0.0f, bool calculate_if_needed=true) {
        bool is_new;
        TensorInfo& info = get_tensor_info(tensor_address, size, associated_tensor, scale_factor, is_new);

        if (!is_new && info.call_count == 0 && info.previous_call_count == 0) {
            printf("Warning: get_absmax_data() with call_count=0 and previous_call_count=0 for existing tensor at %p (size %lu)\n",
                info.call_count, tensor_address, size);
        }

        #if ALWAYS_UPDATE_ABSMAX == true // slow, for sanity checking only
        if (calculate_if_needed) {
            float* absmax_memory = next_absmax_ptr(tensor_address, size, calculate_if_needed);
            get_absmax(absmax_memory, tensor_address, size);
            update_single_absmax(tensor_address, size, calculate_if_needed, 1.0f);
            return (float*)(d_storage + get_storage_offset(info.index) + ABSMAX_HISTORY_SIZE);
        }
        #endif

        if (is_new) {
            if (calculate_if_needed) {
                return calculate_manual_absmax(tensor_address, size, associated_tensor);
            } else {
                return nullptr;
            }
        }
        return (float*)(d_storage + get_storage_offset(info.index) + ABSMAX_HISTORY_SIZE);
    }

    template <typename T>
    float* get_descale_ptr(const T* tensor_address, size_t size, const void* associated_tensor = nullptr, bool calculate_if_needed=false, bool must_be_fp8=false) {
        if constexpr (std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, __nv_fp8_e5m2>::value) {
            float* data = get_absmax_data(tensor_address, size, associated_tensor, 0.0f, calculate_if_needed);
            assert(data);
            return (data + DESCALE_OFFSET);
        }
        assert(!must_be_fp8);
        return nullptr;
    }

    template<typename T>
    float* next_absmax_ptr(const T* tensor_address, size_t size,
                           const void* associated_tensor = nullptr, float scale_factor = 0.0f) {
        bool is_new;
        TensorInfo& info = get_tensor_info(tensor_address, size, associated_tensor, scale_factor, is_new);

        info.call_count++;
        if (info.call_count > 1) {
            printf("Warning: next_absmax_ptr() called %zu times for tensor at %p (size %lu) since last update_all_absmax() [previous call count: %d]\n",
                info.call_count, tensor_address, size, info.previous_call_count);
        }

        uint8_t currentIndex = h_current_indices[info.index];
        h_current_indices[info.index] = (currentIndex + 1) % ABSMAX_HISTORY_SIZE;
        return (float*)(d_storage + get_storage_offset(info.index) + currentIndex);
    }

    template<typename T>
    void set_scale_factor(const T* tensor_address, size_t size, const void* associated_tensor, float scale_factor) {
        TensorKey key{tensor_address, associated_tensor, std::type_index(typeid(T)), size};
        auto it = tensor_info_map.find(key);
        if (it == tensor_info_map.end()) {
            throw std::runtime_error("Tensor not registered");
        }
        it->second.scale_factor = scale_factor;
        size_t offset = get_storage_offset(it->second.index) + ABSMAX_HISTORY_SIZE + SCALING_FACTOR_OFFSET;
        cudaCheck(cudaMemcpy(d_storage + offset, &scale_factor, sizeof(float), cudaMemcpyHostToDevice));
    }

    void update_all_absmax(cudaStream_t stream = 0, float fudge_factor = DEFAULT_FUDGE_FACTOR);

    template<typename T>
    void update_single_absmax(const T* tensor_address, size_t size, const void* associated_tensor,
                              float fudge_factor = DEFAULT_FUDGE_FACTOR, cudaStream_t stream = 0);

    template<typename T>
    float* calculate_manual_absmax(const T* tensor_address, size_t size, const void* associated_tensor, cudaStream_t stream = 0);

    void print_all_tensor_info() {
        std::vector<float> hostData(current_tensor_count * (ABSMAX_HISTORY_SIZE + ABSMAX_VALUES_COUNT));
        cudaCheck(cudaMemcpy(hostData.data(), d_storage,
                             current_tensor_count * (ABSMAX_HISTORY_SIZE + ABSMAX_VALUES_COUNT) * sizeof(float),
                             cudaMemcpyDeviceToHost));

        std::cout << "Tensor Address,Associated Tensor,Type,Size,AbsMax,Scale,Descale,Scale Factor,Current Index";
        for (uint8_t i = 0; i < ABSMAX_HISTORY_SIZE; ++i) {
            std::cout << ",History" << static_cast<int>(i);
        }
        std::cout << std::endl;

        for (const auto& pair : tensor_info_map) {
            size_t offset = get_storage_offset(pair.second.index);
            std::cout << pair.first.address << ","
                      << pair.first.associated_tensor << ","
                      << pair.first.type.name() << ","
                      << pair.first.size << ","
                      << hostData[offset + ABSMAX_HISTORY_SIZE + ABSMAX_OFFSET] << ","
                      << hostData[offset + ABSMAX_HISTORY_SIZE + SCALE_OFFSET] << ","
                      << hostData[offset + ABSMAX_HISTORY_SIZE + DESCALE_OFFSET] << ","
                      << hostData[offset + ABSMAX_HISTORY_SIZE + SCALING_FACTOR_OFFSET] << ","
                      << static_cast<int>(h_current_indices[pair.second.index]) << ",";
            for (uint8_t i = 0; i < ABSMAX_HISTORY_SIZE; ++i) {
                std::cout << std::setprecision(6) << hostData[offset + i];
                if (i < ABSMAX_HISTORY_SIZE - 1) std::cout << ",";
            }
            std::cout << std::endl;
        }
    }

private:
    struct TensorKey {
        const void* address;
        const void* associated_tensor;
        std::type_index type;
        size_t size;

        bool operator==(const TensorKey& other) const {
            return address == other.address &&
                   associated_tensor == other.associated_tensor &&
                   type == other.type && size == other.size;
        }
    };

    struct TensorKeyHash {
        std::size_t operator()(const TensorKey& k) const {
            return std::hash<const void*>()(k.address) ^
                   std::hash<const void*>()(k.associated_tensor) ^
                   std::hash<std::type_index>()(k.type) ^ std::hash<size_t>()(k.size);
        }
    };

    struct TensorInfo {
        size_t index;
        float scale_factor;
        size_t call_count;
        size_t previous_call_count;
    };

    size_t current_tensor_count;
    size_t max_tensor_count;
    float* d_storage;
    uint8_t* d_current_indices;
    std::unordered_map<TensorKey, TensorInfo, TensorKeyHash> tensor_info_map;
    std::vector<uint8_t> h_current_indices;

    void reset_all_call_counts() {
        for (auto& pair : tensor_info_map) {
            pair.second.previous_call_count = pair.second.call_count;
            pair.second.call_count = 0;
        }
    }

    void grow_storage_if_needed(size_t newTensorCount) {
        // todo - not 100% convinced this is safe...
        // what if a kernel has 2xFP8 and this grows when querying for 2nd one, after storing the 1st?!
        if (newTensorCount <= max_tensor_count) return;

        size_t new_max_tensor_count = max_tensor_count == 0 ? INITIAL_TENSOR_COUNT : max_tensor_count * 2;
        while (new_max_tensor_count < newTensorCount) {
            new_max_tensor_count *= 2;
        }

        size_t newSize = new_max_tensor_count * (ABSMAX_HISTORY_SIZE + ABSMAX_VALUES_COUNT) * sizeof(float);
        float* new_storage;
        uint8_t* new_indices;
        cudaCheck(cudaMalloc(&new_storage, newSize));
        cudaCheck(cudaMemset(new_storage, 0, newSize));
        cudaCheck(cudaMalloc(&new_indices, new_max_tensor_count * sizeof(uint8_t)));
        cudaCheck(cudaMemset(new_indices, 0, new_max_tensor_count * sizeof(uint8_t)));
        if (d_storage) {
            cudaCheck(cudaMemcpy(new_storage, d_storage,
                                 max_tensor_count * (ABSMAX_HISTORY_SIZE + ABSMAX_VALUES_COUNT) * sizeof(float),
                                 cudaMemcpyDeviceToDevice));
            cudaCheck(cudaMemcpy(new_indices, d_current_indices, max_tensor_count * sizeof(uint8_t),
                                 cudaMemcpyDeviceToDevice));
            cudaCheck(cudaFree(d_storage));
            cudaCheck(cudaFree(d_current_indices));
            std::cout << "Storage grown to accommodate " << new_max_tensor_count << " tensors." << std::endl;
        }
        d_storage = new_storage;
        d_current_indices = new_indices;
        h_current_indices.resize(new_max_tensor_count);
        max_tensor_count = new_max_tensor_count;
        cudaCheck(cudaGetLastError());
    }

    size_t get_storage_offset(size_t index) const {
        return index * (ABSMAX_HISTORY_SIZE + ABSMAX_VALUES_COUNT);
    }

    template<typename T>
    float get_default_scale_factor() const {
        if (std::is_same<T, __nv_fp8_e4m3>::value) {
            return 1.0f / 448.0f;
        } else if (std::is_same<T, __nv_fp8_e5m2>::value) {
            return 1.0f / 57344.0f;
        } else {
            return 1.0f;
        }
    }

    template<typename T>
    TensorInfo& get_tensor_info(const T* tensor_address, size_t size, const void* associated_tensor,
                                float scale_factor, bool &is_new) {
        TensorKey key{tensor_address, associated_tensor, std::type_index(typeid(T)), size};
        auto it = tensor_info_map.find(key);
        is_new = it == tensor_info_map.end();
        if (is_new) {
            grow_storage_if_needed(current_tensor_count + 1);
            size_t newIndex = current_tensor_count++;
            h_current_indices[newIndex] = 0;

            float actualScaleFactor = scale_factor == 0.0f ? get_default_scale_factor<T>() : scale_factor;
            it = tensor_info_map.emplace(key, TensorInfo{newIndex, actualScaleFactor, 0, 0}).first;

            float allValues[ABSMAX_VALUES_COUNT] = {0.0f, 1.0f, 1.0f, actualScaleFactor};
            cudaCheck(cudaMemcpy(d_storage + get_storage_offset(newIndex) + ABSMAX_HISTORY_SIZE,
                                 &allValues, ABSMAX_VALUES_COUNT * sizeof(float), cudaMemcpyHostToDevice));
        } else if (scale_factor != 0.0f && it->second.scale_factor != scale_factor) {
            set_scale_factor(tensor_address, size, associated_tensor, scale_factor);
        }
        return it->second;
    }
};

__global__ void update_all_absmax_kernel(float* data, uint8_t* currentIndices, size_t tensorCount, float fudgeFactor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensorCount) { return; }

    size_t offset = idx * (ABSMAX_HISTORY_SIZE + ABSMAX_VALUES_COUNT);
    uint8_t currentIndex = currentIndices[idx];
    float maxVal = 0.0f;
    for (uint8_t i = 0; i < ABSMAX_HISTORY_SIZE; ++i) {
        maxVal = max(maxVal, data[offset + i]);
    }
    #if SKIP_UPDATE_ZERO_ABSMAX == true
    if (maxVal == 0.0f) {
        data[offset + ABSMAX_HISTORY_SIZE + ABSMAX_OFFSET]  = 0.0f;
        data[offset + ABSMAX_HISTORY_SIZE + SCALE_OFFSET]   = 1.0f;
        data[offset + ABSMAX_HISTORY_SIZE + DESCALE_OFFSET] = 1.0f;
        return;
    }
    #endif

    maxVal = maxVal == 0.0f ? 1.0f : maxVal;
    float scale_factor = data[offset + ABSMAX_HISTORY_SIZE + SCALING_FACTOR_OFFSET];
    data[offset + ABSMAX_HISTORY_SIZE + ABSMAX_OFFSET]  = maxVal;
    data[offset + ABSMAX_HISTORY_SIZE + SCALE_OFFSET]   = 1.0f / (maxVal * fudgeFactor * scale_factor);
    data[offset + ABSMAX_HISTORY_SIZE + DESCALE_OFFSET] = maxVal * fudgeFactor * scale_factor;
    data[offset + currentIndex] = 0.0f;
}

__global__ void update_single_absmax_kernel(float* data, uint8_t* currentIndex, float fudgeFactor) {
    float maxVal = 0.0f;
    for (uint8_t i = 0; i < ABSMAX_HISTORY_SIZE; ++i) {
        maxVal = max(maxVal, data[i]);
    }
    #if SKIP_UPDATE_ZERO_ABSMAX == true
    if (maxVal == 0.0f) {
        data[ABSMAX_HISTORY_SIZE + ABSMAX_OFFSET]  = 0.0f;
        data[ABSMAX_HISTORY_SIZE + SCALE_OFFSET]   = 1.0f;
        data[ABSMAX_HISTORY_SIZE + DESCALE_OFFSET] = 1.0f;
        return;
    }
    #endif

    maxVal = maxVal == 0.0f ? 1.0f : maxVal;
    float scale_factor = data[ABSMAX_HISTORY_SIZE + SCALING_FACTOR_OFFSET];

    data[ABSMAX_HISTORY_SIZE + ABSMAX_OFFSET]  = maxVal;
    data[ABSMAX_HISTORY_SIZE + SCALE_OFFSET]   = 1.0f / (maxVal * fudgeFactor * scale_factor);
    data[ABSMAX_HISTORY_SIZE + DESCALE_OFFSET] = maxVal * fudgeFactor * scale_factor;
}

void TensorAbsMaxTracker::update_all_absmax(cudaStream_t stream, float fudgeFactor) {
    if (current_tensor_count == 0) {
        return;
    }
    cudaCheck(cudaMemcpyAsync(d_current_indices, h_current_indices.data(),
                              current_tensor_count * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));

    int block_size = 128;
    int grid_size = (current_tensor_count + block_size - 1) / block_size;
    update_all_absmax_kernel<<<grid_size, block_size, 0, stream>>>
                            (d_storage, d_current_indices, current_tensor_count, fudgeFactor);
    cudaCheck(cudaGetLastError());
    reset_all_call_counts();  // Reset call counts after updating
}

// this is literally just a single thread, this should not trigger after the 1st step! (outside of testing)
template<typename T>
void TensorAbsMaxTracker::update_single_absmax(const T* tensor_address, size_t size, const void* associated_tensor,
                                               float fudgeFactor, cudaStream_t stream) {
    TensorKey key{tensor_address, associated_tensor, std::type_index(typeid(T)), size};
    auto it = tensor_info_map.find(key);
    if (it == tensor_info_map.end()) {
        throw std::runtime_error("Tensor not registered");
    }

    size_t index = it->second.index;
    size_t offset = get_storage_offset(index);

    update_single_absmax_kernel<<<1, 1, 0, stream>>>(d_storage + offset, d_current_indices + index, fudgeFactor);
    cudaCheck(cudaGetLastError());

    // reset call count after updating
    it->second.previous_call_count = it->second.call_count;
    it->second.call_count = 0;
}

template<typename T>
float* TensorAbsMaxTracker::calculate_manual_absmax(const T* tensor_address, size_t size,
                                                    const void* associated_tensor, cudaStream_t stream) {
    float* absmax_memory = next_absmax_ptr(tensor_address, size, associated_tensor);
    get_absmax(absmax_memory, tensor_address, size, stream);

    // Use a fudge factor of 1.0f because we are using the real absmax rather than a prediction
    update_single_absmax(tensor_address, size, associated_tensor, 1.0f, stream);
    return get_absmax_data(tensor_address, size, associated_tensor);
}

TensorAbsMaxTracker absmax_tracker;
#endif