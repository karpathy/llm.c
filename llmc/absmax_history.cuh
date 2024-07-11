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

constexpr uint8_t ABSMAX_HISTORY_SIZE = 16;
#define ALWAYS_UPDATE_ABSMAX false // very inefficient, for sanity checking only
#define SKIP_UPDATE_ZERO_ABSMAX true // do not update the scale/descale if max absmax is 0

// Offsets for calculated values
enum CalculatedValueOffset {
    ABSMAX_OFFSET = 0,
    SCALE_OFFSET = 1,
    DESCALE_OFFSET = 2,
    SCALING_FACTOR_OFFSET = 3,
    ABSMAX_VALUES_COUNT = 4
};
const float absmax_two_ones[2] = {1.0f, 1.0f}; // global so memory stays allocated for cudaMemcpyAsync

class TensorAbsMaxTracker {
public:
    static constexpr size_t INITIAL_TENSOR_COUNT = 1024;
    static constexpr float  DEFAULT_FUDGE_FACTOR = 1.0f;

    TensorAbsMaxTracker()
        : currentTensorCount(0), maxTensorCount(0), d_storage(nullptr), d_currentIndices(nullptr) {
        growStorageIfNeeded(INITIAL_TENSOR_COUNT);
    }

    ~TensorAbsMaxTracker() {
        if (d_storage) cudaCheck(cudaFree(d_storage));
        if (d_currentIndices) cudaCheck(cudaFree(d_currentIndices));
    }

    template<typename T>
    float* getCalculatedValuesPtr(const T* tensorAddress, size_t size,
                                  const void* associatedTensor = nullptr,
                                  float scaleFactor = 0.0f, bool calculateIfNeeded=true) {
        bool is_new;
        TensorInfo& info = getOrCreateTensorInfo(tensorAddress, size, associatedTensor, scaleFactor, is_new);

        #if ALWAYS_UPDATE_ABSMAX == true // for sanity checking only
        if (calculateIfNeeded) {
            float* absmax_memory = getNextAbsMaxPtr(tensorAddress, size, associatedTensor);
            get_absmax(absmax_memory, tensorAddress, size);
            updateSingleTensorAbsMax(tensorAddress, size, associatedTensor, 1.0f);
            return (float*)(d_storage + getStorageOffset(info.index) + ABSMAX_HISTORY_SIZE);
        }
        #endif

        if (is_new) {
            if (calculateIfNeeded) {
                return calculateManualAbsMax(tensorAddress, size, associatedTensor);
            }
        }
        return (float*)(d_storage + getStorageOffset(info.index) + ABSMAX_HISTORY_SIZE);
    }

    template<typename T>
    float* getNextAbsMaxPtr(const T* tensorAddress, size_t size,
                            const void* associatedTensor = nullptr, float scaleFactor = 0.0f) {
        bool is_new;
        TensorInfo& info = getOrCreateTensorInfo(tensorAddress, size, associatedTensor, scaleFactor, is_new);
        uint8_t currentIndex = h_currentIndices[info.index];
        h_currentIndices[info.index] = (currentIndex + 1) % ABSMAX_HISTORY_SIZE;
        return (float*)(d_storage + getStorageOffset(info.index) + currentIndex);
    }

    template<typename T>
    void setScaleFactor(const T* tensorAddress, size_t size, const void* associatedTensor, float scaleFactor) {
        TensorKey key{tensorAddress, associatedTensor, std::type_index(typeid(T)), size};
        auto it = tensorInfoMap.find(key);
        if (it == tensorInfoMap.end()) {
            throw std::runtime_error("Tensor not registered");
        }
        it->second.scaleFactor = scaleFactor;
        size_t offset = getStorageOffset(it->second.index) + ABSMAX_HISTORY_SIZE + SCALING_FACTOR_OFFSET;
        cudaCheck(cudaMemcpy(d_storage + offset, &scaleFactor, sizeof(float), cudaMemcpyHostToDevice));
    }

    void updateAbsMax(cudaStream_t stream = 0, float fudgeFactor = DEFAULT_FUDGE_FACTOR);

    template<typename T>
    void updateSingleTensorAbsMax(const T* tensorAddress, size_t size, const void* associatedTensor,
                                  float fudgeFactor = DEFAULT_FUDGE_FACTOR, cudaStream_t stream = 0);

    template<typename T>
    float* calculateManualAbsMax(const T* tensorAddress, size_t size, const void* associatedTensor, cudaStream_t stream = 0);

    void printAllTensorInfo() {
        std::vector<float> hostData(currentTensorCount * (ABSMAX_HISTORY_SIZE + ABSMAX_VALUES_COUNT));
        cudaCheck(cudaMemcpy(hostData.data(), d_storage,
                             currentTensorCount * (ABSMAX_HISTORY_SIZE + ABSMAX_VALUES_COUNT) * sizeof(float),
                             cudaMemcpyDeviceToHost));

        std::cout << "Tensor Address,Associated Tensor,Type,Size,AbsMax,Scale,Descale,Scale Factor,Current Index";
        for (uint8_t i = 0; i < ABSMAX_HISTORY_SIZE; ++i) {
            std::cout << ",History" << static_cast<int>(i);
        }
        std::cout << std::endl;

        for (const auto& pair : tensorInfoMap) {
            size_t offset = getStorageOffset(pair.second.index);
            std::cout << pair.first.address << ","
                      << pair.first.associatedTensor << ","
                      << pair.first.type.name() << ","
                      << pair.first.size << ","
                      << hostData[offset + ABSMAX_HISTORY_SIZE + ABSMAX_OFFSET] << ","
                      << hostData[offset + ABSMAX_HISTORY_SIZE + SCALE_OFFSET] << ","
                      << hostData[offset + ABSMAX_HISTORY_SIZE + DESCALE_OFFSET] << ","
                      << hostData[offset + ABSMAX_HISTORY_SIZE + SCALING_FACTOR_OFFSET] << ","
                      << static_cast<int>(h_currentIndices[pair.second.index]) << ",";
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
        const void* associatedTensor;
        std::type_index type;
        size_t size;

        bool operator==(const TensorKey& other) const {
            return address == other.address &&
                   associatedTensor == other.associatedTensor &&
                   type == other.type && size == other.size;
        }
    };

    struct TensorKeyHash {
        std::size_t operator()(const TensorKey& k) const {
            return std::hash<const void*>()(k.address) ^
                   std::hash<const void*>()(k.associatedTensor) ^
                   std::hash<std::type_index>()(k.type) ^ std::hash<size_t>()(k.size);
        }
    };

    struct TensorInfo {
        size_t index;
        float scaleFactor;
    };

    size_t currentTensorCount;
    size_t maxTensorCount;
    float* d_storage;
    uint8_t* d_currentIndices;
    std::unordered_map<TensorKey, TensorInfo, TensorKeyHash> tensorInfoMap;
    std::vector<uint8_t> h_currentIndices;

    void growStorageIfNeeded(size_t newTensorCount) {
        if (newTensorCount <= maxTensorCount) return;

        size_t newMaxTensorCount = maxTensorCount == 0 ? INITIAL_TENSOR_COUNT : maxTensorCount * 2;
        while (newMaxTensorCount < newTensorCount) {
            newMaxTensorCount *= 2;
        }

        size_t newSize = newMaxTensorCount * (ABSMAX_HISTORY_SIZE + ABSMAX_VALUES_COUNT) * sizeof(float);
        float* new_storage;
        uint8_t* new_indices;
        cudaCheck(cudaMalloc(&new_storage, newSize));
        cudaCheck(cudaMemset(new_storage, 0, newSize));
        cudaCheck(cudaMalloc(&new_indices, newMaxTensorCount * sizeof(uint8_t)));
        cudaCheck(cudaMemset(new_indices, 0, newMaxTensorCount * sizeof(uint8_t)));
        if (d_storage) {
            cudaCheck(cudaMemcpy(new_storage, d_storage,
                                 maxTensorCount * (ABSMAX_HISTORY_SIZE + ABSMAX_VALUES_COUNT) * sizeof(float),
                                 cudaMemcpyDeviceToDevice));
            cudaCheck(cudaMemcpy(new_indices, d_currentIndices, maxTensorCount * sizeof(uint8_t),
                                 cudaMemcpyDeviceToDevice));
            cudaCheck(cudaFree(d_storage));
            cudaCheck(cudaFree(d_currentIndices));
            std::cout << "Storage grown to accommodate " << newMaxTensorCount << " tensors." << std::endl;
        }
        d_storage = new_storage;
        d_currentIndices = new_indices;
        h_currentIndices.resize(newMaxTensorCount);
        maxTensorCount = newMaxTensorCount;
        cudaCheck(cudaGetLastError());
    }

    size_t getStorageOffset(size_t index) const {
        return index * (ABSMAX_HISTORY_SIZE + ABSMAX_VALUES_COUNT);
    }

    template<typename T>
    float getDefaultScaleFactor() const {
        if (std::is_same<T, __nv_fp8_e4m3>::value) {
            return 1.0f / 448.0f;
        } else if (std::is_same<T, __nv_fp8_e5m2>::value) {
            return 1.0f / 57344.0f;
        } else {
            return 1.0f;
        }
    }

    template<typename T>
    TensorInfo& getOrCreateTensorInfo(const T* tensorAddress, size_t size, const void* associatedTensor,
                                      float scaleFactor, bool &is_new) {
        TensorKey key{tensorAddress, associatedTensor, std::type_index(typeid(T)), size};
        auto it = tensorInfoMap.find(key);
        is_new = it == tensorInfoMap.end();
        if (is_new) {
            growStorageIfNeeded(currentTensorCount + 1);
            size_t newIndex = currentTensorCount++;
            h_currentIndices[newIndex] = 0;

            float actualScaleFactor = scaleFactor == 0.0f ? getDefaultScaleFactor<T>() : scaleFactor;
            it = tensorInfoMap.emplace(key, TensorInfo{newIndex, actualScaleFactor}).first;

            float allValues[ABSMAX_VALUES_COUNT] = {0.0f, 1.0f, 1.0f, actualScaleFactor};
            cudaCheck(cudaMemcpy(d_storage + getStorageOffset(newIndex) + ABSMAX_HISTORY_SIZE,
                                 &allValues, ABSMAX_VALUES_COUNT * sizeof(float), cudaMemcpyHostToDevice));
        } else if (scaleFactor != 0.0f && it->second.scaleFactor != scaleFactor) {
            setScaleFactor(tensorAddress, size, associatedTensor, scaleFactor);
        }
        return it->second;
    }
};

__global__ void updateAbsMaxKernel(float* data, uint8_t* currentIndices, size_t tensorCount, float fudgeFactor) {
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
    float scaleFactor = data[offset + ABSMAX_HISTORY_SIZE + SCALING_FACTOR_OFFSET];
    data[offset + ABSMAX_HISTORY_SIZE + ABSMAX_OFFSET]  = maxVal;
    data[offset + ABSMAX_HISTORY_SIZE + SCALE_OFFSET]   = 1.0f / (maxVal * fudgeFactor * scaleFactor);
    data[offset + ABSMAX_HISTORY_SIZE + DESCALE_OFFSET] = maxVal * fudgeFactor * scaleFactor;
    data[offset + currentIndex] = 0.0f;
}

__global__ void updateSingleTensorAbsMaxKernel(float* data, uint8_t* currentIndex, float fudgeFactor) {
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
    float scaleFactor = data[ABSMAX_HISTORY_SIZE + SCALING_FACTOR_OFFSET];

    data[ABSMAX_HISTORY_SIZE + ABSMAX_OFFSET]  = maxVal;
    data[ABSMAX_HISTORY_SIZE + SCALE_OFFSET]   = 1.0f / (maxVal * fudgeFactor * scaleFactor);
    data[ABSMAX_HISTORY_SIZE + DESCALE_OFFSET] = maxVal * fudgeFactor * scaleFactor;
}

void TensorAbsMaxTracker::updateAbsMax(cudaStream_t stream, float fudgeFactor) {
    if (currentTensorCount == 0) {
        return;
    }
    cudaCheck(cudaMemcpyAsync(d_currentIndices, h_currentIndices.data(),
                              currentTensorCount * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));

    int threadsPerBlock = 128;
    int blocksPerGrid = (currentTensorCount + threadsPerBlock - 1) / threadsPerBlock;
    updateAbsMaxKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>
                     (d_storage, d_currentIndices, currentTensorCount, fudgeFactor);
    cudaCheck(cudaGetLastError());
}

template<typename T>
void TensorAbsMaxTracker::updateSingleTensorAbsMax(const T* tensorAddress, size_t size, const void* associatedTensor,
                                                   float fudgeFactor, cudaStream_t stream) {
    TensorKey key{tensorAddress, associatedTensor, std::type_index(typeid(T)), size};
    auto it = tensorInfoMap.find(key);
    if (it == tensorInfoMap.end()) {
        throw std::runtime_error("Tensor not registered");
    }

    size_t index = it->second.index;
    size_t offset = getStorageOffset(index);

    updateSingleTensorAbsMaxKernel<<<1, 1, 0, stream>>>(d_storage + offset, d_currentIndices + index, fudgeFactor);
    cudaCheck(cudaGetLastError());
}

template<typename T>
float* TensorAbsMaxTracker::calculateManualAbsMax(const T* tensorAddress, size_t size,
                                                  const void* associatedTensor, cudaStream_t stream) {
    float* absmax_memory = getNextAbsMaxPtr(tensorAddress, size, associatedTensor);
    get_absmax(absmax_memory, tensorAddress, size, stream);

    // Use a fudge factor of 1.0f because we are using the real absmax rather than a prediction
    updateSingleTensorAbsMax(tensorAddress, size, associatedTensor, 1.0f, stream);
    return getCalculatedValuesPtr(tensorAddress, size, associatedTensor);
}

TensorAbsMaxTracker absmax_tracker;
#endif