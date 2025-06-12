#include "cuda_utils.h"
#include <iostream>

namespace CudaUtils {

void printDeviceInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return;
    }
    
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        
        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
        std::cout << "  Memory clock rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
    }
}

void allocateDeviceMemory(float** d_ptr, size_t size) {
    CUDA_CHECK(cudaMalloc(d_ptr, size * sizeof(float)));
}

void allocateDeviceMemory(int** d_ptr, size_t size) {
    CUDA_CHECK(cudaMalloc(d_ptr, size * sizeof(int)));
}

void allocateDeviceMemory(curandState** d_ptr, size_t size) {
    CUDA_CHECK(cudaMalloc(d_ptr, size * sizeof(curandState)));
}

void copyToDevice(float* d_ptr, const float* h_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size * sizeof(float), cudaMemcpyHostToDevice));
}

void copyToHost(float* h_ptr, const float* d_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost));
}

void copyToDevice(int* d_ptr, const int* h_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size * sizeof(int), cudaMemcpyHostToDevice));
}

void copyToHost(int* h_ptr, const int* d_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, size * sizeof(int), cudaMemcpyDeviceToHost));
}

dim3 calculateGridDim(int width, int height, int blockSizeX, int blockSizeY) {
    int gridX = (width + blockSizeX - 1) / blockSizeX;
    int gridY = (height + blockSizeY - 1) / blockSizeY;
    return dim3(gridX, gridY);
}

dim3 calculateBlockDim(int blockSizeX, int blockSizeY) {
    return dim3(blockSizeX, blockSizeY);
}

dim3 calculate1DGridDim(int numElements, int blockSize) {
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    return dim3(numBlocks);
}

dim3 calculate1DBlockDim(int blockSize) {
    return dim3(blockSize);
}

void freeDeviceMemory(void* d_ptr) {
    if (d_ptr != nullptr) {
        CUDA_CHECK(cudaFree(d_ptr));
    }
}

} // namespace CudaUtils
