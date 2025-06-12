#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <string>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t result = call; \
        if (result != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(result) << std::endl; \
            exit(1); \
        } \
    } while(0)

// CUDA kernel launch error checking
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t result = cudaGetLastError(); \
        if (result != cudaSuccess) { \
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(result) << std::endl; \
            exit(1); \
        } \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)

namespace CudaUtils {
    // Device properties query
    void printDeviceInfo();
    
    // Memory allocation helpers
    void allocateDeviceMemory(float** d_ptr, size_t size);
    void allocateDeviceMemory(int** d_ptr, size_t size);
    void allocateDeviceMemory(curandState** d_ptr, size_t size);
    
    // Memory copy helpers  
    void copyToDevice(float* d_ptr, const float* h_ptr, size_t size);
    void copyToHost(float* h_ptr, const float* d_ptr, size_t size);
    void copyToDevice(int* d_ptr, const int* h_ptr, size_t size);
    void copyToHost(int* h_ptr, const int* d_ptr, size_t size);
    
    // Grid dimension calculations
    dim3 calculateGridDim(int width, int height, int blockSizeX = 16, int blockSizeY = 16);
    dim3 calculateBlockDim(int blockSizeX = 16, int blockSizeY = 16);
    
    // 1D grid calculations for agent processing
    dim3 calculate1DGridDim(int numElements, int blockSize = 256);
    dim3 calculate1DBlockDim(int blockSize = 256);
    
    // GPU memory management
    void freeDeviceMemory(void* d_ptr);
}

// Common GPU constants
namespace GPUConstants {
    constexpr int DEFAULT_BLOCK_SIZE_X = 16;
    constexpr int DEFAULT_BLOCK_SIZE_Y = 16;
    constexpr int DEFAULT_1D_BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    
    // Shared memory tile sizes for RD kernels
    constexpr int TILE_DIM_X = 16;
    constexpr int TILE_DIM_Y = 16;
    constexpr int STENCIL_RADIUS = 1;
    constexpr int TILE_WITH_HALO_X = TILE_DIM_X + 2 * STENCIL_RADIUS;
    constexpr int TILE_WITH_HALO_Y = TILE_DIM_Y + 2 * STENCIL_RADIUS;
}

#endif // CUDA_UTILS_H
