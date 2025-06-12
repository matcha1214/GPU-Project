#include "rd_kernels.h"
#include "cuda_utils.h"
#include "../utils/types.h"
#include <curand_kernel.h>

// CUDA kernel for initializing cuRAND states
__global__ void initCurandStates(curandState* states, unsigned long seed, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        // Each thread gets a unique sequence number
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// CUDA kernel for initializing RD fields with random noise
__global__ void initializeRDFields(float* u_field, float* v_field, curandState* states,
                                   int width, int height, float u_default, float v_default,
                                   float noise_strength = 0.01f) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Get random noise for this cell
        float u_noise = (curand_uniform(&states[idx]) - 0.5f) * noise_strength;
        float v_noise = (curand_uniform(&states[idx]) - 0.5f) * noise_strength;
        
        // Initialize with default values plus small random perturbation
        u_field[idx] = u_default + u_noise;
        v_field[idx] = v_default + v_noise;
        
        // Add a small seed patch in the center for pattern initiation
        int center_x = width / 2;
        int center_y = height / 2;
        int dx = x - center_x;
        int dy = y - center_y;
        float dist_sq = dx * dx + dy * dy;
        
        if (dist_sq < 25) { // 5x5 seed region
            v_field[idx] = 0.25f + v_noise;
            u_field[idx] = 0.5f + u_noise;
        }
    }
}

// Basic RD kernel using global memory (for comparison/testing)
__global__ void reactionDiffusionKernelBasic(const float* u_curr, const float* v_curr,
                                              float* u_next, float* v_next,
                                              int width, int height,
                                              float Du, float Dv, float f, float k, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Periodic boundary conditions using modulo
        int left = y * width + ((x - 1 + width) % width);
        int right = y * width + ((x + 1) % width);
        int up = ((y - 1 + height) % height) * width + x;
        int down = ((y + 1) % height) * width + x;
        
        // 5-point stencil for Laplacian (diffusion term)
        float laplacian_u = u_curr[left] + u_curr[right] + u_curr[up] + u_curr[down] - 4.0f * u_curr[idx];
        float laplacian_v = v_curr[left] + v_curr[right] + v_curr[up] + v_curr[down] - 4.0f * v_curr[idx];
        
        // Current concentrations
        float u = u_curr[idx];
        float v = v_curr[idx];
        
        // Gray-Scott reaction terms
        float reaction_term = u * v * v;
        
        // Update equations with explicit Euler integration
        u_next[idx] = u + dt * (Du * laplacian_u - reaction_term + f * (1.0f - u));
        v_next[idx] = v + dt * (Dv * laplacian_v + reaction_term - (k + f) * v);
        
        // Clamp values to prevent numerical instability
        u_next[idx] = fmaxf(0.0f, fminf(1.0f, u_next[idx]));
        v_next[idx] = fmaxf(0.0f, fminf(1.0f, v_next[idx]));
    }
}

// Optimized RD kernel using shared memory tiling
__global__ void reactionDiffusionKernelOptimized(const float* u_curr, const float* v_curr,
                                                  float* u_next, float* v_next,
                                                  int width, int height,
                                                  float Du, float Dv, float f, float k, float dt) {
    // Shared memory tiles with halo regions
    __shared__ float u_tile[GPUConstants::TILE_WITH_HALO_Y][GPUConstants::TILE_WITH_HALO_X + 1]; // +1 to avoid bank conflicts
    __shared__ float v_tile[GPUConstants::TILE_WITH_HALO_Y][GPUConstants::TILE_WITH_HALO_X + 1];
    
    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Load data into shared memory (including halo regions)
    // Each thread loads its own element plus potentially halo elements
    
    // Load center region
    int shared_x = tx + GPUConstants::STENCIL_RADIUS;
    int shared_y = ty + GPUConstants::STENCIL_RADIUS;
    
    if (x < width && y < height) {
        int global_idx = y * width + x;
        u_tile[shared_y][shared_x] = u_curr[global_idx];
        v_tile[shared_y][shared_x] = v_curr[global_idx];
    } else {
        u_tile[shared_y][shared_x] = 0.0f;
        v_tile[shared_y][shared_x] = 0.0f;
    }
    
    // Load halo regions (boundary threads handle multiple loads)
    // Left halo
    if (tx == 0) {
        int halo_x = ((blockIdx.x * blockDim.x - 1) + width) % width;
        if (y < height) {
            int halo_idx = y * width + halo_x;
            u_tile[shared_y][0] = u_curr[halo_idx];
            v_tile[shared_y][0] = v_curr[halo_idx];
        }
    }
    
    // Right halo
    if (tx == blockDim.x - 1) {
        int halo_x = (blockIdx.x * blockDim.x + blockDim.x) % width;
        if (y < height) {
            int halo_idx = y * width + halo_x;
            u_tile[shared_y][GPUConstants::TILE_WITH_HALO_X - 1] = u_curr[halo_idx];
            v_tile[shared_y][GPUConstants::TILE_WITH_HALO_X - 1] = v_curr[halo_idx];
        }
    }
    
    // Top halo
    if (ty == 0) {
        int halo_y = ((blockIdx.y * blockDim.y - 1) + height) % height;
        if (x < width) {
            int halo_idx = halo_y * width + x;
            u_tile[0][shared_x] = u_curr[halo_idx];
            v_tile[0][shared_x] = v_curr[halo_idx];
        }
    }
    
    // Bottom halo
    if (ty == blockDim.y - 1) {
        int halo_y = (blockIdx.y * blockDim.y + blockDim.y) % height;
        if (x < width) {
            int halo_idx = halo_y * width + x;
            u_tile[GPUConstants::TILE_WITH_HALO_Y - 1][shared_x] = u_curr[halo_idx];
            v_tile[GPUConstants::TILE_WITH_HALO_Y - 1][shared_x] = v_curr[halo_idx];
        }
    }
    
    // Synchronize threads to ensure all shared memory loads are complete
    __syncthreads();
    
    // Perform computation using shared memory
    if (x < width && y < height) {
        // 5-point stencil using shared memory
        float u_center = u_tile[shared_y][shared_x];
        float u_left = u_tile[shared_y][shared_x - 1];
        float u_right = u_tile[shared_y][shared_x + 1];
        float u_up = u_tile[shared_y - 1][shared_x];
        float u_down = u_tile[shared_y + 1][shared_x];
        
        float v_center = v_tile[shared_y][shared_x];
        float v_left = v_tile[shared_y][shared_x - 1];
        float v_right = v_tile[shared_y][shared_x + 1];
        float v_up = v_tile[shared_y - 1][shared_x];
        float v_down = v_tile[shared_y + 1][shared_x];
        
        // Compute Laplacians
        float laplacian_u = u_left + u_right + u_up + u_down - 4.0f * u_center;
        float laplacian_v = v_left + v_right + v_up + v_down - 4.0f * v_center;
        
        // Gray-Scott reaction terms
        float reaction_term = u_center * v_center * v_center;
        
        // Update equations
        float new_u = u_center + dt * (Du * laplacian_u - reaction_term + f * (1.0f - u_center));
        float new_v = v_center + dt * (Dv * laplacian_v + reaction_term - (k + f) * v_center);
        
        // Write to global memory with clamping
        int global_idx = y * width + x;
        u_next[global_idx] = fmaxf(0.0f, fminf(1.0f, new_u));
        v_next[global_idx] = fmaxf(0.0f, fminf(1.0f, new_v));
    }
}

// Host function to manage RD simulation step
extern "C" {
    void launchReactionDiffusionKernel(float* u_curr, float* v_curr, float* u_next, float* v_next,
                                       int width, int height, float Du, float Dv, float f, float k, float dt,
                                       bool use_optimized = true) {
        // Calculate grid and block dimensions
        dim3 blockDim = CudaUtils::calculateBlockDim();
        dim3 gridDim = CudaUtils::calculateGridDim(width, height);
        
        if (use_optimized) {
            reactionDiffusionKernelOptimized<<<gridDim, blockDim>>>(
                u_curr, v_curr, u_next, v_next, width, height, Du, Dv, f, k, dt);
        } else {
            reactionDiffusionKernelBasic<<<gridDim, blockDim>>>(
                u_curr, v_curr, u_next, v_next, width, height, Du, Dv, f, k, dt);
        }
        
        CUDA_CHECK_KERNEL();
    }
    
    void launchInitializeRDFields(float* u_field, float* v_field, curandState* states,
                                  int width, int height, float u_default, float v_default,
                                  unsigned long seed) {
        dim3 blockDim = CudaUtils::calculateBlockDim();
        dim3 gridDim = CudaUtils::calculateGridDim(width, height);
        
        // Initialize cuRAND states
        initCurandStates<<<gridDim, blockDim>>>(states, seed, width, height);
        CUDA_CHECK_KERNEL();
        
        // Initialize RD fields
        initializeRDFields<<<gridDim, blockDim>>>(u_field, v_field, states, width, height, 
                                                  u_default, v_default);
        CUDA_CHECK_KERNEL();
    }
}
