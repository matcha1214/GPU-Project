#ifndef RD_KERNELS_H
#define RD_KERNELS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Host functions for reaction-diffusion kernels
extern "C" {
    // Launch reaction-diffusion simulation step
    void launchReactionDiffusionKernel(float* u_curr, float* v_curr, float* u_next, float* v_next,
                                       int width, int height, float Du, float Dv, float f, float k, float dt,
                                       bool use_optimized = true);
    
    // Initialize RD fields with random noise
    void launchInitializeRDFields(float* u_field, float* v_field, curandState* states,
                                  int width, int height, float u_default, float v_default,
                                  unsigned long seed);
}

// Device functions for cuRAND initialization
__global__ void initCurandStates(curandState* states, unsigned long seed, int width, int height);

#endif // RD_KERNELS_H 