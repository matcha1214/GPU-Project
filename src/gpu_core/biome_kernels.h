#ifndef BIOME_KERNELS_H
#define BIOME_KERNELS_H

#include <cuda_runtime.h>
#include "../utils/types.h"

// GPU biome classification functions
extern "C" {
    // Launch biome classification kernel
    void launchBiomeClassificationKernel(const float* u_field, const float* v_field, 
                                         int* biome_map, int width, int height);
    
    // Launch visualization kernel that overlays agents on biome map
    void launchVisualizationKernel(const float* v_field, const float* agent_x_positions,
                                   const float* agent_y_positions, const int* agent_states,
                                   float* output_field, int width, int height, int num_agents);
}

// Device functions for biome classification (called from other kernels)
__device__ BiomeType classifyBiome(float u_value, float v_value);
__device__ int biomeToColorIndex(BiomeType biome);

#endif // BIOME_KERNELS_H
