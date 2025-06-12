#include "biome_kernels.h"
#include "cuda_utils.h"
#include "../utils/types.h"

// Device function to classify biome based on U and V concentrations
__device__ BiomeType classifyBiome(float u_value, float v_value) {
    // Classification thresholds based on Gray-Scott parameter regimes
    // These can be adjusted based on the specific parameter set being used
    
    if (v_value < 0.1f) {
        if (u_value > 0.8f) {
            return BiomeType::DESERT;      // High U, low V - arid conditions
        } else {
            return BiomeType::GRASSLAND;   // Medium U, low V - sparse vegetation
        }
    } else if (v_value < 0.3f) {
        return BiomeType::GRASSLAND;       // Medium V - grassland
    } else if (v_value < 0.6f) {
        return BiomeType::FOREST;          // High V - dense vegetation/forest
    } else {
        if (u_value < 0.3f) {
            return BiomeType::WATER;       // Very high V, low U - water body
        } else {
            return BiomeType::MOUNTAIN;    // Very high V, high U - rocky/mountainous
        }
    }
}

// Device function to convert biome type to a color index for visualization
__device__ int biomeToColorIndex(BiomeType biome) {
    switch (biome) {
        case BiomeType::DESERT:    return 0;  // Yellow/tan
        case BiomeType::GRASSLAND: return 1;  // Light green
        case BiomeType::FOREST:    return 2;  // Dark green
        case BiomeType::WATER:     return 3;  // Blue
        case BiomeType::MOUNTAIN:  return 4;  // Gray/brown
        default:                   return 5;  // Unknown - black
    }
}

// CUDA kernel for biome classification
__global__ void biomeClassificationKernel(const float* u_field, const float* v_field,
                                           int* biome_map, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        float u_val = u_field[idx];
        float v_val = v_field[idx];
        
        BiomeType biome = classifyBiome(u_val, v_val);
        biome_map[idx] = static_cast<int>(biome);
    }
}

// CUDA kernel for creating visualization with agents overlaid on V field
__global__ void visualizationKernel(const float* v_field, const float* agent_x_positions,
                                     const float* agent_y_positions, const int* agent_states,
                                     float* output_field, int width, int height, int num_agents) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Start with the V field value
        output_field[idx] = v_field[idx];
        
        // Check if any agent is at this location
        for (int i = 0; i < num_agents; i++) {
            int agent_x = static_cast<int>(agent_x_positions[i]);
            int agent_y = static_cast<int>(agent_y_positions[i]);
            
            // If agent is at this grid cell, modify the output value
            if (agent_x == x && agent_y == y) {
                // Color code agents based on their state
                AgentBehaviorState state = static_cast<AgentBehaviorState>(agent_states[i]);
                switch (state) {
                    case AgentBehaviorState::FORAGING:
                        output_field[idx] = 1.2f;  // Bright color for foraging
                        break;
                    case AgentBehaviorState::FLEEING:
                        output_field[idx] = 1.5f;  // Very bright for fleeing
                        break;
                    case AgentBehaviorState::SOCIALIZING:
                        output_field[idx] = 0.9f;  // Moderate color for socializing
                        break;
                    case AgentBehaviorState::MOVING:
                        output_field[idx] = 1.0f;  // Standard bright for moving
                        break;
                    case AgentBehaviorState::IDLE:
                        output_field[idx] = 0.7f;  // Dimmer for idle
                        break;
                    default:
                        output_field[idx] = 0.8f;  // Default agent color
                        break;
                }
                break; // Only show first agent if multiple occupy same cell
            }
        }
    }
}

// Host functions to launch kernels
extern "C" {
    void launchBiomeClassificationKernel(const float* u_field, const float* v_field,
                                          int* biome_map, int width, int height) {
        dim3 blockDim = CudaUtils::calculateBlockDim();
        dim3 gridDim = CudaUtils::calculateGridDim(width, height);
        
        biomeClassificationKernel<<<gridDim, blockDim>>>(u_field, v_field, biome_map, width, height);
        CUDA_CHECK_KERNEL();
    }
    
    void launchVisualizationKernel(const float* v_field, const float* agent_x_positions,
                                   const float* agent_y_positions, const int* agent_states,
                                   float* output_field, int width, int height, int num_agents) {
        dim3 blockDim = CudaUtils::calculateBlockDim();
        dim3 gridDim = CudaUtils::calculateGridDim(width, height);
        
        visualizationKernel<<<gridDim, blockDim>>>(v_field, agent_x_positions, agent_y_positions,
                                                   agent_states, output_field, width, height, num_agents);
        CUDA_CHECK_KERNEL();
    }
}
