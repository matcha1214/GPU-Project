#ifndef ABM_KERNELS_H
#define ABM_KERNELS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../utils/types.h"

// Structure of Arrays (SoA) for agent data on GPU
struct AgentDataGPU {
    // Position arrays
    float* x_positions;
    float* y_positions;
    
    // Agent properties
    float* energies;
    int* behavior_states;
    int* ages;
    float* last_food_times;
    int* ids;
    
    // Perception data
    float* perceived_u_local;
    float* perceived_v_local;
    float* perceived_u_avg;
    float* perceived_v_avg;
    
    // Movement planning
    float* move_dx;
    float* move_dy;
    
    // Social awareness
    int* nearby_agents_count;
    float* nearest_agent_distance;
    int* predator_nearby;
    int* food_competition;
    
    // Alive status (for dynamic population management)
    int* alive_flags;
    
    int capacity;      // Maximum number of agents
    int active_count;  // Current number of active agents
};

// GPU ABM functions
extern "C" {
    // Memory management
    void allocateAgentDataGPU(AgentDataGPU* agent_data, int max_agents);
    void freeAgentDataGPU(AgentDataGPU* agent_data);
    void copyAgentDataToDevice(AgentDataGPU* d_agents, const AgentDataGPU* h_agents, int num_agents);
    void copyAgentDataToHost(AgentDataGPU* h_agents, const AgentDataGPU* d_agents, int num_agents);
    
    // Initialization
    void launchInitializeAgentsKernel(AgentDataGPU* agent_data, int num_agents, 
                                      int width, int height, unsigned long seed);
    
    // Agent simulation kernels
    void launchAgentPerceptionKernel(AgentDataGPU* agent_data, const float* u_field, const float* v_field,
                                     int width, int height, float perception_radius);
    
    void launchAgentDecisionKernel(AgentDataGPU* agent_data, int num_agents, float dt);
    
    void launchAgentActionKernel(AgentDataGPU* agent_data, int num_agents, 
                                 int width, int height, float dt);
    
    void launchAgentEnergyUpdateKernel(AgentDataGPU* agent_data, int num_agents, float dt);
    
    // Interaction kernels
    void launchAgentEnvironmentInteractionKernel(AgentDataGPU* agent_data, 
                                                  const float* u_field, const float* v_field,
                                                  const int* biome_map, int width, int height);
    
    // Population management
    void launchMarkDeadAgentsKernel(AgentDataGPU* agent_data, int num_agents);
    int compactAgentPopulation(AgentDataGPU* agent_data);
    
    // Utility functions
    void launchCountNearbyAgentsKernel(AgentDataGPU* agent_data, int num_agents, float radius);
}

// Device utility functions
__device__ float calculateDistance(float x1, float y1, float x2, float y2);
__device__ void updateAgentPerception(int agent_id, const float* u_field, const float* v_field,
                                      float x, float y, int width, int height, float radius,
                                      float* perceived_u_local, float* perceived_v_local,
                                      float* perceived_u_avg, float* perceived_v_avg);

#endif // ABM_KERNELS_H
