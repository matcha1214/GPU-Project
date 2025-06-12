#include "abm_kernels.h"
#include "biome_kernels.h"
#include "cuda_utils.h"
#include "rd_kernels.h"
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/remove_if.h>
#include <thrust/count.h>

// Device utility functions
__device__ float calculateDistance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

__device__ void updateAgentPerception(int agent_id, const float* u_field, const float* v_field,
                                      float x, float y, int width, int height, float radius,
                                      float* perceived_u_local, float* perceived_v_local,
                                      float* perceived_u_avg, float* perceived_v_avg) {
    int grid_x = static_cast<int>(x);
    int grid_y = static_cast<int>(y);
    
    // Ensure coordinates are within bounds
    grid_x = max(0, min(width - 1, grid_x));
    grid_y = max(0, min(height - 1, grid_y));
    
    int local_idx = grid_y * width + grid_x;
    
    // Local perception (exact position)
    perceived_u_local[agent_id] = u_field[local_idx];
    perceived_v_local[agent_id] = v_field[local_idx];
    
    // Neighborhood average
    float u_sum = 0.0f;
    float v_sum = 0.0f;
    int count = 0;
    
    int radius_int = static_cast<int>(radius) + 1;
    
    for (int dy = -radius_int; dy <= radius_int; dy++) {
        for (int dx = -radius_int; dx <= radius_int; dx++) {
            int nx = grid_x + dx;
            int ny = grid_y + dy;
            
            // Periodic boundary conditions
            nx = (nx + width) % width;
            ny = (ny + height) % height;
            
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist <= radius) {
                int neighbor_idx = ny * width + nx;
                u_sum += u_field[neighbor_idx];
                v_sum += v_field[neighbor_idx];
                count++;
            }
        }
    }
    
    if (count > 0) {
        perceived_u_avg[agent_id] = u_sum / count;
        perceived_v_avg[agent_id] = v_sum / count;
    } else {
        perceived_u_avg[agent_id] = perceived_u_local[agent_id];
        perceived_v_avg[agent_id] = perceived_v_local[agent_id];
    }
}

// Initialize agents with random positions and default values
__global__ void initializeAgentsKernel(AgentDataGPU agent_data, int num_agents, 
                                        int width, int height, curandState* states) {
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (agent_id < num_agents) {
        // Initialize position randomly
        agent_data.x_positions[agent_id] = curand_uniform(&states[agent_id]) * width;
        agent_data.y_positions[agent_id] = curand_uniform(&states[agent_id]) * height;
        
        // Initialize energy and state
        agent_data.energies[agent_id] = Constants::Agent::INITIAL_ENERGY;
        agent_data.behavior_states[agent_id] = static_cast<int>(AgentBehaviorState::IDLE);
        agent_data.ages[agent_id] = 0;
        agent_data.last_food_times[agent_id] = 0.0f;
        agent_data.ids[agent_id] = agent_id;
        
        // Initialize perception
        agent_data.perceived_u_local[agent_id] = 0.0f;
        agent_data.perceived_v_local[agent_id] = 0.0f;
        agent_data.perceived_u_avg[agent_id] = 0.0f;
        agent_data.perceived_v_avg[agent_id] = 0.0f;
        
        // Initialize movement
        agent_data.move_dx[agent_id] = 0.0f;
        agent_data.move_dy[agent_id] = 0.0f;
        
        // Initialize social data
        agent_data.nearby_agents_count[agent_id] = 0;
        agent_data.nearest_agent_distance[agent_id] = Constants::Agent::LARGE_DISTANCE;
        agent_data.predator_nearby[agent_id] = 0;
        agent_data.food_competition[agent_id] = 0;
        
        // Mark as alive
        agent_data.alive_flags[agent_id] = 1;
    }
}

// Agent perception kernel
__global__ void agentPerceptionKernel(AgentDataGPU agent_data, const float* u_field, const float* v_field,
                                      int num_agents, int width, int height, float perception_radius) {
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (agent_id < num_agents && agent_data.alive_flags[agent_id]) {
        float x = agent_data.x_positions[agent_id];
        float y = agent_data.y_positions[agent_id];
        
        updateAgentPerception(agent_id, u_field, v_field, x, y, width, height, perception_radius,
                              agent_data.perceived_u_local, agent_data.perceived_v_local,
                              agent_data.perceived_u_avg, agent_data.perceived_v_avg);
    }
}

// Agent decision-making kernel
__global__ void agentDecisionKernel(AgentDataGPU agent_data, int num_agents, 
                                    curandState* states, float dt) {
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (agent_id < num_agents && agent_data.alive_flags[agent_id]) {
        float energy = agent_data.energies[agent_id];
        float v_local = agent_data.perceived_v_local[agent_id];
        float v_avg = agent_data.perceived_v_avg[agent_id];
        
        AgentBehaviorState new_state = AgentBehaviorState::IDLE;
        float speed_multiplier = Constants::Agent::IDLE_SPEED_MULTIPLIER;
        
        // Decision logic based on energy and environmental conditions
        if (energy < Constants::Agent::LOW_ENERGY_THRESHOLD) {
            // Low energy - need to forage urgently
            if (v_local >= Constants::Agent::FORAGING_V_MIN && v_local <= Constants::Agent::FORAGING_V_MAX) {
                new_state = AgentBehaviorState::FORAGING;
                speed_multiplier = Constants::Agent::ACTIVE_FORAGE_SPEED_MULTIPLIER;
            } else {
                new_state = AgentBehaviorState::MOVING;
                speed_multiplier = Constants::Agent::BASE_SPEED;
            }
        } else if (agent_data.predator_nearby[agent_id]) {
            // Predator nearby - flee!
            new_state = AgentBehaviorState::FLEEING;
            speed_multiplier = Constants::Agent::FLEE_SPEED_MULTIPLIER;
        } else if (energy < Constants::Agent::HUNGRY_THRESHOLD) {
            // Moderately hungry - look for food
            if (v_local >= Constants::Agent::MODERATE_FORAGING_V_MIN && 
                v_local <= Constants::Agent::MODERATE_FORAGING_V_MAX) {
                new_state = AgentBehaviorState::FORAGING;
                speed_multiplier = Constants::Agent::CAREFUL_FORAGE_SPEED_MULTIPLIER;
            } else {
                new_state = AgentBehaviorState::MOVING;
                speed_multiplier = Constants::Agent::BASE_SPEED;
            }
        } else if (agent_data.nearby_agents_count[agent_id] > 0 && 
                   agent_data.nearby_agents_count[agent_id] <= Constants::Agent::MAX_SOCIAL_AGENTS) {
            // Well-fed and other agents nearby - socialize
            new_state = AgentBehaviorState::SOCIALIZING;
            speed_multiplier = Constants::Agent::SOCIAL_SPEED_MULTIPLIER;
        } else {
            // Default behavior - random movement or idle
            if (curand_uniform(&states[agent_id]) < Constants::Agent::MOVEMENT_PROBABILITY) {
                new_state = AgentBehaviorState::MOVING;
                speed_multiplier = Constants::Agent::BASE_SPEED;
            } else {
                new_state = AgentBehaviorState::IDLE;
                speed_multiplier = Constants::Agent::IDLE_SPEED_MULTIPLIER;
            }
        }
        
        agent_data.behavior_states[agent_id] = static_cast<int>(new_state);
        
        // Calculate movement vector based on behavior
        float dx = 0.0f, dy = 0.0f;
        float base_speed = Constants::Agent::BASE_SPEED * speed_multiplier * dt;
        
        switch (new_state) {
            case AgentBehaviorState::FORAGING:
                // Move towards higher V concentrations (gradient following)
                if (v_avg > v_local + 0.01f) {
                    // Move towards average (rough gradient approximation)
                    dx = (curand_uniform(&states[agent_id]) - 0.5f) * base_speed;
                    dy = (curand_uniform(&states[agent_id]) - 0.5f) * base_speed;
                } else {
                    // Local search
                    dx = (curand_uniform(&states[agent_id]) - 0.5f) * base_speed * 0.5f;
                    dy = (curand_uniform(&states[agent_id]) - 0.5f) * base_speed * 0.5f;
                }
                break;
                
            case AgentBehaviorState::FLEEING:
                // Move away from threats (random fast movement for simplicity)
                dx = (curand_uniform(&states[agent_id]) - 0.5f) * base_speed * 2.0f;
                dy = (curand_uniform(&states[agent_id]) - 0.5f) * base_speed * 2.0f;
                break;
                
            case AgentBehaviorState::MOVING:
                // Random movement
                dx = (curand_uniform(&states[agent_id]) - 0.5f) * base_speed;
                dy = (curand_uniform(&states[agent_id]) - 0.5f) * base_speed;
                break;
                
            case AgentBehaviorState::SOCIALIZING:
                // Slow movement towards other agents
                dx = (curand_uniform(&states[agent_id]) - 0.5f) * base_speed * 0.3f;
                dy = (curand_uniform(&states[agent_id]) - 0.5f) * base_speed * 0.3f;
                break;
                
            default: // IDLE
                dx = 0.0f;
                dy = 0.0f;
                break;
        }
        
        agent_data.move_dx[agent_id] = dx;
        agent_data.move_dy[agent_id] = dy;
    }
}

// Agent action (movement) kernel
__global__ void agentActionKernel(AgentDataGPU agent_data, int num_agents, 
                                  int width, int height, float dt) {
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (agent_id < num_agents && agent_data.alive_flags[agent_id]) {
        float new_x = agent_data.x_positions[agent_id] + agent_data.move_dx[agent_id];
        float new_y = agent_data.y_positions[agent_id] + agent_data.move_dy[agent_id];
        
        // Handle boundaries with wrapping
        while (new_x < 0) new_x += width;
        while (new_x >= width) new_x -= width;
        while (new_y < 0) new_y += height;
        while (new_y >= height) new_y -= height;
        
        agent_data.x_positions[agent_id] = new_x;
        agent_data.y_positions[agent_id] = new_y;
        
        // Reset movement vectors
        agent_data.move_dx[agent_id] = 0.0f;
        agent_data.move_dy[agent_id] = 0.0f;
    }
}

// Agent energy and aging update kernel
__global__ void agentEnergyUpdateKernel(AgentDataGPU agent_data, int num_agents, float dt) {
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (agent_id < num_agents && agent_data.alive_flags[agent_id]) {
        AgentBehaviorState state = static_cast<AgentBehaviorState>(agent_data.behavior_states[agent_id]);
        float energy = agent_data.energies[agent_id];
        
        // Base metabolism
        float energy_cost = Constants::Agent::BASE_METABOLISM * dt;
        
        // Behavior-specific energy costs
        switch (state) {
            case AgentBehaviorState::FLEEING:
                energy_cost *= Constants::Agent::FLEE_ENERGY_MULTIPLIER;
                break;
            case AgentBehaviorState::FORAGING:
                energy_cost *= Constants::Agent::FORAGE_ENERGY_MULTIPLIER;
                break;
            case AgentBehaviorState::SOCIALIZING:
                energy_cost *= Constants::Agent::SOCIAL_ENERGY_MULTIPLIER;
                break;
            case AgentBehaviorState::IDLE:
                energy_cost *= Constants::Agent::IDLE_ENERGY_MULTIPLIER;
                break;
            default:
                break;
        }
        
        // Apply energy cost
        energy -= energy_cost;
        
        // Energy gain from foraging
        if (state == AgentBehaviorState::FORAGING) {
            float v_local = agent_data.perceived_v_local[agent_id];
            if (v_local >= Constants::Agent::FORAGING_V_MIN && v_local <= Constants::Agent::FORAGING_V_MAX) {
                energy += Constants::Agent::FORAGING_ENERGY_GAIN_GOOD * dt;
                agent_data.last_food_times[agent_id] = 0.0f; // Reset food timer
            } else if (v_local >= Constants::Agent::MODERATE_FORAGING_V_MIN && 
                       v_local <= Constants::Agent::MODERATE_FORAGING_V_MAX) {
                energy += Constants::Agent::FORAGING_ENERGY_GAIN_MODERATE * dt;
            }
        } else {
            agent_data.last_food_times[agent_id] += dt;
        }
        
        // Clamp energy
        energy = fmaxf(0.0f, fminf(Constants::Agent::MAX_ENERGY, energy));
        agent_data.energies[agent_id] = energy;
        
        // Update age
        agent_data.ages[agent_id]++;
        
        // Check death conditions
        if (energy <= 0.0f || 
            agent_data.ages[agent_id] > Constants::Agent::MAX_AGE ||
            agent_data.last_food_times[agent_id] > Constants::Agent::FOOD_TIMEOUT) {
            agent_data.alive_flags[agent_id] = 0;
            agent_data.behavior_states[agent_id] = static_cast<int>(AgentBehaviorState::DEAD);
        }
    }
}

// Count nearby agents kernel
__global__ void countNearbyAgentsKernel(AgentDataGPU agent_data, int num_agents, float radius) {
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (agent_id < num_agents && agent_data.alive_flags[agent_id]) {
        float my_x = agent_data.x_positions[agent_id];
        float my_y = agent_data.y_positions[agent_id];
        float my_energy = agent_data.energies[agent_id];
        
        int count = 0;
        float nearest_dist = Constants::Agent::LARGE_DISTANCE;
        bool predator_detected = false;
        bool competition_detected = false;
        
        for (int other_id = 0; other_id < num_agents; other_id++) {
            if (other_id != agent_id && agent_data.alive_flags[other_id]) {
                float other_x = agent_data.x_positions[other_id];
                float other_y = agent_data.y_positions[other_id];
                float other_energy = agent_data.energies[other_id];
                
                float dist = calculateDistance(my_x, my_y, other_x, other_y);
                
                if (dist <= radius) {
                    count++;
                    nearest_dist = fminf(nearest_dist, dist);
                    
                    // Check for predators (agents with significantly more energy)
                    if (other_energy > my_energy * Constants::Agent::PREDATOR_ENERGY_RATIO &&
                        dist <= Constants::Agent::PREDATOR_DETECTION_DISTANCE) {
                        predator_detected = true;
                    }
                    
                    // Check for food competition
                    if (dist <= Constants::Agent::COMPETITION_DISTANCE) {
                        competition_detected = true;
                    }
                }
            }
        }
        
        agent_data.nearby_agents_count[agent_id] = count;
        agent_data.nearest_agent_distance[agent_id] = (count > 0) ? nearest_dist : Constants::Agent::LARGE_DISTANCE;
        agent_data.predator_nearby[agent_id] = predator_detected ? 1 : 0;
        agent_data.food_competition[agent_id] = competition_detected ? 1 : 0;
    }
}

// Mark dead agents kernel
__global__ void markDeadAgentsKernel(AgentDataGPU agent_data, int num_agents) {
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (agent_id < num_agents) {
        if (agent_data.energies[agent_id] <= 0.0f || 
            agent_data.ages[agent_id] > Constants::Agent::MAX_AGE) {
            agent_data.alive_flags[agent_id] = 0;
        }
    }
}

// Host functions
extern "C" {

void allocateAgentDataGPU(AgentDataGPU* agent_data, int max_agents) {
    agent_data->capacity = max_agents;
    agent_data->active_count = 0;
    
    CudaUtils::allocateDeviceMemory(&agent_data->x_positions, max_agents);
    CudaUtils::allocateDeviceMemory(&agent_data->y_positions, max_agents);
    CudaUtils::allocateDeviceMemory(&agent_data->energies, max_agents);
    CudaUtils::allocateDeviceMemory(&agent_data->behavior_states, max_agents);
    CudaUtils::allocateDeviceMemory(&agent_data->ages, max_agents);
    CudaUtils::allocateDeviceMemory(&agent_data->last_food_times, max_agents);
    CudaUtils::allocateDeviceMemory(&agent_data->ids, max_agents);
    
    CudaUtils::allocateDeviceMemory(&agent_data->perceived_u_local, max_agents);
    CudaUtils::allocateDeviceMemory(&agent_data->perceived_v_local, max_agents);
    CudaUtils::allocateDeviceMemory(&agent_data->perceived_u_avg, max_agents);
    CudaUtils::allocateDeviceMemory(&agent_data->perceived_v_avg, max_agents);
    
    CudaUtils::allocateDeviceMemory(&agent_data->move_dx, max_agents);
    CudaUtils::allocateDeviceMemory(&agent_data->move_dy, max_agents);
    
    CudaUtils::allocateDeviceMemory(&agent_data->nearby_agents_count, max_agents);
    CudaUtils::allocateDeviceMemory(&agent_data->nearest_agent_distance, max_agents);
    CudaUtils::allocateDeviceMemory(&agent_data->predator_nearby, max_agents);
    CudaUtils::allocateDeviceMemory(&agent_data->food_competition, max_agents);
    
    CudaUtils::allocateDeviceMemory(&agent_data->alive_flags, max_agents);
}

void freeAgentDataGPU(AgentDataGPU* agent_data) {
    CudaUtils::freeDeviceMemory(agent_data->x_positions);
    CudaUtils::freeDeviceMemory(agent_data->y_positions);
    CudaUtils::freeDeviceMemory(agent_data->energies);
    CudaUtils::freeDeviceMemory(agent_data->behavior_states);
    CudaUtils::freeDeviceMemory(agent_data->ages);
    CudaUtils::freeDeviceMemory(agent_data->last_food_times);
    CudaUtils::freeDeviceMemory(agent_data->ids);
    
    CudaUtils::freeDeviceMemory(agent_data->perceived_u_local);
    CudaUtils::freeDeviceMemory(agent_data->perceived_v_local);
    CudaUtils::freeDeviceMemory(agent_data->perceived_u_avg);
    CudaUtils::freeDeviceMemory(agent_data->perceived_v_avg);
    
    CudaUtils::freeDeviceMemory(agent_data->move_dx);
    CudaUtils::freeDeviceMemory(agent_data->move_dy);
    
    CudaUtils::freeDeviceMemory(agent_data->nearby_agents_count);
    CudaUtils::freeDeviceMemory(agent_data->nearest_agent_distance);
    CudaUtils::freeDeviceMemory(agent_data->predator_nearby);
    CudaUtils::freeDeviceMemory(agent_data->food_competition);
    
    CudaUtils::freeDeviceMemory(agent_data->alive_flags);
}

void launchInitializeAgentsKernel(AgentDataGPU* agent_data, int num_agents, 
                                  int width, int height, unsigned long seed) {
    // Allocate and initialize cuRAND states
    curandState* states;
    CudaUtils::allocateDeviceMemory(&states, num_agents);
    
    dim3 blockDim = CudaUtils::calculate1DBlockDim();
    dim3 gridDim = CudaUtils::calculate1DGridDim(num_agents);
    
    // Initialize cuRAND states
    initCurandStates<<<gridDim, blockDim>>>(states, seed, num_agents, 1);
    CUDA_CHECK_KERNEL();
    
    // Initialize agents
    initializeAgentsKernel<<<gridDim, blockDim>>>(*agent_data, num_agents, width, height, states);
    CUDA_CHECK_KERNEL();
    
    agent_data->active_count = num_agents;
    
    CudaUtils::freeDeviceMemory(states);
}

void launchAgentPerceptionKernel(AgentDataGPU* agent_data, const float* u_field, const float* v_field,
                                 int width, int height, float perception_radius) {
    dim3 blockDim = CudaUtils::calculate1DBlockDim();
    dim3 gridDim = CudaUtils::calculate1DGridDim(agent_data->active_count);
    
    agentPerceptionKernel<<<gridDim, blockDim>>>(*agent_data, u_field, v_field, 
                                                 agent_data->active_count, width, height, perception_radius);
    CUDA_CHECK_KERNEL();
}

void launchAgentDecisionKernel(AgentDataGPU* agent_data, int num_agents, float dt) {
    // Need cuRAND states for decision making
    curandState* states;
    CudaUtils::allocateDeviceMemory(&states, num_agents);
    
    dim3 blockDim = CudaUtils::calculate1DBlockDim();
    dim3 gridDim = CudaUtils::calculate1DGridDim(num_agents);
    
    // Initialize cuRAND states
    initCurandStates<<<gridDim, blockDim>>>(states, time(nullptr), num_agents, 1);
    CUDA_CHECK_KERNEL();
    
    agentDecisionKernel<<<gridDim, blockDim>>>(*agent_data, num_agents, states, dt);
    CUDA_CHECK_KERNEL();
    
    CudaUtils::freeDeviceMemory(states);
}

void launchAgentActionKernel(AgentDataGPU* agent_data, int num_agents, 
                             int width, int height, float dt) {
    dim3 blockDim = CudaUtils::calculate1DBlockDim();
    dim3 gridDim = CudaUtils::calculate1DGridDim(num_agents);
    
    agentActionKernel<<<gridDim, blockDim>>>(*agent_data, num_agents, width, height, dt);
    CUDA_CHECK_KERNEL();
}

void launchAgentEnergyUpdateKernel(AgentDataGPU* agent_data, int num_agents, float dt) {
    dim3 blockDim = CudaUtils::calculate1DBlockDim();
    dim3 gridDim = CudaUtils::calculate1DGridDim(num_agents);
    
    agentEnergyUpdateKernel<<<gridDim, blockDim>>>(*agent_data, num_agents, dt);
    CUDA_CHECK_KERNEL();
}

void launchCountNearbyAgentsKernel(AgentDataGPU* agent_data, int num_agents, float radius) {
    dim3 blockDim = CudaUtils::calculate1DBlockDim();
    dim3 gridDim = CudaUtils::calculate1DGridDim(num_agents);
    
    countNearbyAgentsKernel<<<gridDim, blockDim>>>(*agent_data, num_agents, radius);
    CUDA_CHECK_KERNEL();
}

void launchMarkDeadAgentsKernel(AgentDataGPU* agent_data, int num_agents) {
    dim3 blockDim = CudaUtils::calculate1DBlockDim();
    dim3 gridDim = CudaUtils::calculate1DGridDim(num_agents);
    
    markDeadAgentsKernel<<<gridDim, blockDim>>>(*agent_data, num_agents);
    CUDA_CHECK_KERNEL();
}

int compactAgentPopulation(AgentDataGPU* agent_data) {
    // Use Thrust for stream compaction
    thrust::device_ptr<int> alive_ptr(agent_data->alive_flags);
    int new_count = thrust::count(alive_ptr, alive_ptr + agent_data->active_count, 1);
    
    // Update active count
    agent_data->active_count = new_count;
    return new_count;
}

} // extern "C"
