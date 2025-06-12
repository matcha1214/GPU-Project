#ifndef GPU_SIMULATION_MANAGER_H
#define GPU_SIMULATION_MANAGER_H

#include "rd_kernels.h"
#include "biome_kernels.h"
#include "abm_kernels.h"
#include "cuda_utils.h"
#include "../utils/types.h"
#include <memory>

/**
 * GPU Simulation Manager - Orchestrates the entire GPU-accelerated simulation
 * 
 * This class implements the heterogeneous computing model described in the guide,
 * where CPU manages control flow and GPU handles data-parallel computations.
 * 
 * Design follows APOD cycle:
 * - Assess: Profile and identify bottlenecks
 * - Parallelize: Map RD and ABM to GPU kernels
 * - Optimize: Use shared memory, double buffering, SoA layouts
 * - Deploy: Integrate components with minimal CPU-GPU transfers
 */
class GPUSimulationManager {
public:
    // Constructor with simulation parameters
    GPUSimulationManager(const SimulationParams& params);
    
    // Destructor to clean up GPU memory
    ~GPUSimulationManager();
    
    // Initialize simulation state on GPU
    void initializeSimulation();
    
    // Run a single simulation step (RD -> Classification -> ABM)
    void runSimulationStep();
    
    // Run multiple steps with periodic output
    void runSimulation(int num_steps, int output_frequency);
    
    // Performance monitoring
    void enableProfiling(bool enable) { profiling_enabled_ = enable; }
    void printPerformanceStats() const;
    
    // Data access for visualization (copies from GPU to CPU)
    void copyFieldsToHost(std::vector<float>& u_field, std::vector<float>& v_field) const;
    void copyBiomeMapToHost(std::vector<int>& biome_map) const;
    void copyAgentDataToHost(std::vector<float>& x_pos, std::vector<float>& y_pos, 
                             std::vector<int>& states, std::vector<float>& energies) const;
    
    // Utility functions
    void swapRDBuffers();
    int getActiveAgentCount() const { return agent_data_gpu_.active_count; }
    void compactAgentPopulation();
    
private:
    // Simulation parameters
    SimulationParams params_;
    
    // GPU memory for reaction-diffusion
    float* d_u_current_;
    float* d_v_current_;
    float* d_u_next_;
    float* d_v_next_;
    curandState* d_rd_states_;
    
    // GPU memory for biome classification
    int* d_biome_map_;
    float* d_visualization_field_;
    
    // GPU memory for agents (Structure of Arrays)
    AgentDataGPU agent_data_gpu_;
    curandState* d_agent_states_;
    
    // Performance monitoring
    bool profiling_enabled_;
    mutable float rd_kernel_time_;
    mutable float biome_kernel_time_;
    mutable float abm_kernel_time_;
    mutable float memory_transfer_time_;
    
    // CUDA events for timing
    cudaEvent_t start_event_, stop_event_;
    
    // Initialization helpers
    void allocateGPUMemory();
    void initializeRDFields();
    void initializeAgents();
    
    // Kernel launch wrappers with timing
    void launchRDKernel();
    void launchBiomeClassificationKernel();
    void launchABMKernels();
    
    // Utility methods
    void startTiming();
    float stopTiming();
    size_t getGridSize() const { return params_.width * params_.height; }
    
    // Prevent copying (RAII for GPU resources)
    GPUSimulationManager(const GPUSimulationManager&) = delete;
    GPUSimulationManager& operator=(const GPUSimulationManager&) = delete;
};

// Factory function for creating GPU simulation manager
std::unique_ptr<GPUSimulationManager> createGPUSimulation(const SimulationParams& params);

// Performance comparison utilities
struct PerformanceReport {
    float rd_speedup;
    float abm_speedup;
    float total_speedup;
    float memory_overhead_percent;
    int grid_size;
    int agent_count;
};

PerformanceReport compareGPUvsCPU(const SimulationParams& params, int test_steps = 100);

#endif // GPU_SIMULATION_MANAGER_H 