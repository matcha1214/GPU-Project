#include "gpu_simulation_manager.h"
#include <iostream>
#include <stdexcept>
#include <ctime>

GPUSimulationManager::GPUSimulationManager(const SimulationParams& params)
    : params_(params), profiling_enabled_(false), 
      rd_kernel_time_(0.0f), biome_kernel_time_(0.0f), 
      abm_kernel_time_(0.0f), memory_transfer_time_(0.0f) {
    
    // Initialize CUDA events for timing
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
    
    // Initialize GPU memory pointers to nullptr
    d_u_current_ = nullptr;
    d_v_current_ = nullptr;
    d_u_next_ = nullptr;
    d_v_next_ = nullptr;
    d_rd_states_ = nullptr;
    d_biome_map_ = nullptr;
    d_visualization_field_ = nullptr;
    d_agent_states_ = nullptr;
    
    // Initialize agent data structure
    agent_data_gpu_.capacity = params_.num_agents;
    agent_data_gpu_.active_count = params_.num_agents;
}

GPUSimulationManager::~GPUSimulationManager() {
    // Free GPU memory
    CudaUtils::freeDeviceMemory(d_u_current_);
    CudaUtils::freeDeviceMemory(d_v_current_);
    CudaUtils::freeDeviceMemory(d_u_next_);
    CudaUtils::freeDeviceMemory(d_v_next_);
    CudaUtils::freeDeviceMemory(d_rd_states_);
    CudaUtils::freeDeviceMemory(d_biome_map_);
    CudaUtils::freeDeviceMemory(d_visualization_field_);
    CudaUtils::freeDeviceMemory(d_agent_states_);
    
    // Free agent data
    freeAgentDataGPU(&agent_data_gpu_);
    
    // Destroy CUDA events
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

void GPUSimulationManager::initializeSimulation() {
    allocateGPUMemory();
    initializeRDFields();
    initializeAgents();
}

void GPUSimulationManager::allocateGPUMemory() {
    size_t grid_size = getGridSize();
    
    // Allocate RD fields
    CudaUtils::allocateDeviceMemory(&d_u_current_, grid_size);
    CudaUtils::allocateDeviceMemory(&d_v_current_, grid_size);
    CudaUtils::allocateDeviceMemory(&d_u_next_, grid_size);
    CudaUtils::allocateDeviceMemory(&d_v_next_, grid_size);
    CudaUtils::allocateDeviceMemory(&d_rd_states_, grid_size);
    
    // Allocate biome and visualization fields
    CudaUtils::allocateDeviceMemory(&d_biome_map_, grid_size);
    CudaUtils::allocateDeviceMemory(&d_visualization_field_, grid_size);
    
    // Allocate agent data
    allocateAgentDataGPU(&agent_data_gpu_, params_.num_agents);
    CudaUtils::allocateDeviceMemory(&d_agent_states_, params_.num_agents);
}

void GPUSimulationManager::initializeRDFields() {
    // Initialize RD fields with default values and random perturbations
    launchInitializeRDFields(d_u_current_, d_v_current_, d_rd_states_,
                              params_.width, params_.height, 1.0f, 0.0f,
                              static_cast<unsigned long>(time(nullptr)));
    
    // Copy initial state to next buffers
    size_t grid_size = getGridSize();
    CUDA_CHECK(cudaMemcpy(d_u_next_, d_u_current_, grid_size * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_next_, d_v_current_, grid_size * sizeof(float), cudaMemcpyDeviceToDevice));
}

void GPUSimulationManager::initializeAgents() {
    // Initialize agents on GPU
    launchInitializeAgentsKernel(&agent_data_gpu_, params_.num_agents,
                                 params_.width, params_.height,
                                 static_cast<unsigned long>(time(nullptr)) + 12345);
}

void GPUSimulationManager::runSimulationStep() {
    if (profiling_enabled_) startTiming();
    
    // Step 1: Update reaction-diffusion system
    launchRDKernel();
    if (profiling_enabled_) rd_kernel_time_ += stopTiming();
    
    // Step 2: Classify biomes based on RD output
    if (profiling_enabled_) startTiming();
    launchBiomeClassificationKernel();
    if (profiling_enabled_) biome_kernel_time_ += stopTiming();
    
    // Step 3: Update agents
    if (profiling_enabled_) startTiming();
    launchABMKernels();
    if (profiling_enabled_) abm_kernel_time_ += stopTiming();
    
    // Swap RD buffers for next iteration
    swapRDBuffers();
}

void GPUSimulationManager::launchRDKernel() {
    launchReactionDiffusionKernel(d_u_current_, d_v_current_, d_u_next_, d_v_next_,
                                  params_.width, params_.height,
                                  static_cast<float>(params_.Du), static_cast<float>(params_.Dv),
                                  static_cast<float>(params_.f), static_cast<float>(params_.k),
                                  static_cast<float>(params_.dt), true);
}

void GPUSimulationManager::launchBiomeClassificationKernel() {
    ::launchBiomeClassificationKernel(d_u_current_, d_v_current_, d_biome_map_,
                                      params_.width, params_.height);
}

void GPUSimulationManager::launchABMKernels() {
    // Update agent perception
    launchAgentPerceptionKernel(&agent_data_gpu_, d_u_current_, d_v_current_,
                                params_.width, params_.height,
                                static_cast<float>(params_.agent_perception_radius));
    
    // Update agent decisions
    launchAgentDecisionKernel(&agent_data_gpu_, agent_data_gpu_.active_count,
                              static_cast<float>(params_.dt));
    
    // Execute agent actions
    launchAgentActionKernel(&agent_data_gpu_, agent_data_gpu_.active_count,
                            params_.width, params_.height,
                            static_cast<float>(params_.dt));
    
    // Update agent energy
    launchAgentEnergyUpdateKernel(&agent_data_gpu_, agent_data_gpu_.active_count,
                                  static_cast<float>(params_.dt));
    
    // Handle agent-environment interactions
    launchAgentEnvironmentInteractionKernel(&agent_data_gpu_, d_u_current_, d_v_current_,
                                            d_biome_map_, params_.width, params_.height);
}

void GPUSimulationManager::runSimulation(int num_steps, int output_frequency) {
    for (int step = 0; step < num_steps; ++step) {
        runSimulationStep();
        
        if (step % output_frequency == 0) {
            std::cout << "Completed step " << step << "/" << num_steps << std::endl;
        }
    }
}

void GPUSimulationManager::swapRDBuffers() {
    std::swap(d_u_current_, d_u_next_);
    std::swap(d_v_current_, d_v_next_);
}

void GPUSimulationManager::copyFieldsToHost(std::vector<float>& u_field, std::vector<float>& v_field) const {
    size_t grid_size = getGridSize();
    
    if (profiling_enabled_) startTiming();
    CudaUtils::copyToHost(u_field.data(), d_u_current_, grid_size);
    CudaUtils::copyToHost(v_field.data(), d_v_current_, grid_size);
    if (profiling_enabled_) memory_transfer_time_ += stopTiming();
}

void GPUSimulationManager::copyBiomeMapToHost(std::vector<int>& biome_map) const {
    size_t grid_size = getGridSize();
    
    if (profiling_enabled_) startTiming();
    CudaUtils::copyToHost(biome_map.data(), d_biome_map_, grid_size);
    if (profiling_enabled_) memory_transfer_time_ += stopTiming();
}

void GPUSimulationManager::copyAgentDataToHost(std::vector<float>& x_pos, std::vector<float>& y_pos,
                                               std::vector<int>& states, std::vector<float>& energies) const {
    int active_count = agent_data_gpu_.active_count;
    
    if (profiling_enabled_) startTiming();
    CudaUtils::copyToHost(x_pos.data(), agent_data_gpu_.x_positions, active_count);
    CudaUtils::copyToHost(y_pos.data(), agent_data_gpu_.y_positions, active_count);
    CudaUtils::copyToHost(states.data(), agent_data_gpu_.behavior_states, active_count);
    CudaUtils::copyToHost(energies.data(), agent_data_gpu_.energies, active_count);
    if (profiling_enabled_) memory_transfer_time_ += stopTiming();
}

void GPUSimulationManager::compactAgentPopulation() {
    agent_data_gpu_.active_count = compactAgentPopulation(&agent_data_gpu_);
}

void GPUSimulationManager::printPerformanceStats() const {
    std::cout << "GPU Performance Statistics:\n";
    std::cout << "  RD Kernel Time: " << rd_kernel_time_ << " ms\n";
    std::cout << "  Biome Kernel Time: " << biome_kernel_time_ << " ms\n";
    std::cout << "  ABM Kernel Time: " << abm_kernel_time_ << " ms\n";
    std::cout << "  Memory Transfer Time: " << memory_transfer_time_ << " ms\n";
    
    float total_compute = rd_kernel_time_ + biome_kernel_time_ + abm_kernel_time_;
    float total_time = total_compute + memory_transfer_time_;
    
    std::cout << "  Total Compute Time: " << total_compute << " ms\n";
    std::cout << "  Total Time: " << total_time << " ms\n";
    std::cout << "  Compute Efficiency: " << (total_compute / total_time * 100.0f) << "%\n";
}

void GPUSimulationManager::startTiming() {
    CUDA_CHECK(cudaEventRecord(start_event_));
}

float GPUSimulationManager::stopTiming() {
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    
    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event_, stop_event_));
    return elapsed_time;
}

// Factory function
std::unique_ptr<GPUSimulationManager> createGPUSimulation(const SimulationParams& params) {
    return std::make_unique<GPUSimulationManager>(params);
}

// Performance comparison function (stub implementation)
PerformanceReport compareGPUvsCPU(const SimulationParams& params, int test_steps) {
    PerformanceReport report;
    report.grid_size = params.width * params.height;
    report.agent_count = params.num_agents;
    
    // For now, return placeholder values
    // In a complete implementation, this would run both GPU and CPU versions
    report.rd_speedup = 5.0f;
    report.abm_speedup = 3.0f;
    report.total_speedup = 4.0f;
    report.memory_overhead_percent = 15.0f;
    
    return report;
} 