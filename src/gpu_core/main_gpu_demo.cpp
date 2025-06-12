#include "gpu_simulation_manager.h"
#include "../utils/config.h"
#include "../utils/ppm_writer.h"
#include "../utils/timer.h"
#include "../utils/types.h"
#include <iostream>
#include <string>
#include <vector>

/**
 * GPU Demo - Basic foundational version
 * 
 * This demonstrates the GPU-accelerated biome-agent simulation system
 * following the comprehensive implementation guide principles:
 * 
 * 1. Reaction-Diffusion on GPU with shared memory optimization
 * 2. Agent-Based Model using Structure of Arrays (SoA)
 * 3. Biome classification from RD output
 * 4. Minimal CPU-GPU data transfers
 * 5. Performance monitoring and comparison
 */

void printUsage(const char* program_name) {
    std::cout << "GPU Biome-Agent Simulation Demo\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --width N           Grid width (default: 128)\n";
    std::cout << "  --height N          Grid height (default: 128)\n";
    std::cout << "  --agents N          Number of agents (default: 100)\n";
    std::cout << "  --steps N           Simulation steps (default: 1000)\n";
    std::cout << "  --output_freq N     Output frequency (default: 50)\n";
    std::cout << "  --preset NAME       Use preset parameters (coral, mitosis, spots)\n";
    std::cout << "  --config FILE       Load configuration from file\n";
    std::cout << "  --profile           Enable detailed performance profiling\n";
    std::cout << "  --compare           Compare GPU vs CPU performance\n";
    std::cout << "  --basic_kernels     Use basic (non-optimized) GPU kernels\n";
    std::cout << "  --help              Show this help message\n\n";
    std::cout << "GPU Features:\n";
    std::cout << "  - Shared memory tiling for RD kernels\n";
    std::cout << "  - Structure of Arrays (SoA) for agents\n";
    std::cout << "  - cuRAND for stochastic behaviors\n";
    std::cout << "  - Double buffering for iterative updates\n";
    std::cout << "  - Atomic operations for environment interaction\n";
}

void saveVisualizationOutputs(GPUSimulationManager& gpu_sim, int step, 
                              const SimulationParams& params) {
    // Copy data from GPU to CPU for visualization
    std::vector<float> u_field(params.width * params.height);
    std::vector<float> v_field(params.width * params.height);
    std::vector<int> biome_map(params.width * params.height);
    
    std::vector<float> agent_x(params.num_agents);
    std::vector<float> agent_y(params.num_agents);
    std::vector<int> agent_states(params.num_agents);
    std::vector<float> agent_energies(params.num_agents);
    
    gpu_sim.copyFieldsToHost(u_field, v_field);
    gpu_sim.copyBiomeMapToHost(biome_map);
    gpu_sim.copyAgentDataToHost(agent_x, agent_y, agent_states, agent_energies);
    
    // Save U field
    std::string u_filename = "output/gpu_u_field_step_" + std::to_string(step) + ".ppm";
    PPMWriter::writeGrayscale(u_filename, u_field, params.width, params.height);
    
    // Save V field (usually more visually interesting)
    std::string v_filename = "output/gpu_v_field_step_" + std::to_string(step) + ".ppm";
    PPMWriter::writeGrayscale(v_filename, v_field, params.width, params.height);
    
    // Save biome map with color coding
    std::string biome_filename = "output/gpu_biome_map_step_" + std::to_string(step) + ".ppm";
    PPMWriter::writeBiomeMap(biome_filename, biome_map, params.width, params.height);
    
    // Save visualization with agents overlaid on V field
    std::vector<float> viz_field = v_field; // Copy V field as base
    
    // Overlay agent positions
    for (int i = 0; i < gpu_sim.getActiveAgentCount(); ++i) {
        int x = static_cast<int>(agent_x[i]);
        int y = static_cast<int>(agent_y[i]);
        
        if (x >= 0 && x < params.width && y >= 0 && y < params.height) {
            int idx = y * params.width + x;
            
            // Color-code based on agent state and energy
            AgentBehaviorState state = static_cast<AgentBehaviorState>(agent_states[i]);
            switch (state) {
                case AgentBehaviorState::FORAGING:
                    viz_field[idx] = 1.2f; // Bright for foraging
                    break;
                case AgentBehaviorState::FLEEING:
                    viz_field[idx] = 1.5f; // Very bright for fleeing
                    break;
                case AgentBehaviorState::SOCIALIZING:
                    viz_field[idx] = 0.9f; // Moderate for socializing
                    break;
                default:
                    viz_field[idx] = 1.0f; // Standard bright for other states
                    break;
            }
        }
    }
    
    std::string agents_filename = "output/gpu_field_with_agents_step_" + std::to_string(step) + ".ppm";
    PPMWriter::writeGrayscale(agents_filename, viz_field, params.width, params.height);
}

void runGPUDemo(const SimulationParams& params, bool enable_profiling, bool use_basic_kernels) {
    std::cout << "\n=== GPU Biome-Agent Simulation Demo ===\n";
    std::cout << "Grid size: " << params.width << "x" << params.height << "\n";
    std::cout << "Agents: " << params.num_agents << "\n";
    std::cout << "Steps: " << params.num_steps << "\n";
    std::cout << "GPU optimizations: " << (use_basic_kernels ? "Basic kernels" : "Optimized kernels") << "\n\n";
    
    // Print GPU device information
    CudaUtils::printDeviceInfo();
    
    // Create GPU simulation manager
    auto gpu_sim = createGPUSimulation(params);
    gpu_sim->enableProfiling(enable_profiling);
    
    std::cout << "\nInitializing GPU simulation...\n";
    Timer init_timer;
    gpu_sim->initializeSimulation();
    float init_time = init_timer.elapsed();
    std::cout << "Initialization time: " << init_time << " seconds\n";
    
    // Create output directory
    system("mkdir -p output");
    
    std::cout << "\nRunning GPU simulation...\n";
    Timer sim_timer;
    
    // Save initial state
    saveVisualizationOutputs(*gpu_sim, 0, params);
    
    // Main simulation loop
    for (int step = 1; step <= params.num_steps; ++step) {
        gpu_sim->runSimulationStep();
        
        // Periodic output
        if (step % params.output_frequency == 0) {
            saveVisualizationOutputs(*gpu_sim, step, params);
            
            int active_agents = gpu_sim->getActiveAgentCount();
            std::cout << "Step " << step << "/" << params.num_steps 
                      << " - Active agents: " << active_agents << "\n";
        }
        
        // Periodic agent population cleanup
        if (step % (params.output_frequency * 2) == 0) {
            gpu_sim->compactAgentPopulation();
        }
    }
    
    float sim_time = sim_timer.elapsed();
    std::cout << "\nSimulation completed!\n";
    std::cout << "Total simulation time: " << sim_time << " seconds\n";
    std::cout << "Average time per step: " << (sim_time / params.num_steps) << " seconds\n";
    
    if (enable_profiling) {
        std::cout << "\n=== Performance Statistics ===\n";
        gpu_sim->printPerformanceStats();
    }
    
    std::cout << "\nOutput files saved to output/ directory:\n";
    std::cout << "  - gpu_u_field_step_*.ppm (U chemical concentration)\n";
    std::cout << "  - gpu_v_field_step_*.ppm (V chemical concentration)\n";
    std::cout << "  - gpu_biome_map_step_*.ppm (Classified biomes)\n";
    std::cout << "  - gpu_field_with_agents_step_*.ppm (Agents on environment)\n";
}

void runPerformanceComparison(const SimulationParams& params) {
    std::cout << "\n=== GPU vs CPU Performance Comparison ===\n";
    
    Timer comp_timer;
    PerformanceReport report = compareGPUvsCPU(params, 100); // Test with 100 steps
    float comparison_time = comp_timer.elapsed();
    
    std::cout << "\nPerformance Report:\n";
    std::cout << "Grid size: " << report.grid_size << " cells\n";
    std::cout << "Agent count: " << report.agent_count << "\n";
    std::cout << "RD speedup: " << report.rd_speedup << "x\n";
    std::cout << "ABM speedup: " << report.abm_speedup << "x\n";
    std::cout << "Total speedup: " << report.total_speedup << "x\n";
    std::cout << "Memory overhead: " << report.memory_overhead_percent << "%\n";
    std::cout << "Comparison time: " << comparison_time << " seconds\n";
}

int main(int argc, char* argv[]) {
    SimulationParams params; // Default parameters
    bool enable_profiling = false;
    bool run_comparison = false;
    bool use_basic_kernels = false;
    std::string config_file;
    std::string preset;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--profile") {
            enable_profiling = true;
        } else if (arg == "--compare") {
            run_comparison = true;
        } else if (arg == "--basic_kernels") {
            use_basic_kernels = true;
        } else if (arg == "--width" && i + 1 < argc) {
            params.width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            params.height = std::stoi(argv[++i]);
        } else if (arg == "--agents" && i + 1 < argc) {
            params.num_agents = std::stoi(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            params.num_steps = std::stoi(argv[++i]);
        } else if (arg == "--output_freq" && i + 1 < argc) {
            params.output_frequency = std::stoi(argv[++i]);
        } else if (arg == "--preset" && i + 1 < argc) {
            preset = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        }
    }
    
    // Load configuration if specified
    if (!config_file.empty()) {
        params = ConfigLoader::loadFromFile(config_file);
    } else if (!preset.empty()) {
        params = ConfigLoader::getPreset(preset);
    }
    
    try {
        if (run_comparison) {
            runPerformanceComparison(params);
        } else {
            runGPUDemo(params, enable_profiling, use_basic_kernels);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 