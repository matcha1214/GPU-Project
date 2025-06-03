#include "reaction_diffusion_cpu.h"
#include "agent_model_cpu.h"
#include "biome_classifier_cpu.h"
#include "../common/ppm_writer.h"
#include "../common/config.h"
#include "../common/timer.h"
#include <iostream>
#include <string>

/**
 * Integrated Biome-Agent Simulation Demo
 */

int main(int argc, char* argv[]) {
    // Parse command line arguments using config system
    // This handles --help, --config files, --preset options, and individual parameters
    if (!g_config.parseCommandLine(argc, argv)) {
        // If parsing failed or help was requested, exit 
        return (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) ? 0 : 1;
    }
    
    // Get the final simulation parameters (after applying config files, presets, CLI args)
    const SimulationParams& params = g_config.getSimulationParams();
    
    std::cout << "=== Integrated Biome-Agent Simulation ===" << std::endl;
    
    // Show what parameters are actually using
    g_config.printConfig();
    std::cout << std::endl;
    
    // Save the config for this run
    if (g_config.saveToFile("output/integrated_demo_config.conf")) {
        std::cout << "Configuration saved to: output/integrated_demo_config.conf" << std::endl;
    }
    
    std::cout << "=== Starting Integrated Simulation ===" << std::endl;
    
    // Start the stopwatch 
    // This will be the cpu baseline 
    g_timer.start("integrated_simulation");
    
    // Create the simulation manager
    SimulationManager sim_manager(params.width, params.height,
                                 params.Du, params.Dv, params.f, params.k, params.dt,
                                 params.num_agents);
    
    // Set up initial conditions: 
    // - RD system starts with U=1, V=0 plus a seed patch
    // - Agents get randomly scattered around the world
    sim_manager.initializeSimulation();
    
    // Run the main simulation
    // Each step: RD evolves -> agents perceive -> agents think -> agents act -> save output
    sim_manager.runSimulation(params.num_steps, params.output_frequency);
    
    // Stop the timer and check the results
    double total_time_ms = g_timer.stop("integrated_simulation");
    double total_time_sec = total_time_ms / 1000.0;
    
    std::cout << std::endl;
    std::cout << "=== Simulation Complete! ===" << std::endl;
    std::cout << "Total simulation time: " << total_time_sec << " seconds" << std::endl;
    std::cout << "Time per step: " << (total_time_sec / params.num_steps) << " seconds" << std::endl;
    std::cout << std::endl;
    
    // Tell the user what files were generated and what they mean
    std::cout << "Generated files in 'output/' directory:" << std::endl;
    std::cout << "  - u_field_step_*.ppm       : U chemical field evolution" << std::endl;
    std::cout << "  - v_field_step_*.ppm       : V chemical field evolution" << std::endl;
    std::cout << "  - biome_map_step_*.ppm     : Classified biome maps" << std::endl;
    std::cout << "  - field_with_agents_step_*.ppm : V field with agent overlay" << std::endl;
    std::cout << "  - integrated_demo_config.conf  : Configuration used for this run" << std::endl;
    std::cout << std::endl;
    
    // Give helpful hints for viewing the results
    std::cout << "View PPM files with image viewers or convert using ImageMagick:" << std::endl;
    int final_step = params.num_steps - params.output_frequency;
    std::cout << "  convert output/biome_map_step_" << final_step << ".ppm output/final_biome_map.png" << std::endl;
    std::cout << "  convert output/field_with_agents_step_" << final_step << ".ppm output/final_agents.png" << std::endl;
    std::cout << std::endl;
    
    // Show examples of how to try different configurations
    std::cout << "Example usage for different configurations:" << std::endl;
    std::cout << "  " << argv[0] << " --preset coral --agents 200 --steps 1000" << std::endl;
    std::cout << "  " << argv[0] << " --width=256 --height=256 --agents=150" << std::endl;
    std::cout << "  " << argv[0] << " --config integrated_config.conf" << std::endl;
    std::cout << "  " << argv[0] << " --help  # For complete parameter list" << std::endl;
    
    return 0;
} 