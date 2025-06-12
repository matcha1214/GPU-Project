#include "reaction_diffusion_cpu.h"
#include "../utils/ppm_writer.h"
#include "../utils/config.h"
#include "../utils/timer.h"
#include <iostream>
#include <string>

/**
 * Reaction-Diffusion CPU Demo
 * 
 * This is a focused demo that runs ONLY the reaction-diffusion system,
 * without any agents. 
 */

int main(int argc, char* argv[]) {
    // Handle command line arguments
    if (!g_config.parseCommandLine(argc, argv)) {
        return (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) ? 0 : 1;
    }
    
    // Get simulation parameters
    const SimulationParams& params = g_config.getSimulationParams();
    
    std::cout << "=== Reaction-Diffusion CPU Demo ===" << std::endl;
    
    // Show what parameters we're using 
    // why you get certain patterns
    g_config.printConfig();
    std::cout << std::endl;
    
    // Save the config if needed in the future
    if (g_config.saveToFile("output/rd_demo_config.conf")) {
        std::cout << "Configuration saved to: output/rd_demo_config.conf" << std::endl;
    }
    
    std::cout << "=== Starting Reaction-Diffusion Simulation ===" << std::endl;
    
    // Start timing
    g_timer.start("simulation");
    
    // Create and set up the reaction-diffusion system
    ReactionDiffusionSystem rd_system(params);
    rd_system.initialize();  // Sets up U=1, V=0, plus a central seed patch
    
    std::cout << "Running simulation for " << params.num_steps << " steps..." << std::endl;
    
    // Main simulation loop
    for (int step = 0; step < params.num_steps; ++step) {
        // Evolve the chemical fields one time step
        rd_system.step();
        
        // Save snapshots at regular intervals
        if (step % params.output_frequency == 0) {
            std::string u_filename = "output/u_field_step_" + std::to_string(step) + ".ppm";
            std::string v_filename = "output/v_field_step_" + std::to_string(step) + ".ppm";
            
            // Convert the concentration fields to grayscale images
            // U field shows where the "substrate" chemical is
            // V field shows where the "catalyst" chemical is
            PPMWriter::writeGrayscale(u_filename, rd_system.getUGrid().getData(), 
                                    params.width, params.height, 0.0, 1.0);
            PPMWriter::writeGrayscale(v_filename, rd_system.getVGrid().getData(), 
                                    params.width, params.height, 0.0, 1.0);
            
            std::cout << "Step " << step << ": Saved " << u_filename << " and " << v_filename << std::endl;
        }
    }
    
    // Show performance results
    double total_time_ms = g_timer.stop("simulation");
    double total_time_sec = total_time_ms / 1000.0;
    
    std::cout << std::endl;
    std::cout << "=== Simulation Complete! ===" << std::endl;
    std::cout << "Total simulation time: " << total_time_sec << " seconds" << std::endl;
    std::cout << "Time per step: " << (total_time_sec / params.num_steps) << " seconds" << std::endl;
    std::cout << std::endl;
    
    // Explain what files were created
    std::cout << "Generated files in 'output/' directory:" << std::endl;
    std::cout << "  - u_field_step_*.ppm        : U chemical field evolution" << std::endl;
    std::cout << "  - v_field_step_*.ppm        : V chemical field evolution" << std::endl;
    std::cout << "  - rd_demo_config.conf       : Configuration used for this run" << std::endl;
    std::cout << std::endl;
    
    // Give tips for viewing and understanding the output
    std::cout << "View PPM files with image viewers or convert using ImageMagick:" << std::endl;
    int final_step = params.num_steps - params.output_frequency;
    std::cout << "  convert output/v_field_step_" << final_step << ".ppm output/v_field_final.png" << std::endl;
    std::cout << "  convert output/u_field_step_" << final_step << ".ppm output/u_field_final.png" << std::endl;
    std::cout << std::endl;
    
    // Show examples of trying different Gray-Scott patterns
    std::cout << "Example usage for different patterns:" << std::endl;
    std::cout << "  " << argv[0] << " --preset mitosis --steps 2000" << std::endl;
    std::cout << "  " << argv[0] << " --preset spots --width=256 --height=256" << std::endl;
    std::cout << "  " << argv[0] << " --config example_config.conf" << std::endl;
    std::cout << "  " << argv[0] << " --help  # For complete parameter list" << std::endl;
    
    return 0;
}
