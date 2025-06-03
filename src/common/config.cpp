#include "config.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>

// Global configuration instance
Config g_config;

// Define the Gray-Scott presets map
const std::map<std::string, std::tuple<double, double, double, double>> Config::gray_scott_presets_ = {
    {"coral",   {Constants::GrayScott::CORAL_DU,   Constants::GrayScott::CORAL_DV,   Constants::GrayScott::CORAL_F,   Constants::GrayScott::CORAL_K}},
    {"mitosis", {Constants::GrayScott::MITOSIS_DU, Constants::GrayScott::MITOSIS_DV, Constants::GrayScott::MITOSIS_F, Constants::GrayScott::MITOSIS_K}},
    {"spots",   {Constants::GrayScott::SPOTS_DU,   Constants::GrayScott::SPOTS_DV,   Constants::GrayScott::SPOTS_F,   Constants::GrayScott::SPOTS_K}}
};

Config::Config() {
    setDefaults();
}

void Config::setDefaults() {
    // Set default simulation parameters
    params_.width = 128;
    params_.height = 128;
    params_.Du = Constants::GrayScott::CORAL_DU;
    params_.Dv = Constants::GrayScott::CORAL_DV;
    params_.f = Constants::GrayScott::CORAL_F;
    params_.k = Constants::GrayScott::CORAL_K;
    params_.dt = 1.0;
    params_.num_steps = 1000;
    params_.output_frequency = 50;
    params_.num_agents = 100;
    params_.agent_speed = 1.0;
    params_.agent_perception_radius = 2.0;
}

bool Config::parseCommandLine(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        
        if (arg == "--help" || arg == "-h") {
            printHelp();
            return false;  // Indicate that we should exit after showing help
        }
        else if (arg == "--config" && i + 1 < argc) {
            // Load configuration from file
            if (!loadFromFile(argv[++i])) {
                std::cerr << "Error: Failed to load configuration from file: " << argv[i] << std::endl;
                return false;
            }
        }
        else if (arg == "--preset" && i + 1 < argc) {
            // Set Gray-Scott preset
            setGrayScottPreset(argv[++i]);
        }
        else if (arg.find("--") == 0) {
            // Handle key=value arguments
            size_t eq_pos = arg.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = arg.substr(2, eq_pos - 2);  // Remove "--"
                std::string value = arg.substr(eq_pos + 1);
                if (!parseKeyValue(key, value)) {
                    std::cerr << "Error: Invalid parameter: " << key << "=" << value << std::endl;
                    return false;
                }
            } else if (i + 1 < argc) {
                // Handle --key value format
                std::string key = arg.substr(2);  // Remove "--"
                std::string value = argv[++i];
                if (!parseKeyValue(key, value)) {
                    std::cerr << "Error: Invalid parameter: " << key << "=" << value << std::endl;
                    return false;
                }
            }
        }
        else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    
    // Validate final configuration
    return validateParameters();
}

bool Config::parseKeyValue(const std::string& key, const std::string& value) {
    try {
        if (key == "width") {
            int width = std::stoi(value);
            if (width <= 0 || width > 10000) {
                std::cerr << "Error: Width must be between 1 and 10000, got: " << width << std::endl;
                return false;
            }
            params_.width = width;
        } else if (key == "height") {
            int height = std::stoi(value);
            if (height <= 0 || height > 10000) {
                std::cerr << "Error: Height must be between 1 and 10000, got: " << height << std::endl;
                return false;
            }
            params_.height = height;
        } else if (key == "Du") {
            double Du = std::stod(value);
            if (Du < 0.0 || Du > 1.0) {
                std::cerr << "Error: Du must be between 0.0 and 1.0, got: " << Du << std::endl;
                return false;
            }
            params_.Du = Du;
        } else if (key == "Dv") {
            double Dv = std::stod(value);
            if (Dv < 0.0 || Dv > 1.0) {
                std::cerr << "Error: Dv must be between 0.0 and 1.0, got: " << Dv << std::endl;
                return false;
            }
            params_.Dv = Dv;
        } else if (key == "f") {
            double f = std::stod(value);
            if (f < 0.0 || f > 1.0) {
                std::cerr << "Error: f must be between 0.0 and 1.0, got: " << f << std::endl;
                return false;
            }
            params_.f = f;
        } else if (key == "k") {
            double k = std::stod(value);
            if (k < 0.0 || k > 1.0) {
                std::cerr << "Error: k must be between 0.0 and 1.0, got: " << k << std::endl;
                return false;
            }
            params_.k = k;
        } else if (key == "dt") {
            double dt = std::stod(value);
            if (dt <= 0.0 || dt > 10.0) {
                std::cerr << "Error: dt must be between 0.0 and 10.0, got: " << dt << std::endl;
                return false;
            }
            params_.dt = dt;
        } else if (key == "steps") {
            int steps = std::stoi(value);
            if (steps <= 0 || steps > 1000000) {
                std::cerr << "Error: steps must be between 1 and 1000000, got: " << steps << std::endl;
                return false;
            }
            params_.num_steps = steps;
        } else if (key == "output_frequency" || key == "output_freq") {
            // Accept both variants for backward compatibility
            int freq = std::stoi(value);
            if (freq <= 0 || freq > 100000) {
                std::cerr << "Error: output_frequency must be between 1 and 100000, got: " << freq << std::endl;
                return false;
            }
            params_.output_frequency = freq;
        } else if (key == "agents") {
            int agents = std::stoi(value);
            if (agents < 0 || agents > 100000) {
                std::cerr << "Error: agents must be between 0 and 100000, got: " << agents << std::endl;
                return false;
            }
            params_.num_agents = agents;
        } else if (key == "agent_speed") {
            double speed = std::stod(value);
            if (speed < 0.0 || speed > 100.0) {
                std::cerr << "Error: agent_speed must be between 0.0 and 100.0, got: " << speed << std::endl;
                return false;
            }
            params_.agent_speed = speed;
        } else if (key == "perception_radius") {
            double radius = std::stod(value);
            if (radius < 0.0 || radius > 50.0) {
                std::cerr << "Error: perception_radius must be between 0.0 and 50.0, got: " << radius << std::endl;
                return false;
            }
            params_.agent_perception_radius = radius;
        } else {
            return false;  // Unknown parameter
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to parse value '" << value << "' for parameter '" << key << "': " << e.what() << std::endl;
        return false;  // Conversion error
    }
}

bool Config::validateParameters() {
    bool valid = true;
    
    // Check for logical consistency
    if (params_.output_frequency > params_.num_steps) {
        std::cerr << "Warning: output_frequency (" << params_.output_frequency 
                  << ") is greater than num_steps (" << params_.num_steps << ")" << std::endl;
        params_.output_frequency = params_.num_steps;
    }
    
    // Check grid size vs agent count
    int total_cells = params_.width * params_.height;
    if (params_.num_agents > total_cells) {
        std::cerr << "Warning: More agents (" << params_.num_agents 
                  << ") than grid cells (" << total_cells << ")" << std::endl;
    }
    
    // Check perception radius vs grid size
    double max_perception = std::min(params_.width, params_.height) / 2.0;
    if (params_.agent_perception_radius > max_perception) {
        std::cerr << "Warning: perception_radius (" << params_.agent_perception_radius 
                  << ") is large relative to grid size" << std::endl;
    }
    
    return valid;
}

bool Config::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open configuration file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    int line_number = 0;
    
    while (std::getline(file, line)) {
        ++line_number;
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        
        // Parse key=value pairs
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            std::cerr << "Warning: Invalid format at line " << line_number << ": " << line << std::endl;
            continue;
        }
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        // Trim whitespace
        key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
        value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
        
        if (!parseKeyValue(key, value)) {
            std::cerr << "Warning: Invalid parameter at line " << line_number << ": " << key << "=" << value << std::endl;
        }
    }
    
    return validateParameters();
}

bool Config::saveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create configuration file: " << filename << std::endl;
        return false;
    }
    
    file << "# Biome Agent Simulation Configuration File\n";
    file << "# Generated automatically\n\n";
    
    file << "# Grid dimensions\n";
    file << "width=" << params_.width << "\n";
    file << "height=" << params_.height << "\n\n";
    
    file << "# Gray-Scott reaction-diffusion parameters\n";
    file << "Du=" << params_.Du << "\n";
    file << "Dv=" << params_.Dv << "\n";
    file << "f=" << params_.f << "\n";
    file << "k=" << params_.k << "\n";
    file << "dt=" << params_.dt << "\n\n";
    
    file << "# Simulation control\n";
    file << "steps=" << params_.num_steps << "\n";
    file << "output_frequency=" << params_.output_frequency << "\n\n";
    
    file << "# Agent parameters\n";
    file << "agents=" << params_.num_agents << "\n";
    file << "agent_speed=" << params_.agent_speed << "\n";
    file << "perception_radius=" << params_.agent_perception_radius << "\n";
    
    return true;
}

const SimulationParams& Config::getSimulationParams() const {
    return params_;
}

SimulationParams& Config::getSimulationParams() {
    return params_;
}

void Config::setGrayScottPreset(const std::string& preset_name) {
    auto it = gray_scott_presets_.find(preset_name);
    if (it != gray_scott_presets_.end()) {
        std::tie(params_.Du, params_.Dv, params_.f, params_.k) = it->second;
        std::cout << "Applied Gray-Scott preset: " << preset_name << std::endl;
    } else {
        std::cerr << "Warning: Unknown Gray-Scott preset: " << preset_name << std::endl;
        std::cerr << "Available presets: ";
        for (const auto& preset : gray_scott_presets_) {
            std::cerr << preset.first << " ";
        }
        std::cerr << std::endl;
    }
}

void Config::printConfig() const {
    std::cout << "=== Current Configuration ===" << std::endl;
    std::cout << "Grid dimensions: " << params_.width << "x" << params_.height << std::endl;
    std::cout << "Gray-Scott parameters:" << std::endl;
    std::cout << "  Du = " << params_.Du << std::endl;
    std::cout << "  Dv = " << params_.Dv << std::endl;
    std::cout << "  f  = " << params_.f << std::endl;
    std::cout << "  k  = " << params_.k << std::endl;
    std::cout << "  dt = " << params_.dt << std::endl;
    std::cout << "Simulation control:" << std::endl;
    std::cout << "  Steps: " << params_.num_steps << std::endl;
    std::cout << "  Output frequency: " << params_.output_frequency << std::endl;
    std::cout << "Agent parameters:" << std::endl;
    std::cout << "  Number of agents: " << params_.num_agents << std::endl;
    std::cout << "  Agent speed: " << params_.agent_speed << std::endl;
    std::cout << "  Perception radius: " << params_.agent_perception_radius << std::endl;
    std::cout << "=============================" << std::endl;
}

void Config::printHelp() {
    std::cout << "Biome Agent Simulation - Configuration Help\n\n";
    std::cout << "Usage: program [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --help, -h                    Show this help message\n";
    std::cout << "  --config <file>               Load configuration from file\n";
    std::cout << "  --preset <name>               Use Gray-Scott preset (coral, mitosis, spots)\n\n";
    
    std::cout << "Parameters (can be set as --key=value or --key value):\n";
    std::cout << "  Grid dimensions:\n";
    std::cout << "    --width <int>               Grid width (default: 128, range: 1-10000)\n";
    std::cout << "    --height <int>              Grid height (default: 128, range: 1-10000)\n\n";
    
    std::cout << "  Gray-Scott parameters:\n";
    std::cout << "    --Du <double>               U diffusion rate (default: 0.16, range: 0.0-1.0)\n";
    std::cout << "    --Dv <double>               V diffusion rate (default: 0.08, range: 0.0-1.0)\n";
    std::cout << "    --f <double>                Feed rate (default: 0.0545, range: 0.0-1.0)\n";
    std::cout << "    --k <double>                Kill rate (default: 0.062, range: 0.0-1.0)\n";
    std::cout << "    --dt <double>               Time step (default: 1.0, range: 0.0-10.0)\n\n";
    
    std::cout << "  Simulation control:\n";
    std::cout << "    --steps <int>               Number of simulation steps (default: 1000, range: 1-1000000)\n";
    std::cout << "    --output_frequency <int>    Output frequency (default: 50, range: 1-100000)\n\n";
    
    std::cout << "  Agent parameters:\n";
    std::cout << "    --agents <int>              Number of agents (default: 100, range: 0-100000)\n";
    std::cout << "    --agent_speed <double>      Agent movement speed (default: 1.0, range: 0.0-100.0)\n";
    std::cout << "    --perception_radius <double> Agent perception radius (default: 2.0, range: 0.0-50.0)\n\n";
    
    std::cout << "Examples:\n";
    std::cout << "  ./program --preset mitosis --steps 2000\n";
    std::cout << "  ./program --config settings.conf --agents 200\n";
    std::cout << "  ./program --width=256 --height=256 --f=0.035 --k=0.065\n\n";
    
    std::cout << "Available Gray-Scott presets:\n";
    for (const auto& preset : gray_scott_presets_) {
        const auto& [Du, Dv, f, k] = preset.second;
        std::cout << "  " << preset.first << ": Du=" << Du << ", Dv=" << Dv 
                  << ", f=" << f << ", k=" << k << std::endl;
    }
} 