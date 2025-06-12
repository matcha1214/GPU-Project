#ifndef COMMON_CONFIG_H
#define COMMON_CONFIG_H

#include "types.h"
#include <string>
#include <map>

class Config {
public:
    Config();
    
    // Load configuration from command line arguments
    bool parseCommandLine(int argc, char* argv[]);
    
    // Load configuration from file
    bool loadFromFile(const std::string& filename);
    
    // Save current configuration to file
    bool saveToFile(const std::string& filename) const;
    
    // Get simulation parameters
    const SimulationParams& getSimulationParams() const;
    SimulationParams& getSimulationParams();
    
    // Set predefined parameter sets
    void setGrayScottPreset(const std::string& preset_name);
    
    // Print current configuration
    void printConfig() const;
    
    // Print help message
    static void printHelp();

private:
    SimulationParams params_;
    
    // Helper methods
    void setDefaults();
    bool parseKeyValue(const std::string& key, const std::string& value);
    bool validateParameters();
    
    // Available presets
    static const std::map<std::string, std::tuple<double, double, double, double>> gray_scott_presets_;
};

// Global configuration instance
extern Config g_config;

#endif // COMMON_CONFIG_H
