#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include "types.h"
#include <vector>
#include <string>
#include <random>

namespace Utils {
    // Math utilities
    double clamp(double value, double min_val, double max_val);
    int clamp(int value, int min_val, int max_val);
    
    // Distance calculations
    double distance(double x1, double y1, double x2, double y2);
    double distanceSquared(double x1, double y1, double x2, double y2);
    
    // Grid utilities
    int coordToIndex(int row, int col, int width);
    Coordinate indexToCoord(int index, int width);
    bool isValidCoord(int row, int col, int width, int height);
    
    // Periodic boundary conditions
    int wrapCoordinate(int coord, int size);
    double wrapCoordinateDouble(double coord, double size);
    
    // Random number generation
    class RandomGenerator {
    public:
        RandomGenerator(unsigned int seed = 0);
        
        // Generate random double in [min, max]
        double uniform(double min = 0.0, double max = 1.0);
        
        // Generate random integer in [min, max]
        int uniformInt(int min, int max);
        
        // Generate random double from normal distribution
        double normal(double mean = 0.0, double stddev = 1.0);
        
        // Generate random boolean
        bool boolean(double probability = 0.5);
        
        // Generate random point within grid bounds
        Coordinate randomGridPoint(int width, int height);
        
        // Seed the generator
        void seed(unsigned int new_seed);
        
    private:
        std::mt19937 generator_;
        std::uniform_real_distribution<double> uniform_dist_;
        std::normal_distribution<double> normal_dist_;
    };
    
    // File utilities
    bool fileExists(const std::string& filename);
    bool createDirectory(const std::string& path);
    std::string getFileExtension(const std::string& filename);
    std::string removeFileExtension(const std::string& filename);
    
    // String utilities
    std::vector<std::string> split(const std::string& str, char delimiter);
    std::string trim(const std::string& str);
    std::string toLower(const std::string& str);
    
    // Grid initialization utilities
    void initializeGridWithNoise(std::vector<double>& grid, int width, int height,
                                 double base_value, double noise_amplitude,
                                 RandomGenerator& rng);
    
    void initializeGridWithPatch(std::vector<double>& grid, int width, int height,
                                double base_value, double patch_value,
                                int patch_x, int patch_y, int patch_size);
    
    // Agent utilities
    std::vector<AgentState> createRandomAgents(int num_agents, int grid_width, int grid_height,
                                              double initial_energy = 100.0);
    
    // Performance utilities
    void printMemoryUsage();
    size_t getMemoryUsage();
}

#endif // COMMON_UTILS_H
