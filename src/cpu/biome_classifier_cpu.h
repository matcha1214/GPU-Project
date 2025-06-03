#ifndef BIOME_CLASSIFIER_CPU_H
#define BIOME_CLASSIFIER_CPU_H

#include "reaction_diffusion_cpu.h"
#include "../common/types.h"
#include <vector>
#include <string>

class BiomeClassifier {
public:
    BiomeClassifier();
    
    // Classify a single cell based on U and V values
    BiomeType classifyCell(double u_value, double v_value) const;
    
    // Get biome type at specific position (with bounds checking)
    BiomeType getBiomeTypeAt(const Grid& u_grid, const Grid& v_grid, int x, int y) const;
    
    // Get biome type at continuous position (with interpolation)
    BiomeType getBiomeTypeAt(const Grid& u_grid, const Grid& v_grid, double x, double y) const;
    
    // Classify entire grid and return biome map
    std::vector<BiomeType> classifyGrid(const Grid& u_grid, const Grid& v_grid) const;
    
    // Save biome map as colored PPM image
    void saveBiomeMapToPPM(const std::vector<BiomeType>& biome_map, 
                           int width, int height, 
                           const std::string& filename) const;
    
    // Get color for biome type (RGB values) 
    Color getBiomeColor(BiomeType biome) const;
    
    // Get biome characteristics for agent decision making
    struct BiomeCharacteristics {
        double food_availability;
        double movement_cost;
        double safety_level;
        bool is_habitable;
    };
    BiomeCharacteristics getBiomeCharacteristics(BiomeType biome) const;

private:
    // Threshold parameters for classification
    double u_threshold_low_;
    double u_threshold_high_;
    double v_threshold_low_;
    double v_threshold_high_;
};

#endif  