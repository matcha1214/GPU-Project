#include "biome_classifier_cpu.h"
#include "../utils/ppm_writer.h"
#include "../utils/utils.h"
#include <iostream>
#include <algorithm>

BiomeClassifier::BiomeClassifier() 
    : u_threshold_low_(0.3), u_threshold_high_(0.7),
      v_threshold_low_(0.2), v_threshold_high_(0.6) {
}

BiomeType BiomeClassifier::classifyCell(double u_value, double v_value) const {
    // Classification based on U and V thresholds
    // This is a simple threshold-based classification scheme
    
    if (v_value < v_threshold_low_) {
        if (u_value > u_threshold_high_) {
            return BiomeType::DESERT;  // High U, low V = desert
        } else {
            return BiomeType::MOUNTAIN;  // Low U, low V = mountain
        }
    } else if (v_value > v_threshold_high_) {
        return BiomeType::WATER;  // High V = water
    } else {
        // Medium V values
        if (u_value < u_threshold_low_) {
            return BiomeType::FOREST;  // Low U, medium V = forest
        } else if (u_value > u_threshold_high_) {
            return BiomeType::GRASSLAND;  // High U, medium V = grassland
        } else {
            return BiomeType::GRASSLAND;  // Medium U, medium V = grassland
        }
    }
}

BiomeType BiomeClassifier::getBiomeTypeAt(const Grid& u_grid, const Grid& v_grid, int x, int y) const {
    // Use periodic boundary conditions for consistency with Grid class
    int wrapped_x = Utils::wrapCoordinate(x, u_grid.getWidth());
    int wrapped_y = Utils::wrapCoordinate(y, u_grid.getHeight());
    
    double u_val = u_grid.get(wrapped_y, wrapped_x);
    double v_val = v_grid.get(wrapped_y, wrapped_x);
    
    return classifyCell(u_val, v_val);
}

BiomeType BiomeClassifier::getBiomeTypeAt(const Grid& u_grid, const Grid& v_grid, double x, double y) const {
    // Convert continuous coordinates to grid indices
    int grid_x = static_cast<int>(std::floor(x));
    int grid_y = static_cast<int>(std::floor(y));
    
    // For now, simple nearest-neighbor interpolation
    // *Could be enhanced with bilinear interpolation if needed
    return getBiomeTypeAt(u_grid, v_grid, grid_x, grid_y);
}

std::vector<BiomeType> BiomeClassifier::classifyGrid(const Grid& u_grid, const Grid& v_grid) const {
    int width = u_grid.getWidth();
    int height = u_grid.getHeight();
    
    if (width != v_grid.getWidth() || height != v_grid.getHeight()) {
        std::cerr << "Error: U and V grids have different dimensions" << std::endl;
        return {};
    }
    
    std::vector<BiomeType> biome_map(width * height);
    
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            double u_val = u_grid.get(r, c);
            double v_val = v_grid.get(r, c);
            
            BiomeType biome = classifyCell(u_val, v_val);
            biome_map[r * width + c] = biome;
        }
    }
    
    return biome_map;
}

void BiomeClassifier::saveBiomeMapToPPM(const std::vector<BiomeType>& biome_map, 
                                       int width, int height, 
                                       const std::string& filename) const {
    if (!PPMWriter::writeBiomeMap(filename, biome_map, width, height)) {
        std::cerr << "Error: Failed to save biome map to " << filename << std::endl;
    }
}

Color BiomeClassifier::getBiomeColor(BiomeType biome) const {
    // returns the common Color type directly
    return PPMWriter::getBiomeColor(biome);
}

BiomeClassifier::BiomeCharacteristics BiomeClassifier::getBiomeCharacteristics(BiomeType biome) const {
    BiomeCharacteristics characteristics;
    
    switch (biome) {
        case BiomeType::FOREST:
            characteristics.food_availability = 0.8;
            characteristics.movement_cost = 0.6;
            characteristics.safety_level = 0.7;
            characteristics.is_habitable = true;
            break;
            
        case BiomeType::GRASSLAND:
            characteristics.food_availability = 0.6;
            characteristics.movement_cost = 0.3;
            characteristics.safety_level = 0.5;
            characteristics.is_habitable = true;
            break;
            
        case BiomeType::WATER:
            characteristics.food_availability = 0.4;
            characteristics.movement_cost = 0.8;
            characteristics.safety_level = 0.6;
            characteristics.is_habitable = false;  // Assuming land-based agents
            break;
            
        case BiomeType::DESERT:
            characteristics.food_availability = 0.1;
            characteristics.movement_cost = 0.4;
            characteristics.safety_level = 0.3;
            characteristics.is_habitable = true;
            break;
            
        case BiomeType::MOUNTAIN:
            characteristics.food_availability = 0.2;
            characteristics.movement_cost = 0.9;
            characteristics.safety_level = 0.8;
            characteristics.is_habitable = true;
            break;
            
        default:
        case BiomeType::UNKNOWN:
            characteristics.food_availability = 0.0;
            characteristics.movement_cost = 1.0;
            characteristics.safety_level = 0.0;
            characteristics.is_habitable = false;
            break;
    }
    
    return characteristics;
}
