#include "ppm_writer.h"
#include <fstream>
#include <iostream>
#include <algorithm>

bool PPMWriter::writeGrayscale(const std::string& filename, 
                              const std::vector<double>& data,
                              int width, int height,
                              double min_val, double max_val) {
    if (data.size() != static_cast<size_t>(width * height)) {
        std::cerr << "Error: Data size doesn't match grid dimensions" << std::endl;
        return false;
    }
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    if (!writePPMHeader(file, width, height)) {
        return false;
    }
    
    std::vector<unsigned char> pixel_data(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        unsigned char gray_val = scaleToChar(data[i], min_val, max_val);
        pixel_data[i * 3 + 0] = gray_val; // R
        pixel_data[i * 3 + 1] = gray_val; // G
        pixel_data[i * 3 + 2] = gray_val; // B
    }
    
    file.write(reinterpret_cast<const char*>(pixel_data.data()), pixel_data.size());
    return file.good();
}

bool PPMWriter::writeRGB(const std::string& filename,
                        const std::vector<Color>& colors,
                        int width, int height) {
    if (colors.size() != static_cast<size_t>(width * height)) {
        std::cerr << "Error: Color data size doesn't match grid dimensions" << std::endl;
        return false;
    }
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    if (!writePPMHeader(file, width, height)) {
        return false;
    }
    
    std::vector<unsigned char> pixel_data(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        pixel_data[i * 3 + 0] = colors[i].r;
        pixel_data[i * 3 + 1] = colors[i].g;
        pixel_data[i * 3 + 2] = colors[i].b;
    }
    
    file.write(reinterpret_cast<const char*>(pixel_data.data()), pixel_data.size());
    return file.good();
}

bool PPMWriter::writeBiomeMap(const std::string& filename,
                             const std::vector<BiomeType>& biome_map,
                             int width, int height) {
    if (biome_map.size() != static_cast<size_t>(width * height)) {
        std::cerr << "Error: Biome map size doesn't match grid dimensions" << std::endl;
        return false;
    }
    
    std::vector<Color> colors(width * height);
    for (int i = 0; i < width * height; ++i) {
        colors[i] = getBiomeColor(biome_map[i]);
    }
    
    return writeRGB(filename, colors, width, height);
}

bool PPMWriter::writeFieldWithAgents(const std::string& filename,
                                    const std::vector<double>& field_data,
                                    const std::vector<AgentState>& agents,
                                    int width, int height,
                                    double field_min, double field_max) {
    if (field_data.size() != static_cast<size_t>(width * height)) {
        std::cerr << "Error: Field data size doesn't match grid dimensions" << std::endl;
        return false;
    }
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    if (!writePPMHeader(file, width, height)) {
        return false;
    }
    
    // Create base image from field data
    std::vector<unsigned char> pixel_data(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        unsigned char gray_val = scaleToChar(field_data[i], field_min, field_max);
        pixel_data[i * 3 + 0] = gray_val; // R
        pixel_data[i * 3 + 1] = gray_val; // G
        pixel_data[i * 3 + 2] = gray_val; // B
    }
    
    // Overlay agents as colored dots
    for (const auto& agent : agents) {
        int x = static_cast<int>(agent.x);
        int y = static_cast<int>(agent.y);
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
            int idx = (y * width + x) * 3;
            // Draw agent as red dot
            pixel_data[idx + 0] = 255; // R
            pixel_data[idx + 1] = 0;   // G
            pixel_data[idx + 2] = 0;   // B
        }
    }
    
    file.write(reinterpret_cast<const char*>(pixel_data.data()), pixel_data.size());
    return file.good();
}

Color PPMWriter::getBiomeColor(BiomeType biome) {
    switch (biome) {
        case BiomeType::DESERT:
            return Color(255, 255, 0);    // Yellow
        case BiomeType::GRASSLAND:
            return Color(0, 255, 0);      // Green
        case BiomeType::FOREST:
            return Color(0, 128, 0);      // Dark Green
        case BiomeType::WATER:
            return Color(0, 0, 255);      // Blue
        case BiomeType::MOUNTAIN:
            return Color(128, 128, 128);  // Gray
        case BiomeType::UNKNOWN:
        default:
            return Color(255, 0, 255);    // Magenta
    }
}

unsigned char PPMWriter::scaleToChar(double value, double min_val, double max_val) {
    if (value <= min_val) return 0;
    if (value >= max_val) return 255;
    return static_cast<unsigned char>(255.0 * (value - min_val) / (max_val - min_val));
}

bool PPMWriter::writePPMHeader(std::ofstream& file, int width, int height) {
    file << "P6\n" << width << " " << height << "\n255\n";
    return file.good();
}
