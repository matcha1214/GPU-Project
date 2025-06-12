#ifndef COMMON_PPM_WRITER_H
#define COMMON_PPM_WRITER_H

#include "types.h"
#include <string>
#include <vector>

class PPMWriter {
public:
    // Write grayscale data to PPM file
    static bool writeGrayscale(const std::string& filename, 
                              const std::vector<double>& data,
                              int width, int height,
                              double min_val = 0.0, double max_val = 1.0);
    
    // Write RGB color data to PPM file
    static bool writeRGB(const std::string& filename,
                        const std::vector<Color>& colors,
                        int width, int height);
    
    // Write biome map with predefined colors
    static bool writeBiomeMap(const std::string& filename,
                             const std::vector<BiomeType>& biome_map,
                             int width, int height);
    
    // Write agent overlay on top of field data
    static bool writeFieldWithAgents(const std::string& filename,
                                    const std::vector<double>& field_data,
                                    const std::vector<AgentState>& agents,
                                    int width, int height,
                                    double field_min = 0.0, double field_max = 1.0);
    
    // Get predefined biome colors
    static Color getBiomeColor(BiomeType biome);
    
private:
    // Helper function to scale double to unsigned char
    static unsigned char scaleToChar(double value, double min_val, double max_val);
    
    // Helper function to write PPM header
    static bool writePPMHeader(std::ofstream& file, int width, int height);
};

#endif // COMMON_PPM_WRITER_H
