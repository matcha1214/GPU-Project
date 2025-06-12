#include "utils.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>

namespace Utils {

// Math utilities
double clamp(double value, double min_val, double max_val) {
    return std::max(min_val, std::min(value, max_val));
}

int clamp(int value, int min_val, int max_val) {
    return std::max(min_val, std::min(value, max_val));
}

// Distance calculations
double distance(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

double distanceSquared(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return dx * dx + dy * dy;
}

// Grid utilities
int coordToIndex(int row, int col, int width) {
    return row * width + col;
}

Coordinate indexToCoord(int index, int width) {
    return {index / width, index % width};
}

bool isValidCoord(int row, int col, int width, int height) {
    return row >= 0 && row < height && col >= 0 && col < width;
}

// Periodic boundary conditions
int wrapCoordinate(int coord, int size) {
    return ((coord % size) + size) % size;
}

double wrapCoordinateDouble(double coord, double size) {
    return coord - size * std::floor(coord / size);
}

// RandomGenerator implementation
RandomGenerator::RandomGenerator(unsigned int seed) 
    : generator_(seed == 0 ? std::chrono::high_resolution_clock::now().time_since_epoch().count() : seed),
      uniform_dist_(0.0, 1.0), normal_dist_(0.0, 1.0) {
}

double RandomGenerator::uniform(double min, double max) {
    return min + (max - min) * uniform_dist_(generator_);
}

int RandomGenerator::uniformInt(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(generator_);
}

double RandomGenerator::normal(double mean, double stddev) {
    return mean + stddev * normal_dist_(generator_);
}

bool RandomGenerator::boolean(double probability) {
    return uniform_dist_(generator_) < probability;
}

Coordinate RandomGenerator::randomGridPoint(int width, int height) {
    return {uniformInt(0, height - 1), uniformInt(0, width - 1)};
}

void RandomGenerator::seed(unsigned int new_seed) {
    generator_.seed(new_seed);
}

// Grid initialization utilities
void initializeGridWithNoise(std::vector<double>& grid, int width, int height,
                             double base_value, double noise_amplitude,
                             RandomGenerator& rng) {
    grid.resize(width * height);
    for (auto& value : grid) {
        value = base_value + rng.uniform(-noise_amplitude, noise_amplitude);
    }
}

void initializeGridWithPatch(std::vector<double>& grid, int width, int height,
                            double base_value, double patch_value,
                            int patch_x, int patch_y, int patch_size) {
    grid.assign(width * height, base_value);
    
    int half_size = patch_size / 2;
    for (int r = patch_y - half_size; r < patch_y + half_size; ++r) {
        for (int c = patch_x - half_size; c < patch_x + half_size; ++c) {
            if (isValidCoord(r, c, width, height)) {
                grid[coordToIndex(r, c, width)] = patch_value;
            }
        }
    }
}

// Agent utilities
std::vector<AgentState> createRandomAgents(int num_agents, int grid_width, int grid_height,
                                         double initial_energy) {
    RandomGenerator rng;
    std::vector<AgentState> agents;
    agents.reserve(num_agents);
    
    for (int i = 0; i < num_agents; ++i) {
        AgentState agent;
        agent.id = i;
        agent.x = rng.uniform(0.0, static_cast<double>(grid_width));
        agent.y = rng.uniform(0.0, static_cast<double>(grid_height));
        agent.energy = initial_energy;
        agent.current_biome = BiomeType::UNKNOWN;
        agent.behavior_state = AgentBehaviorState::IDLE;
        agent.perceived_u = 0.0;
        agent.perceived_v = 0.0;
        agent.age = 0;
        agent.last_food_time = 0.0;
        agents.push_back(agent);
    }
    
    return agents;
}

} // namespace Utils 