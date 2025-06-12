#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <vector>
#include <utility>

// Common type definitions
using GridData = std::vector<double>;
using Coordinate = std::pair<int, int>;
using CoordinateList = std::vector<Coordinate>;

// Simulation parameters structure
struct SimulationParams {
    // Grid dimensions
    int width = 128;
    int height = 128;
    
    // Gray-Scott parameters
    double Du = 0.16;    // Diffusion rate for U
    double Dv = 0.08;    // Diffusion rate for V
    double f = 0.0545;   // Feed rate
    double k = 0.062;    // Kill rate
    double dt = 1.0;     // Time step
    
    // Simulation control
    int num_steps = 1000;
    int output_frequency = 50;
    
    // Agent parameters
    int num_agents = 100;
    double agent_speed = 1.0;
    double agent_perception_radius = 2.0;
};

// Biome types enumeration
enum class BiomeType {
    DESERT = 0,
    GRASSLAND = 1,
    FOREST = 2,
    WATER = 3,
    MOUNTAIN = 4,
    UNKNOWN = 5
};

// Agent behavior state enumeration
enum class AgentBehaviorState {
    IDLE = 0,
    MOVING = 1,
    FORAGING = 2,
    FLEEING = 3,
    SOCIALIZING = 4,
    DEAD = 5
};

// Color structure for visualization
struct Color {
    unsigned char r, g, b;
    
    Color(unsigned char red = 0, unsigned char green = 0, unsigned char blue = 0)
        : r(red), g(green), b(blue) {}
};

// Agent state structure
struct AgentState {
    int id;
    double x, y;        // Position
    double energy;      // Energy level
    BiomeType current_biome;
    AgentBehaviorState behavior_state;
    double perceived_u;
    double perceived_v;
    int age;           // Agent age in simulation steps
    double last_food_time; // Steps since last successful foraging
};

// Constants
namespace Constants {
    constexpr double PI = 3.14159265358979323846;
    constexpr double EPSILON = 1e-10;
    
    // Default Gray-Scott parameter sets
    namespace GrayScott {
        // Coral growth pattern
        constexpr double CORAL_DU = 0.16;
        constexpr double CORAL_DV = 0.08;
        constexpr double CORAL_F = 0.0545;
        constexpr double CORAL_K = 0.062;
        
        // Mitosis pattern
        constexpr double MITOSIS_DU = 0.16;
        constexpr double MITOSIS_DV = 0.08;
        constexpr double MITOSIS_F = 0.0367;
        constexpr double MITOSIS_K = 0.0649;
        
        // Spots pattern
        constexpr double SPOTS_DU = 0.16;
        constexpr double SPOTS_DV = 0.08;
        constexpr double SPOTS_F = 0.035;
        constexpr double SPOTS_K = 0.065;
    }
    
    // Agent behavior constants
    namespace Agent {
        // Energy thresholds
        constexpr double INITIAL_ENERGY = 100.0;
        constexpr double MAX_ENERGY = 200.0;
        constexpr double LOW_ENERGY_THRESHOLD = 30.0;
        constexpr double HUNGRY_THRESHOLD = 50.0;
        constexpr double WELL_FED_THRESHOLD = 100.0;
        
        // Energy costs and gains
        constexpr double BASE_METABOLISM = 0.05;
        constexpr double MOVEMENT_COST_MULTIPLIER = 0.1;
        constexpr double FORAGING_ENERGY_GAIN_GOOD = 0.8;
        constexpr double FORAGING_ENERGY_GAIN_MODERATE = 0.3;
        
        // Behavior multipliers
        constexpr double FLEE_ENERGY_MULTIPLIER = 1.5;
        constexpr double FORAGE_ENERGY_MULTIPLIER = 1.2;
        constexpr double SOCIAL_ENERGY_MULTIPLIER = 0.9;
        constexpr double IDLE_ENERGY_MULTIPLIER = 0.5;
        
        // Distance and perception constants
        constexpr double DEFAULT_PERCEPTION_RADIUS = 2.5;
        constexpr double NEIGHBOR_SEARCH_RADIUS = 5.0;
        constexpr double PREDATOR_DETECTION_DISTANCE = 3.0;
        constexpr double COMPETITION_DISTANCE = 2.0;
        constexpr double SOCIAL_DISTANCE_MIN = 1.0;
        constexpr double SOCIAL_DISTANCE_MAX = 1.5;
        constexpr double LARGE_DISTANCE = 100.0;
        
        // Age and time constants
        constexpr int MAX_AGE = 5000;
        constexpr double FOOD_TIMEOUT = 100.0;
        
        // Behavioral thresholds
        constexpr double PREDATOR_ENERGY_RATIO = 1.5;
        constexpr double FORAGING_V_MIN = 0.2;
        constexpr double FORAGING_V_MAX = 0.6;
        constexpr double MODERATE_FORAGING_V_MIN = 0.1;
        constexpr double MODERATE_FORAGING_V_MAX = 0.8;
        constexpr double POOR_FOOD_V_THRESHOLD = 0.3;
        constexpr double GRADIENT_THRESHOLD = -0.1;
        constexpr int MAX_SOCIAL_AGENTS = 5;
        constexpr int COMPETITION_AGENT_COUNT = 2;
        
        // Movement speeds
        constexpr double BASE_SPEED = 0.5;
        constexpr double FLEE_SPEED_MULTIPLIER = 1.5;
        constexpr double ACTIVE_FORAGE_SPEED_MULTIPLIER = 1.2;
        constexpr double CAREFUL_FORAGE_SPEED_MULTIPLIER = 0.8;
        constexpr double SOCIAL_SPEED_MULTIPLIER = 0.6;
        constexpr double SOCIAL_RETREAT_SPEED_MULTIPLIER = 0.3;
        constexpr double IDLE_SPEED_MULTIPLIER = 0.2;
        
        // Probability constants
        constexpr double MOVEMENT_PROBABILITY = 0.7;
        
        // Cleanup frequency
        constexpr int DEAD_AGENT_CLEANUP_FREQUENCY = 50;
    }
}

#endif // COMMON_TYPES_H
