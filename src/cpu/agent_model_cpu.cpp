#include "agent_model_cpu.h"
#include "../common/utils.h"
#include "../common/ppm_writer.h"
#include "biome_classifier_cpu.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

// Agent class implementation
Agent::Agent(int id, double x, double y, double initial_energy)
    : id_(id), x_(x), y_(y), energy_(initial_energy),
      behavior_state_(AgentBehaviorState::IDLE), age_(0), last_food_time_(0.0),
      perceived_u_local_(0.0), perceived_v_local_(0.0),
      perceived_u_avg_(0.0), perceived_v_avg_(0.0),
      nearby_agents_count_(0), nearest_agent_distance_(Constants::Agent::LARGE_DISTANCE),
      predator_nearby_(false), food_competition_(false),
      move_dx_(0.0), move_dy_(0.0) {
}

void Agent::perceive(const Grid& u_field, const Grid& v_field) {
    // Sample U and V values at agent's current position
    int grid_x = static_cast<int>(x_);
    int grid_y = static_cast<int>(y_);
    
    // Clamp to grid boundaries
    grid_x = Utils::clamp(grid_x, 0, u_field.getWidth() - 1);
    grid_y = Utils::clamp(grid_y, 0, u_field.getHeight() - 1);
    
    perceived_u_local_ = u_field.get(grid_y, grid_x);
    perceived_v_local_ = v_field.get(grid_y, grid_x);
}

void Agent::perceiveNeighborhood(const Grid& u_field, const Grid& v_field, double radius) {
    // Sample U and V values in a neighborhood around the agent
    int center_x = static_cast<int>(x_);
    int center_y = static_cast<int>(y_);
    int radius_int = static_cast<int>(radius);
    
    double sum_u = 0.0;
    double sum_v = 0.0;
    int count = 0;
    
    for (int dy = -radius_int; dy <= radius_int; ++dy) {
        for (int dx = -radius_int; dx <= radius_int; ++dx) {
            // Check if within circular radius
            if (dx * dx + dy * dy <= radius * radius) {
                int sample_x = Utils::wrapCoordinate(center_x + dx, u_field.getWidth());
                int sample_y = Utils::wrapCoordinate(center_y + dy, u_field.getHeight());
                
                sum_u += u_field.get(sample_y, sample_x);
                sum_v += v_field.get(sample_y, sample_x);
                count++;
            }
        }
    }
    
    if (count > 0) {
        perceived_u_avg_ = sum_u / count;
        perceived_v_avg_ = sum_v / count;
    } else {
        perceived_u_avg_ = perceived_u_local_;
        perceived_v_avg_ = perceived_v_local_;
    }
    
    // Also update local perception
    perceive(u_field, v_field);
}

void Agent::perceiveOtherAgents(const std::vector<Agent*>& nearby_agents) {
    nearby_agents_count_ = nearby_agents.size();
    nearest_agent_distance_ = Constants::Agent::LARGE_DISTANCE; // Large initial value
    predator_nearby_ = false;
    food_competition_ = false;
    
    for (const Agent* other : nearby_agents) {
        if (other->getId() == id_) continue; // Skip self
        
        double distance = Utils::distance(x_, y_, other->getX(), other->getY());
        
        if (distance < nearest_agent_distance_) {
            nearest_agent_distance_ = distance;
        }
        
        // Detect threats (agents with much higher energy are potential predators)
        if (other->getEnergy() > energy_ * Constants::Agent::PREDATOR_ENERGY_RATIO && 
            distance < Constants::Agent::PREDATOR_DETECTION_DISTANCE) {
            predator_nearby_ = true;
        }
        
        // Detect food competition (agents in same favorable area)
        if (perceived_v_local_ > Constants::Agent::POOR_FOOD_V_THRESHOLD && 
            perceived_v_local_ < 0.7 && distance < Constants::Agent::COMPETITION_DISTANCE) {
            food_competition_ = true;
        }
    }
}

void Agent::decideAction() {
    Utils::RandomGenerator rng;
    
    // Base movement: influenced by current behavior state
    double random_angle = rng.uniform(0.0, 2.0 * Constants::PI);
    double base_speed = Constants::Agent::BASE_SPEED;
    
    move_dx_ = 0.0;
    move_dy_ = 0.0;
    
    switch (behavior_state_) {
        case AgentBehaviorState::FLEEING:
            // Move away from threats with high speed
            if (predator_nearby_ && nearest_agent_distance_ < Constants::Agent::NEIGHBOR_SEARCH_RADIUS) {
                // Move away from nearest agent
                double escape_angle = random_angle + Constants::PI; // Opposite direction
                move_dx_ = Constants::Agent::FLEE_SPEED_MULTIPLIER * base_speed * std::cos(escape_angle);
                move_dy_ = Constants::Agent::FLEE_SPEED_MULTIPLIER * base_speed * std::sin(escape_angle);
            } else {
                // Random escape movement
                move_dx_ = base_speed * Constants::Agent::ACTIVE_FORAGE_SPEED_MULTIPLIER * std::cos(random_angle);
                move_dy_ = base_speed * Constants::Agent::ACTIVE_FORAGE_SPEED_MULTIPLIER * std::sin(random_angle);
            }
            break;
            
        case AgentBehaviorState::FORAGING: {
            // Move towards better food areas
            double v_gradient = perceived_v_local_ - perceived_v_avg_;
            
            if (perceived_v_local_ < Constants::Agent::POOR_FOOD_V_THRESHOLD) {
                // Poor food area: search actively
                move_dx_ = base_speed * Constants::Agent::ACTIVE_FORAGE_SPEED_MULTIPLIER * std::cos(random_angle);
                move_dy_ = base_speed * Constants::Agent::ACTIVE_FORAGE_SPEED_MULTIPLIER * std::sin(random_angle);
                
                // Add gradient-based bias
                if (v_gradient < Constants::Agent::GRADIENT_THRESHOLD) {
                    move_dx_ += rng.uniform(-1.0, 1.0);
                    move_dy_ += rng.uniform(-1.0, 1.0);
                }
            } else {
                // Good food area: move more carefully
                move_dx_ = base_speed * Constants::Agent::CAREFUL_FORAGE_SPEED_MULTIPLIER * std::cos(random_angle);
                move_dy_ = base_speed * Constants::Agent::CAREFUL_FORAGE_SPEED_MULTIPLIER * std::sin(random_angle);
                
                // Fine-tune using gradient
                move_dx_ += v_gradient * 0.3;
                move_dy_ += v_gradient * 0.3;
            }
            
            // Avoid competition
            if (food_competition_ && nearby_agents_count_ > Constants::Agent::COMPETITION_AGENT_COUNT) {
                move_dx_ += rng.uniform(-0.5, 0.5);
                move_dy_ += rng.uniform(-0.5, 0.5);
            }
            break;
        }
            
        case AgentBehaviorState::SOCIALIZING: {
            // Move towards other agents but not too close
            if (nearby_agents_count_ > 0 && nearest_agent_distance_ > Constants::Agent::SOCIAL_DISTANCE_MAX) {
                // Move towards others
                move_dx_ = base_speed * Constants::Agent::SOCIAL_SPEED_MULTIPLIER * std::cos(random_angle);
                move_dy_ = base_speed * Constants::Agent::SOCIAL_SPEED_MULTIPLIER * std::sin(random_angle);
            } else if (nearest_agent_distance_ < Constants::Agent::SOCIAL_DISTANCE_MIN) {
                // Too close, move away slightly
                move_dx_ = base_speed * Constants::Agent::SOCIAL_RETREAT_SPEED_MULTIPLIER * std::cos(random_angle + Constants::PI);
                move_dy_ = base_speed * Constants::Agent::SOCIAL_RETREAT_SPEED_MULTIPLIER * std::sin(random_angle + Constants::PI);
            }
            break;
        }
            
        case AgentBehaviorState::MOVING: {
            // Standard movement with environmental bias
            move_dx_ = base_speed * std::cos(random_angle);
            move_dy_ = base_speed * std::sin(random_angle);
            
            // Environmental bias
            double v_gradient = perceived_v_local_ - perceived_v_avg_;
            move_dx_ += v_gradient * 0.2;
            move_dy_ += v_gradient * 0.2;
            break;
        }
            
        case AgentBehaviorState::IDLE: {
            // Minimal random movement
            move_dx_ = base_speed * Constants::Agent::IDLE_SPEED_MULTIPLIER * std::cos(random_angle);
            move_dy_ = base_speed * Constants::Agent::IDLE_SPEED_MULTIPLIER * std::sin(random_angle);
            break;
        }
            
        case AgentBehaviorState::DEAD: {
            // No movement
            move_dx_ = 0.0;
            move_dy_ = 0.0;
            break;
        }
    }
}

void Agent::act(double grid_width, double grid_height) {
    if (behavior_state_ == AgentBehaviorState::DEAD) {
        return; // Dead agents don't act
    }
    
    // Apply movement
    x_ += move_dx_;
    y_ += move_dy_;
    
    // Apply boundary conditions (wrap around)
    x_ = Utils::wrapCoordinateDouble(x_, grid_width);
    y_ = Utils::wrapCoordinateDouble(y_, grid_height);
    
    // Update energy and age
    updateEnergyAndAge();
    
    // Determine new behavior state for next step
    determineNewBehaviorState();
    
    // Reset movement for next step
    move_dx_ = 0.0;
    move_dy_ = 0.0;
}

void Agent::updateState(const Grid& u_field, const Grid& v_field,
                       double grid_width, double grid_height,
                       const std::vector<Agent*>& nearby_agents) {
    if (behavior_state_ == AgentBehaviorState::DEAD) {
        return; // Dead agents don't update
    }
    
    perceiveNeighborhood(u_field, v_field, Constants::Agent::DEFAULT_PERCEPTION_RADIUS);
    perceiveOtherAgents(nearby_agents);
    decideAction();
    act(grid_width, grid_height);
}

void Agent::updateEnergyAndAge() {
    // Age the agent
    age_++;
    
    // Energy consumption based on movement and behavior
    double movement_cost = std::sqrt(move_dx_ * move_dx_ + move_dy_ * move_dy_) * Constants::Agent::MOVEMENT_COST_MULTIPLIER;
    
    // Behavior-specific energy costs
    switch (behavior_state_) {
        case AgentBehaviorState::FLEEING:
            movement_cost *= Constants::Agent::FLEE_ENERGY_MULTIPLIER;
            break;
        case AgentBehaviorState::FORAGING:
            movement_cost *= Constants::Agent::FORAGE_ENERGY_MULTIPLIER;
            break;
        case AgentBehaviorState::SOCIALIZING:
            movement_cost *= Constants::Agent::SOCIAL_ENERGY_MULTIPLIER;
            break;
        case AgentBehaviorState::IDLE:
            movement_cost *= Constants::Agent::IDLE_ENERGY_MULTIPLIER;
            break;
        default:
            break;
    }
    
    energy_ -= movement_cost;
    
    // Base metabolism cost
    energy_ -= Constants::Agent::BASE_METABOLISM;
    
    // Energy gain from environment (foraging)
    if (behavior_state_ == AgentBehaviorState::FORAGING) {
        if (perceived_v_local_ > Constants::Agent::FORAGING_V_MIN && 
            perceived_v_local_ < Constants::Agent::FORAGING_V_MAX) {
            // Good foraging area
            energy_ += Constants::Agent::FORAGING_ENERGY_GAIN_GOOD;
            last_food_time_ = 0.0;
        } else if (perceived_v_local_ > Constants::Agent::MODERATE_FORAGING_V_MIN && 
                   perceived_v_local_ < Constants::Agent::MODERATE_FORAGING_V_MAX) {
            // Moderate foraging area
            energy_ += Constants::Agent::FORAGING_ENERGY_GAIN_MODERATE;
        }
    }
    
    // Track time since last food
    last_food_time_++;
    
    // Check for death
    if (energy_ <= 0.0 || age_ > Constants::Agent::MAX_AGE) { // Death by starvation or old age
        behavior_state_ = AgentBehaviorState::DEAD;
        energy_ = 0.0;
    } else {
        energy_ = std::min(energy_, Constants::Agent::MAX_ENERGY); // Cap maximum energy
    }
}

void Agent::determineNewBehaviorState() {
    if (behavior_state_ == AgentBehaviorState::DEAD) {
        return; // Can't change from dead state
    }
    
    // Priority-based state determination
    
    // Highest priority: flee from danger
    if (predator_nearby_) {
        behavior_state_ = AgentBehaviorState::FLEEING;
        return;
    }
    
    // High priority: forage when hungry
    if (energy_ < Constants::Agent::HUNGRY_THRESHOLD || last_food_time_ > Constants::Agent::FOOD_TIMEOUT) {
        behavior_state_ = AgentBehaviorState::FORAGING;
        return;
    }
    
    // Medium priority: socialize when well-fed and others nearby
    if (energy_ > Constants::Agent::WELL_FED_THRESHOLD && 
        nearby_agents_count_ > 0 && nearby_agents_count_ < Constants::Agent::MAX_SOCIAL_AGENTS) {
        behavior_state_ = AgentBehaviorState::SOCIALIZING;
        return;
    }
    
    // Low priority: move around when energy is moderate
    if (energy_ > Constants::Agent::LOW_ENERGY_THRESHOLD) {
        Utils::RandomGenerator rng;
        if (rng.uniform(0.0, 1.0) < Constants::Agent::MOVEMENT_PROBABILITY) {
            behavior_state_ = AgentBehaviorState::MOVING;
        } else {
            behavior_state_ = AgentBehaviorState::IDLE;
        }
        return;
    }
    
    // Default: idle when low energy but not critically low
    behavior_state_ = AgentBehaviorState::IDLE;
}

// Getters
double Agent::getX() const { return x_; }
double Agent::getY() const { return y_; }
int Agent::getId() const { return id_; }
double Agent::getEnergy() const { return energy_; }
AgentBehaviorState Agent::getBehaviorState() const { return behavior_state_; }
int Agent::getAge() const { return age_; }
bool Agent::isAlive() const { return behavior_state_ != AgentBehaviorState::DEAD; }

// Setters
void Agent::setEnergy(double energy) { energy_ = energy; }
void Agent::setBehaviorState(AgentBehaviorState state) { behavior_state_ = state; }

// SimulationManager class implementation
SimulationManager::SimulationManager(int width, int height,
                                   double Du, double Dv, double f, double k, double dt,
                                   int num_agents)
    : rd_system_(width, height, Du, Dv, f, k, dt),
      width_(width), height_(height), current_step_(0), next_agent_id_(num_agents) {
    
    // Create agents
    Utils::RandomGenerator rng;
    agents_.reserve(num_agents * 2); // Reserve space for potential growth
    
    for (int i = 0; i < num_agents; ++i) {
        double x = rng.uniform(0.0, static_cast<double>(width));
        double y = rng.uniform(0.0, static_cast<double>(height));
        agents_.emplace_back(i, x, y, Constants::Agent::INITIAL_ENERGY);
    }
}

void SimulationManager::initializeSimulation() {
    rd_system_.initialize();
    current_step_ = 0;
    
    std::cout << "Initialized simulation with " << getAliveAgentCount() << " agents" << std::endl;
}

void SimulationManager::runStep() {
    // Step 1: Update reaction-diffusion system
    rd_system_.step();
    
    // Step 2: Update all living agents
    for (auto& agent : agents_) {
        if (agent.isAlive()) {
            // Get nearby agents for this agent
            auto nearby = getNeighbors(agent.getX(), agent.getY(), Constants::Agent::NEIGHBOR_SEARCH_RADIUS);
            
            agent.updateState(rd_system_.getUGrid(), rd_system_.getVGrid(),
                             static_cast<double>(width_), static_cast<double>(height_),
                             nearby);
        }
    }
    
    // Step 3: Remove dead agents periodically
    if (current_step_ % Constants::Agent::DEAD_AGENT_CLEANUP_FREQUENCY == 0) {
        removeDeadAgents();
    }
    
    current_step_++;
}

void SimulationManager::runSimulation(int num_steps, int output_frequency) {
    std::cout << "Running enhanced integrated simulation for " << num_steps << " steps..." << std::endl;
    
    BiomeClassifier classifier;
    
    for (int step = 0; step < num_steps; ++step) {
        runStep();
        
        if (step % output_frequency == 0) {
            outputResults("step_" + std::to_string(step));
            
            // Also generate biome map
            auto biome_map = classifier.classifyGrid(rd_system_.getUGrid(), rd_system_.getVGrid());
            std::string biome_filename = "output/biome_map_step_" + std::to_string(step) + ".ppm";
            classifier.saveBiomeMapToPPM(biome_map, width_, height_, biome_filename);
            
            // Generate combined visualization (field + agents)
            std::vector<AgentState> agent_states;
            agent_states.reserve(agents_.size());
            for (const auto& agent : agents_) {
                if (agent.isAlive()) {
                    AgentState state;
                    state.id = agent.getId();
                    state.x = agent.getX();
                    state.y = agent.getY();
                    state.energy = agent.getEnergy();
                    state.behavior_state = agent.getBehaviorState();
                    state.age = agent.getAge();
                    state.current_biome = BiomeType::UNKNOWN; // Could be determined from position
                    agent_states.push_back(state);
                }
            }
            
            std::string combined_filename = "output/field_with_agents_step_" + std::to_string(step) + ".ppm";
            PPMWriter::writeFieldWithAgents(combined_filename, rd_system_.getVGrid().getData(),
                                          agent_states, width_, height_, 0.0, 1.0);
            
            std::cout << "Step " << step << ": Generated output files" << std::endl;
            printAgentStatistics();
        }
    }
}

void SimulationManager::outputResults(const std::string& base_filename) {
    // Save U and V fields
    std::string u_filename = "output/u_field_" + base_filename + ".ppm";
    std::string v_filename = "output/v_field_" + base_filename + ".ppm";
    
    PPMWriter::writeGrayscale(u_filename, rd_system_.getUGrid().getData(),
                            width_, height_, 0.0, 1.0);
    PPMWriter::writeGrayscale(v_filename, rd_system_.getVGrid().getData(),
                            width_, height_, 0.0, 1.0);
}

std::vector<Agent*> SimulationManager::getNeighbors(double x, double y, double radius) {
    std::vector<Agent*> neighbors;
    
    for (auto& agent : agents_) {
        if (agent.isAlive()) {
            double distance = Utils::distance(x, y, agent.getX(), agent.getY());
            if (distance <= radius) {
                neighbors.push_back(&agent);
            }
        }
    }
    
    return neighbors;
}

void SimulationManager::removeDeadAgents() {
    // Use erase-remove idiom to remove dead agents
    agents_.erase(
        std::remove_if(agents_.begin(), agents_.end(),
                      [](const Agent& agent) { return !agent.isAlive(); }),
        agents_.end()
    );
}

void SimulationManager::addAgent(double x, double y, double energy) {
    agents_.emplace_back(next_agent_id_++, x, y, energy);
}

int SimulationManager::getAliveAgentCount() const {
    return std::count_if(agents_.begin(), agents_.end(),
                        [](const Agent& agent) { return agent.isAlive(); });
}

void SimulationManager::printAgentStatistics() const {
    int alive_count = getAliveAgentCount();
    if (alive_count == 0) {
        std::cout << "  No living agents remaining!" << std::endl;
        return;
    }
    
    double avg_energy = 0.0;
    double avg_age = 0.0;
    int state_counts[6] = {0}; // For each AgentBehaviorState
    
    for (const auto& agent : agents_) {
        if (agent.isAlive()) {
            avg_energy += agent.getEnergy();
            avg_age += agent.getAge();
            state_counts[static_cast<int>(agent.getBehaviorState())]++;
        }
    }
    
    // Fixed potential division by zero
    if (alive_count > 0) {
        avg_energy /= alive_count;
        avg_age /= alive_count;
    }
    
    std::cout << "  Alive agents: " << alive_count << "/" << agents_.size() 
              << ", Avg energy: " << avg_energy 
              << ", Avg age: " << avg_age << std::endl;
    std::cout << "  Behaviors - Idle: " << state_counts[0] 
              << ", Moving: " << state_counts[1]
              << ", Foraging: " << state_counts[2] 
              << ", Fleeing: " << state_counts[3]
              << ", Socializing: " << state_counts[4] << std::endl;
}
