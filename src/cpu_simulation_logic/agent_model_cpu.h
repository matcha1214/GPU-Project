#ifndef AGENT_MODEL_CPU_H
#define AGENT_MODEL_CPU_H

#include "reaction_diffusion_cpu.h"
#include "../utils/types.h"
#include <vector>

/**
 * Agent class 
 * 
 * Each agent follows the classic percept-think-act cycle:
 * *I probably would change this afterwards into more interesting or 'animal-related/species-specific' behaviors
 * 
 */
class Agent {
public:
    // Create a new agent with an ID, position, and starting energy
    Agent(int id, double x, double y, double initial_energy = Constants::Agent::INITIAL_ENERGY);
    
    // PERCEPTION PHASE

    // Look at the immediate chemical environment (U and V concentrations)
    void perceive(const Grid& u_field, const Grid& v_field);
    
    // Look around in a bigger radius to get the neighborhood
    void perceiveNeighborhood(const Grid& u_field, const Grid& v_field, 
                             double radius = Constants::Agent::DEFAULT_PERCEPTION_RADIUS);
    
    // Check for other agents nearby (potential friends, competitors, or threats)
    void perceiveOtherAgents(const std::vector<Agent*>& nearby_agents);
    
    // THINKING PHASE
    // Process all that perceived info and decide what to do next
    void decideAction();
    
    // ACTION PHASE
    // Actually execute the decided action (usually movement)
    void act(double grid_width, double grid_height);
    
    // All-in-one update method that does the full perceive-think-act cycle
    // This is the main method called each simulation step
    void updateState(const Grid& u_field, const Grid& v_field,
                     double grid_width, double grid_height,
                     const std::vector<Agent*>& nearby_agents);
    
    // getters for inspecting agent state
    double getX() const;
    double getY() const;
    int getId() const;
    double getEnergy() const;
    AgentBehaviorState getBehaviorState() const;
    int getAge() const;
    bool isAlive() const;
    
    // setters for external modifications
    void setEnergy(double energy);
    void setBehaviorState(AgentBehaviorState state);

private:
    // Basic agent identity and position
    int id_;                    // Unique identifier
    double x_, y_;              // Continuous position
    
    // Life and energy system
    double energy_;             // Current energy level (death when <= 0)
    AgentBehaviorState behavior_state_;  // What the agent is currently doing
    int age_;                   // How old this agent is (in simulation steps)
    double last_food_time_;     // Steps since they last found food
    
    // PERCEIVED INFORMATION
    // Local chemical environment (right where the agent is)
    double perceived_u_local_;
    double perceived_v_local_;
    
    // Neighborhood averages (broader area around the agent)
    double perceived_u_avg_;
    double perceived_v_avg_;
    
    // Social situation (other agents nearby)
    int nearby_agents_count_;
    double nearest_agent_distance_;
    bool predator_nearby_;      // looking for stronger agents
    bool food_competition_;     // looking for agents competing the food
    
    // PLANNED ACTION
    // Movement vector for this step (decided in think phase, executed in act phase)
    double move_dx_ = 0.0;
    double move_dy_ = 0.0;
    
    // Internal helper methods
    void updateEnergyAndAge();        // Handle energy consumption and aging
    void determineNewBehaviorState(); // Figure out what behavior state to switch to
};

/**
 * SimulationManager
 * 
 * This class ties everything together
 */
class SimulationManager {
public:
    // Create the simulation world with specified parameters
    SimulationManager(int width, int height,
                      double Du, double Dv, double f, double k, double dt,
                      int num_agents);
    
    // Set up the initial state (initialize RD system + spawn agents)
    void initializeSimulation();
    
    // Run just one step of the simulation (RD + agents + optional output)
    void runStep();
    
    // Run the full simulation for many steps with periodic output
    void runSimulation(int num_steps, int output_frequency);
    
    // Helper for agent social interactions 
    std::vector<Agent*> getNeighbors(double x, double y, double radius);
    
    // AGENT POPULATION MANAGEMENT
    void removeDeadAgents();    // Clean up agents that have died
    void addAgent(double x, double y, double energy = Constants::Agent::INITIAL_ENERGY);
    int getAliveAgentCount() const;

private:
    // The two main simulation systems
    ReactionDiffusionSystem rd_system_;  // Generates the biome
    std::vector<Agent> agents_;          // Population of virtual creatures
    
    // Simulation state
    int width_, height_;        // World dimensions
    int current_step_ = 0;      // What simulation step 
    int next_agent_id_ = 0;     // For assigning unique IDs to new agents
    
    // Helper methods
    void outputResults(const std::string& base_filename);  // Save PPM images
    void printAgentStatistics() const;                     // Print population stats
};

#endif 