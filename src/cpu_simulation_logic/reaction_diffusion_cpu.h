#ifndef REACTION_DIFFUSION_CPU_H
#define REACTION_DIFFUSION_CPU_H

#include "../utils/types.h"
#include <vector>
#include <string>

/**
 * Grid class 
 */
class Grid {
public:
    // Create a grid and fill it with some initial value (usually 0.0 or 1.0)
    Grid(int width, int height, double initial_value = 0.0);
    
    // Get/set values at row r, column c
    // These automatically handle periodic boundary conditions (wrapping around edges)
    double get(int r, int c) const;
    void set(int r, int c, double value);
    
    // Basic info about our grid
    int getWidth() const;
    int getHeight() const;
    
    // Direct access to the underlying data - for PPM output
    // and eventually for copying to GPU memory
    std::vector<double>& getData();
    const std::vector<double>& getData() const;
    
    // Initialize with default values 
    void initialize(double u_default, double v_default,
                    const CoordinateList& patch_coords,
                    double u_patch, double v_patch);

private:
    int width_, height_;
    GridData data_; // Using common type aias 
};

/**
 * ReactionDiffusionSystem 
 */
class ReactionDiffusionSystem {
public:
    // Constructor with explicit parameters
    ReactionDiffusionSystem(int width, int height,
                            double Du, double Dv, double f, double k, double dt);
    
    // Constructor using the config system
    ReactionDiffusionSystem(const SimulationParams& params);
    
    // Set up initial conditions (U=1 everywhere, V=0 except for a seed patch)
    void initialize();
    
    // Run one time step of the simulation 
    void step();
    
    // Get read-only access to concentration fields
    const Grid& getUGrid() const;
    const Grid& getVGrid() const;

private:
    // Current state grids 
    Grid u_grid_;
    Grid v_grid_;
    
    // Next state grids 
    Grid u_next_grid_;
    Grid v_next_grid_;
    
    // Temporary storage for Laplacian calculations 
    Grid laplacian_u_;
    Grid laplacian_v_;
    
    // Gray-Scott parameters 
    double Du_, Dv_;  // Diffusion rates for U and V
    double f_, k_;    // Feed and kill rates
    double dt_;       // Time step size
    int width_, height_;
    
    // The Laplacian operator
    void compute_laplacian(const Grid& input_grid, Grid& output_laplacian_grid);
};

#endif 