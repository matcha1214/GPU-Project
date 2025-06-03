#include "reaction_diffusion_cpu.h"
#include "../common/ppm_writer.h"
#include "../common/utils.h"
#include <fstream>
#include <algorithm>
#include <iostream>
#include <random>

// ============================================================================
// Grid class implementation - Our 2D wrapper around a 1D vector
// ============================================================================

Grid::Grid(int width, int height, double initial_value) 
    : width_(width), height_(height), data_(width * height, initial_value) {
    // just allocate flat array and fill it
}

double Grid::get(int r, int c) const {
    // periodic boundaries
    // If you ask for coordinates outside the grid, the code wrap them around
    // This makes our world toroidal (like Pac-Man)
    int wrapped_r = Utils::wrapCoordinate(r, height_);
    int wrapped_c = Utils::wrapCoordinate(c, width_);
    return data_[wrapped_r * width_ + wrapped_c];
}

void Grid::set(int r, int c, double value) {
    // Same wrapping logic for setting values
    if (width_ > 0 && height_ > 0) {
        int wrapped_r = Utils::wrapCoordinate(r, height_);
        int wrapped_c = Utils::wrapCoordinate(c, width_);
        data_[wrapped_r * width_ + wrapped_c] = value;
    }
}

int Grid::getWidth() const { return width_; }
int Grid::getHeight() const { return height_; }

std::vector<double>& Grid::getData() { return data_; }
const std::vector<double>& Grid::getData() const { return data_; }

void Grid::initialize(double u_default, double v_default,
                      const CoordinateList& patch_coords,
                      double u_patch, double v_patch) {
    // This method is mainly for U grid initialization, so ignore some V parameters
    (void)v_default;
    (void)v_patch;
    
    // Start with a uniform field of the default value
    std::fill(data_.begin(), data_.end(), u_default);
    
    // Then add some "seed" patches to break the symmetry and start pattern formation
    for (const auto& coord : patch_coords) {
        int r = coord.first;
        int c = coord.second;
        set(r, c, u_patch);  // This handles boundary wrapping automatically
    }
}

// ============================================================================
// ReactionDiffusionSystem class implementation
// ============================================================================

ReactionDiffusionSystem::ReactionDiffusionSystem(int width, int height,
                                                 double Du, double Dv, double f, double k, double dt)
    : u_grid_(width, height, 1.0), v_grid_(width, height, 0.0),  // U starts at 1, V starts at 0
      u_next_grid_(width, height, 0.0), v_next_grid_(width, height, 0.0),
      laplacian_u_(width, height, 0.0), laplacian_v_(width, height, 0.0),
      Du_(Du), Dv_(Dv), f_(f), k_(k), dt_(dt), width_(width), height_(height) {
    // All the grids are ready
}

ReactionDiffusionSystem::ReactionDiffusionSystem(const SimulationParams& params)
    : ReactionDiffusionSystem(params.width, params.height, params.Du, params.Dv, 
                             params.f, params.k, params.dt) {
    // Delegate to the main constructor
}

void ReactionDiffusionSystem::initialize() {
    // Set up the initial conditions 
    // U field starts at 1.0 everywhere (plenty of chemical U available)
    std::fill(u_grid_.getData().begin(), u_grid_.getData().end(), 1.0);
    // V field starts at 0.0 everywhere (no chemical V initially)
    std::fill(v_grid_.getData().begin(), v_grid_.getData().end(), 0.0);
    
    // add a small patch of V in the center
    int patch_size = 10;
    int center_x = width_ / 2;
    int center_y = height_ / 2;
    
    Utils::RandomGenerator rng;  // For adding a bit of noise
    
    // Create a small square patch with some randomness
    for (int r = center_y - patch_size/2; r < center_y + patch_size/2; ++r) {
        for (int c = center_x - patch_size/2; c < center_x + patch_size/2; ++c) {
            // Add some noise to make things more interesting
            u_grid_.set(r, c, 0.5 + rng.uniform(-0.1, 0.1));
            v_grid_.set(r, c, 0.25 + rng.uniform(-0.1, 0.1));
        }
    }
}

void ReactionDiffusionSystem::compute_laplacian(const Grid& input_grid, Grid& output_laplacian_grid) {
    // The Laplacian measures how much a value differs from its neighbors
    // why? diffusio literally means chemicals spread from high to low concentrations
    
    for (int r = 0; r < height_; ++r) {
        for (int c = 0; c < width_; ++c) {
            // use a 3x3 weighted stencil that includes diagonal neighbors
            // smoother, more natural diffusion than just 4-neighbor stencils
            
            // Get wrapped coordinates for all 8 neighbors + center
            int xp1 = Utils::wrapCoordinate(c + 1, width_);   // East
            int xm1 = Utils::wrapCoordinate(c - 1, width_);   // West  
            int yp1 = Utils::wrapCoordinate(r + 1, height_);  // South
            int ym1 = Utils::wrapCoordinate(r - 1, height_);  // North
            
            double center = input_grid.get(r, c);
            
            // The 4 orthogonal neighbors get weight 0.2 each
            double north = input_grid.get(ym1, c);
            double south = input_grid.get(yp1, c);
            double west = input_grid.get(r, xm1);
            double east = input_grid.get(r, xp1);
            
            // The 4 diagonal neighbors get weight 0.05 each
            double northwest = input_grid.get(ym1, xm1);
            double northeast = input_grid.get(ym1, xp1);
            double southwest = input_grid.get(yp1, xm1);
            double southeast = input_grid.get(yp1, xp1);
            
            // Apply the weighted stencil - this is the discrete Laplacian
            // The center gets weight -1.0, so we're measuring how different it is from neighbors
            double laplacian = 0.2 * (north + south + west + east) +
                              0.05 * (northwest + northeast + southwest + southeast) +
                              (-1.0) * center;
            
            output_laplacian_grid.set(r, c, laplacian);
        }
    }
}

void ReactionDiffusionSystem::step() {
    // one time step of the Gray-Scott model
    
    // Step 1: Calculate how much each chemical wants to diffuse
    compute_laplacian(u_grid_, laplacian_u_);
    compute_laplacian(v_grid_, laplacian_v_);
    
    // Step 2: Apply the Gray-Scott equations to every cell
    for (int r = 0; r < height_; ++r) {
        for (int c = 0; c < width_; ++c) {
            // Get current concentrations and their Laplacians
            double u = u_grid_.get(r, c);
            double v = v_grid_.get(r, c);
            double lap_u = laplacian_u_.get(r, c);
            double lap_v = laplacian_v_.get(r, c);
            
            // The reaction term: U + 2V -> 3V 
            // This is what makes V grow by consuming U
            double uvv = u * v * v;
            
            // Gray-Scott equation for U:
            // dU/dt = Du * ∇²U - uvv + f(1-u)
            // - Du * ∇²U: diffusion (spreading)
            // - uvv: gets consumed in reaction with V
            // + f(1-u): fresh U gets fed in at rate f
            double u_next = u + (Du_ * lap_u - uvv + f_ * (1.0 - u)) * dt_;
            
            // Gray-Scott equation for V:
            // dV/dt = Dv * ∇²V + uvv - (f+k)v
            // - Dv * ∇²V: diffusion (spreading)
            // + uvv: gets created from reaction with U
            // - (f+k)v: gets killed off at rate f+k
            double v_next = v + (Dv_ * lap_v + uvv - (f_ + k_) * v) * dt_;
            
            // Store the new values (double buffering - can't modify while reading)
            u_next_grid_.set(r, c, u_next);
            v_next_grid_.set(r, c, v_next);
        }
    }
    
    // Step 3: Swap the buffers - next becomes current
    // for efficiency - just swapping pointers to the underlying data
    u_grid_.getData().swap(u_next_grid_.getData());
    v_grid_.getData().swap(v_next_grid_.getData());
}

// Simple getters for accessing the grids
const Grid& ReactionDiffusionSystem::getUGrid() const { return u_grid_; }
const Grid& ReactionDiffusionSystem::getVGrid() const { return v_grid_; }
