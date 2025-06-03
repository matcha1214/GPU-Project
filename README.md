# CS179 GPU Programming Project: Biome-Agent Simulation

## Project Proposal

### Summary

Inspired by the organic and varied biome maps in games like Minecraft, this project develops a GPU-accelerated two-dimensional reaction-diffusion system to generate evolving environmental patterns. These patterns are classified into biome types, which then serve as an environment for a simple agent-based model simulating life forms that interact with these biomes. The primary benefits of GPU acceleration include significant speedup for both the pattern generation and the agent simulation, enabling larger, more dynamic worlds.

### Background Information

#### Computation Details

The project involves two main computational phases. First, a reaction-diffusion (RD) system on an N×N grid (e.g., targeting 512×512) evolves two fields (U and V) over T time steps. Each step involves O(N²) work for 5-point stencil diffusion and local reactions per field. This RD phase is highly data-parallel, ideal for GPU acceleration, with regular memory access patterns suited for shared memory tiling.

Second, an Agent-Based Model (ABM) will simulate A agents moving and interacting on the biome map derived from the RD system. Each agent's update per time step is also data-parallel. For a large number of agents, GPU acceleration is crucial for real-time interaction or large-scale simulation.

#### Project Explanation

The system will be implemented in C++ with CUDA and will comprise four main components:

1. **Initial Condition Generation (RD)**: The U and V fields for the reaction-diffusion system will be initialized on the GPU using the cuRAND library to create random fields, potentially with basic spatial patterns to encourage diverse evolutions.

2. **Reaction-Diffusion Simulation & Biome Classification (GPU)**: A simplified Gray-Scott model will evolve the U and V fields. A CUDA kernel will use shared memory tiling and double buffering for this. After a set number of RD steps, another CUDA kernel will classify each grid cell into one of several biome types (e.g., "forest," "grassland," "desert") based on U/V thresholds. This generates a biome map.

3. **Agent-Based Model (GPU)**: A population of agents will then be simulated on this biome map.
   - **Agent State**: Each agent will have a position and a simple internal state (e.g., energy).
   - **Agent Behavior**: Agents will perceive the biome type at their location. Their movement will be rule-based (e.g., moving towards preferred biomes to "graze" or find resources, random exploration). They will consume/gain energy based on actions and biome type. Agents might leave a simple trace on a separate "interaction layer" (e.g., "grazed" cells) rather than directly modifying the RD-generated biome types.

4. **Visualization**: The biome map, agent positions, and any interaction layer will be copied to the CPU and visualized as PPM images.

### Technical Challenges

- **RD System**: Efficient shared memory tiling, numerical stability of the Gray-Scott model, and effective use of cuRAND for initial conditions.
- **ABM System**: Managing state for many agents on the GPU; designing parallel-friendly rules for agent perception, decision-making, and movement; handling agent interactions with the environment potentially requiring atomic operations or careful data structuring to avoid race conditions.
- **Coupling RD and ABM**: Ensuring a clean interface and data flow between the RD/classification output and the ABM input.
- **Performance**: Achieving good performance for both RD and ABM components, and balancing their workloads.

### Deliverables and Goals

- **CPU Baseline**: A serial C++ program for the reaction-diffusion and classification parts. A conceptual CPU version of the ABM rules will also be developed for logic validation. PPM output for visualization.
- **GPU Implementation**: CUDA RD kernel (cuRAND ICs, shared memory, double buffering), CUDA biome classification kernel, CUDA ABM kernel(s) for simulating agent behavior.
- **Visualization**: PPM images showing the generated biomes, agent locations, and any environmental interaction layer.
- **Performance Report**: Timing comparisons for the RD/classification part (CPU vs. GPU) aiming for at least 8x speedup on a 512x512 grid. Performance metrics for the ABM part. Analysis of bottlenecks and optimizations.

Success will be defined by a functional GPU-accelerated system where reaction-diffusion generates varied biome maps, and a population of agents demonstrably interacts with this environment according to their rules, all within a reasonable simulation time.

## CPU Demo Introduction

This repository contains a comprehensive CPU demonstration of our biome generation and agent simulation system. The CPU demo serves as both a standalone working implementation and the foundation for our upcoming GPU port.

### What's Implemented

 CPU demo includes:

1. **Gray-Scott Reaction-Diffusion System**: Generates dynamic, organic patterns using the classic two-chemical model
2. **Agent-Based Model**: Virtual creatures that live in and respond to the chemical environment
3. **Biome Classification**: Converts raw chemical concentrations into discrete biome types
4. **Integrated Simulation**: Combines all components into a living ecosystem
5. **Configuration**: Easy parameter tweaking via config files or command line
6. **Visualization**: Multiple PPM output formats for analyzing results

### The Science Behind It

The **Gray-Scott model** simulates two chemicals (U and V) that diffuse and react:
- U + 2V → 3V (V consumes U to reproduce)
- U is fed into the system at rate F
- V is removed at rate (F + k)

By tweaking the diffusion rates (Du, Dv) and reaction rates (F, k), you get wildly different patterns:
- Coral-like growth structures
- Moving spots and stripes  
- Mitosis-like cell division patterns
- Complex maze-like networks

The **agents** then live in this dynamic chemical world, with behaviors like:
- **Foraging**: Moving toward favorable chemical conditions
- **Fleeing**: Avoiding threats (stronger agents) 
- **Socializing**: Interacting with nearby agents
- **Energy management**: Balancing activity with survival

### Building and Running for the CPU demo

```bash
# Build everything
make all

# Run the integrated demo (reaction-diffusion + agents)
./cpu_demo_integrated

# Run reaction-diffusion only
./cpu_demo

# Try different presets
./cpu_demo_integrated --preset coral --agents 200
./cpu_demo --preset mitosis --steps 2000

# Custom parameters
./cpu_demo_integrated --width=256 --height=256 --agents=150 --steps=1000

# Use a config file
./cpu_demo_integrated --config example_config.conf

# See all options
./cpu_demo_integrated --help
```

### What You'll Get

The demos generate several types of visualization files in the `output/` directory:

- **`u_field_step_*.ppm`**: The U chemical concentration (substrate)
- **`v_field_step_*.ppm`**: The V chemical concentration (catalyst) - usually more visually interesting!
- **`biome_map_step_*.ppm`**: Classified biome types with distinct colors
- **`field_with_agents_step_*.ppm`**: V field with agent positions overlaid as colored dots

### Viewing Results

PPM files can be viewed in most image viewers. To convert to more common formats:

```bash
# Convert to PNG using ImageMagick
convert output/v_field_step_1000.ppm output/v_field_final.png
convert output/field_with_agents_step_1000.ppm output/agents_final.png

# Create an animation from the sequence
convert output/v_field_step_*.ppm output/evolution.gif
```

### Opening PPM Files

```bash
# macOS - open with default viewer
open output/biome_map_step_100.ppm
open output/field_with_agents_step_100.ppm

# Other platforms - convert first if needed
convert output/biome_map_step_100.ppm output/biome_map_step_100.png
```

### Available Presets

The system includes several predefined parameter sets that produce different patterns:

- **`coral`** (default): Organic, coral-like growth patterns
- **`mitosis`**: Cell division-like splitting patterns  
- **`spots`**: Stable spotted patterns

```bash
# Try each preset
./cpu_demo --preset coral --steps 1500
./cpu_demo --preset mitosis --steps 2000
./cpu_demo --preset spots --steps 1000
```

### Configuration Files

Create custom configurations in `.conf` files:

```ini
# My custom config
width=256
height=256

# Gray-Scott parameters for coral growth
Du=0.16
Dv=0.08
f=0.0545
k=0.062
dt=1.0

# Simulation control
steps=2000
output_freq=100

# Agent parameters
agents=200
agent_speed=1.5
perception_radius=3.0
```

### Command Line Parameters

Key parameters you can adjust:

#### Grid and Timing
- `--width`, `--height`: Grid dimensions (default: 128x128)
- `--steps`: Number of simulation steps (default: 1000)
- `--output_freq`: How often to save images (default: 50)

#### Gray-Scott Model
- `--Du`, `--Dv`: Diffusion rates for U and V chemicals
- `--f`: Feed rate (how fast U is supplied)
- `--k`: Kill rate (how fast V is removed)
- `--dt`: Time step size (default: 1.0)

#### Agents (integrated demo only)
- `--agents`: Number of agents (default: 100)
- `--agent_speed`: How fast agents move (default: 1.0)
- `--perception_radius`: How far agents can "see" (default: 2.0)

### Performance Analysis

The demos include built-in timing to help establish performance baselines:

```bash
# Example output
=== Simulation Complete! ===
Total simulation time: 12.34 seconds
Time per step: 0.012 seconds
```

This CPU performance will be our baseline for measuring GPU acceleration improvements.

## Visualizations

### Reaction-Diffusion Fields

**U Field (u_field_step_*.ppm)**:
- **Black/Dark**: Low U concentration
- **White/Bright**: High U concentration  
- Usually starts white and develops dark patterns as V consumes U

**V Field (v_field_step_*.ppm)**:
- **Black/Dark**: No V chemical present
- **White/Bright**: High V concentration
- This is typically the most visually interesting - shows the growing patterns

### Biome Classification

**Biome Map (biome_map_step_*.ppm)**:
Different colors represent different biome types based on U/V thresholds:
- **Brown**: Desert (low V, moderate U)
- **Green**: Forest (moderate V, good U)
- **Blue**: Water (high V concentration)
- **Gray**: Mountain (very low V and U)
- **Light Green**: Grassland (balanced concentrations)

### Agent Visualization

**Agents with Field (field_with_agents_step_*.ppm)**:
- Shows the V field as background (grayscale)
- Colored dots represent agents:
  - **Green**: Foraging agents (actively seeking food)
  - **Red**: Fleeing agents (avoiding threats)
  - **Blue**: Socializing agents (interacting with others)
  - **Yellow**: Idle agents (resting/wandering)
  - **White**: Moving agents (general movement)

### Pattern Interpretation

Different Gray-Scott parameters create distinct visual signatures:

**Coral Growth (F≈0.055, k≈0.062)**:
- Branching, tree-like structures
- Organic, natural appearance
- Good for forest-like biomes

**Mitosis (F≈0.037, k≈0.065)**:
- Circular spots that split and divide
- Cell-like behavior
- Creates interesting dynamic environments

**Spots (F≈0.035, k≈0.065)**:
- Stable, evenly-spaced dots
- More static patterns
- Good for resource distribution simulations

## Test and Examples

### Basic Functionality Tests

Create these simple config files to test different aspects:

**test_simple.conf** - Minimal test:
```ini
width=64
height=64
steps=100
output_freq=25
agents=10
```

**test_performance.conf** - Performance test:
```ini
width=256
height=256
steps=500
output_freq=100
agents=200
```

**test_patterns.conf** - Pattern exploration:
```ini
width=200
height=200
Du=0.16
Dv=0.08
f=0.0545
k=0.062
steps=1500
output_freq=75
```

### Running Tests

```bash
# Quick functionality test
./cpu_demo_integrated --config test_simple.conf

# Performance benchmark
./cpu_demo_integrated --config test_performance.conf

# Pattern visualization
./cpu_demo --config test_patterns.conf

# Agent behavior test
./cpu_demo_integrated --agents=50 --steps=500 --perception_radius=5.0
```

### Validation Checks

To verify the implementation is working correctly:

1. **Pattern Formation**: V field should develop from a small seed into complex patterns
2. **Agent Movement**: Agents should move around and change colors based on behavior
3. **Energy System**: Agent count should gradually decrease as some die from old age
4. **Biome Response**: Agents should cluster in favorable biome areas (green/forest regions)

### Expected Performance Baselines

On a modern CPU, expect roughly:
- **128x128 grid**: ~0.005-0.01 seconds per step
- **256x256 grid**: ~0.02-0.04 seconds per step
- **512x512 grid**: ~0.1-0.2 seconds per step

Agent overhead typically adds 20-50% to simulation time depending on population size.


