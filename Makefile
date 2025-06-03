# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# Directories
SRC_DIR = src
CPU_DIR = $(SRC_DIR)/cpu
COMMON_DIR = $(SRC_DIR)/common
OUTPUT_DIR = output

# Source files
CPU_CORE_SOURCES = $(CPU_DIR)/reaction_diffusion_cpu.cpp $(CPU_DIR)/agent_model_cpu.cpp $(CPU_DIR)/biome_classifier_cpu.cpp
COMMON_SOURCES = $(COMMON_DIR)/ppm_writer.cpp $(COMMON_DIR)/utils.cpp $(COMMON_DIR)/config.cpp $(COMMON_DIR)/timer.cpp

# Header files
CPU_HEADERS = $(CPU_DIR)/reaction_diffusion_cpu.h $(CPU_DIR)/agent_model_cpu.h $(CPU_DIR)/biome_classifier_cpu.h
COMMON_HEADERS = $(COMMON_DIR)/types.h $(COMMON_DIR)/ppm_writer.h $(COMMON_DIR)/utils.h $(COMMON_DIR)/config.h $(COMMON_DIR)/timer.h
ALL_HEADERS = $(CPU_HEADERS) $(COMMON_HEADERS)

# Target executables
TARGET_BASIC = cpu_demo
TARGET_INTEGRATED = cpu_demo_integrated

# Default target
all: $(TARGET_BASIC) $(TARGET_INTEGRATED)

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Compile the basic CPU demo (RD only, now with configuration system)
$(TARGET_BASIC): $(CPU_CORE_SOURCES) $(CPU_DIR)/main_RD.cpp $(COMMON_SOURCES) $(ALL_HEADERS) | $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS) -o $(TARGET_BASIC) $(CPU_CORE_SOURCES) $(CPU_DIR)/main_RD.cpp $(COMMON_SOURCES)

# Compile the integrated demo (RD + Agents + Biomes)
$(TARGET_INTEGRATED): $(CPU_CORE_SOURCES) $(CPU_DIR)/main_integrated.cpp $(COMMON_SOURCES) $(ALL_HEADERS) | $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS) -o $(TARGET_INTEGRATED) $(CPU_CORE_SOURCES) $(CPU_DIR)/main_integrated.cpp $(COMMON_SOURCES)

# Clean build artifacts
clean:
	rm -f $(TARGET_BASIC) $(TARGET_INTEGRATED)
	rm -rf $(OUTPUT_DIR)

# Run the basic demo (RD only, now configurable)
run: $(TARGET_BASIC)
	./$(TARGET_BASIC)

# Run integrated demo with agents and biome classification
run-integrated: $(TARGET_INTEGRATED)
	./$(TARGET_INTEGRATED)

# Test with different Gray-Scott parameters using the enhanced basic demo
test-mitosis: $(TARGET_BASIC)
	@echo "Testing with mitosis parameters..."
	./$(TARGET_BASIC) --preset mitosis --steps 1000

test-spots: $(TARGET_BASIC)
	@echo "Testing with spots parameters..."
	./$(TARGET_BASIC) --preset spots --steps 1000

test-large: $(TARGET_BASIC)
	@echo "Testing with larger grid..."
	./$(TARGET_BASIC) --width=256 --height=256 --steps=500

# Demonstration of configuration file usage
demo-config: $(TARGET_BASIC)
	@echo "Creating example configuration file..."
	@echo "# Example configuration for spots pattern" > example_rd_config.conf
	@echo "width=200" >> example_rd_config.conf
	@echo "height=200" >> example_rd_config.conf
	@echo "Du=0.16" >> example_rd_config.conf
	@echo "Dv=0.08" >> example_rd_config.conf
	@echo "f=0.035" >> example_rd_config.conf
	@echo "k=0.065" >> example_rd_config.conf
	@echo "steps=800" >> example_rd_config.conf
	@echo "output_frequency=40" >> example_rd_config.conf
	@echo "Running simulation with configuration file..."
	./$(TARGET_BASIC) --config example_rd_config.conf

# Help target
help:
	@echo "Available targets:"
	@echo "  all               - Build both demos (default)"
	@echo "  clean             - Remove build artifacts and output"
	@echo "  run               - Build and run the configurable RD demo"
	@echo "  run-integrated    - Build and run the integrated demo (RD + Agents + Biomes)"
	@echo "  test-mitosis      - Run RD demo with mitosis pattern parameters"
	@echo "  test-spots        - Run RD demo with spots pattern parameters"
	@echo "  test-large        - Run with larger grid"
	@echo "  demo-config       - Demonstrate configuration file usage"
	@echo "  help              - Show this help message"
	@echo ""
	@echo "For RD demo usage, run: ./$(TARGET_BASIC) --help"
	@echo "The basic demo now includes full configuration support!"

.PHONY: all clean run run-integrated test-mitosis test-spots test-large demo-config help
