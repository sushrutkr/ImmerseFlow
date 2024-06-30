# Define variables
NVCC = nvcc
SRCDIR = include
OBJDIR = obj
TARGET = immerseFlow

# Source files
SRC = $(SRCDIR)/globalVariables.cu $(SRCDIR)/preSim.cu $(SRCDIR)/postSim.cu main.cu

# Object files
OBJ = $(OBJDIR)/globalVariables.o $(OBJDIR)/preSim.o $(OBJDIR)/postSim.o $(OBJDIR)/main.o

# Default target
all: $(TARGET)

# Compile source files to object files
$(OBJDIR)/globalVariables.o: $(SRCDIR)/globalVariables.cu
	$(NVCC) -c $< -o $@

$(OBJDIR)/preSim.o: $(SRCDIR)/preSim.cu
	$(NVCC) -c $< -o $@

$(OBJDIR)/postSim.o: $(SRCDIR)/postSim.cu
	$(NVCC) -c $< -o $@

$(OBJDIR)/main.o: main.cu
	$(NVCC) -c $< -o $@

# Link object files to create the final executable
$(TARGET): $(OBJ)
	$(NVCC) $^ -o $@ -run

# Clean up
clean:
	rm -f $(OBJDIR)/*.o $(TARGET)

# Phony targets
.PHONY: all clean
