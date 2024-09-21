# Define variables
NVCC = nvcc
SRCDIR = include
OBJDIR = obj
TARGET = immerseFlow

# Source files
SRC = $(SRCDIR)/globalVariables.cu $(SRCDIR)/preSim.cu $(SRCDIR)/ADSolver.cu $(SRCDIR)/PPESolver.cu $(SRCDIR)/AD_PPE_Correction.cu $(SRCDIR)/postSim.cu main.cu

# Object files
OBJ = $(OBJDIR)/globalVariables.o $(OBJDIR)/preSim.o $(OBJDIR)/ADSolver.o $(OBJDIR)/PPESolver.o $(OBJDIR)/AD_PPE_Correction.o $(OBJDIR)/postSim.o $(OBJDIR)/main.o

# Default target
all: $(TARGET)

# Compile source files to object files
$(OBJDIR)/globalVariables.o: $(SRCDIR)/globalVariables.cu
		$(NVCC) -c $< -o $@

$(OBJDIR)/preSim.o: $(SRCDIR)/preSim.cu
		$(NVCC) -c $< -o $@

$(OBJDIR)/ADSolver.o: $(SRCDIR)/ADSolver.cu
		$(NVCC) -c $< -o $@

$(OBJDIR)/PPESolver.o: $(SRCDIR)/PPESolver.cu
		$(NVCC) -c $< -o $@

$(OBJDIR)/AD_PPE_Correction.o: $(SRCDIR)/AD_PPE_Correction.cu
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