# Compiler and flags
NVCC = nvcc
CXXFLAGS = -arch=sm_60 -std=c++11

# Directories
INCDIR = include
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Source files
SRCFILES = $(wildcard $(SRCDIR)/*.cu) $(wildcard $(INCDIR)/*.cu)
OBJFILES = $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(wildcard $(SRCDIR)/*.cu)) $(patsubst $(INCDIR)/%.cu, $(OBJDIR)/%.o, $(wildcard $(INCDIR)/*.cu))

# Executable name
EXEC = $(BINDIR)/immerseFlow

# Default target
all: $(EXEC)

# Build executable
$(EXEC): $(OBJFILES)
	mkdir -p $(BINDIR)
	$(NVCC) $(CXXFLAGS) -o $@ $^ -run

# Compile CUDA source files from src directory
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	mkdir -p $(OBJDIR)
	$(NVCC) $(CXXFLAGS) -c -o $@ $<

# Compile CUDA source files from include directory
$(OBJDIR)/%.o: $(INCDIR)/%.cu
	mkdir -p $(OBJDIR)
	$(NVCC) $(CXXFLAGS) -c -o $@ $<

# Clean
clean:
	rm -rf $(OBJDIR)/* $(BINDIR)/*

.PHONY: all clean
