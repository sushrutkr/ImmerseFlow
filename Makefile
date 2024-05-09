# Compiler and flags
NVCC := nvcc
NVCC_FLAGS := -std=c++11

# Directories
SRC_DIR := .
INCLUDE_DIR := include
OBJ_DIR := obj
BIN_DIR := bin

# Files
SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC_FILES))
DEP_FILES := $(OBJ_FILES:.o=.d)

# Target executable
TARGET := $(BIN_DIR)/main

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ_FILES) | $(BIN_DIR)
    $(NVCC) $(NVCC_FLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
    $(NVCC) $(NVCC_FLAGS) -MMD -MP -c $< -o $@

-include $(DEP_FILES)

$(OBJ_DIR):
    mkdir -p $@

$(BIN_DIR):
    mkdir -p $@

clean:
    $(RM) -r $(OBJ_DIR) $(BIN_DIR)
