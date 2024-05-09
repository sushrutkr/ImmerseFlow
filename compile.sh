# Compile include/
nvcc -c include/preSim.cu -o include/preSim.o
nvcc -c include/postSim.cu -o include/postSim.o

# Compile main.cpp
nvcc -c main.cu -o main.o

# Link object files together to create the final executable
nvcc include/preSim.o include/postSim.o main.o -o immerseFlow -run
