rm obj/*

# Compile include/
# nvcc -c include/globalVariables.cu -o obj/globalVariables.o
nvcc -c include/preSim.cu -o obj/preSim.o
# nvcc -c include/postSim.cu -o obj/postSim.o

# Compile main.cpp
nvcc -c main.cu -o obj/main.o

# Link object files together to create the final executable
# nvcc obj/globalVariables.o obj/preSim.o obj/postSim.o main.o -o immerseFlow -run
nvcc obj/* -o immerseFlow -run
