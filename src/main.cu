#include "./header/preSim.cuh"
#include "./header/globalVariables.cuh"
#include <iostream>
#include <fstream>

int main() {
    int nx, ny;

    // Read nx and ny from a file
    std::ifstream inputFile("../inputs/inputs.txt");
    if (inputFile.is_open()) {
        inputFile >> nx >> ny;
        inputFile.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
        return 1;
    }


    // Allocate memory for CFD data
    ImmerseFlow Solver;
    Solver.Input.nx = nx;
    Solver.Input.ny = ny;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&Solver.Data.u.velc, sizeof(float) * Solver.Input.nx * Solver.Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Solver.Data.v.velc, sizeof(float) * Solver.Input.nx * Solver.Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Solver.Data.p, sizeof(float) * Solver.Input.nx * Solver.Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Solver.gridData.x, sizeof(float) * Solver.Input.nx));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Solver.gridData.y, sizeof(float) * Solver.Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Solver.ibm.iBlank, sizeof(float) * Solver.Input.nx * Solver.Input.ny));
    
    // Initialize and print the CFD data using CUDA
    Solver.readGridData();
    Solver.initializeData();

    // Free allocated memory
    CHECK_CUDA_ERROR(cudaFree(Solver.gridData.x));
    CHECK_CUDA_ERROR(cudaFree(Solver.gridData.y));
    CHECK_CUDA_ERROR(cudaFree(Solver.Data.u.velc));
    CHECK_CUDA_ERROR(cudaFree(Solver.Data.v.velc));
    CHECK_CUDA_ERROR(cudaFree(Solver.Data.p));
    CHECK_CUDA_ERROR(cudaFree(Solver.ibm.iBlank));

    return 0;
}
