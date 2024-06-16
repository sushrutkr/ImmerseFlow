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
    CHECK_CUDA_ERROR(cudaMalloc((void**)&devData.u.velc, sizeof(float) * nx * ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&devData.v.velc, sizeof(float) * nx * ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&devData.p, sizeof(float) * nx * ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&ibm.iblank, sizeof(float) * nx * ny));
    
    // Initialize and print the CFD data using CUDA
    initializeCFDData(nx, ny, devData);

    // Free allocated memory
    CHECK_CUDA_ERROR(cudaFree(devData.u.velc));
    CHECK_CUDA_ERROR(cudaFree(devData.v.velc));
    CHECK_CUDA_ERROR(cudaFree(devData.p));
    CHECK_CUDA_ERROR(cudaFree(ibm.iblank));

    return 0;
}
