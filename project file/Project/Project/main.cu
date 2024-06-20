#pragma once
#include "preSim.cuh"
// #include "./header/globalVariables.cuh"


int main() {
    // Read nx and ny from a file
    std::ifstream inputFile("../inputs/inputs.txt");
    int nx, ny;
    if (inputFile.is_open()) {
        inputFile >> nx >> ny;
        inputFile.close();
    }
    else {
        std::cerr << "Unable to open file" << std::endl;
        return 1;
    }


    // Allocate host memory for u, v, p
    ImmerseFlow Solver;
    Solver.Input.nx = nx;
    Solver.Input.ny = ny;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&Solver.Data.u.velc, sizeof(float) * Solver.Input.nx * Solver.Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Solver.Data.v.velc, sizeof(float) * Solver.Input.nx * Solver.Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Solver.Data.p, sizeof(float) * Solver.Input.nx * Solver.Input.ny));

    // Initialize and print the CFD
    // data using CUDA
    Solver.initializeCFDData();
    Solver.printCFDData();

    

    
}



// This worked but global variable is the issue
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <iostream>

// // Namespace containing the array
// namespace MyNamespace {
//   __device__ float myArray[10]; // Use __device__ instead of __constant__
// }

// __global__ void myKernel(int nx_) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < nx_) { // Access only valid elements (array size limit)
//     MyNamespace::myArray[i] = i; // Now it's legal to write to myArray
//   }
// }

// __global__ void printKernel(int nx_) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < nx_){
//     printf("Result[%d]: %f\n", i, MyNamespace::myArray[i]);
//   }
// }

// int main() {
//   // Assign a value to nx
//   int nx = 10; // Update nx to 10 to match the array size

//   // Kernel launch parameters (adjust based on your needs)
//   int threadsPerBlock = 256;
//   int blocksPerGrid = (nx + threadsPerBlock - 1) / threadsPerBlock;

//   // Launch the kernel
//   myKernel<<<blocksPerGrid, threadsPerBlock>>>(nx);
//   cudaDeviceSynchronize(); // Ensure myKernel has finished execution

//   printKernel<<<blocksPerGrid, threadsPerBlock>>>(nx);
//   cudaDeviceSynchronize(); // Ensure printKernel has finished execution

//   // Check for errors
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     std::cerr << "Error launching kernel: " << cudaGetErrorString(err) << std::endl;
//     return 1;
//   }

//   return 0;
// }





// #include "./header/globalVariables.cuh"
// #include "./header/preSim.cuh"

// #include <iostream>

// // __global__ void printThread(){
// //   int i = threadIdx.x + blockIdx.x*blockDim.x;
// //   printf("ThreadIdx : %d \n",i);

// // }

// int main() {
//   Initialize Initialize;
//   Grid Grid;
//   // printCudaArray<<<4,4>>>();
//   printThread<<<4,4>>>();
//   cudaError_t cudaStatus = cudaGetLastError();
//   if (cudaStatus != cudaSuccess) {
//       fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//       return 1;
//   }

//   return 0;
// }


// // #include "./header/preSim.cuh"
// // #include "header/postSim.cuh"
// #include "./header/globalVariables.cuh"
// #include <fstream>

// // Main Function
// int main(){
//   GlobalVariables::Device::N = 6;
//   printf("dimensions are %i\n", GlobalVariables::Device::N);

//   return 0;
// }

// #include "GlobalVariables.h"

// __global__ void myKernel() {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < GlobalVariables::Device::N) {
//         GlobalVariables::Device::dataArray[idx] = idx * 2.0;
//     }
// }

// int main() {
//     const int N = 1000;
//     GlobalVariables::initialize(N);

//     // Launch kernel
//     myKernel<<<(N + 255) / 256, 256>>>();

//     // Cleanup
//     GlobalVariables::cleanup();

//     return 0;
// }

