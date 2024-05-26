#ifndef PRESIM_CUH
#define PRESIM_CUH

#include <cuda_runtime.h>

// Define a struct to hold the CFD arrays
struct CFDData {
  float* u;
  float* v;
  float* p;
};

// Declare a global instance of the struct as a device variable
__device__ CFDData deviceData;

// Function prototypes
void initializeCFDData(int nx, int ny);
void printCFDData(int nx, int ny);

#endif // PRESIM_CUH

// #ifndef PRESIM_CUH
// #define PRESIM_CUH

// #include <cuda_runtime.h>

// // namespace MyNamespace {
// //   // Declare the global device pointer
// //   __device__ float* d_myArray;
// // }

// // Function prototypes
// void initializeArray(int nx);
// void printArray(int nx);

// #endif // PRESIM_CUH




// #ifndef INPUT_READER_H_
// #define INPUT_READER_H_

// #include <fstream>
// #include <iostream>
// #include "globalVariables.cuh"
// #include <cuda_runtime.h>
// #include <cstdio>

// // void setZero(float* vec, int rows, int cols);
// __global__ void printThread();

// class Initialize {
// private:
//     int N_;
//     int nx_;
//     int ny_;

// public:
//     Initialize();
//     void InputReader();
//     void InitializeArrays();
// };

// class Grid {
// private:
//     int nx_;
//     int ny_;
//     float* x_;
//     float* y_;

// public:
//     Grid();
//     // ~Grid(); // Destructor to release memory
//     void createGrid();
    
//     // void calculateiBlank();
// };

// #endif // INPUT_READER_H_
