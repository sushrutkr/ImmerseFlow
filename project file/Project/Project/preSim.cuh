#pragma once
#ifndef PRESIM_CUH
#define PRESIM_CUH

#include "MACRO.h"

struct vel {
    float* velf, * velc, * velInter;
};

// Define a struct to hold the CFD arrays
struct CFDData {
    float* p;
    vel u, v;
};


struct CFDInput {
    //Restart
    int Restart;
    int Restart_Time;

    //Domain Information
    int nx, ny;
    double Lx, Ly;

    //Iterative Solver Settings
    int w_AD, w_PPE, AD_itermax, PPE_itermax, AD_solver, PPE_solver;

    //Simulation Settings
    double ErrorMax, tmax, dt, Re, mu;

    //Data write
    int Write_Interval;
};


struct ImmerseFlow{
    CFDData Data;
    CFDInput  Input;
    // Function prototypes
    void initializeCFDData();
    void printCFDData();
};




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
