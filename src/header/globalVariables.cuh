// GlobalVariables.h
#ifndef GLOBAL_VARIABLES_H
#define GLOBAL_VARIABLES_H

namespace domain {
    int nx;
    // int ny;
    // int N; // Declare N as an external variable
    __device__ float* x;
    __device__ float* y;
    __device__ float* d_myArray;
}

namespace ibm {
    __device__ float* iBlank;
}

namespace flow {

}

#endif // GLOBAL_VARIABLES_H


// #ifndef VARIABLES_H_
// #define VARIABLES_H_

// namespace GlobalVariables {
//     namespace Device {
//         // float *iBlank;
//         // float *x;
//         // float *y;
//         extern int N;
//         // int nx;
//         // int ny;
//     }

//     // void initialize(int size);
//     // void cleanup();
// }

// #endif