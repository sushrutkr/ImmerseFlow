// #include "../header/globalVariables.cuh"
#include "preSim.cuh"
#include <iostream>

// Define the global instance of the struct as a device variable
// __device__ CFDData deviceData;

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(int* g_idata, int* g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void initializeKernel(int nx, int ny, CFDData deviceData) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i + j * nx;

    if (i < nx && j < ny) {
        deviceData.u.velc[idx] = static_cast<float>(idx); // Initialize u
        deviceData.v.velc[idx] = static_cast<float>(idx); // Initialize v
        deviceData.p[idx] = static_cast<float>(idx); // Initialize p
    }
}

__global__ void printKernel(int nx, int ny, CFDData deviceData) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i + j * nx;

    if (i < nx && j < ny) {
        printf("u[%d,%d]: %f, v[%d,%d]: %f, p[%d,%d]: %f\n", i, j, deviceData.u.velc[idx], i, j, deviceData.v.velc[idx], i, j, deviceData.p[idx]);
    }
}



void ImmerseFlow:: initializeCFDData() {
    // Kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((Input.nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (Input.ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the initialization kernel
    initializeKernel << <blocksPerGrid, threadsPerBlock >> > (Input.nx, Input.ny, Data);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize(); 
}

void ImmerseFlow::printCFDData() {
    // Kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((Input.nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (Input.ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the print kernel
    printKernel << <blocksPerGrid, threadsPerBlock >> > (Input.nx, Input.ny, Data);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize(); 
}



// __global__ void myKernel(int nx_) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < nx_) {
//     domain::d_myArray[i] = static_cast<float>(i); // Initialize the array
//   }
// }

// __global__ void printKernel(int nx_) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < nx_) {
//     printf("Result[%d]: %f\n", i, domain::d_myArray[i]);
//   }
// }

// void initializeArray(int nx) {
//   // Allocate device memory for d_myArray
//   float* d_myArray_host;
//   cudaMalloc(&d_myArray_host, sizeof(float) * nx);

//   // Copy the pointer to the device symbol
//   cudaMemcpyToSymbol(domain::d_myArray, &d_myArray_host, sizeof(float*));

//   // Kernel launch parameters
//   int threadsPerBlock = 256;
//   int blocksPerGrid = (nx + threadsPerBlock - 1) / threadsPerBlock;

//   // Launch the initialization kernel
//   myKernel<<<blocksPerGrid, threadsPerBlock>>>(nx);
//   cudaDeviceSynchronize(); // Ensure myKernel has finished execution

//   // Check for errors
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     std::cerr << "Error launching kernel: " << cudaGetErrorString(err) << std::endl;
//     cudaFree(d_myArray_host);
//     return;
//   }

//   // Store the device pointer for later use
//   cudaMemcpyToSymbol(domain::d_myArray, &d_myArray_host, sizeof(float*));
// }

// void printArray(int nx) {
//   // Kernel launch parameters
//   int threadsPerBlock = 256;
//   int blocksPerGrid = (nx + threadsPerBlock - 1) / threadsPerBlock;

//   // Launch the print kernel
//   printKernel<<<blocksPerGrid, threadsPerBlock>>>(nx);
//   cudaDeviceSynchronize(); // Ensure printKernel has finished execution

//   // Check for errors
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     std::cerr << "Error launching kernel: " << cudaGetErrorString(err) << std::endl;
//     return;
//   }
// }


// #include <fstream>
// #include <cmath>
// #include "../header/globalVariables.cuh"
// #include "../header/preSim.cuh"

// void setZero(float* vec, int N) {
//     for (int i = 0; i < N; i++) {
//       vec[i] = 0.0f;
//     }
// }

// __global__ void printThread(){
//   int i = threadIdx.x + blockIdx.x*blockDim.x;
//   printf("ThreadIdx : %f \n",&GlobalVariables::Device::x[i]);

// }

// Initialize::Initialize(){
//   InputReader();         // Call member function InputReader()
//   InitializeArrays();    // Call member function InitializeArrays()
// }

// void Initialize::InputReader(){
//   nx_ = 181;
//   ny_ = 129;
//   N_ = nx_ * ny_;

//   GlobalVariables::Device::nx = nx_;
//   GlobalVariables::Device::ny = ny_;
//   GlobalVariables::Device::N = N_;
// }

// void Initialize::InitializeArrays(){


//   cudaMalloc(&GlobalVariables::Device::x, nx_ * sizeof(float));
//   cudaMemset(GlobalVariables::Device::x, 0, GlobalVariables::Device::nx * sizeof(float));

//   cudaMalloc(&GlobalVariables::Device::y, ny_ * sizeof(float));
//   cudaMemset(GlobalVariables::Device::y, 0, GlobalVariables::Device::ny * sizeof(float));

//   cudaMalloc(&GlobalVariables::Device::iBlank, N_ * sizeof(float));
//   cudaMemset(GlobalVariables::Device::iBlank, 0, GlobalVariables::Device::N * sizeof(float));
// }


// Grid::Grid() {
//   // Fill x and y arrays
//   createGrid();
//   // Calculate iBlank
//   // calculateiBlank();
// }

// void Grid::createGrid(){
//   nx_ = GlobalVariables::Device::nx;
//   ny_ = GlobalVariables::Device::ny;

//   x_ = new float[nx_];
//   y_ = new float[ny_];

//   // Read X coordinate data
//   std::ifstream infile_x("../inputs/xgrid.dat");
//   if (!infile_x){
//       std::cerr << "Error Opening Xgrid.dat" << std::endl;
//   }
//   int dummy;
//   for (int i = 0; i < nx_; ++i){
//       infile_x >> dummy >> x_[i];
//   }
//   infile_x.close();

//   // Read Y coordinate data
//   std::ifstream infile_y("../inputs/ygrid.dat");
//   if (!infile_y){
//       std::cerr << "Error Opening Ygrid.dat" << std::endl;
//   }
//   for (int i = 0; i < ny_; ++i){
//       infile_y >> dummy >> y_[i];
//   }
//   infile_y.close(); 

//   cudaMemcpy(GlobalVariables::Device::x, x_, nx_ * sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(GlobalVariables::Device::y, y_, ny_ * sizeof(float), cudaMemcpyHostToDevice);

// }

// // void Grid::calculateiBlank() {
// //     // Set all elements of iBlank to zero
// //     setZero(iBlank_, nx_, ny_);

// //     // Iterate over each element of iBlank and set it to 1.0f if it satisfies the condition
// //     for (int i = 0; i < nx_; ++i) {
// //         for (int j = 0; j < ny_; ++j) {
// //             if (pow((x_[i] - 3), 2) + pow((y_[j] - 2.5), 2) <= pow(0.5, 2)) {
// //                 iBlank_[i * ny_ + j] = 1.0f;
// //             }
// //         }
// //     }
// // }