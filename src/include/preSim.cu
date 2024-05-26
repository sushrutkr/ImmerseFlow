// #include "../header/globalVariables.cuh"
#include "../header/preSim.cuh"
#include <iostream>

// Define the global instance of the struct as a device variable
// __device__ CFDData deviceData;

__global__ void initializeKernel(int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i + j * nx;

  if (i < nx && j < ny) {
    deviceData.u[idx] = static_cast<float>(idx); // Initialize u
    deviceData.v[idx] = static_cast<float>(idx); // Initialize v
    deviceData.p[idx] = static_cast<float>(idx); // Initialize p
  }
}

__global__ void printKernel(int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = i + j * nx;

  if (i < nx && j < ny) {
    printf("u[%d,%d]: %f, v[%d,%d]: %f, p[%d,%d]: %f\n", i, j, deviceData.u[idx], i, j, deviceData.v[idx], i, j, deviceData.p[idx]);
  }
}

void initializeCFDData(int nx, int ny) {
  // Allocate host memory for u, v, p
  CFDData hostData;
  cudaMalloc(&hostData.u, sizeof(float) * nx * ny);
  cudaMalloc(&hostData.v, sizeof(float) * nx * ny);
  cudaMalloc(&hostData.p, sizeof(float) * nx * ny);

  // Check if memory allocation was successful
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Error allocating device memory: " << cudaGetErrorString(err) << std::endl;
    return;
  }

  // Copy the host struct to the device
  cudaMemcpyToSymbol(deviceData, &hostData, sizeof(CFDData));

  // Kernel launch parameters
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                     (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // Launch the initialization kernel
  initializeKernel<<<blocksPerGrid, threadsPerBlock>>>(nx, ny);
  cudaDeviceSynchronize(); // Ensure initializeKernel has finished execution

  // Check for errors
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Error launching initialization kernel: " << cudaGetErrorString(err) << std::endl;
    cudaFree(hostData.u);
    cudaFree(hostData.v);
    cudaFree(hostData.p);
    return;
  }
}

void printCFDData(int nx, int ny) {
  // Kernel launch parameters
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                     (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // Launch the print kernel
  printKernel<<<blocksPerGrid, threadsPerBlock>>>(nx, ny);
  cudaDeviceSynchronize(); // Ensure printKernel has finished execution

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Error launching print kernel: " << cudaGetErrorString(err) << std::endl;
    return;
  }
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