#include "../header/preSim.cuh"
#include "../header/postSim.cuh"
#include "../header/globalVariables.cuh"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    if (tid == 0) printf("output=%f %d %d\n", g_odata[blockIdx.x], threadIdx.x, blockIdx.x);
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

__global__ void iBlankComputeKernel(int nx, int ny, Grid gridData, IBM ibm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i + j * nx;

    if (i < nx && j < ny) {
        // Access x and y from gridData directly
        float x = gridData.x[i];
        float y = gridData.y[j];

        // Calculate distance from center (3, 2.5)
        float distance = sqrtf(powf((x - 3.0f), 2) + powf((y - 2.5f), 2));

        // Check if within radius of 0.5
        if (distance <= 0.5f) {
            ibm.iBlank[idx] = 1.0f; // Assuming linear indexing
        } else {
            ibm.iBlank[idx] = 0.0f;
        }
    }
}


void copyDataToHost(int nx, int ny, const Grid& gridData, const IBM& ibm, float* host_x, float* host_y, float* host_iBlank) {
    // Copy data from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(host_x, gridData.x, sizeof(float) * nx, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(host_y, gridData.y, sizeof(float) * ny, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(host_iBlank, ibm.iBlank, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost));
}


void ImmerseFlow:: initializeData() {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((Input.nx + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (Input.ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Initialize kernel
    initializeKernel<<<blocksPerGrid, threadsPerBlock>>>(Input.nx, Input.ny, Data);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Initialize iBlank to zero
    CHECK_CUDA_ERROR(cudaMemset(ibm.iBlank, 0, sizeof(float) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Compute iBlank kernel
    iBlankComputeKernel<<<blocksPerGrid, threadsPerBlock>>>(Input.nx, Input.ny, gridData, ibm);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Allocate host memory
    float* host_x = (float*)malloc(sizeof(float) * Input.nx);
    float* host_y = (float*)malloc(sizeof(float) * Input.ny);
    float* host_iBlank = (float*)malloc(sizeof(float) * Input.nx * Input.ny);

    // Check allocation
    if (host_x == nullptr || host_y == nullptr || host_iBlank == nullptr) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return;
    }

    // Copy data from device to host
    copyDataToHost(Input.nx, Input.ny, gridData, ibm, host_x, host_y, host_iBlank);

    // Write results to file
    write_results_to_file(host_x, host_y, host_iBlank, Input.nx, Input.ny, "../results/final_results.dat");

    // Free host memory
    free(host_x);
    free(host_y);
    free(host_iBlank);

    float* g_idata, * g_odata;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&g_odata, sizeof(float) * 1));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&g_idata, sizeof(float) * Input.nx*Input.ny));
    CHECK_LAST_CUDA_ERROR();


    //Test
    const int BlocksPerGrid = 2;
    const int ThreadsPerBlock = 64;

    
    switch (ThreadsPerBlock)
    {
    case 512:
        reduce6<512> << < BlocksPerGrid, ThreadsPerBlock >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
    case 256:
        reduce6<256> << < BlocksPerGrid, ThreadsPerBlock >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
    case 128:
        reduce6<128> << < BlocksPerGrid, ThreadsPerBlock >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
    case 64:
        reduce6< 64> << < BlocksPerGrid, ThreadsPerBlock >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
    case 32:
        reduce6< 32> << < BlocksPerGrid, ThreadsPerBlock >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
    case 16:
        reduce6< 16> << < BlocksPerGrid, ThreadsPerBlock >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
    case 8:
        reduce6< 8> << < BlocksPerGrid, ThreadsPerBlock >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
    case 4:
        reduce6< 4> << < BlocksPerGrid, ThreadsPerBlock >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
    case 2:
        reduce6< 2> << < BlocksPerGrid, ThreadsPerBlock >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
    case 1:
        reduce6< 1> << < BlocksPerGrid, ThreadsPerBlock >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
    }
}




void ImmerseFlow :: readGridData() {
    float *x;
    float *y;
    int idx, idy;
    std::ifstream infile;

    x = (float*)malloc(Input.nx * sizeof(float));
    y = (float*)malloc(Input.ny * sizeof(float));

    // Check if memory allocation was successful
    if (x == nullptr || y == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        exit(1);
    }

    // Read values from xgrid.dat
    infile.open("../inputs/xgrid.dat");
    if (!infile) {
        std::cerr << "Error opening xgrid.dat" << std::endl;
        free(x);
        free(y);
        exit(1);
    }
    for (int i = 0; i < Input.nx; i++) {
        infile >> idx >> x[i];
    }
    infile.close();

    // Read values from ygrid.dat
    infile.open("../inputs/ygrid.dat");
    if (!infile) {
        std::cerr << "Error opening ygrid.dat" << std::endl;
        free(x);
        free(y);
        exit(1);
    }
    for (int i = 0; i < Input.ny; i++) {
        infile >> idy >> y[i];
    }
    infile.close();

    // Copy data from CPU to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(gridData.x, x, Input.nx * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gridData.y, y, Input.ny * sizeof(float), cudaMemcpyHostToDevice));

    // Free CPU memory
    free(x);
    free(y);
}