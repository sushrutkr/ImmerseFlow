#include "../header/preSim.cuh"
#include <iostream>

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
        ibm.iblank[idx] = static_cast<float>(i);
    }
}

void initializeCFDData(int nx, int ny, CFDData devData) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeKernel<<<blocksPerGrid, threadsPerBlock>>>(nx, ny, devData);
    CHECK_LAST_CUDA_ERROR();

    printKernel<<<blocksPerGrid, threadsPerBlock>>>(nx, ny, devData);
    CHECK_LAST_CUDA_ERROR();

    cudaDeviceSynchronize();
}

void computeIBM(int nx, int ny, IBM ibm) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printKernel<<<blocksPerGrid, threadsPerBlock>>>(nx, ny, devData);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
}

void readGridData(){

}
