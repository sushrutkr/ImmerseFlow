#include "../header/preSim.cuh"
#include "../header/globalVariables.cuh"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>


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

__global__ void iBlankComputeKernel(int nx, int ny, Grid gridData, IBM& ibm) {
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
        }
    }
}


void initializeData(int nx, int ny, CFDData& devData,  IBM& ibm, Grid gridData) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeKernel<<<blocksPerGrid, threadsPerBlock>>>(nx, ny, devData);
    CHECK_LAST_CUDA_ERROR();

    printKernel<<<blocksPerGrid, threadsPerBlock>>>(nx, ny, devData);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
}


void readGridData(int nx, int ny, Grid& gridData) {
    float *x;
    float *y;
    int idx, idy;
    std::ifstream infile;

    x = (float*)malloc(nx * sizeof(float));
    y = (float*)malloc(ny * sizeof(float));

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
    for (int i = 0; i < nx; i++) {
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
    for (int i = 0; i < ny; i++) {
        infile >> idy >> y[i];
    }
    infile.close();

    // Copy data from CPU to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(gridData.x, x, nx * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gridData.y, y, ny * sizeof(float), cudaMemcpyHostToDevice));

    // Free CPU memory
    free(x);
    free(y);
}