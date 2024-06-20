#include "../header/preSim.cuh"
#include "../header/postSim.cuh"
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