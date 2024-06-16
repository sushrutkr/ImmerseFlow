#pragma once
#include "preSim.cu"


void FlowSolver::CUDAinitialize()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((Input.nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (Input.ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the initialization kernel
    initializeKernel << <blocksPerGrid, threadsPerBlock >> > (Input.nx, Input.ny, Data);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize(); // Ensure initializeKernel has finished execution
};

void FlowSolver::CUDAprint()
{
    // Kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((Input.nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (Input.ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the print kernel
    printKernel << <blocksPerGrid, threadsPerBlock >> > (Input.nx, Input.ny, Data);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize(); // Ensure printKernel has finished execution
};

void FlowSolver::CUDAallocation()
{
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.u.velc, sizeof(float) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.v.velc, sizeof(float) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.p, sizeof(float) * Input.nx * Input.ny));
};