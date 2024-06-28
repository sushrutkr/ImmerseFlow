#include "../header/preSim.cuh"
#include "../header/postSim.cuh"
#include "../header/globalVariables.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>

template <unsigned int blockSize>
__device__ void warpReduce(volatile REALTYPE* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(REALTYPE* g_idata, REALTYPE* g_odata, unsigned int n) {
    extern __shared__ REALTYPE sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata[tid] = 0;
    while (i  < n) {
        if (i + blockSize < n)
        {
            sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize;
        }
        else
        {
            sdata[tid] += g_idata[i]; i += gridSize;
        }
    }

    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    
}


__global__ void initializeKernel(int nx, int ny, CFDData deviceData) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nGrid = blockDim.x * gridDim.x;

    while (idx < nx*ny)
    {
        if (idx < nx * ny)
        {
            deviceData.u.velc[idx] = 0.0;
            deviceData.v.velc[idx] = 0.0;
            deviceData.u.velInter[idx] = 0.0;
            deviceData.v.velInter[idx] = 0.0;
            deviceData.u.velf[idx] = 0.0;
            deviceData.v.velf[idx] = 0.0;
            deviceData.p[idx] = 0.0;
            
        }

        
        idx = idx + nGrid;
    }

    idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < (nx - 2) * ny)
    {
        if (idx < (nx - 2) * ny)
        {
            deviceData.v.velf[idx] = 0.0;
        }
        idx = idx + nGrid;
    }

    idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < nx * (ny - 2))
    {
        if (idx < nx * (ny - 2))
        {
            deviceData.u.velf[idx] = 0.0;
        }
        idx = idx + nGrid;
    }
}

__global__ void printKernel(int nx, int ny, CFDData deviceData) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nGrid = blockDim.x * gridDim.x;
    
    while (idx < nx * ny)
    {
       printf("u[%d,%d]: %f, v[%d,%d]: %f, p[%d,%d]: %f\n", idx/nx, idx % nx, deviceData.u.velc[idx], idx / nx, idx % nx, deviceData.v.velc[idx], idx / nx, idx % nx, deviceData.p[idx]);
       idx = idx + nGrid;
    }
}

__global__ void iBlankComputeKernel(int nx, int ny, Grid gridData, IBM ibm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nGrid = blockDim.x * gridDim.x;

    while (idx < nx * ny)
    {
        int i = idx % nx;
        int j = idx / nx;
        // Access x and y from gridData directly
        REALTYPE x = gridData.x[i];
        REALTYPE y = gridData.y[j];

        // Calculate distance from center (3, 2.5)
        REALTYPE distance = sqrtf(powf((x - 3.0f), 2) + powf((y - 2.5f), 2));

        // Check if within radius of 0.5
        if (distance <= 0.5) {
            ibm.iBlank[idx] = 1.0; // Assuming linear indexing
        } else {
            ibm.iBlank[idx] = 0.0;
        }
        idx = idx + nGrid;
    }
}


void copyDataToHost(int nx, int ny, const Grid& gridData, const IBM& ibm, REALTYPE* host_x, REALTYPE* host_y, REALTYPE* host_iBlank) {
    // Copy data from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(host_x, gridData.x, sizeof(REALTYPE) * nx, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(host_y, gridData.y, sizeof(REALTYPE) * ny, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(host_iBlank, ibm.iBlank, sizeof(REALTYPE) * nx * ny, cudaMemcpyDeviceToHost));
}

void ImmerseFlow::allocation() {
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.u.velc, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.v.velc, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.u.velInter, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.v.velInter, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.u.velf, sizeof(REALTYPE) * Input.nx * (Input.ny-2)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.v.velf, sizeof(REALTYPE) * (Input.nx-2) * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.p, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gridData.x, sizeof(REALTYPE) * Input.nx));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gridData.y, sizeof(REALTYPE) * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&ibm.iBlank, sizeof(REALTYPE) * Input.nx * Input.ny));
}

void ImmerseFlow::freeAllocation() {
    // Free allocated memory
    CHECK_CUDA_ERROR(cudaFree(Data.u.velc));
    CHECK_CUDA_ERROR(cudaFree(Data.v.velc));
    CHECK_CUDA_ERROR(cudaFree(Data.u.velInter));
    CHECK_CUDA_ERROR(cudaFree(Data.v.velInter));
    CHECK_CUDA_ERROR(cudaFree(Data.u.velf));
    CHECK_CUDA_ERROR(cudaFree(Data.v.velf));
    CHECK_CUDA_ERROR(cudaFree(Data.p));
    CHECK_CUDA_ERROR(cudaFree(gridData.x));
    CHECK_CUDA_ERROR(cudaFree(gridData.y));
    CHECK_CUDA_ERROR(cudaFree(ibm.iBlank));
}

void ImmerseFlow::CUDAQuery() {
    cudaDeviceProp prop;    
    cudaGetDeviceProperties(&prop, 0);
    CUDAData.threadsPerBlock = prop.maxThreadsPerBlock;
    //Do we need to change this for cell face case or just waste some cores
    CUDAData.blocksPerGrid = (Input.nx * Input.ny + CUDAData.threadsPerBlock - 1) / CUDAData.threadsPerBlock;
    printf("Maximum number of threads = %d\n", prop.maxThreadsPerBlock);
    printf("Maximum number of blocks = %d\n", prop.maxThreadsDim[0]);
}

void ImmerseFlow:: initializeData() {
    // Initialize kernel
    initializeKernel<<<CUDAData.blocksPerGrid, CUDAData.threadsPerBlock>>>(Input.nx, Input.ny, Data);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Initialize iBlank to zero
    CHECK_CUDA_ERROR(cudaMemset(ibm.iBlank, 0, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Compute iBlank kernel
    iBlankComputeKernel<<<CUDAData.blocksPerGrid, CUDAData.threadsPerBlock >>>(Input.nx, Input.ny, gridData, ibm);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Allocate host memory
    REALTYPE* host_x = (REALTYPE*)malloc(sizeof(REALTYPE) * Input.nx);
    REALTYPE* host_y = (REALTYPE*)malloc(sizeof(REALTYPE) * Input.ny);
    REALTYPE* host_iBlank = (REALTYPE*)malloc(sizeof(REALTYPE) * Input.nx * Input.ny);

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


    //Test
    int BlocksPerGrid = 4;
    int ThreadsPerBlock = 1024;

    

    REALTYPE* g_idata, * g_odata;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&g_odata, sizeof(REALTYPE) * ThreadsPerBlock));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&g_idata, sizeof(REALTYPE) * Input.nx*Input.ny));
    CHECK_LAST_CUDA_ERROR();


    

    REALTYPE h_odata;
    
	switch (ThreadsPerBlock)
	{
    case 1024:
        reduce6<1024> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
	case 512:
		reduce6<512> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
	case 256:
		reduce6<256> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
	case 128:
		reduce6<128> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
	case 64:
		reduce6< 64> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
	case 32:
		reduce6< 32> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
	case 16:
		reduce6< 16> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
	case 8:
		reduce6< 8> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
	case 4:
		reduce6< 4> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
	case 2:
		reduce6< 2> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
	case 1:
		reduce6< 1> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (Data.u.velc, g_odata, Input.nx * Input.ny); break;
	}

    BlocksPerGrid = 1;


    switch (ThreadsPerBlock)
    {
    case 1024:
        reduce6<1024> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata, Input.nx * Input.ny); break;
    case 512:
        reduce6<512> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata, Input.nx * Input.ny); break;
    case 256:
        reduce6<256> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata, Input.nx * Input.ny); break;
    case 128:
        reduce6<128> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata, Input.nx * Input.ny); break;
    case 64:
        reduce6< 64> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata, Input.nx * Input.ny); break;
    case 32:
        reduce6< 32> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata, Input.nx * Input.ny); break;
    case 16:
        reduce6< 16> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata, Input.nx * Input.ny); break;
    case 8:
        reduce6< 8> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata, Input.nx * Input.ny); break;
    case 4:
        reduce6< 4> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata, Input.nx * Input.ny); break;
    case 2:
        reduce6< 2> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata, Input.nx * Input.ny); break;
    case 1:
        reduce6< 1> << < BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata, Input.nx * Input.ny); break;
    }

    CHECK_CUDA_ERROR(cudaMemcpy(&h_odata, g_odata, sizeof(REALTYPE) * 1, cudaMemcpyDeviceToHost));
    printf("%f\n", h_odata);

}




void ImmerseFlow :: readGridData() {
    REALTYPE *x;
    REALTYPE*y;
    int idx, idy;
    std::ifstream infile;

    x = (REALTYPE*)malloc(Input.nx * sizeof(REALTYPE));
    y = (REALTYPE*)malloc(Input.ny * sizeof(REALTYPE));

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
    CHECK_CUDA_ERROR(cudaMemcpy(gridData.x, x, Input.nx * sizeof(REALTYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gridData.y, y, Input.ny * sizeof(REALTYPE), cudaMemcpyHostToDevice));


    // Free CPU memory
    free(x);
    free(y);
}