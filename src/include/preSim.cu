#include "../header/preSim.cuh"
#include "../header/postSim.cuh"
#include "../header/globalVariables.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>

#define idx(i, j, nx) ((i) + (j) * (nx))



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
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int nGrid = blockDim.x * gridDim.x;

    while (id < nx*ny)
    {
        if (id < nx * ny)
        {
            deviceData.u.velc[id] = 0.0;
            deviceData.v.velc[id] = 0.0;
            deviceData.u.velInter[id] = 0.0;
            deviceData.v.velInter[id] = 0.0;
            deviceData.p[id] = 0.0;
            
        }

        
        id = id + nGrid;
    }

    id = blockIdx.x * blockDim.x + threadIdx.x;
    while (id < (nx - 2) * ny)
    {
        if (id < (nx - 2) * ny)
        {
            deviceData.v.velf[id] = 0.0;
        }
        id = id + nGrid;
    }

    id = blockIdx.x * blockDim.x + threadIdx.x;
    while (id < nx * (ny - 2))
    {
        if (id < nx * (ny - 2))
        {
            deviceData.u.velf[id] = 0.0;
        }
        id = id + nGrid;
    }
}

__global__ void printKernel(int nx, int ny, CFDData deviceData) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int nGrid = blockDim.x * gridDim.x;
    
    while (id < nx * ny)
    {
       printf("u[%d,%d]: %f, v[%d,%d]: %f, p[%d,%d]: %f\n", id/nx, id % nx, deviceData.u.velc[id], id / nx, id % nx, deviceData.v.velc[id], id / nx, id % nx, deviceData.p[id]);
       id = id + nGrid;
    }
}

__global__ void iBlankComputeKernel(int nx, int ny, Grid gridData, IBM ibm) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int nGrid = blockDim.x * gridDim.x;

    while (id < nx * ny)
    {
        int i = id % nx;
        int j = id / nx;
        // Access x and y from gridData directly
        REALTYPE x = gridData.xc[i];
        REALTYPE y = gridData.yc[j];

        // Calculate distance from center (3, 2.5)
        REALTYPE distance = sqrtf(powf((x - 3.0f), 2) + powf((y - 2.5f), 2));

        // Check if within radius of 0.5
        if (distance <= 0.5) {
            ibm.iBlank[id] = 1.0; // Assuming linear indexing
        } else {
            ibm.iBlank[id] = 0.0;
        }
        id = id + nGrid;
    }
}

__global__ void printGridDataKernel(REALTYPE *xf, int nNodes) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int nGrid = blockDim.x * gridDim.x;
    if (id < nNodes) {
        printf("gridData[%d] = %f\n", id, xf[id]);
        id = id + nGrid;
    }
}

__global__ void jacobiIteration(int nx, int ny, Grid gridData, coeffPPE coeff, REALTYPE* p, REALTYPE* p_new) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int nGrid = blockDim.x * gridDim.x;

    while (id < nx * ny) {
        int i = id % nx;
        int j = id / nx;

        p_new[id] = p[id];

        if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
            p_new[id] = - (p[(i+1) + j * nx] * coeff.coeff_dx2_p1[id] +
                           p[(i-1) + j * nx] * coeff.coeff_dx2_m1[id] +
                           p[i + (j+1) * nx] * coeff.coeff_dy2_p1[id] + 
                           p[i + (j-1) * nx] * coeff.coeff_dy2_m1[id]) / coeff.coeff_ppe[id];
        }
        id += nGrid;
    }
}

__global__ void Compute_Residual(int nx, int ny, Grid gridData, coeffPPE coeff, REALTYPE* p, REALTYPE* Residual) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int nGrid = blockDim.x * gridDim.x;

    while (id < nx * ny) {
        int i = id % nx;
        int j = id / nx;

        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
           Residual[id] = p[id] * coeff.coeff_ppe[id] + 
                p[(i + 1) + j * nx] * coeff.coeff_dx2_p1[id] +
                p[(i - 1) + j * nx] * coeff.coeff_dx2_m1[id] +
                p[i + (j + 1) * nx] * coeff.coeff_dy2_p1[id] +
                p[i + (j - 1) * nx] * coeff.coeff_dy2_m1[id] ;
        }
        id += nGrid;
    }
}

__global__ void set_pressure_BC(int nx, int ny, REALTYPE *p){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int nGrid = blockDim.x * gridDim.x;
    
    while (id < nx *ny){
        int i = id % nx;
        int j = id / nx;

        if (i==0){
            p[id] = 100.0;
        }

        if (j==0){
            p[id] = 100.0;
        }
        id += nGrid;
    }
}

__global__ void calculatePPECoefficients(int nx, int ny, Grid gridData, coeffPPE coeff) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int nGrid = blockDim.x * gridDim.x;

    while (id < nx * ny) {
        int i = id % nx;
        int j = id / nx;

        coeff.coeff_ppe[id] = 1;
        coeff.coeff_dx2_m1[id] = 1;
        coeff.coeff_dx2_p1[id] = 1;
        coeff.coeff_dy2_m1[id] = 1;
        coeff.coeff_dy2_p1[id] = 1;

        if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
            REALTYPE dx_i = gridData.dx[idx(i,j,nx)];
            REALTYPE dx_ip1 = gridData.dx[(i+1) + j * nx];
            REALTYPE dx_im1 = gridData.dx[(i-1) + j * nx];
            REALTYPE dy_j = gridData.dy[idx(i,j,nx)];
            REALTYPE dy_jp1 = gridData.dy[i + (j+1) * nx];
            REALTYPE dy_jm1 = gridData.dy[i + (j-1) * nx];

            coeff.coeff_ppe[id] = -1 * (((2 / (dx_i * (dx_i + dx_ip1))) + (2 / (dx_i * (dx_i + dx_im1)))) +
                                        ((2 / (dy_j * (dy_j + dy_jp1))) + (2 / (dy_j * (dy_j + dy_jm1)))));

            coeff.coeff_dx2_m1[id] = (2 / (dx_i * (dx_i + dx_im1)));
            coeff.coeff_dx2_p1[id] = (2 / (dx_i * (dx_i + dx_ip1)));
            coeff.coeff_dy2_m1[id] = (2 / (dy_j * (dy_j + dy_jm1)));
            coeff.coeff_dy2_p1[id] = (2 / (dy_j * (dy_j + dy_jp1)));
        }

        id += nGrid;
    }
}

void ImmerseFlow::allocation() {
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.u.velc, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.v.velc, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.u.velInter, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.v.velInter, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.u.velf, sizeof(REALTYPE) * Input.nx * (Input.ny-2)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&Data.v.velf, sizeof(REALTYPE) * (Input.nx-2) * Input.ny));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&Data.p, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&gridData.xc, sizeof(REALTYPE) * Input.nx));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gridData.yc, sizeof(REALTYPE) * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gridData.xf, sizeof(REALTYPE) * Input.nxf));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gridData.yf, sizeof(REALTYPE) * Input.nyf));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gridData.dx, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gridData.dy, sizeof(REALTYPE) * Input.nx * Input.ny));
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
    CHECK_CUDA_ERROR(cudaFree(gridData.xc));
    CHECK_CUDA_ERROR(cudaFree(gridData.yc));
    CHECK_CUDA_ERROR(cudaFree(gridData.xf));
    CHECK_CUDA_ERROR(cudaFree(gridData.yf));
    CHECK_CUDA_ERROR(cudaFree(ibm.iBlank));
}

void copyDataToHost(int nx, int ny, const Grid& gridData, const IBM& ibm, REALTYPE* host_x, REALTYPE* host_y, REALTYPE* host_iBlank) {
    CHECK_CUDA_ERROR(cudaMemcpy(host_x, gridData.xc, sizeof(REALTYPE) * nx, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(host_y, gridData.yc, sizeof(REALTYPE) * ny, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(host_iBlank, ibm.iBlank, sizeof(REALTYPE) * nx * ny, cudaMemcpyDeviceToHost));
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

    saveDataToFile(Input.nx, Input.ny, gridData.xc, gridData.yc, ibm.iBlank, "../results/final_results.dat");
    
    // // Allocate host memory
    // REALTYPE* host_x = (REALTYPE*)malloc(sizeof(REALTYPE) * Input.nx);
    // REALTYPE* host_y = (REALTYPE*)malloc(sizeof(REALTYPE) * Input.ny);
    // REALTYPE* host_iBlank = (REALTYPE*)malloc(sizeof(REALTYPE) * Input.nx * Input.ny);

    // // Check allocation
    // if (host_x == nullptr || host_y == nullptr || host_iBlank == nullptr) {
    //     std::cerr << "Host memory allocation failed" << std::endl;
    //     return;
    // }

    // // Copy data from device to host
    // copyDataToHost(Input.nx, Input.ny, gridData, ibm, host_x, host_y, host_iBlank);

    // // Write results to file
    // write_results_to_file(host_x, host_y, host_iBlank, Input.nx, Input.ny, "../results/final_results.dat");

    // // Free host memory
    // free(host_x);
    // free(host_y);
    // free(host_iBlank);
}

void ImmerseFlow::readGridData() {
    REALTYPE *xf;
    REALTYPE *yf;
    int id, idy;
    std::ifstream infile;

    xf = (REALTYPE*)malloc(Input.nxf * sizeof(REALTYPE));
    yf = (REALTYPE*)malloc(Input.nyf * sizeof(REALTYPE));

    REALTYPE *x_centers = (REALTYPE*)malloc(Input.nx * sizeof(REALTYPE));
    REALTYPE *y_centers = (REALTYPE*)malloc(Input.ny * sizeof(REALTYPE));

    // Check if memory allocation was successful
    if (xf == nullptr || yf == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        exit(1);
    }

    // Check if memory allocation was successful
    if (x_centers == nullptr || y_centers == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        exit(1);
    }

    // Read values from xgrid.dat
    infile.open("../inputs/xgrid.dat2");
    if (!infile) {
        std::cerr << "Error opening xgrid.dat" << std::endl;
        free(xf);
        free(yf);
        exit(1);
    }
    for (int i = 0; i < Input.nxf; i++) {
        infile >> id >> xf[i];
    }
    infile.close();

    // Read values from ygrid.dat
    infile.open("../inputs/ygrid.dat2");
    if (!infile) {
        std::cerr << "Error opening ygrid.dat" << std::endl;
        free(xf);
        free(yf);
        exit(1);
    }
    for (int i = 0; i < Input.nyf; i++) {
        infile >> idy >> yf[i];
    }
    infile.close();

    // Calculate x centers
    for (int i = 1; i < Input.nx-1; i++) {
        x_centers[i] = (xf[i-1] + xf[i]) / 2.0;
    }

    // Calculate y centers
    for (int i = 1; i < Input.ny-1; i++) {
        y_centers[i] = (yf[i-1] + yf[i]) / 2.0;
    }

    // Handle boundary conditions
    x_centers[0] = -1 * x_centers[1];
    y_centers[0] = -1 * y_centers[1];
    x_centers[Input.nx - 1] = xf[Input.nxf - 1] + (xf[Input.nxf - 1] - x_centers[Input.nx - 2]);
    y_centers[Input.ny - 1] = yf[Input.nyf - 1] + (yf[Input.nyf - 1] - y_centers[Input.ny - 2]);

    // for (int i = 0; i < Input.nxf; ++i) {
    //     std::cout << "x[" << i << "] = " << xf[i] << std::endl;
    // }


    // Compute Grid Spacing
    // Allocate memory for dx and dy
    REALTYPE *dx = (REALTYPE*)malloc(Input.nx * Input.ny * sizeof(REALTYPE));
    REALTYPE *dy = (REALTYPE*)malloc(Input.nx * Input.ny * sizeof(REALTYPE));

    if (dx == nullptr || dy == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        free(xf);
        free(yf);
        free(x_centers);
        free(y_centers);
        exit(1);
    }

    // Compute grid spacings
    for (int j = 1; j < Input.ny-1; j++) {
        for (int i = 1; i < Input.nx-1; i++) {
            dx[i + j * Input.nx] = xf[i] - xf[i-1];
            dy[i + j * Input.nx] = yf[j] - yf[j-1];  // Corrected indexing
        }
    }

    // Handle boundary conditions for dx and dy
    for (int j = 0; j < Input.ny; j++) {
        dx[idx(0, j, Input.nx)] = dx[idx(1, j, Input.nx)];
        dx[idx(Input.nx-1,j,Input.nx)] = dx[idx(Input.nx-2,j,Input.nx)];
    }

    for (int i = 0; i < Input.nx; i++) {
        dx[idx(i, 0, Input.nx)] = dx[idx(i, 1, Input.nx)];
        dx[idx(i, (Input.ny-1), Input.nx)] = dx[idx(i, (Input.ny-2), Input.nx)];
    }
    
    for (int j = 0; j < Input.ny; j++) {
        dy[idx(0, j, Input.nx)] = dy[idx(1, j, Input.nx)];
        dy[idx(Input.nx-1,j,Input.nx)] = dy[idx(Input.nx-2,j,Input.nx)];
    }

    for (int i = 0; i < Input.nx; i++) {
        dy[idx(i, 0, Input.nx)] = dy[idx(i, 1, Input.nx)];
        dy[idx(i, (Input.ny-1), Input.nx)] = dy[idx(i, (Input.ny-2), Input.nx)];
    }

    // Copy data from CPU to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(gridData.xf, xf, Input.nxf * sizeof(REALTYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gridData.yf, yf, Input.nyf * sizeof(REALTYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gridData.xc, x_centers, Input.nx * sizeof(REALTYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gridData.yc, y_centers, Input.ny * sizeof(REALTYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gridData.dx, dx, Input.nx * Input.ny * sizeof(REALTYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gridData.dy, dy, Input.nx * Input.ny * sizeof(REALTYPE), cudaMemcpyHostToDevice));

    // Free CPU memory
    free(xf);
    free(yf);
    free(x_centers);
    free(y_centers);
    free(dx);
    free(dy);
}

void ImmerseFlow::PPESolver() {

    REALTYPE* pTemp, * pResidue;
    
    coeffPPE coeff;
    
    CHECK_CUDA_ERROR(cudaMallocManaged(&pTemp, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMallocManaged(&pResidue, sizeof(REALTYPE) * Input.nx * Input.ny));
    CHECK_CUDA_ERROR(cudaMallocManaged(&coeff.coeff_ppe, Input.nx * Input.ny * sizeof(REALTYPE)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&coeff.coeff_dx2_p1, Input.nx * Input.ny * sizeof(REALTYPE)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&coeff.coeff_dx2_m1, Input.nx * Input.ny * sizeof(REALTYPE)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&coeff.coeff_dy2_p1, Input.nx * Input.ny * sizeof(REALTYPE)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&coeff.coeff_dy2_m1, Input.nx * Input.ny * sizeof(REALTYPE)));

    //Compute Coefficient Matrix
    calculatePPECoefficients<<<CUDAData.blocksPerGrid, CUDAData.threadsPerBlock>>>(Input.nx, Input.ny, gridData, coeff);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    //Set Boundary Conditions    
    set_pressure_BC<<<CUDAData.blocksPerGrid, CUDAData.threadsPerBlock>>>(Input.nx, Input.ny, Data.p);    
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());      
    
    //Setting arbitrary large number to start iterations
    REALTYPE Residual = 1.0;

    int iter=0;
    while(Residual > pow(10.0,-6.0) && iter < Input.PPE_itermax) {
    //for (int iter = 0; iter < Input.PPE_itermax; iter++) {
        jacobiIteration<<<CUDAData.blocksPerGrid, CUDAData.threadsPerBlock>>>(Input.nx, Input.ny, gridData, coeff, Data.p, pTemp);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
       
        // Swap pointers
        REALTYPE *temp = Data.p;
        Data.p = pTemp;
        pTemp = temp;

        Compute_Residual << <CUDAData.blocksPerGrid, CUDAData.threadsPerBlock >> > (Input.nx, Input.ny, gridData, coeff, Data.p, pResidue);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        Reduction(pResidue, &Residual);
        
        iter += 1;
        // printf("iter = %d %f\n", iter, Residual);
    }
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    //Set Boundary Conditions
    set_pressure_BC<<<CUDAData.blocksPerGrid, CUDAData.threadsPerBlock>>>(Input.nx, Input.ny, Data.p);

    saveDataToFile(Input.nx, Input.ny, gridData.xc, gridData.yc, Data.p, "../results/p.dat");

    CHECK_CUDA_ERROR(cudaFree(pTemp));
    CHECK_CUDA_ERROR(cudaFree(coeff.coeff_ppe));
    CHECK_CUDA_ERROR(cudaFree(coeff.coeff_dx2_p1));
    CHECK_CUDA_ERROR(cudaFree(coeff.coeff_dx2_m1));
    CHECK_CUDA_ERROR(cudaFree(coeff.coeff_dy2_p1));
    CHECK_CUDA_ERROR(cudaFree(coeff.coeff_dy2_m1));
}

void ImmerseFlow::Reduction(REALTYPE *input, REALTYPE* Residual)
{
    REALTYPE* g_odata, * g_odata2;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&g_odata, sizeof(REALTYPE) * CUDAData.threadsPerBlock));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&g_odata2, sizeof(REALTYPE) * CUDAData.threadsPerBlock));
    CHECK_LAST_CUDA_ERROR();

    // Calculate Error
    switch (CUDAData.threadsPerBlock)
    {
    case 1024:
        reduce6<1024> << < CUDAData.blocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (input, g_odata, Input.nx * Input.ny); break;
    case 512:
        reduce6<512> << < CUDAData.blocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (input, g_odata, Input.nx * Input.ny); break;
    case 256:
        reduce6<256> << < CUDAData.blocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (input, g_odata, Input.nx * Input.ny); break;
    case 128:
        reduce6<128> << < CUDAData.blocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (input, g_odata, Input.nx * Input.ny); break;
    case 64:
        reduce6< 64> << < CUDAData.blocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (input, g_odata, Input.nx * Input.ny); break;
    case 32:
        reduce6< 32> << < CUDAData.blocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (input, g_odata, Input.nx * Input.ny); break;
    case 16:
        reduce6< 16> << < CUDAData.blocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (input, g_odata, Input.nx * Input.ny); break;
    case 8:
        reduce6< 8> << < CUDAData.blocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (input, g_odata, Input.nx * Input.ny); break;
    case 4:
        reduce6< 4> << < CUDAData.blocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (input, g_odata, Input.nx * Input.ny); break;
    case 2:
        reduce6< 2> << < CUDAData.blocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (input, g_odata, Input.nx * Input.ny); break;
    case 1:
        reduce6< 1> << < CUDAData.blocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (input, g_odata, Input.nx * Input.ny); break;
    }

    int BlocksPerGrid = 1;


    switch (CUDAData.threadsPerBlock)
    {
    case 1024:
        reduce6<1024> << < BlocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata2, CUDAData.threadsPerBlock); break;
    case 512:
        reduce6<512> << < BlocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata2, CUDAData.threadsPerBlock); break;
    case 256:
        reduce6<256> << < BlocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata2, CUDAData.threadsPerBlock); break;
    case 128:
        reduce6<128> << < BlocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata2, CUDAData.threadsPerBlock); break;
    case 64:
        reduce6< 64> << < BlocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata2, CUDAData.threadsPerBlock); break;
    case 32:
        reduce6< 32> << < BlocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata2, CUDAData.threadsPerBlock); break;
    case 16:
        reduce6< 16> << < BlocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata2, CUDAData.threadsPerBlock); break;
    case 8:
        reduce6< 8> << < BlocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata2, CUDAData.threadsPerBlock); break;
    case 4:
        reduce6< 4> << < BlocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata2, CUDAData.threadsPerBlock); break;
    case 2:
        reduce6< 2> << < BlocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata2, CUDAData.threadsPerBlock); break;
    case 1:
        reduce6< 1> << < BlocksPerGrid, CUDAData.threadsPerBlock, CUDAData.threadsPerBlock * sizeof(REALTYPE) >> > (g_odata, g_odata2, CUDAData.threadsPerBlock); break;
    }

    CHECK_CUDA_ERROR(cudaMemcpy(Residual, g_odata2, sizeof(REALTYPE) * 1, cudaMemcpyDeviceToHost));
}