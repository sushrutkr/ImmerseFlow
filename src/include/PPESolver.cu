#include "../header/preSim.cuh"
#include "../header/postSim.cuh"
#include "../header/globalVariables.cuh"
#include "../header/PPESolver.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>

#define idx(i, j, nx) ((i) + (j) * (nx))

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
              p[i + (j - 1) * nx] * coeff.coeff_dy2_m1[id];
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