#include "../header/preSim.cuh"
#include "../header/postSim.cuh"
#include "../header/globalVariables.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>

#define idx(i, j, nx) ((i) + (j) * (nx))

__global__ void calculateADCoefficients(int nx, int ny, Grid gridData, REALTYPE dt, coefficient coeff, REALTYPE Re) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int nGrid = blockDim.x * gridDim.x;

  while (id < nx * ny) {
      int i = id % nx;
      int j = id / nx;

      coeff.coeff[id] = 1;
      coeff.coeff_dx2_m1[id] = 1;
      coeff.coeff_dx2_p1[id] = 1;
      coeff.coeff_dy2_m1[id] = 1;
      coeff.coeff_dy2_p1[id] = 1;

      if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
          REALTYPE dx_i = gridData.dx[id];
          REALTYPE dx_ip1 = gridData.dx[id + 1];
          REALTYPE dx_im1 = gridData.dx[id - 1];
          REALTYPE dy_j = gridData.dy[id];
          REALTYPE dy_jp1 = gridData.dy[id + nx];
          REALTYPE dy_jm1 = gridData.dy[id - nx];

          coeff.coeff[id] = 1.0 + (dt / Re) * ((2 / (dx_i * (dx_i + dx_ip1))) + (2 / (dx_i * (dx_i + dx_im1))))
                                + (dt / Re) * ((2 / (dy_j * (dy_j + dy_jp1))) + (2 / (dy_j * (dy_j + dy_jm1))));
          coeff.coeff_dx2_m1[id] = (dt / Re) * (2 / (dx_i * (dx_i + dx_im1)));
          coeff.coeff_dx2_p1[id] = (dt / Re) * (2 / (dx_i * (dx_i + dx_ip1)));
          coeff.coeff_dy2_m1[id] = (dt / Re) * (2 / (dy_j * (dy_j + dy_jm1)));
          coeff.coeff_dy2_p1[id] = (dt / Re) * (2 / (dy_j * (dy_j + dy_jp1)));
      }
      
      id += nGrid;
  }
}

__global__ void ADSource(int nx, int ny, Grid gridData, REALTYPE dt, REALTYPE* u, REALTYPE* v, REALTYPE* uf, REALTYPE* vf,
                        REALTYPE* sx, REALTYPE* sy) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int nGrid = blockDim.x * gridDim.x;

  while (id < nx * ny) {
      int i = id % nx;
      int j = id / nx;

      sx[id] = 0.0;
      sy[id] = 0.0;

      if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
          REALTYPE dx_i = gridData.dx[idx(i, j, nx)];
          REALTYPE dx_ip1 = gridData.dx[(i + 1) + j * nx];
          REALTYPE dx_im1 = gridData.dx[(i - 1) + j * nx];
          REALTYPE dy_j = gridData.dy[idx(i, j, nx)];
          REALTYPE dy_jp1 = gridData.dy[i + (j + 1) * nx];
          REALTYPE dy_jm1 = gridData.dy[i + (j - 1) * nx];

          sx[id] = (u[id] - (dt / dx_i)*0.5 * (((1 / (dx_i + dx_ip1)) * (u[id+1 ] * dx_i   + u[id   ] * dx_ip1) * uf[i + (j - 1) * (nx-1)])
                                             - ((1 / (dx_i + dx_im1)) * (u[id   ] * dx_im1 + u[id-1 ] * dx_i  ) * uf[i - 1 + (j - 1) * (nx - 1)]))
                          - (dt / dy_j)*0.5 * (((1 / (dy_j + dy_jp1)) * (u[id+nx] * dy_j   + u[id   ] * dy_jp1) * vf[i - 1 + j * (nx-2)])
                                             - ((1 / (dy_j + dy_jm1)) * (u[id   ] * dy_jm1 + u[id-nx] * dy_j  ) * vf[i - 1 + (j - 1) * (nx-2)])));

          sy[id] = (v[id] - (dt / dx_i)*0.5 * (((1 / (dx_i + dx_ip1)) * (v[id+1 ] * dx_i   + v[id   ] * dx_ip1) * uf[i + (j - 1) * (nx - 1)])
                                             - ((1 / (dx_i + dx_im1)) * (v[id   ] * dx_im1 + v[id-1 ] * dx_i  ) * uf[i - 1 + (j - 1) * (nx - 1)]))
                          - (dt / dy_j)*0.5 * (((1 / (dy_j + dy_jp1)) * (v[id+nx] * dy_j   + v[id   ] * dy_jp1) * vf[i - 1 + j * (nx - 2)])
                                             - ((1 / (dy_j + dy_jm1)) * (v[id   ] * dy_jm1 + v[id-nx] * dy_j  ) * vf[i - 1 + (j - 1) * (nx - 2)])));
      }
      id += nGrid;

  }
}

__global__ void ADusolver_kernel(int nx, int ny, Grid gridData, coefficient coeff, IBM ibm, CFDInput input,
  REALTYPE dt, REALTYPE* u, REALTYPE* uf, REALTYPE* vf, REALTYPE* unew, REALTYPE* sx) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int nGrid = blockDim.x * gridDim.x;

  while (id < nx * ny) {
      int i = id % nx;
      int j = id / nx;

      if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
          unew[id] = sx[id] + coeff.coeff_dx2_p1[id] * u[id + 1 ] + coeff.coeff_dx2_m1[id] * u[id-1 ]
                            + coeff.coeff_dy2_p1[id] * u[id + nx] + coeff.coeff_dy2_m1[id] * u[id-nx];
          
          unew[id] = ibm.iBlank[id] * unew[id] /coeff.coeff[id];
          //unew[id] = (1.0 - input.w_AD) * u[id] + input.w_AD * unew[id]; 
      }
      id += nGrid;
  }
  
}

__global__ void ADvsolver_kernel(int nx, int ny, Grid gridData, coefficient coeff, IBM ibm, CFDInput input,
  REALTYPE dt, REALTYPE* v, REALTYPE* uf, REALTYPE* vf, REALTYPE* vnew, REALTYPE* sy) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int nGrid = blockDim.x * gridDim.x;

  while (id < nx * ny) {
      int i = id % nx;
      int j = id / nx;

      if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
          vnew[id] = sy[id] + coeff.coeff_dx2_p1[i + j * nx] * v[(i + 1) + j * nx] + coeff.coeff_dx2_m1[i + j * nx] * v[(i - 1) + j * nx]
                            + coeff.coeff_dy2_p1[i + j * nx] * v[i + (j + 1) * nx] + coeff.coeff_dy2_m1[i + j * nx] * v[i + (j - 1) * nx];
         
          vnew[id] = ibm.iBlank[i + j * nx] * vnew[id] / coeff.coeff[i + j * nx];
          //vnew[id] = (1.0 - input.w_AD) * v[id] + input.w_AD * vnew[id];
      }
      id += nGrid;
  }
}

__global__ void Compute_uResidual_AD(int nx, int ny, Grid gridData, coefficient coeff, IBM ibm, CFDInput input,
  REALTYPE dt, REALTYPE* u, REALTYPE* unew,  REALTYPE* uResidue, REALTYPE* sx) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int nGrid = blockDim.x * gridDim.x;

  while (id < nx * ny) {
      int i = id % nx;
      int j = id / nx;
      uResidue[id] = 0.0;
      if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && ibm.iBlank[id] == 1) {
          REALTYPE tmp = sx[id] + coeff.coeff_dx2_p1[i + j * nx] * u[(i + 1) + j * nx] + coeff.coeff_dx2_m1[i + j * nx] * u[(i - 1) + j * nx]
                                + coeff.coeff_dy2_p1[i + j * nx] * u[i + (j + 1) * nx] + coeff.coeff_dy2_m1[i + j * nx] * u[i + (j - 1) * nx];

          //uResidue[id] = abs(ibm.iBlank[i + j * nx] * tmp - u[id] * coeff.coeff[i + j * nx]); 
          uResidue[id] = abs(unew[id]-u[id]);
      }
      id += nGrid;
  }
}

__global__ void Compute_vResidual_AD(int nx, int ny, Grid gridData, coefficient coeff, IBM ibm, CFDInput input,
  REALTYPE dt, REALTYPE* v, REALTYPE* vnew, REALTYPE* vResidue, REALTYPE* sy) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int nGrid = blockDim.x * gridDim.x;

  while (id < nx * ny) {
      int i = id % nx;
      int j = id / nx;
      vResidue[id] = 0.0;
      if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && ibm.iBlank[id] == 1) {
          REALTYPE tmp = sy[id] + coeff.coeff_dx2_p1[i + j * nx] * v[(i + 1) + j * nx] + coeff.coeff_dx2_m1[i + j * nx] * v[(i - 1) + j * nx]
                                + coeff.coeff_dy2_p1[i + j * nx] * v[i + (j + 1) * nx] + coeff.coeff_dy2_m1[i + j * nx] * v[i + (j - 1) * nx];

          //vResidue[id] = abs(ibm.iBlank[i + j * nx] * tmp - v[id] *coeff.coeff[i + j * nx]);
          vResidue[id] = abs(vnew[id] - v[id]);
      }
      id += nGrid;
  }
}

__global__ void Compute_velf(int nx, int ny, Grid gridData, coefficient coeff, IBM ibm, CFDInput input,
  REALTYPE* u, REALTYPE* v, REALTYPE* uf, REALTYPE* vf) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int nGrid = blockDim.x * gridDim.x;

  while (id < (nx-1)*(ny-2)) {
      int i = id % (nx-1);
      int j = id / (nx-1);

      int idc = j * nx + i;

      uf[id] = 1.0 / (gridData.dx[idc + nx] + gridData.dx[idc + nx + 1])
             * (u[idc + nx + 1] * gridData.dx[idc + nx] + u[idc + nx] * gridData.dx[idc + nx + 1]);

      id += nGrid;
  }

  while (id < (nx - 2) * (ny-1)) {
      int i = id % (nx - 2);
      int j = id / (nx - 2);

      int idc = j * nx + i;
      vf[id] = 1.0 / (gridData.dy[idc + nx + 1] + gridData.dy[idc + 1])
             * (v[idc + nx + 1] * gridData.dy[idc + 1] + v[idc + 1] * gridData.dy[idc + nx + 1]);
      
      id += nGrid;
  }
}

__global__ void set_velocity_BC(int nx, int ny, REALTYPE* u, REALTYPE* v, REALTYPE* uf, REALTYPE* vf) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int nGrid = blockDim.x * gridDim.x;

  while (id < nx * ny) {
      int i = id % nx;
      int j = id / nx;

      if (i == 0) {
          u[id] = -u[id + 1] + 1.0 * 2.0; 
          v[id] = -v[id + 1] + 0.0 * 2.0;
      }

      if (j == 0) {
          u[id] = -u[id + nx] + 1.0*2.0;
          v[id] = -v[id + nx] + 0.0*2.0;
      }

      if (i == nx-1) {
          u[id] = -u[id - 1] + 1.0*2.0;
          v[id] = -v[id - 1] + 0.0*2.0; 
      }

      if (j == ny-1) {
          u[id] = -u[id - nx] + 1.0*2.0;
          v[id] = -v[id - nx] + 0.0*2.0;
      }
      id += nGrid;
  }

  while (id < (nx-1) * (ny-2)) {
      int i = id % (nx-1);
      int j = id / (nx-1);

      if (i == 0) {
          uf[id] = 1.0;
      }

      if (j == 0) {
         uf[id] = 1.0;
      }

      if (i == nx - 2) {
         uf[id] = 1.0;
      }

      if (j == ny - 3) {
          uf[id] = 1.0;
      }
      id += nGrid;
  }

  while (id < (nx-2) * (ny-1)) {
      int i = id % (nx-2);
      int j = id / (nx-2);

      if (i == 0) {
          vf[id] = 0.0;
      }

      if (j == 0) {
          vf[id] = 0.0;
      }

      if (i == nx - 3) {
          vf[id] = 0.0;
      }

      if (j == ny - 2) {
          vf[id] = 0.0;
      }
      id += nGrid;
  }
}



void ImmerseFlow::ADsolver()
{
  coefficient coeff;
  REALTYPE* uTemp, * uResidue;
  REALTYPE* vTemp, * vResidue;
  REALTYPE* sx, * sy;
  printf("dt=%f\n", Input.dt);
  CHECK_CUDA_ERROR(cudaMalloc(&uTemp, sizeof(REALTYPE) * Input.nx * Input.ny));
  CHECK_CUDA_ERROR(cudaMalloc(&uResidue, sizeof(REALTYPE) * Input.nx * Input.ny));
  CHECK_CUDA_ERROR(cudaMalloc(&vTemp, sizeof(REALTYPE) * Input.nx * Input.ny));
  CHECK_CUDA_ERROR(cudaMalloc(&vResidue, sizeof(REALTYPE) * Input.nx * Input.ny));

  CHECK_CUDA_ERROR(cudaMalloc(&coeff.coeff, Input.nx * Input.ny * sizeof(REALTYPE)));
  CHECK_CUDA_ERROR(cudaMalloc(&coeff.coeff_dx2_p1, Input.nx * Input.ny * sizeof(REALTYPE)));
  CHECK_CUDA_ERROR(cudaMalloc(&coeff.coeff_dx2_m1, Input.nx * Input.ny * sizeof(REALTYPE)));
  CHECK_CUDA_ERROR(cudaMalloc(&coeff.coeff_dy2_p1, Input.nx * Input.ny * sizeof(REALTYPE)));
  CHECK_CUDA_ERROR(cudaMalloc(&coeff.coeff_dy2_m1, Input.nx * Input.ny * sizeof(REALTYPE)));
  
  CHECK_CUDA_ERROR(cudaMalloc(&sx, Input.nx * Input.ny * sizeof(REALTYPE)));
  CHECK_CUDA_ERROR(cudaMalloc(&sy, Input.nx * Input.ny * sizeof(REALTYPE)));

  calculateADCoefficients << <CUDAData.blocksPerGrid, CUDAData.threadsPerBlock >> > (Input.nx, Input.ny, gridData, Input.dt, coeff, Input.Re);
  CHECK_CUDA_ERROR(cudaGetLastError()); 
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  REALTYPE uResidual = 1.0;
  REALTYPE vResidual = 1.0;
  int iter = 0;


  Compute_velf << <CUDAData.blocksPerGrid, CUDAData.threadsPerBlock >> > (Input.nx, Input.ny, gridData, coeff, ibm, Input,
                                                                          Data.u.velc, Data.v.velc, Data.u.velf, Data.v.velf);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  //Set Boundary Conditions    
  set_velocity_BC << <CUDAData.blocksPerGrid, CUDAData.threadsPerBlock >> > (Input.nx, Input.ny, Data.u.velc, Data.v.velc, Data.u.velf, Data.v.velf);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  ADSource << <CUDAData.blocksPerGrid, CUDAData.threadsPerBlock >> > (Input.nx, Input.ny, gridData, Input.dt, Data.u.velc, Data.v.velc, 
                                                                      Data.u.velf, Data.v.velf, sx, sy);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  printf("________AD slover________\n");
 
  while (uResidual + vResidual > pow(10.0, -6.0) && iter < Input.AD_itermax) {     
      //Set Boundary Conditions    
      set_velocity_BC << <CUDAData.blocksPerGrid, CUDAData.threadsPerBlock >> > (Input.nx, Input.ny, Data.u.velc, Data.v.velc, Data.u.velf, Data.v.velf);
      CHECK_CUDA_ERROR(cudaGetLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());

      Compute_velf << <CUDAData.blocksPerGrid, CUDAData.threadsPerBlock >> > (Input.nx, Input.ny, gridData, coeff, ibm, Input,
                       Data.u.velc, Data.v.velc, Data.u.velf, Data.v.velf);
      CHECK_CUDA_ERROR(cudaGetLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());

      ADusolver_kernel << <CUDAData.blocksPerGrid, CUDAData.threadsPerBlock >> > (Input.nx, Input.ny, gridData, coeff, ibm, Input, 
                                                                                  Input.dt, Data.u.velc, Data.u.velf, Data.v.velf, uTemp, sx);
      CHECK_CUDA_ERROR(cudaGetLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());

      ADvsolver_kernel << <CUDAData.blocksPerGrid, CUDAData.threadsPerBlock >> > (Input.nx, Input.ny, gridData, coeff, ibm, Input, 
                                                                                  Input.dt, Data.v.velc, Data.u.velf, Data.v.velf, vTemp, sy);
      CHECK_CUDA_ERROR(cudaGetLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());

      //Set Boundary Conditions    
      set_velocity_BC << <CUDAData.blocksPerGrid, CUDAData.threadsPerBlock >> > (Input.nx, Input.ny, Data.u.velc, Data.v.velc, Data.u.velf, Data.v.velf);
      CHECK_CUDA_ERROR(cudaGetLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());

       //Swap pointers
      REALTYPE* temp = Data.u.velc;
      Data.u.velc = uTemp;
      uTemp = temp;
      
      temp = Data.v.velc;
      Data.v.velc = vTemp;
      vTemp = temp;

      Compute_uResidual_AD << <CUDAData.blocksPerGrid, CUDAData.threadsPerBlock >> > (Input.nx, Input.ny, gridData, coeff, ibm, Input, 
                                                                                      Input.dt, Data.u.velc, uTemp, uResidue, sx);
      CHECK_CUDA_ERROR(cudaGetLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());

      Compute_vResidual_AD << <CUDAData.blocksPerGrid, CUDAData.threadsPerBlock >> > (Input.nx, Input.ny, gridData, coeff, ibm, Input,
                                                                                      Input.dt, Data.v.velc, vTemp, vResidue, sy);
      CHECK_CUDA_ERROR(cudaGetLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());

      Reduction(uResidue, &uResidual);
      CHECK_CUDA_ERROR(cudaGetLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());

      Reduction(vResidue, &vResidual);
      CHECK_CUDA_ERROR(cudaGetLastError());
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());
      
      iter += 1;
      printf("iter = %d %f %f\n", iter, uResidual, vResidual);
  }

  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

 
  
  
  saveDataToFile(Input.nx, Input.ny, gridData.xc, gridData.yc, Data.u.velc, "../results/uc.dat");
  saveDataToFile(Input.nx, Input.ny, gridData.xc, gridData.yc, Data.v.velc, "../results/vc.dat");
  

  CHECK_CUDA_ERROR(cudaFree(uTemp));
  CHECK_CUDA_ERROR(cudaFree(uResidue));
  CHECK_CUDA_ERROR(cudaFree(vTemp));
  CHECK_CUDA_ERROR(cudaFree(vResidue));

  CHECK_CUDA_ERROR(cudaFree(coeff.coeff));
  CHECK_CUDA_ERROR(cudaFree(coeff.coeff_dx2_p1));
  CHECK_CUDA_ERROR(cudaFree(coeff.coeff_dx2_m1));
  CHECK_CUDA_ERROR(cudaFree(coeff.coeff_dy2_p1));
  CHECK_CUDA_ERROR(cudaFree(coeff.coeff_dy2_m1));

  CHECK_CUDA_ERROR(cudaFree(sx));
  CHECK_CUDA_ERROR(cudaFree(sy));
}