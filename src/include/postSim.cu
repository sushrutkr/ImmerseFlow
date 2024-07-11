#include "../header/postSim.cuh"
#include "../header/globalVariables.cuh"
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>

void saveDataToFile(int nx, 
                    int ny, 
                    const REALTYPE* x, 
                    const REALTYPE* y, 
                    const REALTYPE* dataToSave, 
                    const char* filename) {
  // Allocate host memory
  REALTYPE* host_x = (REALTYPE*)malloc(sizeof(REALTYPE) * nx);
  REALTYPE* host_y = (REALTYPE*)malloc(sizeof(REALTYPE) * ny);
  REALTYPE* host_iBlank = (REALTYPE*)malloc(sizeof(REALTYPE) * nx * ny);

  // Check allocation
  if (host_x == NULL || host_y == NULL || host_iBlank == NULL) {
      std::cerr << "Host memory allocation failed" << std::endl;
      return;
  }

  // Copy data from device to host
  CHECK_CUDA_ERROR(cudaMemcpy(host_x, x, sizeof(REALTYPE) * nx, cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(host_y, y, sizeof(REALTYPE) * ny, cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(host_iBlank, dataToSave, sizeof(REALTYPE) * nx * ny, cudaMemcpyDeviceToHost));

  // Write results to file
  write_results_to_file(host_x, host_y, host_iBlank, nx, ny, filename);

  // Free host memory
  free(host_x);
  free(host_y);
  free(host_iBlank);
}

void write_results_to_file(const REALTYPE* x,
                           const REALTYPE* y,
                           const REALTYPE* final_data,
                           int ni, int nj,
                           const char* filename){

  FILE* fp = fopen(filename, "w"); // Open in write mode

  if (fp == NULL) {
    printf("Error opening file: %s\n", filename);
    return;
  }

  fprintf(fp, "TITLE = \"Post Processing Tecplot\"\n");  // String
  fprintf(fp, "VARIABLES = \"X\",\"Y\",\"T\"\n");  // String
  fprintf(fp, "ZONE T=\"BIG ZONE\", I=%d, J=%d, DATAPACKING=POINT\n", ni, nj);  // Integers

  // Write data in row-major order
  for (int j = 0; j < nj; j++) {
    for (int i = 0; i < ni; i++) {
      fprintf(fp, "%f,%f,%f\n", x[i], y[j], final_data[i + j * ni]);
    }
  }

  fclose(fp);
}
