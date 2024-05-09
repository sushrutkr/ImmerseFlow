#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void printThread(){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  printf("ThreadIdx : %d \n",i);
  
}


int main(){

  size_t size = 16*sizeof(int);
	int *array;

	cudaError_t cudaStatus = cudaMallocManaged(&array, size);

	if (cudaStatus != cudaSuccess){
		std::cerr << "cudaMallocManaged failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    return 1;
	}

  printThread<<<4,4>>>();
  cudaDeviceSynchronize();

	cudaFree(array);

  return 0;
}