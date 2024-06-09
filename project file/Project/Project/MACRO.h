#pragma once
#include "preSim.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <fstream>

//reference
//https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
inline void checkCuda(cudaError_t err, const char* const func, const char* const file,const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
            << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
       
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLastCuda(__FILE__, __LINE__)
inline void checkLastCuda(const char* const file, const int line)
{
    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
            << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
       
        std::exit(EXIT_FAILURE);
    }
}