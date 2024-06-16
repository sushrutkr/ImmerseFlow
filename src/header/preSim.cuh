#ifndef PRESIM_CUH
#define PRESIM_CUH

#include "globalVariables.cuh"

// Function declarations
void initializeCFDData(int nx, int ny, CFDData devData);
void computeIBM(int nx, int ny, IBM ibm);

#endif // PRESIM_CUH
