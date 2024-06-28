#pragma once
#include "../header/globalVariables.cuh"
void write_results_to_file(const REALTYPE* x,
  const REALTYPE* y,
  const REALTYPE* final_data,
  int ni, int nj,
  const char* filename);