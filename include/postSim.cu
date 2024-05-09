#include "postSim.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdio.h>

using namespace std;


// Postprocessing
void write_results_to_file(const vector<float>& x, 
                           const vector<float>& y, 
                           const vector<vector<float>>& final_data, 
                           const char* filename){
  
  const int ni = size(x);
  const int nj = size(y);                          
  
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
      fprintf(fp, "%f,%f,%f\n", x[i], y[j], final_data[i][j]);
    }
  }

  fclose(fp);
}