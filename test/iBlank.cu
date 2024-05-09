#include <iostream>
#include <vector>
#include <stdio.h>
#include <math.h>

using namespace std;

//Postprocessing function
void write_results_to_file(int ni, int nj, const vector<vector<float>>& final_data, const char* filename) {
  FILE* fp = fopen(filename, "w"); // Open in write mode

  if (fp == NULL) {
      printf("Error opening file: %s\n", filename);
      return;
  }

  fprintf(fp, "TITLE = \"Post Processing Tecplot\"\n");  // String
  fprintf(fp, "VARIABLES = \"T\"\n");  // String
  fprintf(fp, "ZONE T=\"BIG ZONE\", I=%d, J=%d, DATAPACKING=POINT\n", ni, nj);  // Integers

  // Write data in row-major order
  for (int j = 0; j < nj; j++) {
      for (int i = 0; i < ni; i++) {
          fprintf(fp, "%i,%i,%f\n", i, j, final_data[i][j]);
      }
  }

  fclose(fp);
}

// Class to create models



// Main Function
int main(){

  // Cylinder Parameter

  // Grid Parameters
  const float lx = 10;
  const float ly = 5;
  const int nx = 128;
  const int ny = 64;

  printf("nx,ny = %i,%i\n",nx,ny);

  vector<vector<float>> grid(nx, vector<float>(ny,0.0f));

  for (int i=0; i<nx; ++i){
    for (int j=0; j<ny; ++j){
      grid[i][j] = j;
    }
  }

  write_results_to_file(nx, ny, grid, "final_results.dat");

  
  return 0;
}