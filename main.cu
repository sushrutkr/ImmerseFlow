#include <iostream>
#include <vector>
#include "include/preSim.cuh"
#include "include/postSim.cuh"

// Main Function
int main(){

  // Read Input
  InputReader reader;
  int nx = reader.Getnx();
  int ny = reader.Getny();
  printf("nx,ny = %i,%i\n",nx,ny);

  // Set Grid
  Grid g(nx,ny);
  std::vector<float> x = g.Getx();
  std::vector<float> y = g.Gety();
  std::vector<std::vector<float>> iBlank = g.GetiBlank();

  write_results_to_file(x, y, iBlank, "results/final_results.dat");

  return 0;
}