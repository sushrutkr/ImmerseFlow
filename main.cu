#include <iostream>
#include <vector>
#include "include/preSim.cuh"
#include "include/postSim.cuh"

using namespace std;

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

  vector<vector<float>> iBlank(nx, vector<float>(ny,0.0f));

  for (int i=0; i<nx; ++i){
    for (int j=0; j<ny; ++j){
      iBlank[i][j] = y[j];
    }
  }

  write_results_to_file(x, y, iBlank, "results/final_results.dat");

  return 0;
}

// #include <iostream>
// #include "include/preSim.cuh"

// int main() {
//     std::cout << addSum(10, 20) << std::endl;
//     return 0;
// }
