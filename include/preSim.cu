#include "preSim.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

void setZero(std::vector<std::vector<float>>& vec) {
  size_t rows = vec.size();
  size_t cols = vec.empty() ? 0 : vec[0].size();
  vec.assign(rows, std::vector<float>(cols, 0.0f));
}


InputReader::InputReader(){
  nx_ = 181;
  ny_ = 129;
}

int InputReader::Getnx() {
  return nx_;
}
  
int InputReader::Getny() {
  return ny_;
}

Grid::Grid(int nx, int ny) : nx_(nx), ny_(ny) {
  //Create x, y, and iBlank
  x_.resize(nx_);
  y_.resize(ny_);
  int dummy;

  std::ifstream infile;

  //Reading X coordinate data
  infile.open("inputs/xgrid.dat");
  if (!infile){
    std::cerr << "Error Opening Xgrid.dat" << std::endl;
  }
  for (int i=0; i<nx_; ++i){
    infile >> dummy >> x_[i];
  }
  infile.close();

  //Reading Y coordinate data
  infile.open("inputs/ygrid.dat");
  if (!infile){
    std::cerr << "Error Opening Ygrid.dat" << std::endl;
  }
  for (int i=0; i<ny_; ++i){
    infile >> dummy >> y_[i];
  }
  infile.close();
  this -> calculateiBlank();

}

void Grid::calculateiBlank() {
  iBlank_.resize(nx_, std::vector<float>(ny_, 0.0f));

  // Iterate over each element of iBlank_ and set it to 1.0f if it satisfies the condition
  for (int i = 0; i < nx_; ++i) {
      for (int j = 0; j < ny_; ++j) {
        if (pow((x_[i] - 3), 2) + pow((y_[j] - 2.5), 2) <= pow(0.5, 2)) {
              iBlank_[i][j] = 1.0f;
          }
      }
  }
}

const std::vector<std::vector<float>>& Grid::GetiBlank() const {
  return iBlank_;
}

const std::vector<float>& Grid::Getx() const {
  return x_;
}

const std::vector<float>& Grid::Gety() const {
  return y_;
}