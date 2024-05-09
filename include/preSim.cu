#include "preSim.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>


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

Grid::Grid(int nx, int ny){
  //Create x, y, and iBlank
  x_.resize(nx);
  y_.resize(ny);
  int dummy;

  std::ifstream infile;

  //Reading X coordinate data
  infile.open("inputs/xgrid.dat");
  if (!infile){
    std::cerr << "Error Opening Xgrid.dat" << std::endl;
  }
  for (int i=0; i<nx; ++i){
    infile >> dummy >> x_[i];
  }
  infile.close();

  //Reading Y coordinate data
  infile.open("inputs/ygrid.dat");
  if (!infile){
    std::cerr << "Error Opening Ygrid.dat" << std::endl;
  }
  for (int i=0; i<ny; ++i){
    infile >> dummy >> y_[i];
  }
  infile.close();

}

const std::vector<float>& Grid::Getx() const {
  return x_;
}

const std::vector<float>& Grid::Gety() const {
  return y_;
}

//Inputs
// class InputReader {
// private:
//   // std::ifstream file_;
//   int nx_;
//   int ny_;

// public:

//   // Read input files
//   InputReader(){
//     // file_.open("input.txt", std::ios::in);

//     // if (!file_.is_open()) {
//     //   std::cerr << "Could not open input.txt" << std::endl;
//     //   return;
//     // }

//     // std::string line;
//     // std::getline(file_,line);
//     // std:sscanf(line.c_str(),"%d, %d", &nx_, %ny_);
//     nx_ = 181;
//     ny_ = 129;
//   }

//   // ~InputReader(){
//   //   file_.close();
//   // }

//   // Getter function - const to not allow any change
//   int Getnx() const { return nx_; }
//   int Getny() const { return ny_; }
// };