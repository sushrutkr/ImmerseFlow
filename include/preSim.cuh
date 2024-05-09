#ifndef INPUT_READER_H_
#define INPUT_READER_H_

#include <fstream>
#include <iostream>
#include <vector>

class InputReader{
private:
  int nx_;
  int ny_;

public:
  InputReader();

  int Getnx();
  
  int Getny();
};

class Grid{
private:
  std::vector<float> x_;
  std::vector<float> y_;

public:
  Grid(int nx, int ny);

  const std::vector<float>& Getx() const; 
  const std::vector<float>& Gety() const; 
};

#endif // INPUT_READER_H_
