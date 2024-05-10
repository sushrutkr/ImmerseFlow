#ifndef INPUT_READER_H_
#define INPUT_READER_H_

#include <fstream>
#include <iostream>
#include <vector>

void setZero(std::vector<std::vector<float>>& vec);

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
  int nx_;
  int ny_;
  std::vector<float> x_;
  std::vector<float> y_;
  std::vector<std::vector<float>> iBlank_;

public:
  Grid(int nx, int ny);
  void calculateiBlank();
  const std::vector<std::vector<float>>& GetiBlank() const; 
  const std::vector<float>& Getx() const; 
  const std::vector<float>& Gety() const; 
};

#endif // INPUT_READER_H_
