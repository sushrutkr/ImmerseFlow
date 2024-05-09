#pragma once
#include <vector>
using namespace std;

void write_results_to_file(const vector<float>& x, 
                           const vector<float>& y, 
                           const vector<vector<float>>& final_data, 
                           const char* filename);
