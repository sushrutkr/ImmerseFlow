#include "./header/preSim.cuh"
#include "./header/globalVariables.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

void readInputFile(const std::string& filename, ImmerseFlow& Solver) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    while (getline(inputFile, line)) {
        if (line.empty() || line[0] == '=' || line[0] == '_') {
            continue;  // Skip header or separator lines
        }

        std::istringstream iss(line);
        std::string keyword;

        // Reading and parsing line by line
        if (line.find("Restart") != std::string::npos) {
            getline(inputFile, line);  // Read the next line
            iss.str(line); iss.clear();
            iss >> Solver.Input.Restart >> Solver.Input.Restart_Time;
        } else if (line.find("nx") != std::string::npos) {
            getline(inputFile, line);  // Read the next line
            iss.str(line); iss.clear();
            iss >> Solver.Input.nx >> Solver.Input.ny;
        } else if (line.find("Lx") != std::string::npos) {
            getline(inputFile, line);  // Read the next line
            iss.str(line); iss.clear();
            iss >> Solver.Input.Lx >> Solver.Input.Ly;
        } else if (line.find("w-AD") != std::string::npos) {
            getline(inputFile, line);  // Read the next line
            iss.str(line); iss.clear();
            iss >> Solver.Input.w_AD >> Solver.Input.w_PPE >> Solver.Input.AD_itermax >> Solver.Input.PPE_itermax >> Solver.Input.AD_solver >> Solver.Input.PPE_solver;
        } else if (line.find("ErrorMax") != std::string::npos) {
            getline(inputFile, line);  // Read the next line
            iss.str(line); iss.clear();
            iss >> Solver.Input.ErrorMax >> Solver.Input.tmax >> Solver.Input.dt >> Solver.Input.Re >> Solver.Input.mu;
        } else if (line.find("Write Interval") != std::string::npos) {
            getline(inputFile, line);  // Read the next line
            iss.str(line); iss.clear();
            iss >> Solver.Input.Write_Interval;
        }
    }

    inputFile.close();
    Solver.Input.nxf = Solver.Input.nx + 1;
    Solver.Input.nyf = Solver.Input.ny + 1;
    Solver.Input.nx += 2;
    Solver.Input.ny += 2;
}

void printLargeText() {
    std::string text = R"(
  _____                                                __  _                               
 |_   _|                                              / _|| |                   _      _   
   | |   _ __ ___   _ __ ___    ___  _ __  ___   ___ | |_ | |  ___ __      __ _| |_  _| |_ 
   | |  | '_ ` _ \ | '_ ` _ \  / _ \| '__|/ __| / _ \|  _|| | / _ \\ \ /\ / /|_   _||_   _|
  _| |_ | | | | | || | | | | ||  __/| |   \__ \|  __/| |  | || (_) |\ V  V /   |_|    |_|  
 |_____||_| |_| |_||_| |_| |_| \___||_|   |___/ \___||_|  |_| \___/  \_/\_/                
                                                                                           
                                                                                           
)";
    std::cout << text << std::endl;
}

int main() {
    printLargeText();
    ImmerseFlow Solver;

    // Read input from file
    readInputFile("../inputs/inputs.txt", Solver);
     

  

    // Allocate memory for CFD data
    Solver.CUDAQuery();
    Solver.allocation();
    // Initialize and print the CFD data using CUDA
    Solver.readGridData();
    Solver.initializeData();
    
    
    for (int timestep = 0; timestep < 20; ++timestep)
    {
        Solver.ADsolver();
    }
   
    //Solver.PPESolver();
    Solver.freeAllocation();

    return 0;
}
