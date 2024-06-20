#ifndef GLOBALVARIABLES_CUH
#define GLOBALVARIABLES_CUH

struct vel {
    float* velf, * velc, * velInter;
};

// Define a struct to hold the CFD arrays
struct CFDData {
    float* p;
    vel u, v;
};

struct CFDInput {
    //Restart
    int Restart;
    int Restart_Time;

    //Domain Information
    int nx, ny;
    double Lx, Ly;

    //Iterative Solver Settings
    int w_AD, w_PPE, AD_itermax, PPE_itermax, AD_solver, PPE_solver;

    //Simulation Settings
    double ErrorMax, tmax, dt, Re, mu;

    //Data write
    int Write_Interval;
};

struct Grid {
    float* x;
    float* y;
};

struct IBM {
    float* iBlank;
};

struct ImmerseFlow {
    CFDData   Data;
    CFDInput  Input;
    Grid      gridData;
    IBM       ibm;
    // Function prototypes
    void initializeCFDData();
    void printCFDData();
    void readGridData();
    void initializeData();
};

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) {                                \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
        std::cerr << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        exit(1);                                                \
    }                                                           \
}

#define CHECK_LAST_CUDA_ERROR() {                               \
    cudaError_t err = cudaGetLastError();                       \
    if (err != cudaSuccess) {                                   \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
        std::cerr << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        exit(1);                                                \
    }                                                           \
}

#endif // GLOBALVARIABLES_CUH
