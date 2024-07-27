#ifndef GLOBALVARIABLES_CUH
#define GLOBALVARIABLES_CUH
#define  REALTYPE   double

struct vel {
    REALTYPE* velf, * velc, * velInter;
};

// Define a struct to hold the CFD arrays
struct CFDData {
    REALTYPE* p;
    vel u, v;
};

struct CFDInput {
    //Restart
    int Restart;
    int Restart_Time;

    //Domain Information
    int nx, ny;
    int nxf, nyf;
    REALTYPE Lx, Ly;

    //Iterative Solver Settings
    int w_AD, w_PPE, AD_itermax, PPE_itermax, AD_solver, PPE_solver;

    //Simulation Settings
    REALTYPE ErrorMax, tmax, dt, Re, mu;

    //Data write
    int Write_Interval;
};

struct BC {
    REALTYPE u_bc_w, u_bc_e, u_bc_n, u_bc_s,
             v_bc_w, v_bc_e, v_bc_n, v_bc_s,
             p_bc_w, p_bc_e, p_bc_n, p_bc_s;
};

struct Grid {
    REALTYPE* xf;
    REALTYPE* yf;
    REALTYPE* xc;
    REALTYPE* yc;
    REALTYPE* dx, * dy;

};

struct IBM {
    REALTYPE* iBlank;
};

struct CUDAInfo {
    int blocksPerGrid, threadsPerBlock;
};

struct coeffPPE {
    REALTYPE* coeff_dx2_p1,* coeff_dx2_m1,
            * coeff_dy2_p1,* coeff_dy2_m1;
    REALTYPE* coeff_ppe;
};

struct coefficient {
    REALTYPE* coeff_dx2_p1, * coeff_dx2_m1,
        * coeff_dy2_p1, * coeff_dy2_m1;
    REALTYPE* coeff;
};

struct ImmerseFlow {
    CFDData   Data;
    CFDInput  Input;
    Grid      gridData;
    IBM       ibm;
    BC        BcData;
    CUDAInfo  CUDAData;
    // Function prototypes
    void initializeCFDData();
    void printCFDData();
    void readGridData();
    void initializeData();
    void allocation();
    void freeAllocation();
    void CUDAQuery();
    void PPESolver();
    void ADsolver();
    void Reduction(REALTYPE* input, REALTYPE* output);
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
