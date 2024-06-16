#ifndef GLOBALVARIABLES_CUH
#define GLOBALVARIABLES_CUH

struct CFDData {
    struct {
        float* velc;
    } u, v;
    float* p;
};

struct Grid {
    float* x;
    float* y;
};

struct IBM {
    float* iblank;
};

extern CFDData devData;  
extern Grid gridData;
extern IBM ibm;

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
