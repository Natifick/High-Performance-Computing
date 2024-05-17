#include <array>
#include <iostream>
#include <stdio.h>

#include "utils.cuh"

static const size_t N = 1<<6;

// GPU code

// Check all errors that occur
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void initArrayKernel(float* array, int size) {
    int gid = getGid();
    if (gid < size) {
        array[gid] = gid;
    }
}

__global__ void zeroKernel(float* array, int size) {
    int gid = getGid();
    if (gid < size) {
        array[gid] = 0;
    }
}

__global__ void squareKernel(float* inArray, float* outArray, int size) {
    int gid = getGid();
    if (gid < size) {
        outArray[gid] = inArray[gid] * inArray[gid];
    }
}

__global__ void adderKernel(float* array, int size) {
    int gid = getGid();
    if (gid < size) {
        //array[0] = array[0] + 1;
        atomicAdd(&array[0], 1);
    }
}

// CPU code
int main() {

    std::array<float, N> h_array;

    float* d_inArray;
    cudaMalloc(&d_inArray, N * sizeof(float));

    float* d_outArray;
    cudaMalloc(&d_outArray, N * sizeof(float));

    initArrayKernel<<<1024, 1024>>>(d_inArray, N);
    zeroKernel<<<1024, 1024>>>(d_outArray, N);
    //squareKernel<<<1, N>>>(d_inArray, d_outArray, N);
    adderKernel<<<1024, 1024>>>(d_outArray, N);

    cudaMemcpy(&h_array[0], d_outArray, N * sizeof(float), cudaMemcpyDeviceToHost);

    for(auto n = 0; n < N; ++n) {
        std::cout << h_array[n] << "\t";
    }
    std::cout << std::endl;

    std::cout << "Done from CPU\n";
    cudaFree(d_inArray);
    cudaFree(d_outArray);
    return 0;
}
