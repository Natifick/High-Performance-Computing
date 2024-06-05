

#include <stdio.h>
#include <array>
#include <iostream>

#include "utils.cuh"

static const size_t N = 1024;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void reduceKernel(float* a, size_t N) {
    size_t gid = getGid();
    size_t s = N / 2;
    float dummy;

    while (s > 0) {
        if (gid < s) {
            a[gid] = a[gid] + a[gid + s];
        }
        s = s / 2;
        __syncthreads();
    }

    // a[0] contains the result
}

__global__ void reduceKernel_shared(float* a, size_t N) {
    size_t gid = getGid();
    size_t s = N / 2;
    float dummy;
    __shared__ float shmem[1024];

    shmem[gid] = a[gid];


    while (s > 0) {
        __syncthreads();
        if (gid < s) {
            shmem[gid] = shmem[gid] + shmem[gid + s];
        }
        s = s / 2;
    }

    if  (gid==0) {
        a[gid] = shmem[gid];
    }
    // a[0] contains the result
}

int main() {
    float* h_a = (float*)malloc(N*sizeof(float));
    for(auto n = 0; n < N; ++n) {
        h_a[n] = n;
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    float* d_a;
    gpuErrchk(cudaMalloc(&d_a, (N-3) * sizeof(float)));

    gpuErrchk(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    reduceKernel<<<1, N>>>(d_a, N);
    cudaEventRecord(end);

    //cudaEventSynchronize(start);
    cudaEventSynchronize(end);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, end);

    cudaMemcpy(h_a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result is " << h_a[0] << std::endl;
    std::cout << "Reduce time is " << milliseconds << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_a);
    return 0;
}
