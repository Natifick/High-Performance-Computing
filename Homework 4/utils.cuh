__device__ int getGid() {

    int tid = threadIdx.x + blockDim.x * threadIdx.y +
              blockDim.x * blockDim.y * threadIdx.z;

    int bid = blockIdx.x + gridDim.x * blockIdx.y +
              gridDim.x * gridDim.y * blockIdx.z;

    int gid = bid * blockDim.x * blockDim.y * blockDim.z + tid;

    return gid;
}

