//==============================================================
// CUDA sum reduction with Dynamic Parallelism code referenced from Robert_Crovella NVidia forum post:
// https://forums.developer.nvidia.com/t/reduction-sum-using-dynamic-parallelism-working-upto-a-magical-number/79695/2
//
// Compile on NVidia with CUDA SDK
// nvcc -rdc=true sumReductionCudaDP.cu
// =============================================================

#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

const int N = 1048576*256+37;
const int BLOCK = 1024;

__device__ int blockID = 0;
__device__ int numK = 0;

// tree reduction kernel with CUDA Dynamic Parallelism
__global__ void treeReduceKernelCDP(int* d_in, int n) {
    __shared__ int sdata[BLOCK];

    int tID = threadIdx.x;
    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID == 0) atomicAdd(&numK, 1);

    if (gridDim.x == 1){ // when kernel has only 1 block use classic parallel reduction
        sdata[tID] = d_in[tID];
        if ((tID + BLOCK) < n) sdata[tID] += d_in[tID + BLOCK];
        for (int i = BLOCK/2; i > 0; i/=2){
            __syncthreads();
            if (tID < i) sdata[tID] += sdata[tID + i];
        }
        if (tID == 0) d_in[0] = sdata[0];
    } else { // tree reduction
        int offset = n/2 + (n&1);
        if ((ID + offset) < n) d_in[ID] += d_in[ID + offset];
        __threadfence();
        // Launch dynamic kernel in last block
        if (tID == 0) {
            int currentBlockID = atomicAdd(&blockID, 1);  // get current block ID
            if (currentBlockID == gridDim.x - 1){ // if current block ID is last block
                blockID = 0;
                __threadfence();
                int nHalf = offset/2 + (offset&1);
                int numBlocks = (nHalf + BLOCK - 1) / BLOCK;
                treeReduceKernelCDP <<<numBlocks, BLOCK>>> (d_in, offset);
            }
        }
        // END dynamic kernel launch
    }   
}

int main(){

    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev, 0);
    std::cout << "Device: " << dev.name << "\n";
    
    int *data, *h_data = (int *)malloc(N*sizeof(int));
    cudaMalloc(&data, N*sizeof(int));
    for (int i = 0; i < N; i++) h_data[i] = 1;
    cudaMemcpy(data, h_data, N*sizeof(int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Perform tree reduction recursively using Cuda Dynamic Parallelism kernel
    int n = N;
    int nHalf = n/2 + (n&1);
    int numBlocks = (nHalf + BLOCK - 1) / BLOCK;
    int numKernelLaunch = 0;
    treeReduceKernelCDP <<<numBlocks, BLOCK>>> (data, n);
    cudaDeviceSynchronize();
    // END reduction

    auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
    std::cout << "Kernel Duration: " << duration / 1e+9 << " seconds\n";

    cudaMemcpyFromSymbol(&numKernelLaunch, numK, sizeof(int));
    std::cout << numKernelLaunch << " Kernels Launched\n";
    cudaMemcpy(h_data, data, sizeof(int), cudaMemcpyDeviceToHost);
    (h_data[0] == N) ? std::cout << "PASS: " : std::cout << "FAIL: ";
    std::cout << "Sum = "<< h_data[0] << ", N = " << N << "\n";
    cudaFree(data);
    free(h_data);
    return 0;
}

