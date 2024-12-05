//==============================================================
// CUDA Sum Tree Reduction using recursive kernel launches
//
// Compile on NVidia with CUDA SDK
// nvcc sumReductionCuda.cu
// =============================================================

#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

const int N = 1048576*256+37;
const int BLOCK = 1024;

// tree reduction kernel
__global__ void treeReduceKernel(int* d_in, int n) {
    __shared__ int sdata[BLOCK];

    int tID = threadIdx.x;
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

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

    // Perform tree reduction recursively
    int n = N;
    int nHalf = n/2 + (n&1);
    int numBlocks = (nHalf + BLOCK - 1) / BLOCK;
    int numKernelLaunch = 0;
    while (numBlocks >= 1) {
        numKernelLaunch++;
        treeReduceKernel <<<numBlocks, BLOCK>>> (data, n);
        cudaDeviceSynchronize();
        if (numBlocks == 1) break; // quit after last block reduction
        // half array size and numBlocks to launch new kernel
        n = n/2 + (n&1);
        nHalf = n/2 + (n&1);
        numBlocks = (nHalf + BLOCK - 1) / BLOCK;
    }
    // END reduction

    auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
    std::cout << "Kernel Duration: " << duration / 1e+9 << " seconds\n";
    
    std::cout << numKernelLaunch << " Kernels Launched\n";
    cudaMemcpy(h_data, data, sizeof(int), cudaMemcpyDeviceToHost);
    (h_data[0] == N) ? std::cout << "PASS: " : std::cout << "FAIL: ";
    std::cout << "Sum = "<< h_data[0] << ", N = " << N << "\n";
    cudaFree(data);
    free(h_data);
    return 0;
}


