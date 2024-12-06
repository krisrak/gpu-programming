# Removing CUDA Dynamic Parallelism

This document explains how CUDA Dynamic Parallelism was removed to support porting CUDA code to SYCL code.

_NOTE: This process of removing CUDA Dynamic Parallelism is done since as of 12/2024 SYCL does not support similar functionality as CUDA Dynamic Parallelism and the only way to port to SYCL is to first remove CUDA Dynamic Parallelism. This is not needed once SYCL supports recursive kernel launch from kernel in future_

### Sum Reduction with CUDA Dynamic Parallelism

The snippet belwo from [sumRecutionCudaDP.cu](sumRecutionCudaDP.cu) code shows cuda kernel with CUDA Dynamic Parallelism, where the kernel code recursively calls the same CUDA kernel to perform sum reduction.

##### {code-snippet 1}
```cpp
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
```
The above kernel is just called once from host as shown below and the kernel will call the kernel recursively to perform sum reduction of large arrays.

##### {code-snippet 2}
```cpp
treeReduceKernelCDP <<<numBlocks, BLOCK>>> (data, n);
```

The snippet below shows the exact lines of kernel code that launches kernel recursively, this is called CUDA Dynamic Parallelism.

##### {code-snippet 3}
```cpp
...
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
...
```
The above logic has to be moved to host side to launch kernel recursively from host instead so the CUDA Dynamic Parallelism is removed, which will enable migrating to SYCL. The section below explains how this is accomplished.

### Sum Reduction without CUDA Dynamic Parallelism

The snippet below from [sumRecutionCuda.cu](sumRecutionCuda.cu) code shows cuda kernel without CUDA Dynamic Parallelism, where the kernel code just does one pass of tree reduction. We will then implement host code to call the kernel recursively to achieve the same result of sum reduction.

##### {code-snippet 4}
```cpp
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
```
The code snippet below from [sumRecutionCuda.cu](sumRecutionCuda.cu) code shows launching kernel recursively to achive the result as CUDA Dynamic Parallelism.

##### {code-snippet 5}
```cpp
...
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
...
```
The main difference is that the recursive kernel launch is moved from kernel to host. 

Note that {code-snipped 3} and {code-snippet 5} does the same configuration to launch kernel, but the main differnece is one is done in kernel and other is done at host.








