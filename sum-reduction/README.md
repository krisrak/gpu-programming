# Sum Reduction with and without CUDA Dynamic Parallelism

The CUDA code examples demostrate how CUDA Dynamic Parallelism works for simple Tree Reduction algorithm to calculate sum: [sumReductionCudaDP.cu](sumReductionCudaDP.cu)

The same algorithm is implemented without using CUDA Dynamic Parallelism, and instead launching kernel recursively in host code: [sumReductionCuda.cu](sumReductionCuda.cu)

As of 12/2024 SYCL does not support functionality similar to CUDA Dynamic Parallelism, the example below shows how to port code that uses CUDA Dynamic Parallelism to SYCL, by first moving CUDA Dynamic Parallelism logic to host side recursive kernel launch and then porting to SYCL: [sumReductionSYCL.cpp](sumReductionSYCL.cpp).

#### CUDA Dynamic Parallelism --> CUDA --> SYCL

The document [RemoveCDP](RemoveCDP.md) shows how CUDA Dynamic Parallelism is removed by moving the recursive launching of kernel in host code. Once CUDA Dynamic Parallelism si removed, the CUDA code can be ported to SYCL code either using SYCLomatic or porting CUDA APIs manually to SYCL API equivalents.

| Filename | Description | Language |
| --- | --- | --- |
| [sumReductionCudaDP.cu](sumReductionCudaDP.cu) | sum reduction example with CUDA Dynamic Parallelism | CUDA |
| [sumReductionCuda.cu](sumReductionCuda.cu) | Remove CUDA Dynamic Parallelism and instead launch reduction kernel recursively from host | CUDA
| [sumReductionSYCL.cpp](sumReductionSYCL.cpp) | Ported CUDA code without Dynamic Parallelism to SYCL | SYCL |

### Reference
- The sum reduction with CUDA Dynamic Parallelism logic is modification from this [NVidia forum post](https://forums.developer.nvidia.com/t/reduction-sum-using-dynamic-parallelism-working-upto-a-magical-number/79695)
- Read more about CUDA Dynamic Parallelism [Part 1](https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/), [Part 2](https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/), [mandelbrot example](https://github.com/canonizer/mandelbrot-dyn)

### Requirements

- NVidia GPU
- NVidia CUDA SDK
- Intel oneAPI Base Toolkit + oneAPI plugin for NVidia
