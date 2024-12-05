//==============================================================
// SYCL Sum Tree Reduction using recursive kernel launches
//
// Compile on NVidia with oneAPI compiler + oneAPI plugin for NVidia
// icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda sumReductionSYCL.cpp
// =============================================================

#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>

const int N = 1048576*256+37;
const int BLOCK = 1024;

// tree reduction kernel
void treeReduceKernel(int *sdata, sycl::nd_item<1> &item, int *d_in, int n) {
    int tID = item.get_local_id(0);
    int ID = item.get_global_id(0);

    if (item.get_group_range(0) == 1){  // when kernel has only 1 block use classic parallel reduction
        sdata[tID] = d_in[tID];
        if ((tID + BLOCK) < n) sdata[tID] += d_in[tID + BLOCK];
        for (int i = BLOCK/2; i > 0; i/=2){
            sycl::group_barrier(item.get_group());
            if (tID < i) sdata[tID] += sdata[tID + i];
        }
        if (tID == 0) d_in[0] = sdata[0];
    } else { // tree reduction
        int offset = n/2 + (n&1);
        if ((ID + offset) < n) d_in[ID] += d_in[ID + offset];
    }
}

int main(){
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    int *data, *h_data = (int *)malloc(N*sizeof(int));
    data = sycl::malloc_device<int>(N, q);
    for (int i = 0; i < N; i++) h_data[i] = 1;
    q.memcpy(data, h_data, N*sizeof(int)).wait();

    auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Perform tree reduction recursively
    int n = N;
    int nHalf = n/2 + (n&1);
    int numBlocks = (nHalf + BLOCK - 1) / BLOCK;
    int numKernelLaunch = 0;
    while (numBlocks >= 1) {
        numKernelLaunch++;
        q.submit([&](sycl::handler &h){
            sycl::local_accessor<int, 1> sdata(sycl::range<1>(BLOCK), h);
            h.parallel_for(sycl::nd_range<1>{numBlocks * BLOCK, BLOCK}, [=](sycl::nd_item<1> item){
                treeReduceKernel(sdata.get_multi_ptr<sycl::access::decorated::no>().get(), item, data, n);
            });
        }).wait();
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
    q.memcpy(h_data, data, sizeof(int)).wait();    
    (h_data[0] == N) ? std::cout << "PASS: " : std::cout << "FAIL: ";
    std::cout << "Sum = "<< h_data[0] << ", N = " << N << "\n";
    sycl::free(data, q);
    free(h_data);
    return 0;
}


