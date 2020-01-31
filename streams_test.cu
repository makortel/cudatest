// -*- c++ -*-
// nvcc -o transfer transfer.cu

#include <cuda_runtime.h>
#include <iostream>
#include <tuple>
#include <thread>
#include <mutex>
#include <cassert>
#include <vector>

constexpr size_t STREAMS = 64;
//constexpr size_t STREAMS = 256;
//constexpr size_t STREAMS = 50000;
constexpr size_t CUDATHREADS = 32; // 1 warp
constexpr size_t EVENTS = 1;
constexpr size_t KERNELS = 2;

__global__ void kernel()
{
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   int stride = gridDim.x * blockDim.x;
   volatile float x = 0;

   for (int j = idx; j < 1024*1024*64; j += stride) {
       #pragma unroll
       for (int i = 0; i < 512; ++i) {
           x += float(i)*float(i);
       }
   }
}

int main() {
  std::vector<cudaStream_t> streams(STREAMS);
  for(auto& s: streams) {
    cudaStreamCreate(&s);
  }

  for(size_t iev=0; iev<EVENTS; ++iev) {
    for(size_t ist=0; ist<STREAMS; ++ist) {
      for(size_t ik=0; ik<KERNELS; ++ ik) {
        kernel<<<1, CUDATHREADS, 0, streams[ist]>>>();
      }
    }
  }

  for(auto& s: streams) {
    cudaStreamSynchronize(s);
  }

  for(auto& s: streams) {
    cudaStreamDestroy(s);
  }

  return 0;
}
