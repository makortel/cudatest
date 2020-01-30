// -*- c++ -*-
// nvcc -std=c++14 -o callback2 callback2.cu

// as callback2 but use custom synchronization with the callback instead of cudaDeviceSynchronize

#include <iostream>
#include <cstdio>
#include <atomic>

__global__ void kernel()
{
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   int stride = gridDim.x * blockDim.x;
   volatile float x = 0;

   for (int j = idx; j < 1024*1024*16; j += stride) {
       #pragma unroll
       for (int i = 0; i < 128; ++i) {
           x += float(i)*float(i);
       }
   }
}

constexpr size_t NSTREAMS = 7;

std::atomic<bool> canContinue[NSTREAMS];

void CUDART_CB cb(cudaStream_t stream, cudaError_t status, void *data) {
  canContinue[reinterpret_cast<size_t>(data)].store(true);
}

int main(int argc, char* argv[])
{
   cudaStream_t streams[NSTREAMS];

   for (int i = 0; i < NSTREAMS; ++i) {
       cudaStreamCreate(&streams[i]);
       canContinue[i].store(false);
   }

   cudaDeviceSynchronize();

   int numThreads = 256;
   int numBlocks = 20;

   for (int j = 0; j < 4; ++j) {
       for (size_t i = 0; i < NSTREAMS; ++i) {
           kernel<<<numBlocks, numThreads, 0, streams[i]>>>();
           kernel<<<numBlocks, numThreads, 0, streams[i]>>>();
           if (j == 2)
               cudaStreamAddCallback(streams[i], cb, (void*) i, 0);
       }
       if(j== 2) {
         for(size_t i=0; i<NSTREAMS; ++i) {
           while(not canContinue[i].load()) {}
         }
       }
       else {
         //cudaDeviceSynchronize();
       }
   }
   cudaDeviceSynchronize();

   for (int i = 0; i < NSTREAMS; ++i) {
       cudaStreamDestroy(streams[i]);
   }

   return 0;
}
