// -*- c++ -*-
// nvcc -std=c++14 -o callback2 callback2.cu

#include <iostream>
#include <cstdio>

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

void CUDART_CB cb(cudaStream_t stream, cudaError_t status, void *data) {}

int main(int argc, char* argv[])
{
   cudaStream_t streams[8];

   for (int i = 0; i < 8; ++i) {
       cudaStreamCreate(&streams[i]);
   }

   cudaDeviceSynchronize();

   int numThreads = 256;
   int numBlocks = 20;

   for (int j = 0; j < 4; ++j) {
       for (size_t i = 0; i < 8; ++i) {
           kernel<<<numBlocks, numThreads, 0, streams[i]>>>();
           kernel<<<numBlocks, numThreads, 0, streams[i]>>>();
           if (j == 2)
               cudaStreamAddCallback(streams[i], cb, (void*) i, 0);
       }
       cudaDeviceSynchronize();
   }

   for (int i = 0; i < 8; ++i) {
       cudaStreamDestroy(streams[i]);
   }

   return 0;
}
