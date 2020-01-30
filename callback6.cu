// -*- c++ -*-
// nvcc -std=c++14 -o callback2 callback2.cu

// as callback4 but with longer running kernel

#include <iostream>
#include <cstdio>
#include <atomic>
#include <vector>
#include <thread>

//#define SYNCHRONIZE
#define CALLBACK
#define ULTRASHORT
//#define SHORT

__global__ void kernel()
{
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   int stride = gridDim.x * blockDim.x;
   volatile float x = 0;

#ifdef ULTRASHORT
   for (int j = idx; j < 1024*64; j += stride) {
#elif defined SHORT
   for (int j = idx; j < 1024*1024*16; j += stride) {
#else
   for (int j = idx; j < 1024*1024*256; j += stride) {
#endif
       #pragma unroll
       for (int i = 0; i < 128; ++i) {
           x += float(i)*float(i);
       }
   }
}

constexpr size_t NSTREAMS = 12;
constexpr int numThreads = 32;
constexpr int numBlocks = 4;

std::atomic<bool> canStart;
std::atomic<bool> canContinue[NSTREAMS];

void CUDART_CB cb(cudaStream_t stream, cudaError_t status, void *data) {
  canContinue[reinterpret_cast<size_t>(data)].store(true);
}

int main(int argc, char* argv[])
{
   cudaStream_t streams[NSTREAMS];

   canStart = false;
   for (int i = 0; i < NSTREAMS; ++i) {
       cudaStreamCreate(&streams[i]);
       canContinue[i].store(false);
   }

   cudaDeviceSynchronize();

   std::vector<std::thread> threads(NSTREAMS);
   for (size_t i = 0; i < NSTREAMS; ++i) {
     threads[i] = std::thread{[index=i, &streams]() {
         while(not canStart.load()) {}

         for (int j = 0; j < 4; ++j) {
           kernel<<<numBlocks, numThreads, 0, streams[index]>>>();
           kernel<<<numBlocks, numThreads, 0, streams[index]>>>();
           if(j==2) {
#ifdef SYNCRONIZE
             cudaStreamSynchronize(streams[index]);
#elif defined CALLBACK
             cudaStreamAddCallback(streams[index], cb, (void*) index, 0);
             while(not canContinue[index].load()) {}
#endif
           }
         }

         cudaStreamSynchronize(streams[index]);
       }
     };
   }
   canStart.store(true);

   for(auto& th: threads) {
     th.join();
   }

   cudaDeviceSynchronize();

   for (int i = 0; i < NSTREAMS; ++i) {
       cudaStreamDestroy(streams[i]);
   }

   return 0;
}
