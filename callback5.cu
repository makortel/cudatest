// -*- c++ -*-
// nvcc -std=c++14 -o callback2 callback2.cu

// as callback4 but with multiple streams per thread

#include <iostream>
#include <cstdio>
#include <atomic>
#include <vector>
#include <thread>

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

constexpr size_t NSTREAMS = 32;
constexpr size_t NTHREADS = 1;
constexpr size_t STREAMS_PER_THREAD = NSTREAMS/NTHREADS;
constexpr int numThreads = 256;
constexpr int numBlocks = 20;

//#define CREATE_THREADS
//#define ADD_CALLBACK

std::atomic<bool> canStart;
std::atomic<bool> canContinue[NSTREAMS];
std::atomic<int> waiting;

void CUDART_CB cb(cudaStream_t stream, cudaError_t status, void *data) {
  canContinue[reinterpret_cast<size_t>(data)].store(true);
  --waiting;
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

   waiting.store(NSTREAMS);
#ifdef CREATE_THREADS
   std::vector<std::thread> threads(NTHREADS);
   for (size_t i = 0; i < NTHREADS; ++i) {
     threads[i] = std::thread{[ith=i, &streams]() {
         while(not canStart.load()) {}
#else
         const size_t ith = 0;
#endif // CREATE_THREADS

         for (int j = 0; j < 4; ++j) {
           for(size_t k = 0; k<STREAMS_PER_THREAD; ++k) {
             const size_t ist = ith*STREAMS_PER_THREAD+k;
             kernel<<<numBlocks, numThreads, 0, streams[ist]>>>();
             //kernel<<<numBlocks, numThreads, 0, streams[ist]>>>();
#ifdef ADD_CALLBACK
             if (j == 2) {
               cudaStreamAddCallback(streams[ist], cb, (void*) ist, 0);
             }
#endif // ADD_CALLBACK
           }
#ifdef ADD_CALLBACK
           if(j == 2) {
             for(size_t k = 0; k<STREAMS_PER_THREAD; ++k) {
               const size_t ist = ith*STREAMS_PER_THREAD+k;
               while(not canContinue[ist].load()) {}
               while(not waiting.load() == 0) {}
             }
           }
#endif // ADD_CALLBACK
         }

         for(size_t k = 0; k<STREAMS_PER_THREAD; ++k) {
           const size_t ist = ith*STREAMS_PER_THREAD+k;
           cudaStreamSynchronize(streams[ist]);
         }
#ifdef CREATE_THREADS
       }};
   }
   canStart.store(true);

   for(auto& th: threads) {
     th.join();
   }
#endif // CREATE_THREADS

   cudaDeviceSynchronize();

   for (int i = 0; i < NSTREAMS; ++i) {
       cudaStreamDestroy(streams[i]);
   }

   return 0;
}
