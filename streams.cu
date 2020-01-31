// -*- c++ -*-
// nvcc -o transfer transfer.cu

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <tuple>
#include <thread>
#include <mutex>
#include <cassert>

constexpr size_t STREAMS = 64;
//constexpr size_t STREAMS = 256;
//constexpr size_t STREAMS = 50000;
constexpr size_t CUDATHREADS = 32; // 1 warp
constexpr size_t ELEMENTS = STREAMS*CUDATHREADS;
constexpr size_t LOOPS = 1000000;

__global__ void kernel_looping(float *a) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  for(size_t iloop=0; iloop<LOOPS; ++iloop) {
    a[idx] = a[idx] + 1.0f;
  }
}

int main() {
  std::vector<cudaStream_t> streams(STREAMS);
  for(auto& s: streams) {
    cudaStreamCreate(&s);
  }

  float *data_d;
  float *data_h;
  cudaMalloc(&data_d, ELEMENTS*sizeof(float));
  cudaMallocHost(&data_h, ELEMENTS*sizeof(float));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(1e-5, 100.);
  for(size_t i=0; i<ELEMENTS; ++i) {
    data_h[i] = dis(gen);
  }
  cudaMemcpyAsync(data_d, data_h, ELEMENTS*sizeof(float), cudaMemcpyDefault, streams[0]);
  cudaStreamSynchronize(streams[0]);

  for(size_t i=0; i<STREAMS; ++i) {
    kernel_looping<<<1, CUDATHREADS, 0, streams[i]>>>(data_d+i*CUDATHREADS);
  }

  for(auto& s: streams) {
    cudaStreamSynchronize(s);
  }

  for(auto& s: streams) {
    cudaStreamDestroy(s);
  }

  cudaFreeHost(data_h);
  cudaFree(data_d);

  return 0;
}
