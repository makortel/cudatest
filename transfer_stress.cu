// -*- c++ -*-
// nvcc -o transfer transfer.cu

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <tuple>

constexpr size_t TIMES = 1;
constexpr size_t MAX = 1<<27; // 128Melements = 512 MB
constexpr size_t MAXOPS = 1000000;

double transfer(float *a_d, float *a_h, cudaStream_t& stream) {
  auto start = std::chrono::high_resolution_clock::now();

  for(size_t i=0, j=0; i<MAXOPS; ++i) {
    cudaMemcpyAsync(a_d+j, a_h+j, sizeof(float), cudaMemcpyDefault, stream);
    j = (j+1)%MAX;
  }

  auto stop = std::chrono::high_resolution_clock::now();
  return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count())/1e6;
}


int main() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float *a_d;
  float *a_h;
  cudaMalloc(&a_d, MAX*sizeof(float));
  cudaMallocHost(&a_h, MAX*sizeof(float));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(1e-5, 100.);
  for(size_t i=0; i<MAX; ++i) {
    a_h[i] = dis(gen);
  }

  double total = 0;
  for(size_t i=0; i<TIMES; ++i) {
    total += transfer(a_d, a_h, stream);
  }
  total = total / TIMES;

  std::cout << "Ops " << MAXOPS << " time " << total << " ops/s " << (MAXOPS/total) << " us/op " << (total/MAXOPS*1e6) << std::endl;

  cudaStreamSynchronize(stream);

  cudaFreeHost(a_h);
  cudaFree(a_d);

  cudaStreamDestroy(stream);

  return 0;
}
