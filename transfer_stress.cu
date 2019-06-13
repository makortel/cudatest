// -*- c++ -*-
// nvcc -o transfer transfer.cu

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <tuple>
#include <thread>

constexpr size_t THREADS = 1;
constexpr size_t TIMES = 10;
constexpr size_t MAX = 1<<27; // 128Melements = 512 MB
constexpr size_t MAXOPS = 1000000;

struct Data {
  Data() {
    cudaStreamCreate(&stream);

    cudaMalloc(&a_d, MAX*sizeof(float));
    cudaMallocHost(&a_h, MAX*sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1e-5, 100.);
    for(size_t i=0; i<MAX; ++i) {
      a_h[i] = dis(gen);
    }
  }
  ~Data() {
    cudaFreeHost(a_h);
    cudaFree(a_d);
    cudaStreamDestroy(stream);
  }
  cudaStream_t stream;
  float *a_d;
  float *a_h;
};

double transfer(Data& data) {
  auto start = std::chrono::high_resolution_clock::now();

  for(size_t i=0, j=0; i<MAXOPS; ++i) {
    cudaMemcpyAsync(data.a_d+j, data.a_h+j, sizeof(float), cudaMemcpyDefault, data.stream);
    j = (j+1)%MAX;
  }

  auto stop = std::chrono::high_resolution_clock::now();
  return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count())/1e6;
}

int main() {
  Data data;

  double total = 0;
  for(size_t i=0; i<TIMES; ++i) {
    total += transfer(data);
  }
  total = total / TIMES;

  std::cout << "Ops " << MAXOPS << " time " << total << " ops/s " << (MAXOPS/total) << " us/op " << (total/MAXOPS*1e6) << std::endl;

  cudaStreamSynchronize(data.stream);

  return 0;
}
