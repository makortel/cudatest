// -*- c++ -*-
// nvcc -o transfer transfer.cu

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <tuple>

constexpr int TIMES = 10;
constexpr size_t MAX = 1<<25; // 32Melements = 128 MB

std::tuple<double, double> transfer(cudaStream_t& stream, size_t num) {
  int *data_d, *data_h;
  cudaMalloc(&data_d, num*sizeof(int));
  cudaMallocHost(&data_h, num*sizeof(int));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(-250, 250);
  for(size_t i=0; i<num; ++i) {
    data_h[i] = dis(gen);
  }

  unsigned long total1 = 0;
  unsigned long total2 = 0;

  for(int i=0; i<TIMES; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(data_d, data_h, num*sizeof(int), cudaMemcpyDefault, stream);
    auto stop1 = std::chrono::high_resolution_clock::now();
    cudaStreamSynchronize(stream);
    auto stop2 = std::chrono::high_resolution_clock::now();

    total1 += std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start).count();
    total2 += std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start).count();
  }

  cudaFreeHost(data_h);
  cudaFree(data_d);

  return std::make_tuple(static_cast<double>(total1)/TIMES,
                         static_cast<double>(total2)/TIMES);
}


int main() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  for(size_t i=1; i<=MAX; i = i<<1) {
    auto times = transfer(stream, i);
    //std::cout << "Transferring " << i*sizeof(int) << " bytes, memcpy call took " << std::get<0>(times) << " us transfer took " << std::get<1>(times) << " us" << std::endl;
    std::cout << i*sizeof(int) << " " << std::get<0>(times) << " " << std::get<1>(times) << std::endl;
  }

  cudaStreamDestroy(stream);

  return 0;
}
