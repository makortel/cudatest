// -*- c++ -*-
// nvcc -o transfer transfer.cu

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <tuple>
#include <thread>

constexpr size_t MAXTHREADS = 8;
constexpr size_t TIMES = 2;
constexpr size_t MAX = 1<<27; // 128Melements = 512 MB
constexpr size_t MAXOPS = 1000000;

class Data;
void transfer(Data *data);

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
    cudaStreamSynchronize(stream);
    cudaFreeHost(a_h);
    cudaFree(a_d);
    cudaStreamDestroy(stream);
  }

  void transferAsync() {
    thread = std::thread{transfer, this};
  }

  double wait() {
    thread.join();
    return time;
  }

  std::thread thread;
  cudaStream_t stream;
  double time;
  float *a_d;
  float *a_h;
};

void transfer(Data *data) {
  auto start = std::chrono::high_resolution_clock::now();

  for(size_t i=0, j=0; i<MAXOPS; ++i) {
    cudaMemcpyAsync(data->a_d+j, data->a_h+j, sizeof(float), cudaMemcpyDefault, data->stream);
    j = (j+1)%MAX;
  }

  auto stop = std::chrono::high_resolution_clock::now();
  data->time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count())/1e6;
}

int main() {
  std::vector<Data> threads(MAXTHREADS);

  for(size_t nth=1; nth<=MAXTHREADS; ++nth) {
    std::cout << "Number of threads " << nth << std::endl;
    double total = 0;
    for(size_t i=0; i<TIMES; ++i) {
      std::cout << "Trial " << i << std::endl;
      for(size_t j=0; j<nth; ++j) {
        threads[j].transferAsync();
      }
      for(size_t j=0; j<nth; ++j) {
        total += threads[j].wait();
      }
    }
    total = total / TIMES;
    std::cout << "Ops " << (MAXOPS*nth) << " time/trial " << total << " ops/s " << (MAXOPS*nth/total) << " us/op " << (total/(MAXOPS*nth)*1e6) << std::endl;
  }

  return 0;
}
