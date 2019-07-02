// -*- c++ -*-
// nvcc -std=c++14 -o callback callback.cu

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <tuple>
#include <thread>
#include <mutex>
#include <cassert>
#include <atomic>

constexpr size_t MAXTHREADS = 8;
constexpr size_t MAXCHUNKS = 1000;
constexpr size_t TIMES = 5;

class Data;
void work(Data *data, size_t i);

std::atomic<int> countdownToStartTimer;
std::atomic<bool> startProcessing;
decltype(std::chrono::high_resolution_clock::now()) globalStart;

struct Data {
  Data() {
    cudaStreamCreate(&stream);

  }
  ~Data() {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }

  void workAsync(size_t index) {
    thread = std::thread{work, this, index};
  }

  void wait() {
    thread.join();
  }

  std::thread thread;
  cudaStream_t stream;
  std::atomic<bool> canContinue;
  size_t chunk_i = 0;
  char unused[32]; // to fill a full cacheline per element
};

void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void *data) {
  auto& data2 = *reinterpret_cast<Data *>(data);

  //std::cout << "callback " << data2.chunk_i << std::endl;
  data2.canContinue.store(true);
};

void work(Data *data, size_t index) {
  if(--countdownToStartTimer == 0) {
    globalStart = std::chrono::high_resolution_clock::now();
    startProcessing.store(true);
  }
  else {
    while(not startProcessing.load()) {}
  }

  //std::cout << "Thread " << index << " started processing" << std::endl;

  for(size_t i=0; i<MAXCHUNKS; ++i) {
    data->canContinue.store(false);
    data->chunk_i = i;
    //std::cout << "Thread " << index << " queue callback " << i << std::endl;
    cudaStreamAddCallback(data->stream, callback, data, 0);
    //std::cout << "Thread " << index << " waiting" << std::endl;
    while(not data->canContinue.load()) {}
    //std::cout << "Thread " << index << " continues" << std::endl;
  }
}

int main() {
  std::cout << "sizeof(Data) " << sizeof(Data) << std::endl;

  std::vector<Data> threads(MAXTHREADS);

  for(size_t nth=1; nth<=MAXTHREADS; ++nth) {
    std::cout << "Number of threads " << nth << std::endl;
    double total = 0;
    for(size_t i=0; i<TIMES; ++i) {
      std::cout << "Trial " << i << std::endl;
      countdownToStartTimer.store(nth);
      startProcessing.store(false);
      for(size_t j=0; j<nth; ++j) {
        threads[j].workAsync(j);
      }
      for(size_t j=0; j<nth; ++j) {
        threads[j].wait();
      }
      auto stop = std::chrono::high_resolution_clock::now();
      total += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(stop-globalStart).count())/1e6;
    }
    total = total / TIMES;
    std::cout << "Chunks " << (MAXCHUNKS*nth)
              << " time/trial " << total
              << " chunks/s " << (MAXCHUNKS*nth/total)
              << std::endl;
  }

  return 0;
}
