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

constexpr size_t MAXTHREADS = 8;
constexpr size_t TIMES = 10;
constexpr size_t LOOPS = 1024;
constexpr size_t CUDATHREADS = 32; // 1 warp
constexpr size_t SIZE = LOOPS*CUDATHREADS;
//constexpr size_t SIZE = 1<<27; // 128Melements = 512 MB
constexpr size_t MAXCHUNKS = 25000;

class Data;
void work(Data *data, size_t loops);

std::mutex cudaMutex;

struct Data {
  Data() {
    cudaStreamCreate(&stream);

    cudaMalloc(&a_d, SIZE*sizeof(float));
    cudaMallocHost(&a_h, SIZE*sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1e-5, 100.);
    for(size_t i=0; i<SIZE; ++i) {
      a_h[i] = dis(gen);
    }
  }
  ~Data() {
    cudaStreamSynchronize(stream);
    cudaFreeHost(a_h);
    cudaFree(a_d);
    cudaStreamDestroy(stream);
  }

  void workAsync(size_t loops) {
    thread = std::thread{work, this, loops};
  }

  std::tuple<double, double> wait() {
    thread.join();
    return std::make_tuple(time, launch_time);
  }

  std::thread thread;
  cudaStream_t stream;
  double time;
  double launch_time;
  float *a_d;
  float *a_h;
};

__global__ void kernel_looping(float *a, unsigned int size, size_t loops) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  for(size_t iloop=0; iloop<loops; ++iloop) {
    size_t ind = iloop*gridDim.x+idx;
    if(ind < size) {
      a[ind] = a[ind] + 4.0f;
    }
  }
}

void work(Data *data, size_t loops) {
  double launch_time = 0;

  auto start = std::chrono::high_resolution_clock::now();

  for(size_t i=0; i<MAXCHUNKS; ++i) {
    auto starti = std::chrono::high_resolution_clock::now();
    //{
      std::lock_guard<std::mutex> lock{cudaMutex};
      cudaMemcpyAsync(data->a_d, data->a_h, SIZE*sizeof(float), cudaMemcpyDefault, data->stream);
      kernel_looping<<<1, CUDATHREADS, 0, data->stream>>>(data->a_d, SIZE, loops);
      cudaMemcpyAsync(data->a_h, data->a_d, SIZE*sizeof(float), cudaMemcpyDefault, data->stream);
      //}
    auto stopi = std::chrono::high_resolution_clock::now();
    launch_time += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(stopi-starti).count())/1e6;
    cudaStreamSynchronize(data->stream);
  }

  auto stop = std::chrono::high_resolution_clock::now();
  data->time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count())/1e6;
  data->launch_time = launch_time;
}

int main() {
  std::vector<Data> threads(MAXTHREADS);

  for(size_t nth=1; nth<=MAXTHREADS; ++nth) {
    std::cout << "Number of threads " << nth << std::endl;
    double total = 0;
    double total_launch = 0;
    for(size_t i=0; i<TIMES; ++i) {
      std::cout << "Trial " << i << std::endl;
      for(size_t j=0; j<nth; ++j) {
        threads[j].workAsync(LOOPS);
      }
      for(size_t j=0; j<nth; ++j) {
        auto times = threads[j].wait();
        total += std::get<0>(times);
        total_launch += std::get<1>(times);
      }
    }
    total = total / TIMES;
    total_launch = total_launch / TIMES;
    std::cout << "Chunks " << (MAXCHUNKS*nth)
              << " time/trial " << total
              << " chunks/s " << (MAXCHUNKS*nth/total)
              << " us/chunk " << (total/(MAXCHUNKS*nth)*1e6)
              << " launch us/chunk " << (total_launch/(MAXCHUNKS*nth)*1e6)
              << std::endl;
  }

  return 0;
}
