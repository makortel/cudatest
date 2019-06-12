// -*- c++ -*-
// nvcc -o transfer transfer.cu

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <tuple>

constexpr int TIMES = 10;
//constexpr size_t MAX = 1<<10; // 32Melements = 128 MB
constexpr size_t MAX = 1<<27; // 128Melements = 512 MB

// does N(loops)*N(size)/N(gridDim) multiplications
__global__ void kernel_looping(float *a, float *b, float *c, unsigned int size, size_t loops) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  for(size_t iloop=0; iloop<loops; ++iloop) {
    size_t ind = iloop*gridDim.x+idx;
    if(ind < size) {
      a[ind] = b[ind]*c[ind];
    }
  }
}


std::tuple<double, double> kernel(float *a_d, float *b_d, float *c_d, cudaStream_t& stream, size_t loops) {
  unsigned long total1 = 0;
  unsigned long total2 = 0;

  for(int i=0; i<TIMES; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    kernel_looping<<<1, 32, 0, stream>>>(a_d, b_d, c_d, MAX*sizeof(float), loops);
    auto stop1 = std::chrono::high_resolution_clock::now();
    cudaStreamSynchronize(stream);
    auto stop2 = std::chrono::high_resolution_clock::now();

    total1 += std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start).count();
    total2 += std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start).count();
  }

  return std::make_tuple(static_cast<double>(total1)/TIMES,
                         static_cast<double>(total2)/TIMES);
}


int main() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float *a_d, *b_d, *c_d;
  float *b_h, *c_h;
  cudaMalloc(&a_d, MAX*sizeof(float));
  cudaMalloc(&b_d, MAX*sizeof(float));
  cudaMalloc(&c_d, MAX*sizeof(float));
  cudaMallocHost(&b_h, MAX*sizeof(float));
  cudaMallocHost(&c_h, MAX*sizeof(float));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(1e-5, 100.);
  for(size_t i=0; i<MAX; ++i) {
    b_h[i] = dis(gen);
    c_h[i] = dis(gen);
  }

  cudaMemcpyAsync(b_d, b_h, MAX*sizeof(float), cudaMemcpyDefault, stream);
  cudaMemcpyAsync(c_d, c_h, MAX*sizeof(float), cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);

  for(size_t i=1; i<=(MAX/32); i = i<<1) {
    auto times = kernel(a_d, b_d, c_d, stream, i);
    //std::cout << "Computing " << i << " loops, kernel launch took " << std::get<0>(times) << " us total took " << std::get<1>(times) << " us" << std::endl;
    std::cout << i << " " << std::get<0>(times) << " " << std::get<1>(times) << std::endl;
  }
  cudaFreeHost(c_h);
  cudaFreeHost(b_h);
  cudaFree(c_d);
  cudaFree(b_d);
  cudaFree(a_d);

  cudaStreamDestroy(stream);

  return 0;
}
