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
#include <atomic>
#include <condition_variable>

constexpr size_t MAXTHREADS = 1;
constexpr size_t TIMES = 200;
//constexpr size_t LOOPS = 1;
constexpr size_t LOOPS = 1024000;
constexpr size_t CUDATHREADS = 32; // 1 warp
constexpr size_t SIZE = LOOPS*CUDATHREADS;
//constexpr size_t SIZE = 1<<27; // 128Melements = 512 MB
//constexpr size_t MAXCHUNKS = 200000;
constexpr size_t MAXCHUNKS = 1;


#define CUDA_STREAM_SYNCHRONIZE
//#define CUDA_EVENT_SYNCHRONIZE
//#define CUDA_EVENT_POLL
//#define CUDA_CALLBACK
//#define CUDA_HOST_FUNC

class Data;
void work(Data *data, size_t loops);

std::atomic<int> countdownToStartTimer;
std::atomic<bool> startProcessing;
decltype(std::chrono::high_resolution_clock::now()) globalStart;

struct Data {
  Data() {
    cudaStreamCreate(&stream);
    cudaEventCreate(&event);

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
    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
  }

  void workAsync(size_t loops) {
    thread = std::thread{work, this, loops};
  }

  void wait() {
    thread.join();
  }

  std::thread thread;
  cudaStream_t stream;
  cudaEvent_t event;
  std::mutex mut;
  std::condition_variable cv;
  bool canContinue;
  float *a_d;
  float *a_h;
};

#ifdef CUDA_CALLBACK
void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void *data) {
#else
void CUDART_CB callback(void *data) {
#endif
  auto& data2 = *reinterpret_cast<Data *>(data);

  std::unique_lock<std::mutex> lk(data2.mut);
  data2.canContinue = true;
  lk.unlock();
  data2.cv.notify_one();
};

/*
struct DataCPU {
  void workAsync() {
    threads = std::thread{};
  }

  void wait() {
    thread.join();
  }

  std::thread thread;
};
*/

__global__ void kernel_looping(float *a, unsigned int size, size_t loops) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  for(int i=0; i<TIMES; ++i) {
    for(size_t iloop=0; iloop<loops; ++iloop) {
      size_t ind = iloop*gridDim.x+idx;
      if(ind < size) {
        a[ind] = a[ind] + 4.0f;
      }
    }
  }
}

void work(Data *data, size_t loops) {
  if(--countdownToStartTimer == 0) {
    globalStart = std::chrono::high_resolution_clock::now();
    startProcessing.store(true);
  }
  else {
    while(not startProcessing.load()) {}
  }

  for(size_t i=0; i<MAXCHUNKS; ++i) {
    {
      //std::lock_guard<std::mutex> lock{cudaMutex};
      cudaMemcpyAsync(data->a_d, data->a_h, SIZE*sizeof(float), cudaMemcpyDefault, data->stream);
      kernel_looping<<<1, CUDATHREADS, 0, data->stream>>>(data->a_d, SIZE, loops);
      cudaMemcpyAsync(data->a_h, data->a_d, SIZE*sizeof(float), cudaMemcpyDefault, data->stream);
    }
#ifdef CUDA_STREAM_SYNCHRONIZE
    cudaStreamSynchronize(data->stream);
#elif defined CUDA_EVENT_SYNCHRONIZE
    cudaEventRecord(data->event, data->stream);
    cudaEventSynchronize(data->event);
#elif defined CUDA_EVENT_POLL
    cudaEventRecord(data->event, data->stream);
    while(cudaEventQuery(data->event) == cudaErrorNotReady) {
      //std::this_thread::yield();
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(10ms);
    }
#elif defined CUDA_CALLBACK 
    data->canContinue = false;
    std::unique_lock<std::mutex> lk(data->mut);
    cudaStreamAddCallback(data->stream, callback, data, 0);
    data->cv.wait(lk, [data](){return data->canContinue;});
#elif defined CUDA_HOST_FUNC 
    data->canContinue = false;
    std::unique_lock<std::mutex> lk(data->mut);
    cudaLaunchHostFunc(data->stream, callback, data);
    data->cv.wait(lk, [data](){return data->canContinue;});
#endif
  }
}


int main() {
  std::cout << "sizeof(Data) " << sizeof(Data) << std::endl;

  //cudaSetDeviceFlags(cudaDeviceScheduleSpin);
  //cudaSetDeviceFlags(cudaDeviceScheduleYield);
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

  std::vector<Data> threads(MAXTHREADS);

  std::cout << "Number of threads " << MAXTHREADS << std::endl;
  std::cout << "Trial " << 0 << std::endl;
  countdownToStartTimer.store(MAXTHREADS);
  startProcessing.store(false);
  for(auto& th: threads) {
    th.workAsync(LOOPS);
  }
  for(auto& th: threads) {
    th.wait();
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto total= static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(stop-globalStart).count())/1e6;
  std::cout << "Chunks " << (MAXCHUNKS*MAXTHREADS)
            << " time/trial " << total
            << " chunks/s " << (MAXCHUNKS*MAXTHREADS/total)
            << " us/chunk " << (total/(MAXCHUNKS*MAXTHREADS)*1e6)
            << std::endl;

  return 0;
}
