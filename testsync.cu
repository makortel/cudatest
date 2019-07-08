// -*- c++ -*-
// nvcc -o testsync testsync.cu -lnvToolsExt
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

#include <atomic>
#include <iostream>
#include <cassert>
#include <chrono>
#include <thread>
#include <vector>
#define NUM_STREAM  16
#define ARRAY_SIZE  2000000
#define NLOOPS 1

nvtxRangeId_t nvtxDomainRangeStart(nvtxDomainHandle_t domain, const char* message) {
  nvtxEventAttributes_t eventAttrib = { 0 };
  eventAttrib.version         = NVTX_VERSION;
  eventAttrib.size            = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType     = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii   = message;
  return nvtxDomainRangeStartEx(domain, &eventAttrib);
}


__global__ void kernel_looping(float* point, unsigned int num) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  for(int iloop=0; iloop<NLOOPS; ++iloop) {
    for (size_t offset = idx; offset < num; offset += gridDim.x * blockDim.x) {
      point[offset] += 1;
    }
  }
}

__global__ void kernel_assert(float* point, unsigned int num) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx >= num)
    return;

  point[idx] += 1;
  assert(point[idx] > 0);
}

std::atomic<int> f{0};
void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void *arg) {
  f++;
  //std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void testNormal(float *dev_points[NUM_STREAM], float *host_points[NUM_STREAM], cudaStream_t streams[NUM_STREAM]) {
  for( int j = 0; j < 4; ++j) {
    for (size_t i = 0; i < NUM_STREAM; ++i) {
      cudaMemcpyAsync(dev_points[i], host_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyHostToDevice, streams[i]);
      kernel_looping<<<1, 16, 0, streams[i]>>>(dev_points[i], ARRAY_SIZE);
      cudaMemcpyAsync(host_points[i], dev_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyDeviceToHost, streams[i]);
    }
  }
  cudaDeviceSynchronize();
}

void testCallback(float *dev_points[NUM_STREAM], float *host_points[NUM_STREAM], cudaStream_t streams[NUM_STREAM]) {
  for( int j = 0; j < 4; ++j) {
    for (size_t i = 0; i < NUM_STREAM; ++i) {
      cudaMemcpyAsync(dev_points[i], host_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyHostToDevice, streams[i]);
      kernel_looping<<<1, 16, 0, streams[i]>>>(dev_points[i], ARRAY_SIZE);
      cudaMemcpyAsync(host_points[i], dev_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyDeviceToHost, streams[i]);

      cudaStreamAddCallback(streams[i], callback, NULL, 0);
    }
  }
  cudaDeviceSynchronize();
}

void testAssert(float *dev_points[NUM_STREAM], float *host_points[NUM_STREAM], cudaStream_t streams[NUM_STREAM]) {
  for( int j = 0; j < 4; ++j) {
    for (size_t i = 0; i < NUM_STREAM; ++i) {
      cudaMemcpyAsync(dev_points[i], host_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyHostToDevice, streams[i]);
      kernel_looping<<<1, 16, 0, streams[i]>>>(dev_points[i], ARRAY_SIZE);
      kernel_assert<<<200, 512, 0, streams[i]>>>(dev_points[i], ARRAY_SIZE);
      cudaMemcpyAsync(host_points[i], dev_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyDeviceToHost, streams[i]);
    }
  }
  cudaDeviceSynchronize();
}

void testCudaMalloc(float *dev_points[NUM_STREAM], float *host_points[NUM_STREAM], cudaStream_t streams[NUM_STREAM]) {
  std::vector<float *> tmp;
  tmp.reserve(4*NUM_STREAM);

  for( int j = 0; j < 4; ++j) {
    for (size_t i = 0; i < NUM_STREAM; ++i) {
      cudaMemcpyAsync(dev_points[i], host_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyHostToDevice, streams[i]);
      kernel_looping<<<1, 16, 0, streams[i]>>>(dev_points[i], ARRAY_SIZE);
      cudaMemcpyAsync(host_points[i], dev_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyDeviceToHost, streams[i]);
      float *ptr;
      cudaMalloc(&ptr, 1024*sizeof(float));
      tmp.push_back(ptr);
    }
  }

  cudaDeviceSynchronize();
  for(auto ptr: tmp) {
    cudaFree(ptr);
  }
}

void testCudaMallocMemcpy(float *dev_points[NUM_STREAM], float *host_points[NUM_STREAM], cudaStream_t streams[NUM_STREAM]) {
  std::vector<float *> tmp;
  tmp.reserve(4*NUM_STREAM);

  for( int j = 0; j < 4; ++j) {
    for (size_t i = 0; i < NUM_STREAM; ++i) {
      cudaMemcpyAsync(dev_points[i], host_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyHostToDevice, streams[i]);
      kernel_looping<<<1, 16, 0, streams[i]>>>(dev_points[i], ARRAY_SIZE);
      cudaMemcpyAsync(host_points[i], dev_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyDeviceToHost, streams[i]);
      float *ptr;
      cudaMalloc(&ptr, 1024*sizeof(float));
      cudaMemcpyAsync(ptr, host_points[i], 
                      sizeof(float), 
                      cudaMemcpyHostToDevice, streams[i]);
      cudaMemcpyAsync(host_points[i], ptr, 
                      sizeof(float), 
                      cudaMemcpyDeviceToHost, streams[i]);
      tmp.push_back(ptr);
    }
  }

  cudaDeviceSynchronize();
  for(auto ptr: tmp) {
    cudaFree(ptr);
  }
}


void testCudaMallocFree(float *dev_points[NUM_STREAM], float *host_points[NUM_STREAM], cudaStream_t streams[NUM_STREAM]) {
  for( int j = 0; j < 4; ++j) {
    for (size_t i = 0; i < NUM_STREAM; ++i) {
      cudaMemcpyAsync(dev_points[i], host_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyHostToDevice, streams[i]);
      kernel_looping<<<1, 16, 0, streams[i]>>>(dev_points[i], ARRAY_SIZE);
      cudaMemcpyAsync(host_points[i], dev_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyDeviceToHost, streams[i]);
      float *ptr;
      cudaMalloc(&ptr, 1024*sizeof(float));
      cudaFree(ptr);
    }
  }

  cudaDeviceSynchronize();
}


void testCudaMallocManaged(float *dev_points[NUM_STREAM], float *host_points[NUM_STREAM], cudaStream_t streams[NUM_STREAM]) {
  std::vector<float *> tmp;
  tmp.reserve(4*NUM_STREAM);

  for( int j = 0; j < 4; ++j) {
    for (size_t i = 0; i < NUM_STREAM; ++i) {
      cudaMemcpyAsync(dev_points[i], host_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyHostToDevice, streams[i]);
      kernel_looping<<<1, 16, 0, streams[i]>>>(dev_points[i], ARRAY_SIZE);
      cudaMemcpyAsync(host_points[i], dev_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyDeviceToHost, streams[i]);
      float *ptr;
      cudaMallocManaged(&ptr, 1024*sizeof(float));
      tmp.push_back(ptr);
    }
  }

  cudaDeviceSynchronize();
  for(auto ptr: tmp) {
    cudaFree(ptr);
  }
}

void testCudaMallocManagedFree(float *dev_points[NUM_STREAM], float *host_points[NUM_STREAM], cudaStream_t streams[NUM_STREAM]) {
  for( int j = 0; j < 4; ++j) {
    for (size_t i = 0; i < NUM_STREAM; ++i) {
      cudaMemcpyAsync(dev_points[i], host_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyHostToDevice, streams[i]);
      kernel_looping<<<1, 16, 0, streams[i]>>>(dev_points[i], ARRAY_SIZE);
      cudaMemcpyAsync(host_points[i], dev_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyDeviceToHost, streams[i]);
      float *ptr;
      cudaMallocManaged(&ptr, 1024*sizeof(float));
      cudaFree(ptr);
    }
  }

  cudaDeviceSynchronize();
}

void testCudaHostRegister(float *dev_points[NUM_STREAM], float *host_points[NUM_STREAM], cudaStream_t streams[NUM_STREAM]) {
  std::vector<float *> tmp;
  tmp.reserve(4*NUM_STREAM);
  for(int i=0; i<4*NUM_STREAM; ++i) {
    tmp.push_back(reinterpret_cast<float *>(malloc(1024*sizeof(float))));
  }


  for( int j = 0; j < 4; ++j) {
    for (size_t i = 0; i < NUM_STREAM; ++i) {
      cudaMemcpyAsync(dev_points[i], host_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyHostToDevice, streams[i]);
      kernel_looping<<<1, 16, 0, streams[i]>>>(dev_points[i], ARRAY_SIZE);
      cudaMemcpyAsync(host_points[i], dev_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyDeviceToHost, streams[i]);
      float *ptr = tmp[j*NUM_STREAM + i];
      cudaHostRegister(ptr, 1024*sizeof(float), cudaHostRegisterDefault);
    }
  }

  cudaDeviceSynchronize();
  for(auto ptr: tmp) {
    cudaHostUnregister(ptr);
  }

  for(auto ptr: tmp) {
    free(ptr);
  }
}

void testCudaHostRegisterUnregister(float *dev_points[NUM_STREAM], float *host_points[NUM_STREAM], cudaStream_t streams[NUM_STREAM]) {
  std::vector<float *> tmp;
  tmp.reserve(4*NUM_STREAM);
  for(int i=0; i<4*NUM_STREAM; ++i) {
    tmp.push_back(reinterpret_cast<float *>(malloc(1024*sizeof(float))));
  }


  for( int j = 0; j < 4; ++j) {
    for (size_t i = 0; i < NUM_STREAM; ++i) {
      cudaMemcpyAsync(dev_points[i], host_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyHostToDevice, streams[i]);
      kernel_looping<<<1, 16, 0, streams[i]>>>(dev_points[i], ARRAY_SIZE);
      cudaMemcpyAsync(host_points[i], dev_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyDeviceToHost, streams[i]);
      float *ptr = tmp[j*NUM_STREAM + i];
      cudaHostRegister(ptr, 1024*sizeof(float), cudaHostRegisterDefault);
      cudaHostUnregister(ptr);
    }
  }

  cudaDeviceSynchronize();
  for(auto ptr: tmp) {
    free(ptr);
  }
}

int main() {
  float *dev_points[NUM_STREAM];
  float *host_points[NUM_STREAM];
  cudaStream_t streams[NUM_STREAM];

  for (size_t i = 0; i < NUM_STREAM; ++i) {
    cudaMalloc(dev_points + i, ARRAY_SIZE * sizeof(float));
    cudaMallocHost(host_points + i, ARRAY_SIZE * sizeof(float));
    cudaStreamCreateWithFlags(streams + i, cudaStreamNonBlocking);
    for (size_t j = 0; j < ARRAY_SIZE; ++j) {
      host_points[i][j] = static_cast<float>(i + j);
    }
  }

  /*
  for( int j = 0; j < 4; ++j) {
    for (size_t i = 0; i < NUM_STREAM; ++i) {
      cudaMemcpyAsync(dev_points[i], host_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyHostToDevice, streams[i]);
      kernel_looping<<<1, 16, 0, streams[i]>>>(dev_points[i], ARRAY_SIZE);
      //kernel_assert<<<200, 512, 0, streams[i]>>>(dev_points[i], ARRAY_SIZE);
      //cudaMalloc(test_ptrs + i, sizeof(float)*1024*1024);
      //cudaMallocHost(test_ptrs + i, sizeof(float));
      cudaMallocManaged(test_ptrs + i, sizeof(float));
      cudaMemcpyAsync(host_points[i], dev_points[i], 
                      ARRAY_SIZE * sizeof(float), 
                      cudaMemcpyDeviceToHost, streams[i]);

      cudaFree(test_ptrs[i]);
      //cudaFreeHost(test_ptrs[i]);

      //cudaStreamAddCallback(streams[i], callback, NULL, 0);
      std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
  }
  */

  nvtxDomainHandle_t handle = nvtxDomainCreate("Test case");
  auto id = nvtxDomainRangeStart(handle, "Normal");
  testNormal(dev_points, host_points, streams);
  nvtxDomainRangeEnd(handle, id);

  id = nvtxDomainRangeStart(handle, "Callback");
  testCallback(dev_points, host_points, streams);
  nvtxDomainRangeEnd(handle, id);

  id = nvtxDomainRangeStart(handle, "Assert");
  testAssert(dev_points, host_points, streams);
  nvtxDomainRangeEnd(handle, id);

  id = nvtxDomainRangeStart(handle, "cudaMalloc");
  testCudaMalloc(dev_points, host_points, streams);
  nvtxDomainRangeEnd(handle, id);

  id = nvtxDomainRangeStart(handle, "cudaMallocMemcpy");
  testCudaMallocMemcpy(dev_points, host_points, streams);
  nvtxDomainRangeEnd(handle, id);

  id = nvtxDomainRangeStart(handle, "cudaMallocFree");
  testCudaMallocFree(dev_points, host_points, streams);
  nvtxDomainRangeEnd(handle, id);


  id = nvtxDomainRangeStart(handle, "cudaMallocManaged");
  testCudaMallocManaged(dev_points, host_points, streams);
  nvtxDomainRangeEnd(handle, id);

  id = nvtxDomainRangeStart(handle, "cudaMallocManagedFree");
  testCudaMallocManagedFree(dev_points, host_points, streams);
  nvtxDomainRangeEnd(handle, id);


  id = nvtxDomainRangeStart(handle, "cudaHostRegister");
  testCudaHostRegister(dev_points, host_points, streams);
  nvtxDomainRangeEnd(handle, id);

  id = nvtxDomainRangeStart(handle, "cudaHostRegisterUnregister");
  testCudaHostRegisterUnregister(dev_points, host_points, streams);
  nvtxDomainRangeEnd(handle, id);


  nvtxDomainDestroy(handle);
  std::cout << f << std::endl;
  for (size_t i = 0; i < NUM_STREAM; ++i) {
    cudaFree(dev_points[i]);
    cudaFreeHost(host_points[i]);
    cudaStreamDestroy(streams[i]);
  }

  return 0;
}
