// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "p2p_allreduce.hpp"
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd)                                                          \
  do {                                                                          \
    hipError_t error = (cmd);                                                   \
    if (error != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error: '%s'(%d) at %s:%d\n",                        \
              hipGetErrorString(error), error, __FILE__, __LINE__);             \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

using namespace p2p_allreduce;

void TestAllReduceSum(int rank, int worldSize, size_t count) {
  std::cout << "Rank " << rank << ": Testing AllReduce SUM with " << count << " elements" << std::endl;
  
  // Allocate and initialize data on GPU
  float* d_sendbuf = nullptr;
  float* d_recvbuf = nullptr;
  HIP_CHECK(hipMalloc(&d_sendbuf, count * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_recvbuf, count * sizeof(float)));
  
  // Initialize sendbuf with rank value
  std::vector<float> h_sendbuf(count, static_cast<float>(rank + 1));
  HIP_CHECK(hipMemcpy(d_sendbuf, h_sendbuf.data(), count * sizeof(float), hipMemcpyHostToDevice));
  
  // Initialize bootstrap
  MPIBootstrap bootstrap;
  bootstrap.Initialize();
  
  // Create memory manager
  SymmMemManager memManager(bootstrap);
  
  // Create AllReduce instance
  AllReduce allreduce(bootstrap, memManager);
  
  // Execute AllReduce
  allreduce.Execute(d_sendbuf, d_recvbuf, count, HIP_R_32F, ReduceOp::SUM, 0);
  
  // Copy result back to host
  std::vector<float> h_recvbuf(count);
  HIP_CHECK(hipMemcpy(h_recvbuf.data(), d_recvbuf, count * sizeof(float), hipMemcpyDeviceToHost));
  
  // Verify result
  // Expected: sum of (1 + 2 + ... + worldSize) = worldSize * (worldSize + 1) / 2
  float expected = static_cast<float>(worldSize * (worldSize + 1) / 2);
  bool success = true;
  for (size_t i = 0; i < count; i++) {
    if (std::abs(h_recvbuf[i] - expected) > 1e-5) {
      std::cerr << "Rank " << rank << ": Mismatch at index " << i 
                << ": expected " << expected << ", got " << h_recvbuf[i] << std::endl;
      success = false;
      break;
    }
  }
  
  if (success) {
    std::cout << "Rank " << rank << ": AllReduce SUM test PASSED!" << std::endl;
  } else {
    std::cout << "Rank " << rank << ": AllReduce SUM test FAILED!" << std::endl;
  }
  
  // Cleanup
  HIP_CHECK(hipFree(d_sendbuf));
  HIP_CHECK(hipFree(d_recvbuf));
  bootstrap.Finalize();
}

void TestAllReduceMax(int rank, int worldSize, size_t count) {
  std::cout << "Rank " << rank << ": Testing AllReduce MAX with " << count << " elements" << std::endl;
  
  // Allocate and initialize data on GPU
  float* d_sendbuf = nullptr;
  float* d_recvbuf = nullptr;
  HIP_CHECK(hipMalloc(&d_sendbuf, count * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_recvbuf, count * sizeof(float)));
  
  // Initialize sendbuf with rank value
  std::vector<float> h_sendbuf(count, static_cast<float>(rank));
  HIP_CHECK(hipMemcpy(d_sendbuf, h_sendbuf.data(), count * sizeof(float), hipMemcpyHostToDevice));
  
  // Initialize bootstrap
  MPIBootstrap bootstrap;
  bootstrap.Initialize();
  
  // Create memory manager
  SymmMemManager memManager(bootstrap);
  
  // Create AllReduce instance
  AllReduce allreduce(bootstrap, memManager);
  
  // Execute AllReduce
  allreduce.Execute(d_sendbuf, d_recvbuf, count, HIP_R_32F, ReduceOp::MAX, 0);
  
  // Copy result back to host
  std::vector<float> h_recvbuf(count);
  HIP_CHECK(hipMemcpy(h_recvbuf.data(), d_recvbuf, count * sizeof(float), hipMemcpyDeviceToHost));
  
  // Verify result
  // Expected: max of (0, 1, 2, ..., worldSize-1) = worldSize - 1
  float expected = static_cast<float>(worldSize - 1);
  bool success = true;
  for (size_t i = 0; i < count; i++) {
    if (std::abs(h_recvbuf[i] - expected) > 1e-5) {
      std::cerr << "Rank " << rank << ": Mismatch at index " << i 
                << ": expected " << expected << ", got " << h_recvbuf[i] << std::endl;
      success = false;
      break;
    }
  }
  
  if (success) {
    std::cout << "Rank " << rank << ": AllReduce MAX test PASSED!" << std::endl;
  } else {
    std::cout << "Rank " << rank << ": AllReduce MAX test FAILED!" << std::endl;
  }
  
  // Cleanup
  HIP_CHECK(hipFree(d_sendbuf));
  HIP_CHECK(hipFree(d_recvbuf));
  bootstrap.Finalize();
}

int main(int argc, char** argv) {
  // Initialize MPI
  MPIBootstrap bootstrap;
  bootstrap.Initialize();
  
  int rank = bootstrap.GetRank();
  int worldSize = bootstrap.GetWorldSize();
  
  std::cout << "Rank " << rank << " of " << worldSize << " initialized" << std::endl;
  
  // Test with small message (recursive doubling)
  TestAllReduceSum(rank, worldSize, 1024);
  
  // Test with large message (ring algorithm)
  TestAllReduceSum(rank, worldSize, 1024 * 1024);
  
  // Test MAX operation
  TestAllReduceMax(rank, worldSize, 1024);
  
  bootstrap.Finalize();
  
  return 0;
}
