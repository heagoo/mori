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
#include <chrono>
#include <iomanip>
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
using namespace std::chrono;

struct BenchmarkResult {
  size_t count;
  size_t bytes;
  double avgTimeMs;
  double minTimeMs;
  double maxTimeMs;
  double bandwidthGBs;
  double algBandwidthGBs;  // Algorithm bandwidth (considering data movement)
};

void PrintHeader() {
  std::cout << std::setw(12) << "Count"
            << std::setw(12) << "Size(KB)"
            << std::setw(12) << "Avg(ms)"
            << std::setw(12) << "Min(ms)"
            << std::setw(12) << "Max(ms)"
            << std::setw(15) << "BW(GB/s)"
            << std::setw(15) << "AlgBW(GB/s)"
            << std::endl;
  std::cout << std::string(88, '-') << std::endl;
}

void PrintResult(const BenchmarkResult& result) {
  std::cout << std::setw(12) << result.count
            << std::setw(12) << std::fixed << std::setprecision(2) 
            << result.bytes / 1024.0
            << std::setw(12) << std::fixed << std::setprecision(3)
            << result.avgTimeMs
            << std::setw(12) << result.minTimeMs
            << std::setw(12) << result.maxTimeMs
            << std::setw(15) << std::fixed << std::setprecision(2)
            << result.bandwidthGBs
            << std::setw(15) << result.algBandwidthGBs
            << std::endl;
}

BenchmarkResult BenchmarkAllReduce(AllReduce& allreduce, Bootstrap& bootstrap,
                                   size_t count, int warmup, int iterations) {
  int rank = bootstrap.GetRank();
  int worldSize = bootstrap.GetWorldSize();
  
  // Allocate memory
  float* d_sendbuf = nullptr;
  float* d_recvbuf = nullptr;
  size_t bytes = count * sizeof(float);
  
  HIP_CHECK(hipMalloc(&d_sendbuf, bytes));
  HIP_CHECK(hipMalloc(&d_recvbuf, bytes));
  HIP_CHECK(hipMemset(d_sendbuf, 0, bytes));
  
  // Create HIP events for timing
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  
  // Warmup
  for (int i = 0; i < warmup; i++) {
    allreduce.Execute(d_sendbuf, d_recvbuf, count, HIP_R_32F, ReduceOp::SUM, 0);
  }
  HIP_CHECK(hipDeviceSynchronize());
  bootstrap.Barrier();
  
  // Benchmark
  std::vector<double> times;
  for (int i = 0; i < iterations; i++) {
    HIP_CHECK(hipEventRecord(start, 0));
    allreduce.Execute(d_sendbuf, d_recvbuf, count, HIP_R_32F, ReduceOp::SUM, 0);
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    times.push_back(milliseconds);
    
    bootstrap.Barrier();
  }
  
  // Calculate statistics
  double sum = 0;
  double minTime = times[0];
  double maxTime = times[0];
  for (double t : times) {
    sum += t;
    minTime = std::min(minTime, t);
    maxTime = std::max(maxTime, t);
  }
  double avgTime = sum / iterations;
  
  // Calculate bandwidth
  // Bus bandwidth: data size / time
  double bandwidthGBs = (bytes / 1e9) / (avgTime / 1000.0);
  
  // Algorithm bandwidth for ring: 2 * (P-1) / P * data
  // For reduce-scatter and allgather phases
  double algFactor = 2.0 * (worldSize - 1) / worldSize;
  double algBandwidthGBs = (bytes * algFactor / 1e9) / (avgTime / 1000.0);
  
  // Cleanup
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_sendbuf));
  HIP_CHECK(hipFree(d_recvbuf));
  
  return BenchmarkResult{
    count, bytes, avgTime, minTime, maxTime, 
    bandwidthGBs, algBandwidthGBs
  };
}

int main(int argc, char** argv) {
  // Initialize
  MPIBootstrap bootstrap;
  bootstrap.Initialize();
  
  int rank = bootstrap.GetRank();
  int worldSize = bootstrap.GetWorldSize();
  
  if (rank == 0) {
    std::cout << "\n=== P2P AllReduce Benchmark ===\n" << std::endl;
    std::cout << "World Size: " << worldSize << std::endl;
    std::cout << "Operation: AllReduce SUM (float)" << std::endl;
    std::cout << "Warmup iterations: 5" << std::endl;
    std::cout << "Benchmark iterations: 20" << std::endl;
    std::cout << std::endl;
  }
  
  // Create memory manager and allreduce
  SymmMemManager memManager(bootstrap);
  AllReduce allreduce(bootstrap, memManager);
  
  // Test various sizes
  std::vector<size_t> sizes = {
    1024,           // 4 KB
    4096,           // 16 KB
    16384,          // 64 KB
    65536,          // 256 KB
    262144,         // 1 MB
    1048576,        // 4 MB
    4194304,        // 16 MB
    16777216,       // 64 MB
    67108864,       // 256 MB
  };
  
  if (rank == 0) {
    PrintHeader();
  }
  
  for (size_t count : sizes) {
    BenchmarkResult result = BenchmarkAllReduce(allreduce, bootstrap, count, 5, 20);
    
    if (rank == 0) {
      PrintResult(result);
    }
  }
  
  if (rank == 0) {
    std::cout << "\nNotes:" << std::endl;
    std::cout << "- BW(GB/s): Bus bandwidth = DataSize / Time" << std::endl;
    std::cout << "- AlgBW(GB/s): Algorithm bandwidth considering ring data movement" << std::endl;
    std::cout << "- For ring: Total data moved = 2 * (P-1) / P * DataSize" << std::endl;
    std::cout << std::endl;
  }
  
  bootstrap.Finalize();
  return 0;
}
