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
#include "kernels.hpp"
#include <algorithm>
#include <cmath>

#define HIP_CHECK(cmd)                                                          \
  do {                                                                          \
    hipError_t error = (cmd);                                                   \
    if (error != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error: '%s'(%d) at %s:%d\n",                        \
              hipGetErrorString(error), error, __FILE__, __LINE__);             \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

namespace p2p_allreduce {

AllReduce::AllReduce(Bootstrap& bootstrap, SymmMemManager& memManager)
    : bootstrap_(bootstrap), memManager_(memManager) {
  // Allocate step counters as SymmMemObj for P2P access
  // Each rank has its own counter that peers can read
  stepCounters_ = memManager_.Malloc(sizeof(int));
}

AllReduce::~AllReduce() {
  if (workspace_) {
    memManager_.Free(workspace_);
    workspace_ = nullptr;
  }
  if (stepCounters_) {
    memManager_.Free(stepCounters_);
    stepCounters_ = nullptr;
  }
}

void AllReduce::EnsureWorkspace(size_t requiredSize) {
  if (workspaceSize_ >= requiredSize && workspace_) {
    return;
  }
  
  // Free old workspace if it exists
  if (workspace_) {
    memManager_.Free(workspace_);
  }
  
  // Allocate new workspace
  workspaceSize_ = requiredSize;
  workspace_ = memManager_.Malloc(workspaceSize_);
}

size_t GetTypeSize(hipDataType datatype) {
  switch (datatype) {
    case HIP_R_32F: return sizeof(float);
    case HIP_R_64F: return sizeof(double);
    case HIP_R_16F: return sizeof(__half);
    case HIP_R_32I: return sizeof(int32_t);
    case HIP_R_64I: return sizeof(int64_t);
    default: return 1;
  }
}

void AllReduce::Execute(const void* sendbuf, void* recvbuf, size_t count,
                        hipDataType datatype, ReduceOp op, hipStream_t stream) {
  size_t typeSize = GetTypeSize(datatype);
  size_t totalBytes = count * typeSize;
  
  // Choose algorithm based on message size
  if (totalBytes <= SMALL_MSG_THRESHOLD) {
    RecursiveDoublingAllReduce(sendbuf, recvbuf, count, datatype, op, stream);
  } else {
    RingAllReduce(sendbuf, recvbuf, count, datatype, op, stream);
  }
}

void AllReduce::RingAllReduce(const void* sendbuf, void* recvbuf, size_t count,
                              hipDataType datatype, ReduceOp op, hipStream_t stream) {
  int worldSize = bootstrap_.GetWorldSize();
  int rank = bootstrap_.GetRank();
  size_t typeSize = GetTypeSize(datatype);
  
  // Ensure workspace is large enough
  EnsureWorkspace(count * typeSize);
  
  // Calculate chunk size for ring algorithm
  size_t chunkSize = (count + worldSize - 1) / worldSize;
  
  // Determine block and grid size
  int blockSize = 256;
  int gridSize = (chunkSize + blockSize - 1) / blockSize;
  
  // Reset local step counter to 0
  HIP_CHECK(hipMemsetAsync(stepCounters_->GetAs<int>(), 0, sizeof(int), stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  
  // Synchronize all ranks before starting (barrier)
  bootstrap_.Barrier();
  
  // Launch kernel ONCE for entire AllReduce operation
  switch (datatype) {
    case HIP_R_32F:
      hipLaunchKernelGGL(RingAllReduceKernel<float>, gridSize, blockSize, 0, stream,
                        workspace_, static_cast<const float*>(sendbuf),
                        static_cast<float*>(recvbuf), count,
                        rank, worldSize, chunkSize, op, stepCounters_);
      break;
    case HIP_R_64F:
      hipLaunchKernelGGL(RingAllReduceKernel<double>, gridSize, blockSize, 0, stream,
                        workspace_, static_cast<const double*>(sendbuf),
                        static_cast<double*>(recvbuf), count,
                        rank, worldSize, chunkSize, op, stepCounters_);
      break;
    case HIP_R_32I:
      hipLaunchKernelGGL(RingAllReduceKernel<int32_t>, gridSize, blockSize, 0, stream,
                        workspace_, static_cast<const int32_t*>(sendbuf),
                        static_cast<int32_t*>(recvbuf), count,
                        rank, worldSize, chunkSize, op, stepCounters_);
      break;
    default:
      fprintf(stderr, "Unsupported data type in RingAllReduce\n");
      return;
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
}

void AllReduce::RecursiveDoublingAllReduce(const void* sendbuf, void* recvbuf,
                                           size_t count, hipDataType datatype,
                                           ReduceOp op, hipStream_t stream) {
  int worldSize = bootstrap_.GetWorldSize();
  int rank = bootstrap_.GetRank();
  size_t typeSize = GetTypeSize(datatype);
  
  // Ensure workspace is large enough
  EnsureWorkspace(count * typeSize);
  
  // Determine block and grid size
  int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  
  // Reset local step counter to 0
  HIP_CHECK(hipMemsetAsync(stepCounters_->GetAs<int>(), 0, sizeof(int), stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  
  // Synchronize all ranks before starting (barrier)
  bootstrap_.Barrier();
  
  // Launch kernel ONCE for entire AllReduce operation
  switch (datatype) {
    case HIP_R_32F:
      hipLaunchKernelGGL(RecursiveDoublingAllReduceKernel<float>, gridSize, blockSize, 0, stream,
                        workspace_, static_cast<const float*>(sendbuf),
                        static_cast<float*>(recvbuf), count,
                        rank, worldSize, op, stepCounters_);
      break;
    case HIP_R_64F:
      hipLaunchKernelGGL(RecursiveDoublingAllReduceKernel<double>, gridSize, blockSize, 0, stream,
                        workspace_, static_cast<const double*>(sendbuf),
                        static_cast<double*>(recvbuf), count,
                        rank, worldSize, op, stepCounters_);
      break;
    case HIP_R_32I:
      hipLaunchKernelGGL(RecursiveDoublingAllReduceKernel<int32_t>, gridSize, blockSize, 0, stream,
                        workspace_, static_cast<const int32_t*>(sendbuf),
                        static_cast<int32_t*>(recvbuf), count,
                        rank, worldSize, op, stepCounters_);
      break;
    default:
      fprintf(stderr, "Unsupported data type in RecursiveDoublingAllReduce\n");
      return;
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
}

}  // namespace p2p_allreduce
