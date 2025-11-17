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
    : bootstrap_(bootstrap), memManager_(memManager) {}

AllReduce::~AllReduce() {
  if (workspace_) {
    memManager_.Free(workspace_);
    workspace_ = nullptr;
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
  
  // Phase 1: Reduce-Scatter
  // In each step, reduce data from left neighbor
  for (int step = 1; step < worldSize; step++) {
    switch (datatype) {
      case HIP_R_32F:
        hipLaunchKernelGGL(RingReduceScatterKernel<float>, gridSize, blockSize, 0, stream,
                          workspace_, static_cast<const float*>(sendbuf), count,
                          rank, worldSize, step, chunkSize, op);
        break;
      case HIP_R_64F:
        hipLaunchKernelGGL(RingReduceScatterKernel<double>, gridSize, blockSize, 0, stream,
                          workspace_, static_cast<const double*>(sendbuf), count,
                          rank, worldSize, step, chunkSize, op);
        break;
      case HIP_R_32I:
        hipLaunchKernelGGL(RingReduceScatterKernel<int32_t>, gridSize, blockSize, 0, stream,
                          workspace_, static_cast<const int32_t*>(sendbuf), count,
                          rank, worldSize, step, chunkSize, op);
        break;
      default:
        fprintf(stderr, "Unsupported data type in RingAllReduce\n");
        return;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  
  // Phase 2: AllGather
  // Gather the reduced chunks to all ranks
  for (int step = 1; step < worldSize; step++) {
    switch (datatype) {
      case HIP_R_32F:
        hipLaunchKernelGGL(RingAllgatherKernel<float>, gridSize, blockSize, 0, stream,
                          workspace_, static_cast<float*>(recvbuf), count,
                          rank, worldSize, step, chunkSize);
        break;
      case HIP_R_64F:
        hipLaunchKernelGGL(RingAllgatherKernel<double>, gridSize, blockSize, 0, stream,
                          workspace_, static_cast<double*>(recvbuf), count,
                          rank, worldSize, step, chunkSize);
        break;
      case HIP_R_32I:
        hipLaunchKernelGGL(RingAllgatherKernel<int32_t>, gridSize, blockSize, 0, stream,
                          workspace_, static_cast<int32_t*>(recvbuf), count,
                          rank, worldSize, step, chunkSize);
        break;
      default:
        fprintf(stderr, "Unsupported data type in RingAllReduce\n");
        return;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  
  // Copy final result (my reduced chunk) to recvbuf
  int myChunk = rank;
  size_t chunkStart = myChunk * chunkSize;
  size_t chunkEnd = std::min(chunkStart + chunkSize, count);
  size_t chunkCount = chunkEnd - chunkStart;
  
  gridSize = (chunkCount + blockSize - 1) / blockSize;
  switch (datatype) {
    case HIP_R_32F:
      hipLaunchKernelGGL(CopyToOutputKernel<float>, gridSize, blockSize, 0, stream,
                        workspace_->GetAs<float>() + chunkStart,
                        static_cast<float*>(recvbuf) + chunkStart,
                        chunkCount, op, worldSize);
      break;
    case HIP_R_64F:
      hipLaunchKernelGGL(CopyToOutputKernel<double>, gridSize, blockSize, 0, stream,
                        workspace_->GetAs<double>() + chunkStart,
                        static_cast<double*>(recvbuf) + chunkStart,
                        chunkCount, op, worldSize);
      break;
    case HIP_R_32I:
      hipLaunchKernelGGL(CopyToOutputKernel<int32_t>, gridSize, blockSize, 0, stream,
                        workspace_->GetAs<int32_t>() + chunkStart,
                        static_cast<int32_t*>(recvbuf) + chunkStart,
                        chunkCount, op, worldSize);
      break;
    default:
      break;
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
  
  // Number of steps in recursive doubling
  int numSteps = static_cast<int>(std::ceil(std::log2(worldSize)));
  
  for (int step = 0; step < numSteps; step++) {
    int distance = 1 << step;
    int partnerRank = rank ^ distance;
    
    // Skip if partner is out of bounds
    if (partnerRank >= worldSize) continue;
    
    switch (datatype) {
      case HIP_R_32F:
        hipLaunchKernelGGL(RecursiveDoublingReduceKernel<float>, gridSize, blockSize, 0, stream,
                          workspace_, static_cast<const float*>(sendbuf),
                          static_cast<float*>(recvbuf), count,
                          rank, worldSize, step, partnerRank, op);
        break;
      case HIP_R_64F:
        hipLaunchKernelGGL(RecursiveDoublingReduceKernel<double>, gridSize, blockSize, 0, stream,
                          workspace_, static_cast<const double*>(sendbuf),
                          static_cast<double*>(recvbuf), count,
                          rank, worldSize, step, partnerRank, op);
        break;
      case HIP_R_32I:
        hipLaunchKernelGGL(RecursiveDoublingReduceKernel<int32_t>, gridSize, blockSize, 0, stream,
                          workspace_, static_cast<const int32_t*>(sendbuf),
                          static_cast<int32_t*>(recvbuf), count,
                          rank, worldSize, step, partnerRank, op);
        break;
      default:
        fprintf(stderr, "Unsupported data type in RecursiveDoublingAllReduce\n");
        return;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  
  // Copy final result to recvbuf
  switch (datatype) {
    case HIP_R_32F:
      hipLaunchKernelGGL(CopyToOutputKernel<float>, gridSize, blockSize, 0, stream,
                        workspace_->GetAs<float>(),
                        static_cast<float*>(recvbuf),
                        count, op, worldSize);
      break;
    case HIP_R_64F:
      hipLaunchKernelGGL(CopyToOutputKernel<double>, gridSize, blockSize, 0, stream,
                        workspace_->GetAs<double>(),
                        static_cast<double*>(recvbuf),
                        count, op, worldSize);
      break;
    case HIP_R_32I:
      hipLaunchKernelGGL(CopyToOutputKernel<int32_t>, gridSize, blockSize, 0, stream,
                        workspace_->GetAs<int32_t>(),
                        static_cast<int32_t*>(recvbuf),
                        count, op, worldSize);
      break;
    default:
      break;
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
}

}  // namespace p2p_allreduce
