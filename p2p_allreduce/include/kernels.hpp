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

#pragma once

#include <hip/hip_runtime.h>
#include "p2p_allreduce.hpp"

namespace p2p_allreduce {

// Single-launch Ring AllReduce kernel
// Manages all steps internally using cooperative groups and atomics
template <typename T>
__global__ void RingAllReduceKernel(
    SymmMemObj* workspace,
    const T* sendbuf,
    T* recvbuf,
    size_t count,
    int rank,
    int worldSize,
    size_t chunkSize,
    ReduceOp op,
    int* globalStepCounter) {
  
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Phase 1: Reduce-Scatter
  // Process all steps in a single kernel launch
  for (int step = 1; step < worldSize; step++) {
    // Calculate which chunk this rank is responsible for in this step
    int recvRank = (rank - step + worldSize) % worldSize;
    int recvChunk = (recvRank - 1 + worldSize) % worldSize;
    
    size_t chunkStart = recvChunk * chunkSize;
    size_t chunkEnd = min(chunkStart + chunkSize, count);
    size_t chunkCount = chunkEnd - chunkStart;
    
    // Get pointer to peer's data
    int peerRank = (rank - 1 + worldSize) % worldSize;
    const T* peerData = workspace->GetPeerAs<T>(peerRank) + chunkStart;
    
    // Local workspace
    T* localData = workspace->GetAs<T>() + chunkStart;
    
    // If this is the first step, copy from sendbuf
    if (step == 1 && idx < chunkCount) {
      localData[idx] = sendbuf[chunkStart + idx];
    }
    
    // Synchronize with peer GPU using atomic counter
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      // Signal that we're ready
      atomicAdd(globalStepCounter, 1);
      // Wait for peer to be ready (busy wait)
      while (atomicAdd(globalStepCounter, 0) < (step * worldSize)) {
        // Busy wait
      }
    }
    __syncthreads();
    
    // Reduce from peer
    if (idx < chunkCount) {
      T val = peerData[idx];
      switch (op) {
        case ReduceOp::SUM:
          localData[idx] += val;
          break;
        case ReduceOp::PROD:
          localData[idx] *= val;
          break;
        case ReduceOp::MIN:
          localData[idx] = (localData[idx] < val) ? localData[idx] : val;
          break;
        case ReduceOp::MAX:
          localData[idx] = (localData[idx] > val) ? localData[idx] : val;
          break;
        case ReduceOp::AVG:
          localData[idx] += val;
          break;
      }
    }
    __syncthreads();
  }
  
  // Phase 2: AllGather
  for (int step = 1; step < worldSize; step++) {
    // Calculate which chunk to send/receive
    int sendChunk = (rank - step + 1 + worldSize) % worldSize;
    
    size_t chunkStart = sendChunk * chunkSize;
    size_t chunkEnd = min(chunkStart + chunkSize, count);
    size_t chunkCount = chunkEnd - chunkStart;
    
    // Get pointer to peer's data
    int peerRank = (rank - 1 + worldSize) % worldSize;
    const T* peerData = workspace->GetPeerAs<T>(peerRank) + chunkStart;
    
    // Synchronize with peer GPU
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      atomicAdd(globalStepCounter, 1);
      while (atomicAdd(globalStepCounter, 0) < ((worldSize - 1 + step) * worldSize)) {
        // Busy wait
      }
    }
    __syncthreads();
    
    // Copy to local recvbuf
    if (idx < chunkCount) {
      recvbuf[chunkStart + idx] = peerData[idx];
    }
    __syncthreads();
  }
  
  // Copy final result (my reduced chunk) to recvbuf
  int myChunk = rank;
  size_t chunkStart = myChunk * chunkSize;
  size_t chunkEnd = min(chunkStart + chunkSize, count);
  size_t chunkCount = chunkEnd - chunkStart;
  
  if (idx < chunkCount) {
    T val = workspace->GetAs<T>()[chunkStart + idx];
    if (op == ReduceOp::AVG) {
      val /= static_cast<T>(worldSize);
    }
    recvbuf[chunkStart + idx] = val;
  }
}

// Single-launch Recursive Doubling AllReduce kernel
template <typename T>
__global__ void RecursiveDoublingAllReduceKernel(
    SymmMemObj* workspace,
    const T* sendbuf,
    T* recvbuf,
    size_t count,
    int rank,
    int worldSize,
    ReduceOp op,
    int* globalStepCounter) {
  
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Number of steps in recursive doubling
  int numSteps = 0;
  int temp = worldSize - 1;
  while (temp > 0) {
    numSteps++;
    temp >>= 1;
  }
  
  // Initialize workspace with sendbuf
  if (idx < count) {
    workspace->GetAs<T>()[idx] = sendbuf[idx];
  }
  __syncthreads();
  
  for (int step = 0; step < numSteps; step++) {
    int distance = 1 << step;
    int partnerRank = rank ^ distance;
    
    // Skip if partner is out of bounds
    if (partnerRank >= worldSize) continue;
    
    // Synchronize with partner
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      atomicAdd(globalStepCounter, 1);
      while (atomicAdd(globalStepCounter, 0) < ((step + 1) * worldSize)) {
        // Busy wait
      }
    }
    __syncthreads();
    
    if (idx < count) {
      // Get pointer to partner's data
      const T* partnerData = workspace->GetPeerAs<T>(partnerRank);
      T* localData = workspace->GetAs<T>();
      
      T val = partnerData[idx];
      switch (op) {
        case ReduceOp::SUM:
          localData[idx] += val;
          break;
        case ReduceOp::PROD:
          localData[idx] *= val;
          break;
        case ReduceOp::MIN:
          localData[idx] = (localData[idx] < val) ? localData[idx] : val;
          break;
        case ReduceOp::MAX:
          localData[idx] = (localData[idx] > val) ? localData[idx] : val;
          break;
        case ReduceOp::AVG:
          localData[idx] += val;
          break;
      }
    }
    __syncthreads();
  }
  
  // Copy final result to recvbuf
  if (idx < count) {
    T val = workspace->GetAs<T>()[idx];
    if (op == ReduceOp::AVG) {
      val /= static_cast<T>(worldSize);
    }
    recvbuf[idx] = val;
  }
}

}  // namespace p2p_allreduce
