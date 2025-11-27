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

// Device-side helper to get pointer from peer array
template <typename T>
__device__ inline T* GetPeerPtr(uintptr_t* d_peerPtrs, int pe) {
  return reinterpret_cast<T*>(d_peerPtrs[pe]);
}

// Single-launch Ring AllReduce kernel
// Manages all steps internally using P2P step counters for peer-to-peer synchronization
// Each rank only waits for the rank it depends on by checking that rank's counter
// Uses d_peerPtrs arrays for device-accessible peer pointers
template <typename T>
__global__ void RingAllReduceKernel(
    uintptr_t* workspacePeerPtrs,    // Device array of workspace peer pointers
    const T* sendbuf,
    T* recvbuf,
    size_t count,
    int rank,
    int worldSize,
    size_t chunkSize,
    ReduceOp op,
    uintptr_t* stepCounterPeerPtrs)  // Device array of step counter peer pointers
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  
  // Get local step counter pointer
  volatile int* myCounter = GetPeerPtr<int>(stepCounterPeerPtrs, rank);
  
  // Get local workspace pointer
  T* localWorkspace = GetPeerPtr<T>(workspacePeerPtrs, rank);
  
  // Get peer rank (the one we read from in the ring)
  int peerRank = (rank - 1 + worldSize) % worldSize;
  
  // Step 0: Initialize workspace with sendbuf
  for (size_t i = idx; i < count; i += stride) {
    localWorkspace[i] = sendbuf[i];
  }
  __syncthreads();
  
  // Signal that data is ready and wait for peer to be ready
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence_system();
    atomicAdd((int*)myCounter, 1);  // Step 0 complete
    __threadfence_system();
    
    // Wait for peer to finish initialization (step 0)
    volatile int* peerCounter = GetPeerPtr<int>(stepCounterPeerPtrs, peerRank);
    while (*peerCounter < 1) {
      // Busy wait
    }
  }
  __syncthreads();
  
  // Phase 1: Reduce-Scatter
  // Each step, we reduce a different chunk from our peer into our workspace
  for (int step = 0; step < worldSize - 1; step++) {
    // Calculate which chunk we're working on in this step
    // In ring reduce-scatter, each rank reduces the chunk that will eventually
    // belong to rank (rank - step - 1 + worldSize) % worldSize
    int chunkIdx = (rank - step - 1 + worldSize) % worldSize;
    
    size_t chunkStart = chunkIdx * chunkSize;
    size_t chunkEnd = min(chunkStart + chunkSize, count);
    
    // Get peer's workspace pointer for this chunk
    const T* peerData = GetPeerPtr<T>(workspacePeerPtrs, peerRank) + chunkStart;
    T* localData = localWorkspace + chunkStart;
    
    // Reduce peer's data into our local workspace
    for (size_t i = idx; i < chunkEnd - chunkStart; i += stride) {
      T peerVal = peerData[i];
      T localVal = localData[i];
      switch (op) {
        case ReduceOp::SUM:
        case ReduceOp::AVG:
          localData[i] = localVal + peerVal;
          break;
        case ReduceOp::PROD:
          localData[i] = localVal * peerVal;
          break;
        case ReduceOp::MIN:
          localData[i] = (localVal < peerVal) ? localVal : peerVal;
          break;
        case ReduceOp::MAX:
          localData[i] = (localVal > peerVal) ? localVal : peerVal;
          break;
      }
    }
    __syncthreads();
    
    // Signal step completion and wait for peer
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      __threadfence_system();
      atomicAdd((int*)myCounter, 1);  // Step (step+1) complete
      __threadfence_system();
      
      // Wait for peer to finish this step
      volatile int* peerCounter = GetPeerPtr<int>(stepCounterPeerPtrs, peerRank);
      while (*peerCounter < (step + 2)) {
        // Busy wait
      }
    }
    __syncthreads();
  }
  
  // Phase 2: AllGather
  // Each step, we copy a fully-reduced chunk from our peer to our workspace
  for (int step = 0; step < worldSize - 1; step++) {
    // Calculate which chunk to gather in this step
    // Each rank has its fully reduced chunk at index = rank
    // In allgather, we copy from peer's chunk (peerRank - step + worldSize) % worldSize
    int chunkIdx = (rank - step - 2 + 2 * worldSize) % worldSize;
    
    size_t chunkStart = chunkIdx * chunkSize;
    size_t chunkEnd = min(chunkStart + chunkSize, count);
    
    // Get peer's workspace pointer for this chunk
    const T* peerData = GetPeerPtr<T>(workspacePeerPtrs, peerRank) + chunkStart;
    T* localData = localWorkspace + chunkStart;
    
    // Copy peer's fully-reduced chunk to our workspace
    for (size_t i = idx; i < chunkEnd - chunkStart; i += stride) {
      localData[i] = peerData[i];
    }
    __syncthreads();
    
    // Signal step completion and wait for peer
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      __threadfence_system();
      atomicAdd((int*)myCounter, 1);  // Allgather step complete
      __threadfence_system();
      
      // Wait for peer to finish this step
      volatile int* peerCounter = GetPeerPtr<int>(stepCounterPeerPtrs, peerRank);
      int expectedStep = worldSize + step + 1;
      while (*peerCounter < expectedStep) {
        // Busy wait
      }
    }
    __syncthreads();
  }
  
  // Copy final result from workspace to recvbuf
  for (size_t i = idx; i < count; i += stride) {
    T val = localWorkspace[i];
    if (op == ReduceOp::AVG) {
      val /= static_cast<T>(worldSize);
    }
    recvbuf[i] = val;
  }
}

// Single-launch Recursive Doubling AllReduce kernel
// Each rank only waits for its partner rank by checking that rank's counter
// Uses d_peerPtrs arrays for device-accessible peer pointers
template <typename T>
__global__ void RecursiveDoublingAllReduceKernel(
    uintptr_t* workspacePeerPtrs,    // Device array of workspace peer pointers
    const T* sendbuf,
    T* recvbuf,
    size_t count,
    int rank,
    int worldSize,
    ReduceOp op,
    uintptr_t* stepCounterPeerPtrs)  // Device array of step counter peer pointers
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  
  // Get local step counter pointer
  volatile int* myCounter = GetPeerPtr<int>(stepCounterPeerPtrs, rank);
  
  // Get local workspace pointer
  T* localWorkspace = GetPeerPtr<T>(workspacePeerPtrs, rank);
  
  // Number of steps in recursive doubling
  int numSteps = 0;
  int temp = worldSize - 1;
  while (temp > 0) {
    numSteps++;
    temp >>= 1;
  }
  
  // Step 0: Initialize workspace with sendbuf
  for (size_t i = idx; i < count; i += stride) {
    localWorkspace[i] = sendbuf[i];
  }
  __syncthreads();
  
  // Signal initialization complete and wait for all peers
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence_system();
    atomicAdd((int*)myCounter, 1);  // Step 0 complete
    __threadfence_system();
  }
  __syncthreads();
  
  for (int step = 0; step < numSteps; step++) {
    int distance = 1 << step;
    int partnerRank = rank ^ distance;
    
    // Skip if partner is out of bounds
    if (partnerRank >= worldSize) {
      // Still need to update counter for ranks waiting on us
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        atomicAdd((int*)myCounter, 1);
        __threadfence_system();
      }
      __syncthreads();
      continue;
    }
    
    // Wait for partner to be ready for this step
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      volatile int* partnerCounter = GetPeerPtr<int>(stepCounterPeerPtrs, partnerRank);
      while (*partnerCounter < (step + 1)) {
        // Busy wait for partner
      }
    }
    __syncthreads();
    
    // Reduce partner's data into our workspace
    const T* partnerData = GetPeerPtr<T>(workspacePeerPtrs, partnerRank);
    for (size_t i = idx; i < count; i += stride) {
      T peerVal = partnerData[i];
      T localVal = localWorkspace[i];
      switch (op) {
        case ReduceOp::SUM:
        case ReduceOp::AVG:
          localWorkspace[i] = localVal + peerVal;
          break;
        case ReduceOp::PROD:
          localWorkspace[i] = localVal * peerVal;
          break;
        case ReduceOp::MIN:
          localWorkspace[i] = (localVal < peerVal) ? localVal : peerVal;
          break;
        case ReduceOp::MAX:
          localWorkspace[i] = (localVal > peerVal) ? localVal : peerVal;
          break;
      }
    }
    __syncthreads();
    
    // Signal step completion
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      __threadfence_system();
      atomicAdd((int*)myCounter, 1);  // Step (step+1) complete
      __threadfence_system();
    }
    __syncthreads();
  }
  
  // Copy final result to recvbuf
  for (size_t i = idx; i < count; i += stride) {
    T val = localWorkspace[i];
    if (op == ReduceOp::AVG) {
      val /= static_cast<T>(worldSize);
    }
    recvbuf[i] = val;
  }
}

}  // namespace p2p_allreduce
