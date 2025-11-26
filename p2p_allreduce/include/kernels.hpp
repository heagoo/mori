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

// Kernel state enum to control different stages
enum class KernelState {
  RING_REDUCE_SCATTER,   // Ring algorithm reduce-scatter phase
  RING_ALLGATHER,        // Ring algorithm allgather phase
  RECURSIVE_DOUBLING,    // Recursive doubling reduction
  COPY_TO_OUTPUT         // Final copy to output buffer
};

// Unified kernel for all AllReduce operations
// Uses shared memory flag to control the operation state
template <typename T>
__global__ void UnifiedAllReduceKernel(
    KernelState state,
    SymmMemObj* workspace,
    const T* sendbuf,
    T* recvbuf,
    size_t count,
    int rank,
    int worldSize,
    int step,
    size_t chunkSize,
    int partnerRank,
    ReduceOp op) {
  
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Process based on state
  switch (state) {
    case KernelState::RING_REDUCE_SCATTER: {
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
      break;
    }
    
    case KernelState::RING_ALLGATHER: {
      // Calculate which chunk to send/receive
      int sendChunk = (rank - step + 1 + worldSize) % worldSize;
      
      size_t chunkStart = sendChunk * chunkSize;
      size_t chunkEnd = min(chunkStart + chunkSize, count);
      size_t chunkCount = chunkEnd - chunkStart;
      
      // Get pointer to peer's data
      int peerRank = (rank - 1 + worldSize) % worldSize;
      const T* peerData = workspace->GetPeerAs<T>(peerRank) + chunkStart;
      
      // Copy to local recvbuf
      if (idx < chunkCount) {
        recvbuf[chunkStart + idx] = peerData[idx];
      }
      break;
    }
    
    case KernelState::RECURSIVE_DOUBLING: {
      // Initialize with sendbuf on first step
      if (step == 0 && idx < count) {
        workspace->GetAs<T>()[idx] = sendbuf[idx];
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
      break;
    }
    
    case KernelState::COPY_TO_OUTPUT: {
      if (idx < count) {
        T val = sendbuf[idx];  // Source data is in sendbuf parameter for this state
        // For average, divide by world size
        if (op == ReduceOp::AVG) {
          val /= static_cast<T>(worldSize);
        }
        recvbuf[idx] = val;
      }
      break;
    }
  }
}

}  // namespace p2p_allreduce
