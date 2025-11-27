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

namespace p2p_allreduce {

// Optimized memory copy utilities
namespace utils {

// Vectorized copy for aligned memory
template <typename T>
__device__ inline void VectorizedCopy(T* dst, const T* src, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  
  // Try to use vector loads/stores for better memory bandwidth
  if constexpr (sizeof(T) == 4) {  // float or int32
    size_t vec_count = count / 4;
    size_t remainder = count % 4;
    
    // Process 128-bit (4x32-bit) chunks
    for (size_t i = idx; i < vec_count; i += stride) {
      reinterpret_cast<float4*>(dst)[i] = reinterpret_cast<const float4*>(src)[i];
    }
    
    // Handle remainder
    for (size_t i = vec_count * 4 + idx; i < count; i += stride) {
      dst[i] = src[i];
    }
  } else if constexpr (sizeof(T) == 8) {  // double or int64
    size_t vec_count = count / 2;
    size_t remainder = count % 2;
    
    // Process 128-bit (2x64-bit) chunks
    for (size_t i = idx; i < vec_count; i += stride) {
      reinterpret_cast<double2*>(dst)[i] = reinterpret_cast<const double2*>(src)[i];
    }
    
    // Handle remainder
    for (size_t i = vec_count * 2 + idx; i < count; i += stride) {
      dst[i] = src[i];
    }
  } else {
    // Fallback for other types
    for (size_t i = idx; i < count; i += stride) {
      dst[i] = src[i];
    }
  }
}

// Vectorized reduction
template <typename T, typename Op>
__device__ inline void VectorizedReduce(T* dst, const T* src, size_t count, Op op) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  
  for (size_t i = idx; i < count; i += stride) {
    dst[i] = op(dst[i], src[i]);
  }
}

// Warp-level reduction for better utilization
template <typename T>
__device__ inline T WarpReduce(T val, T (*op)(T, T)) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    T other = __shfl_down(val, offset);
    val = op(val, other);
  }
  return val;
}

// Block-level reduction using shared memory
template <typename T>
__device__ inline T BlockReduce(T val, T (*op)(T, T), T* shared) {
  int tid = threadIdx.x;
  int lane = tid % warpSize;
  int wid = tid / warpSize;
  
  // Warp-level reduce
  val = WarpReduce(val, op);
  
  // Write warp results to shared memory
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  
  // Block-level reduce from warp results
  if (tid < blockDim.x / warpSize) {
    val = shared[tid];
  } else {
    val = T{};
  }
  
  if (wid == 0) {
    val = WarpReduce(val, op);
  }
  
  return val;
}

}  // namespace utils

// Optimized reduction operators as device functions
namespace ops {

template <typename T>
__device__ inline T Sum(T a, T b) { return a + b; }

template <typename T>
__device__ inline T Prod(T a, T b) { return a * b; }

template <typename T>
__device__ inline T Min(T a, T b) { return (a < b) ? a : b; }

template <typename T>
__device__ inline T Max(T a, T b) { return (a > b) ? a : b; }

}  // namespace ops

}  // namespace p2p_allreduce
