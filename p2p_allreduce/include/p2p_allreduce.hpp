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
#include <stddef.h>
#include <stdint.h>
#include <vector>

namespace p2p_allreduce {

// Symmetric memory object structure
// Manages memory that is accessible across all processes via P2P
struct SymmMemObj {
  void* localPtr{nullptr};           // Local memory pointer
  uintptr_t* peerPtrs{nullptr};      // Array of peer memory pointers (one per rank)
  hipIpcMemHandle_t* ipcHandles{nullptr};  // IPC handles for peer access
  size_t size{0};                    // Size of the memory region

  // Get local pointer
  inline __device__ __host__ void* Get() const { return localPtr; }
  
  // Get pointer to peer memory
  inline __device__ __host__ void* GetPeer(int pe) const {
    return reinterpret_cast<void*>(peerPtrs[pe]);
  }
  
  template <typename T>
  inline __device__ __host__ T* GetAs() const {
    return reinterpret_cast<T*>(localPtr);
  }
  
  template <typename T>
  inline __device__ __host__ T* GetPeerAs(int pe) const {
    return reinterpret_cast<T*>(peerPtrs[pe]);
  }
};

// Bootstrap interface for initialization and collective operations
class Bootstrap {
 public:
  virtual ~Bootstrap() = default;
  
  virtual void Initialize() = 0;
  virtual void Finalize() = 0;
  
  virtual int GetWorldSize() const = 0;
  virtual int GetRank() const = 0;
  
  // Allgather: each rank contributes sendbuf, receives all contributions in recvbuf
  virtual void Allgather(const void* sendbuf, void* recvbuf, size_t sendcount) = 0;
  
  // Barrier: synchronization point for all ranks
  virtual void Barrier() = 0;
};

// Symmetric memory manager
class SymmMemManager {
 public:
  SymmMemManager(Bootstrap& bootstrap);
  ~SymmMemManager();
  
  // Allocate GPU memory and register it for P2P access
  SymmMemObj* Malloc(size_t size);
  
  // Free GPU memory and deregister
  void Free(SymmMemObj* obj);
  
  // Register existing memory for P2P access
  SymmMemObj* Register(void* ptr, size_t size);
  
  // Deregister memory
  void Deregister(SymmMemObj* obj);
  
 private:
  Bootstrap& bootstrap_;
  std::vector<SymmMemObj*> allocatedObjs_;
};

// Reduction operations
enum class ReduceOp {
  SUM,
  PROD,
  MIN,
  MAX,
  AVG
};

// AllReduce implementation
class AllReduce {
 public:
  AllReduce(Bootstrap& bootstrap, SymmMemManager& memManager);
  ~AllReduce();
  
  // Perform all-reduce operation
  // sendbuf: input buffer (GPU memory)
  // recvbuf: output buffer (GPU memory)
  // count: number of elements
  // datatype: HIP data type (e.g., hipFloat, hipInt, etc.)
  // op: reduction operation
  // stream: HIP stream for async execution
  void Execute(const void* sendbuf, void* recvbuf, size_t count,
               hipDataType datatype, ReduceOp op, hipStream_t stream = 0);
  
 private:
  Bootstrap& bootstrap_;
  SymmMemManager& memManager_;
  
  // Workspace for temporary buffers
  SymmMemObj* workspace_{nullptr};
  size_t workspaceSize_{0};
  
  // Algorithm selection threshold
  static constexpr size_t SMALL_MSG_THRESHOLD = 32 * 1024;  // 32KB
  
  // Ring-based all-reduce for large messages
  void RingAllReduce(const void* sendbuf, void* recvbuf, size_t count,
                     hipDataType datatype, ReduceOp op, hipStream_t stream);
  
  // Recursive doubling for small messages
  void RecursiveDoublingAllReduce(const void* sendbuf, void* recvbuf,
                                  size_t count, hipDataType datatype,
                                  ReduceOp op, hipStream_t stream);
  
  // Ensure workspace is allocated
  void EnsureWorkspace(size_t requiredSize);
};

// MPI-based bootstrap implementation
class MPIBootstrap : public Bootstrap {
 public:
  MPIBootstrap();
  ~MPIBootstrap() override;
  
  void Initialize() override;
  void Finalize() override;
  
  int GetWorldSize() const override { return worldSize_; }
  int GetRank() const override { return rank_; }
  
  void Allgather(const void* sendbuf, void* recvbuf, size_t sendcount) override;
  void Barrier() override;
  
 private:
  int worldSize_{0};
  int rank_{0};
  bool initialized_{false};
};

}  // namespace p2p_allreduce
