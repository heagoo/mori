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
#include <cassert>
#include <cstdlib>
#include <cstring>

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

SymmMemManager::SymmMemManager(Bootstrap& bootstrap)
    : bootstrap_(bootstrap) {}

SymmMemManager::~SymmMemManager() {
  // Clean up all allocated objects
  for (auto obj : allocatedObjs_) {
    if (obj) {
      Deregister(obj);
      delete obj;
    }
  }
  allocatedObjs_.clear();
}

SymmMemObj* SymmMemManager::Malloc(size_t size) {
  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, size));
  HIP_CHECK(hipMemset(ptr, 0, size));
  
  SymmMemObj* obj = Register(ptr, size);
  allocatedObjs_.push_back(obj);
  
  return obj;
}

void SymmMemManager::Free(SymmMemObj* obj) {
  if (!obj) return;
  
  // Find and remove from allocated list
  auto it = std::find(allocatedObjs_.begin(), allocatedObjs_.end(), obj);
  if (it != allocatedObjs_.end()) {
    allocatedObjs_.erase(it);
  }
  
  // Free the GPU memory
  if (obj->localPtr) {
    HIP_CHECK(hipFree(obj->localPtr));
  }
  
  Deregister(obj);
  delete obj;
}

SymmMemObj* SymmMemManager::Register(void* ptr, size_t size) {
  int worldSize = bootstrap_.GetWorldSize();
  int rank = bootstrap_.GetRank();
  
  SymmMemObj* obj = new SymmMemObj();
  obj->localPtr = ptr;
  obj->size = size;
  obj->worldSize = worldSize;
  
  // Allocate arrays for peer information (host memory)
  obj->peerPtrs = static_cast<uintptr_t*>(calloc(worldSize, sizeof(uintptr_t)));
  obj->ipcHandles = static_cast<hipIpcMemHandle_t*>(
      calloc(worldSize, sizeof(hipIpcMemHandle_t)));
  
  // Get IPC handle for local memory
  hipIpcMemHandle_t handle;
  HIP_CHECK(hipIpcGetMemHandle(&handle, ptr));
  
  // Exchange IPC handles using bootstrap allgather
  bootstrap_.Allgather(&handle, obj->ipcHandles, sizeof(hipIpcMemHandle_t));
  
  // Open IPC handles for all peers
  for (int i = 0; i < worldSize; i++) {
    if (i == rank) {
      // Local pointer
      obj->peerPtrs[i] = reinterpret_cast<uintptr_t>(ptr);
    } else {
      // Open remote peer's memory
      void* peerPtr = nullptr;
      HIP_CHECK(hipIpcOpenMemHandle(&peerPtr, obj->ipcHandles[i],
                                    hipIpcMemLazyEnablePeerAccess));
      obj->peerPtrs[i] = reinterpret_cast<uintptr_t>(peerPtr);
    }
  }
  
  // Allocate device-side peer pointer array and copy data
  HIP_CHECK(hipMalloc(&obj->d_peerPtrs, worldSize * sizeof(uintptr_t)));
  HIP_CHECK(hipMemcpy(obj->d_peerPtrs, obj->peerPtrs, 
                      worldSize * sizeof(uintptr_t), hipMemcpyHostToDevice));
  
  return obj;
}

void SymmMemManager::Deregister(SymmMemObj* obj) {
  if (!obj) return;
  
  int worldSize = bootstrap_.GetWorldSize();
  int rank = bootstrap_.GetRank();
  
  // Free device-side peer pointer array
  if (obj->d_peerPtrs) {
    HIP_CHECK(hipFree(obj->d_peerPtrs));
    obj->d_peerPtrs = nullptr;
  }
  
  // Close IPC handles for all peers
  if (obj->peerPtrs) {
    for (int i = 0; i < worldSize; i++) {
      if (i != rank && obj->peerPtrs[i] != 0) {
        void* peerPtr = reinterpret_cast<void*>(obj->peerPtrs[i]);
        HIP_CHECK(hipIpcCloseMemHandle(peerPtr));
      }
    }
    free(obj->peerPtrs);
    obj->peerPtrs = nullptr;
  }
  
  if (obj->ipcHandles) {
    free(obj->ipcHandles);
    obj->ipcHandles = nullptr;
  }
}

}  // namespace p2p_allreduce
