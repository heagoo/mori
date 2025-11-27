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
#include <mpi.h>
#include <cassert>
#include <cstring>

namespace p2p_allreduce {

MPIBootstrap::MPIBootstrap() {}

MPIBootstrap::~MPIBootstrap() {
  if (initialized_ && shouldFinalize_) {
    Finalize();
  }
}

void MPIBootstrap::Initialize() {
  if (initialized_) return;
  
  // Check if MPI is already initialized by someone else
  int mpiInitialized = 0;
  MPI_Initialized(&mpiInitialized);
  
  if (!mpiInitialized) {
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    assert(provided >= MPI_THREAD_SERIALIZED);
    shouldFinalize_ = true;  // We initialized MPI, so we should finalize it
  } else {
    shouldFinalize_ = false;  // MPI was already initialized, don't finalize
  }
  
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  
  initialized_ = true;
}

void MPIBootstrap::Finalize() {
  if (!initialized_) return;
  
  // Only finalize MPI if we were the one who initialized it
  if (shouldFinalize_) {
    int mpiFinalized = 0;
    MPI_Finalized(&mpiFinalized);
    if (!mpiFinalized) {
      MPI_Finalize();
    }
  }
  
  initialized_ = false;
}

void MPIBootstrap::Allgather(const void* sendbuf, void* recvbuf, size_t sendcount) {
  MPI_Allgather(sendbuf, sendcount, MPI_BYTE,
                recvbuf, sendcount, MPI_BYTE,
                MPI_COMM_WORLD);
}

void MPIBootstrap::Barrier() {
  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace p2p_allreduce
