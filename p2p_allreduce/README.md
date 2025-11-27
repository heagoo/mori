# P2P AllReduce - Standalone Implementation

A high-performance, standalone AllReduce implementation using direct P2P (peer-to-peer) memory access via HIP IPC (Inter-Process Communication). This project is inspired by code patterns from the MORI project but is completely independent and self-contained.

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[BUILD.md](BUILD.md)** - Detailed build and installation instructions  
- **[DESIGN.md](DESIGN.md)** - Technical design, algorithms, and optimizations
- **README.md** (this file) - Overview and API reference

## Features

- **Direct P2P Memory Access**: Uses HIP IPC for zero-copy data transfer between GPUs
- **Multiple Algorithms**: 
  - Ring algorithm for large messages (bandwidth-optimal)
  - Recursive doubling for small messages (latency-optimal)
- **Optimized for AMD GPUs**: Built on ROCm/HIP
- **Multiple Reduction Operations**: SUM, PROD, MIN, MAX, AVG
- **MPI Bootstrap**: Uses MPI for initialization and coordination
- **Symmetric Memory Management**: Automatic P2P memory registration

## Architecture

### Components

1. **Bootstrap** (`bootstrap.cpp`): MPI-based initialization and collective operations
2. **Symmetric Memory Manager** (`symmetric_memory.cpp`): Manages GPU memory that is accessible across all processes via P2P
3. **AllReduce Implementation** (`allreduce.cpp`): Core reduction algorithms
4. **GPU Kernels** (`kernels.hpp`): Optimized HIP kernels for data movement and reduction

### Algorithms

#### Ring Algorithm (for large messages > 32KB)
- **Phase 1 - Reduce-Scatter**: Data is divided into chunks, each rank reduces one chunk
- **Phase 2 - AllGather**: Reduced chunks are gathered to all ranks
- **Complexity**: O(N) bandwidth, O(P) latency where N is data size, P is number of processes
- **Optimal for**: Large messages where bandwidth is the bottleneck

#### Recursive Doubling (for small messages ≤ 32KB)
- Ranks exchange data in log(P) steps with increasing distances
- **Complexity**: O(N*log(P)) bandwidth, O(log(P)) latency
- **Optimal for**: Small messages where latency is the bottleneck

### Key Optimizations

1. **Zero-Copy P2P Transfer**: Direct GPU-to-GPU memory access without host involvement
2. **Symmetric Memory**: Pre-registered memory regions for fast P2P access
3. **Algorithm Selection**: Automatic choice based on message size
4. **Asynchronous Execution**: Support for HIP streams
5. **Coalesced Memory Access**: Optimized memory access patterns in kernels

## Prerequisites

- ROCm 5.0 or later
- HIP runtime
- MPI implementation (OpenMPI, MPICH, etc.)
- CMake 3.19 or later
- C++17 compatible compiler

## Building

```bash
cd p2p_allreduce
mkdir build && cd build
cmake ..
make
```

## Usage

### Basic Example

```cpp
#include "p2p_allreduce.hpp"
#include <hip/hip_runtime.h>

using namespace p2p_allreduce;

int main(int argc, char** argv) {
  // Initialize MPI bootstrap
  MPIBootstrap bootstrap;
  bootstrap.Initialize();
  
  int rank = bootstrap.GetRank();
  int worldSize = bootstrap.GetWorldSize();
  
  // Allocate GPU memory
  float* d_sendbuf, *d_recvbuf;
  size_t count = 1024 * 1024;  // 1M elements
  hipMalloc(&d_sendbuf, count * sizeof(float));
  hipMalloc(&d_recvbuf, count * sizeof(float));
  
  // Initialize data (e.g., all ranks contribute rank+1)
  // ... (copy data to d_sendbuf)
  
  // Create memory manager and allreduce instance
  SymmMemManager memManager(bootstrap);
  AllReduce allreduce(bootstrap, memManager);
  
  // Execute AllReduce SUM
  allreduce.Execute(d_sendbuf, d_recvbuf, count, 
                    HIP_R_32F, ReduceOp::SUM, 0);
  
  // Result is now in d_recvbuf
  
  // Cleanup
  hipFree(d_sendbuf);
  hipFree(d_recvbuf);
  bootstrap.Finalize();
  
  return 0;
}
```

### Running

```bash
# Run with 4 processes (requires 4 GPUs with P2P capability)
mpirun -np 4 ./build/test_allreduce
```

## API Reference

### Bootstrap Interface

```cpp
class Bootstrap {
  virtual void Initialize() = 0;
  virtual void Finalize() = 0;
  virtual int GetWorldSize() const = 0;
  virtual int GetRank() const = 0;
  virtual void Allgather(const void* sendbuf, void* recvbuf, size_t sendcount) = 0;
  virtual void Barrier() = 0;
};
```

### SymmMemManager

```cpp
class SymmMemManager {
  SymmMemManager(Bootstrap& bootstrap);
  
  // Allocate GPU memory and register for P2P
  SymmMemObj* Malloc(size_t size);
  void Free(SymmMemObj* obj);
  
  // Register existing memory for P2P
  SymmMemObj* Register(void* ptr, size_t size);
  void Deregister(SymmMemObj* obj);
};
```

### AllReduce

```cpp
class AllReduce {
  AllReduce(Bootstrap& bootstrap, SymmMemManager& memManager);
  
  void Execute(const void* sendbuf, void* recvbuf, size_t count,
               hipDataType datatype, ReduceOp op, hipStream_t stream = 0);
};

enum class ReduceOp {
  SUM,   // Sum reduction
  PROD,  // Product reduction
  MIN,   // Minimum reduction
  MAX,   // Maximum reduction
  AVG    // Average reduction
};
```

### Supported Data Types

- `HIP_R_32F` - float (32-bit)
- `HIP_R_64F` - double (64-bit)
- `HIP_R_32I` - int32_t
- `HIP_R_64I` - int64_t

## Performance Considerations

1. **P2P Capability**: Ensure GPUs support P2P access (check with `hipDeviceCanAccessPeer`)
2. **Message Size**: The implementation automatically selects the best algorithm based on size
3. **GPU Topology**: Performance depends on GPU interconnect (XGMI, NVLink, PCIe)
4. **Memory Alignment**: Use aligned memory allocations for better performance

## Limitations

1. Requires MPI for bootstrap operations
2. All GPUs must support P2P access to each other
3. World size must be a power of 2 for optimal recursive doubling performance
4. Maximum tested with up to 8 GPUs

## Code Structure

```
p2p_allreduce/
├── include/
│   ├── p2p_allreduce.hpp  # Main API header
│   └── kernels.hpp         # GPU kernel declarations
├── src/
│   ├── bootstrap.cpp       # MPI bootstrap implementation
│   ├── symmetric_memory.cpp # Memory management
│   └── allreduce.cpp       # AllReduce algorithms
├── examples/
│   └── test_allreduce.cpp  # Example and test program
├── CMakeLists.txt          # Build configuration
└── README.md               # This file
```

## Design Principles

This implementation follows these key principles borrowed from MORI:

1. **Direct P2P Access**: Use HIP IPC for zero-copy GPU-to-GPU transfers
2. **Symmetric Memory**: All ranks allocate and register memory symmetrically
3. **Minimal Host Involvement**: Computation and data movement happen on GPU
4. **Modular Design**: Separate concerns (bootstrap, memory, algorithms)

## References

- MORI Project: https://github.com/ROCm/mori
- HIP Programming Guide: https://rocm.docs.amd.com/projects/HIP
- MPI Standard: https://www.mpi-forum.org/

## License

MIT License - See LICENSE file for details

## Contributing

This is a standalone project demonstrating P2P AllReduce implementation. Feel free to use and modify for your needs.

## Acknowledgments

Code patterns and design inspired by the MORI project (Modular RDMA Interface).
