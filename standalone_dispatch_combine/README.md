# Standalone Dispatch-Combine Implementation

A simple, standalone implementation of the dispatch-combine algorithm for expert parallelism (MoE models), using P2P communication within nodes and RDMA across nodes.

## Features

- **P2P Communication**: Direct GPU-to-GPU memory access within the same node using HIP IPC (Inter-Process Communication)
- **RDMA Communication**: Remote Direct Memory Access for inter-node communication
- **GPU-Initiated Transfers**: All data movement happens directly from GPU kernels
- **MPI Bootstrap**: Uses MPI for initialization and coordination (called once per process)
- **Zero-Copy**: GPU kernels directly access peer GPU memory
- **Optimized Memory Access**: Data stays in GPU memory throughout dispatch and combine operations

## Architecture

### Dispatch Phase
1. Each rank has input tokens to process
2. Based on expert assignments, tokens are routed to appropriate ranks
3. **Intra-node**: Direct GPU memory writes using P2P (via HIP IPC handles)
4. **Inter-node**: GPU-initiated RDMA writes to remote GPU memory
5. All transfers happen in parallel from GPU kernels

### Combine Phase
1. Each rank receives expert outputs from multiple sources
2. For each original token, gather outputs from all assigned experts
3. Weighted aggregation using routing weights
4. Output written to local GPU memory

## Dependencies

- **ROCm/HIP**: For GPU programming (ROCm 5.0+)
- **MPI**: For process initialization and coordination (OpenMPI, MPICH, etc.)
- **CMake**: For building (3.16+)
- **C++17**: Compiler with C++17 support

Optional (for full RDMA functionality):
- **libibverbs**: For RDMA operations
- **RDMA-capable NICs**: InfiniBand, RoCE, etc.

## Building

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j

# Install (optional)
make install
```

## Usage

### Basic Example

```cpp
#include "dispatch_combine.hpp"

int main(int argc, char** argv) {
    // Initialize MPI (once per process)
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Configure
    simple_dispatch_combine::Config config;
    config.world_size = world_size;
    config.rank = rank;
    config.hidden_dim = 4096;
    config.max_tokens_per_rank = 128;
    config.num_experts_per_rank = 8;
    config.num_experts_per_token = 2;
    config.gpu_per_node = 8;
    
    // Create context and initialize
    simple_dispatch_combine::DispatchCombineContext ctx(config);
    ctx.Initialize();
    
    // Create handle
    simple_dispatch_combine::DispatchCombineHandle handle(&ctx);
    
    // Prepare input data (GPU pointers)
    // tokens: [num_tokens, hidden_dim]
    // expert_ids: [num_tokens, num_experts_per_token]
    // weights: [num_tokens, num_experts_per_token]
    handle.PrepareInference(d_tokens, d_expert_ids, d_weights, num_tokens);
    
    // Execute dispatch-combine
    handle.LaunchDispatch();  // Route tokens to experts
    handle.LaunchCombine();   // Aggregate expert outputs
    
    // Get output
    void* output = handle.GetOutputBuffer();
    
    // Cleanup
    ctx.Finalize();
    MPI_Finalize();
    
    return 0;
}
```

### Running the Example

```bash
# Single node with 8 GPUs
mpirun -np 8 ./example

# Multi-node (2 nodes, 8 GPUs each)
mpirun -np 16 -hostfile hostfile ./example

# With custom parameters
mpirun -np 8 ./example 4096 128 8 2
#                      ^    ^   ^ ^
#                      |    |   | expert_per_token
#                      |    |   expert_per_rank
#                      |    max_tokens_per_rank
#                      hidden_dim
```

## Implementation Details

### Memory Management

The implementation uses **symmetric memory** - memory allocated on each GPU that is accessible by all other GPUs:

- **Within Node (P2P)**: Uses `hipIpcGetMemHandle()` and `hipIpcOpenMemHandle()` to enable direct GPU-to-GPU access
- **Across Nodes (RDMA)**: Registers memory with RDMA subsystem (ibverbs) and exchanges remote keys

### GPU Kernels

All data movement is GPU-initiated:

1. **DispatchIntraNodeKernel**: Directly writes tokens to peer GPU memory using P2P pointers
2. **DispatchInterNodeKernel**: Stages data and initiates RDMA writes (in full implementation)
3. **CombineKernel**: Reads expert outputs and performs weighted aggregation

### Optimizations

- **Coalesced Memory Access**: Kernels are designed for optimal memory bandwidth
- **Parallel Execution**: Multiple tokens and experts processed simultaneously
- **Zero-Copy**: No intermediate CPU buffers or unnecessary copies
- **Batched Transfers**: Multiple tokens transferred together to amortize latency
- **Direct GPU Access**: All operations happen on GPU, minimizing CPU involvement

## Limitations and Future Work

This is a simplified implementation for educational purposes. A production-ready version would need:

1. **Full RDMA Support**: Integration with ibverbs or vendor-specific libraries (MLX5, BNXT)
2. **GPU-Initiated RDMA**: Use GPU Direct RDMA (GDR) and GPU-Direct Async (GDA)
3. **Error Handling**: Robust error checking and recovery
4. **Flow Control**: Handle congestion and rate limiting
5. **Memory Registration Caching**: Reuse registered memory regions
6. **Dynamic Topology**: Automatic detection of P2P vs RDMA paths
7. **Performance Tuning**: Kernel optimization for specific hardware
8. **Multi-GPU Support**: Handle multiple GPUs per process

## Code Structure

```
standalone_dispatch_combine/
├── include/
│   └── dispatch_combine.hpp     # Public API
├── src/
│   ├── dispatch_combine.cpp     # Core implementation
│   └── kernels.hip.cpp          # GPU kernels
├── examples/
│   └── example.cpp              # Usage example
├── CMakeLists.txt               # Build configuration
└── README.md                    # This file
```

## Comparison with MORI Project

This standalone implementation borrows key concepts from the MORI project:

- **Bootstrap Pattern**: Similar to `torch_bootstrap.cpp` for MPI initialization
- **Symmetric Memory**: Based on `RegisterSymmMemObj()` from `symmetric_memory.cpp`
- **Dispatch-Combine Algorithm**: Simplified version of MORI-EP operations

However, it is:
- **Independent**: No dependency on MORI libraries
- **Simpler**: Easier to understand and modify
- **Educational**: Focus on clarity over performance
- **Portable**: Can be adapted to different hardware/software stacks

## References

- **MORI Project**: https://github.com/heagoo/mori
- **HIP Programming Guide**: https://rocm.docs.amd.com/
- **MPI Standard**: https://www.mpi-forum.org/
- **InfiniBand/RDMA**: https://www.openfabrics.org/

## License

MIT License - See LICENSE file for details.

## Contributing

This is a minimal reference implementation. For production use, consider using the full MORI project or adapting this code for your specific needs.
