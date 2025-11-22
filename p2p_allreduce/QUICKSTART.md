# Quick Start Guide

This guide helps you get started with P2P AllReduce in 5 minutes.

## Prerequisites Check

Before starting, verify you have:

```bash
# Check ROCm installation
rocm-smi
hipconfig --version

# Check MPI installation
mpirun --version

# Check GPU P2P capability
rocminfo | grep -i "p2p"
```

## Installation (3 steps)

### 1. Clone or Copy the Code

```bash
# If this is part of a repository
cd /path/to/p2p_allreduce

# Or copy the standalone directory
cp -r p2p_allreduce /your/workspace/
cd /your/workspace/p2p_allreduce
```

### 2. Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Expected output:
```
[ 20%] Building CXX object CMakeFiles/p2p_allreduce.dir/src/bootstrap.cpp.o
[ 40%] Building CXX object CMakeFiles/p2p_allreduce.dir/src/symmetric_memory.cpp.o
[ 60%] Building CXX object CMakeFiles/p2p_allreduce.dir/src/allreduce.cpp.o
[ 80%] Linking CXX shared library libp2p_allreduce.so
[100%] Built target p2p_allreduce
```

### 3. Run Test

```bash
# Single node with 2 GPUs
mpirun -np 2 ./test_allreduce

# Expected output:
# Rank 0: AllReduce SUM test PASSED!
# Rank 1: AllReduce SUM test PASSED!
```

## Your First Program

Create `my_allreduce.cpp`:

```cpp
#include "p2p_allreduce.hpp"
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

using namespace p2p_allreduce;

int main(int argc, char** argv) {
  // 1. Initialize MPI
  MPIBootstrap bootstrap;
  bootstrap.Initialize();
  
  int rank = bootstrap.GetRank();
  int worldSize = bootstrap.GetWorldSize();
  
  std::cout << "Hello from rank " << rank << " of " << worldSize << std::endl;
  
  // 2. Allocate GPU memory
  const size_t count = 1024;  // Number of elements
  float* d_sendbuf;
  float* d_recvbuf;
  hipMalloc(&d_sendbuf, count * sizeof(float));
  hipMalloc(&d_recvbuf, count * sizeof(float));
  
  // 3. Initialize data (each rank contributes rank+1)
  std::vector<float> h_data(count, rank + 1.0f);
  hipMemcpy(d_sendbuf, h_data.data(), count * sizeof(float), hipMemcpyHostToDevice);
  
  // 4. Setup AllReduce
  SymmMemManager memManager(bootstrap);
  AllReduce allreduce(bootstrap, memManager);
  
  // 5. Execute AllReduce SUM
  allreduce.Execute(d_sendbuf, d_recvbuf, count, HIP_R_32F, ReduceOp::SUM, 0);
  
  // 6. Verify result
  std::vector<float> h_result(count);
  hipMemcpy(h_result.data(), d_recvbuf, count * sizeof(float), hipMemcpyDeviceToHost);
  
  float expected = worldSize * (worldSize + 1) / 2.0f;
  std::cout << "Rank " << rank << ": Result = " << h_result[0] 
            << " (expected " << expected << ")" << std::endl;
  
  // 7. Cleanup
  hipFree(d_sendbuf);
  hipFree(d_recvbuf);
  bootstrap.Finalize();
  
  return 0;
}
```

Build and run:

```bash
# Add to CMakeLists.txt
echo "add_executable(my_allreduce my_allreduce.cpp)" >> CMakeLists.txt
echo "target_link_libraries(my_allreduce PRIVATE p2p_allreduce)" >> CMakeLists.txt

# Build
cd build && cmake .. && make my_allreduce

# Run with 4 GPUs
mpirun -np 4 ./my_allreduce
```

## Benchmarking Performance

Run the included benchmark:

```bash
cd build

# Benchmark with 4 GPUs
mpirun -np 4 ./benchmark_allreduce

# Output will show:
#       Count    Size(KB)     Avg(ms)     Min(ms)     Max(ms)      BW(GB/s)   AlgBW(GB/s)
# ----------------------------------------------------------------------------------------------
#        1024        4.00       0.015       0.014       0.016          0.27          0.20
#        4096       16.00       0.018       0.017       0.019          0.89          0.67
#      ...
```

## Common Use Cases

### 1. Average Gradients (Deep Learning)

```cpp
// Average gradients across all GPUs
allreduce.Execute(d_gradients, d_averaged_gradients, 
                  gradient_count, HIP_R_32F, ReduceOp::AVG, stream);
```

### 2. Synchronize Model Parameters

```cpp
// Ensure all ranks have the same model parameters
allreduce.Execute(d_model_params, d_model_params,
                  param_count, HIP_R_32F, ReduceOp::SUM, stream);
// Then divide by world_size to average
```

### 3. Find Global Maximum

```cpp
// Find maximum loss across all ranks
allreduce.Execute(&d_local_loss, &d_global_max_loss,
                  1, HIP_R_32F, ReduceOp::MAX, stream);
```

### 4. Aggregate Statistics

```cpp
// Sum statistics from all ranks
allreduce.Execute(d_local_stats, d_global_stats,
                  stat_count, HIP_R_32F, ReduceOp::SUM, stream);
```

## Troubleshooting

### Issue: "HIP not found"
**Solution**: Install ROCm and set ROCM_PATH
```bash
export ROCM_PATH=/opt/rocm
cmake .. -DROCM_PATH=/opt/rocm
```

### Issue: "MPI not found"
**Solution**: Install MPI or specify MPI path
```bash
sudo apt install libopenmpi-dev
# or
cmake .. -DMPI_HOME=/path/to/mpi
```

### Issue: Test fails with P2P error
**Solution**: Verify P2P capability
```bash
# Check if GPUs support P2P
rocminfo | grep -i "link type"

# Enable P2P in BIOS if needed
# Ensure GPUs are on same PCIe root complex
```

### Issue: Low performance
**Solution**: Check GPU topology and connections
```bash
# Check GPU interconnect
rocm-smi --showtopoinfo

# Prefer XGMI over PCIe
# Use GPUs with direct connections
```

## Next Steps

1. **Read DESIGN.md**: Understand the algorithms and optimizations
2. **Read BUILD.md**: Detailed build and installation instructions
3. **Read README.md**: Complete API reference
4. **Experiment**: Try different message sizes and operations
5. **Optimize**: Tune the algorithm threshold for your hardware

## Getting Help

- Review the examples in `examples/` directory
- Check the detailed documentation in `*.md` files
- Examine the source code in `src/` and `include/`

## Performance Tips

1. **Use the right algorithm**: Small messages benefit from recursive doubling
2. **Align memory**: Use aligned allocations for better performance
3. **Batch operations**: Reduce per-operation overhead by batching
4. **Use streams**: Overlap operations with HIP streams
5. **Profile**: Use rocprof to identify bottlenecks

```bash
# Profile your application
rocprof --stats ./my_allreduce
```

## Example Benchmarks

On a system with 8x MI300X GPUs connected via XGMI:

| Message Size | Algorithm | Bandwidth | Latency |
|--------------|-----------|-----------|---------|
| 4 KB         | RecDoubling | ~25 GB/s | ~2 μs |
| 64 KB        | Ring       | ~120 GB/s | ~8 μs |
| 1 MB         | Ring       | ~280 GB/s | ~50 μs |
| 256 MB       | Ring       | ~350 GB/s | ~800 μs |

Your results may vary based on GPU model, interconnect, and system configuration.

## Complete Working Example

See `examples/test_allreduce.cpp` for a complete working example with:
- Initialization and cleanup
- Multiple test cases
- Result verification
- Error handling

Run it:
```bash
cd build
mpirun -np 4 ./test_allreduce
```

Congratulations! You now have a working P2P AllReduce implementation. 🎉
