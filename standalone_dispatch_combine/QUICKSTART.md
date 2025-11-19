# Quick Start Guide

## Prerequisites

Before building, ensure you have:

```bash
# Check ROCm/HIP installation
hipcc --version

# Check MPI installation  
mpicc --version

# Check CMake
cmake --version  # Requires 3.16+
```

## Build Instructions

### Quick Build
```bash
cd standalone_dispatch_combine
./build.sh
```

### Manual Build
```bash
cd standalone_dispatch_combine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Options
```bash
# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Custom install location
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install

# Verbose build
make VERBOSE=1
```

## Running Examples

### Single Node (8 GPUs)
```bash
cd build
mpirun -np 8 ./example
```

### Multi-Node (2 nodes, 8 GPUs each)
```bash
# Create hostfile
cat > hostfile << EOF
node1 slots=8
node2 slots=8
EOF

# Run
mpirun -np 16 -hostfile hostfile ./example
```

### With Custom Parameters
```bash
mpirun -np 8 ./example \
    4096 \  # hidden_dim
    128  \  # max_tokens_per_rank
    8    \  # num_experts_per_rank
    2       # num_experts_per_token
```

## Verifying Installation

### Test P2P Capability
```bash
# Check if GPUs support peer access
rocm-smi --showpids

# Verify P2P topology
rocm-smi --showtopo
```

### Test RDMA Capability (if available)
```bash
# Check RDMA devices
ibv_devices

# Check GPU Direct RDMA
ls /sys/kernel/mm/memory_peers/nv_mem/
```

## Troubleshooting

### Issue: "hipIpcGetMemHandle failed"
**Solution**: Ensure GPUs support P2P. Check with:
```bash
rocminfo | grep -i "large bar"
```

### Issue: "MPI not initialized"
**Solution**: Ensure `MPI_Init()` is called before creating context:
```cpp
MPI_Init(&argc, &argv);  // Must be first
DispatchCombineContext ctx(config);
ctx.Initialize();
```

### Issue: "Cannot open IPC memory handle"
**Solution**: 
1. Check GPU peer accessibility: `rocm-smi --showtopo`
2. Verify processes are on same node for P2P
3. Check memory limits: `ulimit -l` (should be unlimited)

### Issue: Poor Performance
**Solution**:
1. Check GPU clocks: `rocm-smi --showclocks`
2. Verify NUMA affinity: `numactl --hardware`
3. Enable GPU performance mode: `rocm-smi --setperflevel high`

## Environment Variables

### HIP Variables
```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Select GPUs
export HSA_ENABLE_SDMA=0  # Disable SDMA (may improve P2P)
```

### MPI Variables
```bash
export OMPI_MCA_btl_openib_allow_ib=1  # Enable InfiniBand
export OMPI_MCA_btl_openib_if_include=mlx5_0:1  # Specify IB device
```

### ROCm Variables
```bash
export HSA_FORCE_FINE_GRAIN_PCIE=1  # Fine-grain PCIe memory
export GPU_MAX_HW_QUEUES=8  # Increase GPU queue limit
```

## Performance Tuning

### For Latency
- Use fewer tokens per rank
- Reduce block size
- Use IntraNode kernel for local communication

### For Bandwidth
- Use more tokens per rank
- Increase block size
- Use vectorized kernels
- Enable all optimizations

### For Multi-Node
- Tune RDMA parameters
- Use appropriate MTU size
- Consider network topology

## Integration Guide

### Using in Your Application

```cpp
#include "dispatch_combine.hpp"

// 1. Initialize MPI once at program start
MPI_Init(&argc, &argv);

// 2. Create configuration
simple_dispatch_combine::Config config;
config.world_size = world_size;
config.rank = rank;
// ... set other parameters

// 3. Create context and handle
simple_dispatch_combine::DispatchCombineContext ctx(config);
ctx.Initialize();
simple_dispatch_combine::DispatchCombineHandle handle(&ctx);

// 4. In inference loop
handle.PrepareInference(tokens, expert_ids, weights, num_tokens);
handle.LaunchDispatch();
handle.LaunchCombine();
void* output = handle.GetOutputBuffer();

// 5. Cleanup
ctx.Finalize();
MPI_Finalize();
```

## Next Steps

- See README.md for detailed API documentation
- See DESIGN.md for architecture and optimization details
- Check examples/ directory for more code samples
- Modify kernels in src/ for custom optimizations
