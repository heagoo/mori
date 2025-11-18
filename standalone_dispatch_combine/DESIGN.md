# Design Document: Standalone Dispatch-Combine Implementation

## Overview

This document describes the design and implementation of a standalone dispatch-combine algorithm for expert parallelism (Mixture-of-Experts models) using P2P and RDMA communication.

## Architecture

### 1. Communication Model

#### Intra-Node Communication (P2P)
- **Mechanism**: HIP IPC (Inter-Process Communication)
- **Setup**:
  1. Each GPU allocates memory: `hipMalloc(&ptr, size)`
  2. Get IPC handle: `hipIpcGetMemHandle(&handle, ptr)`
  3. Exchange handles via MPI: `MPI_Allgather(handle, ...)`
  4. Open peer handles: `hipIpcOpenMemHandle(&peer_ptr, handle, flags)`
- **Access**: Direct GPU memory access through peer pointers
- **Advantages**: Zero-copy, low latency, high bandwidth

#### Inter-Node Communication (RDMA)
- **Mechanism**: GPU Direct RDMA (in full implementation)
- **Setup**:
  1. Register GPU memory with RDMA NIC: `ibv_reg_mr(pd, ptr, size, flags)`
  2. Extract lkey and rkey from memory region
  3. Exchange rkeys via MPI: `MPI_Allgather(rkey, ...)`
  4. Create queue pairs for each peer
- **Access**: GPU-initiated RDMA writes (requires vendor support)
- **Advantages**: Direct GPU-to-GPU across nodes, no CPU involvement

### 2. Symmetric Memory Management

All ranks allocate the same memory layout:
```
Rank 0: [local_ptr] -> accessible by all via peer_ptrs[0] or RDMA
Rank 1: [local_ptr] -> accessible by all via peer_ptrs[1] or RDMA
...
```

**Key Properties**:
- Every rank can access every other rank's memory
- Intra-node: via P2P pointers
- Inter-node: via RDMA with remote keys
- Symmetric allocation ensures deterministic addressing

### 3. Dispatch Algorithm

**Goal**: Route tokens to expert ranks based on expert assignments

**Steps**:
1. **Compute Routing**:
   - For each token, determine destination ranks based on expert IDs
   - Count tokens going to each destination
   - Compute offsets for writing

2. **Transfer Phase** (GPU Kernel):
   ```
   for each token:
       for each expert assigned to token:
           dest_rank = expert_id / num_experts_per_rank
           if same_node(dest_rank):
               write_to_p2p(peer_ptrs[dest_rank], token_data)
           else:
               write_via_rdma(dest_rank, token_data, rkey[dest_rank])
   ```

3. **Synchronization**:
   - GPU kernel completion ensures all writes finished
   - MPI barrier ensures global consistency (if needed)

### 4. Combine Algorithm

**Goal**: Aggregate expert outputs back to original token locations

**Steps**:
1. **Gather Phase** (GPU Kernel):
   ```
   for each original token:
       output = 0
       for each expert assigned to token:
           expert_output = read_from_dispatch_buffer(expert_rank, offset)
           weight = routing_weights[token, expert]
           output += weight * expert_output
       write_output(token, output)
   ```

2. **Normalization**:
   - Weights are normalized per token
   - Can be done on-the-fly during aggregation

## Optimizations

### 1. Vectorized Memory Access

**Problem**: Scalar loads/stores underutilize memory bandwidth

**Solution**: Use `float4` (128-bit) vectorized access
```cpp
float4* dest_vec = reinterpret_cast<float4*>(dest);
float4* src_vec = reinterpret_cast<float4*>(src);
for (int i = tid; i < size/4; i += stride) {
    dest_vec[i] = src_vec[i];  // 4x fewer transactions
}
```

**Benefits**:
- 4x reduction in memory transactions
- Better utilization of memory bus width
- Significant bandwidth improvement

### 2. Warp-Level Primitives

**Problem**: Shared memory has overhead and limited capacity

**Solution**: Use warp shuffle instructions
```cpp
__device__ void WarpReduce(float* val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        *val += __shfl_down(*val, offset);
    }
}
```

**Benefits**:
- No shared memory usage
- Lower latency than shared memory
- Better occupancy

### 3. Double Buffering

**Problem**: Sequential copy-then-send creates idle time

**Solution**: Overlap operations with staging buffers
```
Time:  [Copy to Buffer 0] [Send Buffer 0]
            [Copy to Buffer 1] [Send Buffer 1]
                [Copy to Buffer 0] [Send Buffer 0]
```

**Benefits**:
- Hides communication latency
- Better GPU utilization
- Higher effective bandwidth

### 4. Bank Conflict Avoidance

**Problem**: Shared memory bank conflicts reduce bandwidth

**Solution**: Add padding to shared memory layout
```cpp
// Without padding: 32-way bank conflict for stride-32 access
float shared[1024];

// With padding: no conflicts
float shared[1024 + 32];  // One extra element per warp
```

**Benefits**:
- Eliminates bank conflicts
- Full shared memory bandwidth
- Better latency for reductions

### 5. Fused Operations

**Problem**: Multiple kernel launches have overhead

**Solution**: Fuse normalization into combine kernel
```cpp
// Instead of:
//   Kernel1: normalize weights
//   Kernel2: combine with normalized weights
// Do:
__global__ void CombineFused(...) {
    float sum = compute_weight_sum();
    for (...) {
        output += (weight / sum) * expert_output;
    }
}
```

**Benefits**:
- Reduced kernel launch overhead
- Better data locality
- Fewer memory round-trips

### 6. All-to-All Pattern Optimization

**Problem**: Random scatter pattern has poor memory coalescing

**Solution**: Restructure as structured all-to-all
```cpp
// Group tokens by destination rank
for dest_rank in all_ranks:
    batch_transfer(tokens_for[dest_rank], dest_rank)
```

**Benefits**:
- Coalesced memory access
- Better batching for RDMA
- Predictable communication pattern

## GPU Memory Access Patterns

### Coalesced Access Pattern
```
Thread 0: access address 0
Thread 1: access address 1
Thread 2: access address 2
...
Thread 31: access address 31
--> Single 128-byte memory transaction
```

### Non-coalesced Access Pattern
```
Thread 0: access address 0
Thread 1: access address 32
Thread 2: access address 64
...
--> 32 separate memory transactions (32x slower!)
```

**Our Approach**: Structure all access patterns to be coalesced

## MPI Usage Pattern

### Initialization (Once per Process)
```cpp
int main(int argc, char** argv) {
    // Call MPI_Init ONCE at program start
    MPI_Init(&argc, &argv);
    
    // ... application logic ...
    
    // Call MPI_Finalize ONCE at program end
    MPI_Finalize();
}
```

### Why MPI_Init is Called Once
1. **Process Group**: Establishes communication between all ranks
2. **Resources**: Allocates network resources (expensive operation)
3. **Thread Safety**: May affect threading model
4. **Standard**: MPI standard mandates single init/finalize

### MPI in Our Implementation
- Used only for bootstrapping (exchange IPC handles, RDMA keys)
- Optional barrier for synchronization
- Not used in critical path (dispatch/combine are pure GPU operations)

## GPU Direct RDMA (Full Implementation)

### Requirements
1. **Hardware**: RDMA-capable NIC with GPU Direct support
2. **Driver**: GPU Direct RDMA kernel module (nvidia_peermem for NVIDIA, amd_peermem for AMD)
3. **Library**: libibverbs or vendor-specific (mlx5, bnxt)

### Setup Flow
```cpp
// 1. Open RDMA device
ibv_context* ctx = ibv_open_device(device);

// 2. Create protection domain
ibv_pd* pd = ibv_alloc_pd(ctx);

// 3. Register GPU memory
ibv_mr* mr = ibv_reg_mr(pd, gpu_ptr, size, 
                        IBV_ACCESS_LOCAL_WRITE | 
                        IBV_ACCESS_REMOTE_WRITE);

// 4. Create queue pair for communication
ibv_qp* qp = ibv_create_qp(pd, qp_init_attr);

// 5. Post RDMA write from GPU kernel
// (Requires vendor-specific extensions)
```

### GPU-Initiated RDMA (MLX5 Example)
```cpp
__global__ void DispatchRDMAKernel(...) {
    // Use MLX5 device memory API
    mlx5_post_rdma_write(qp_handle, 
                         local_gpu_addr,
                         remote_gpu_addr,
                         size,
                         rkey);
}
```

## Data Flow Example

### Scenario: 2 nodes, 4 GPUs per node, top-2 experts

```
Node 0:              Node 1:
+--------+          +--------+
| GPU 0  |<--P2P--->| GPU 1  |
| Token0 |          | Token1 |
+--------+          +--------+
    |                   |
    +-----RDMA----------+
    |                   |
+--------+          +--------+
| GPU 2  |<--P2P--->| GPU 3  |
| Token2 |          | Token3 |
+--------+          +--------+
```

**Token Routing**:
- Token0 -> Experts on GPU 0 (P2P, self) and GPU 2 (RDMA)
- Token1 -> Experts on GPU 1 (P2P, self) and GPU 3 (P2P)
- Token2 -> Experts on GPU 2 (P2P, self) and GPU 0 (RDMA)
- Token3 -> Experts on GPU 3 (P2P, self) and GPU 1 (P2P)

## Performance Considerations

### Bandwidth Estimates
- **P2P (XGMI)**: ~300-400 GB/s per link
- **RDMA (InfiniBand)**: ~50-100 GB/s per link
- **PCIe**: ~32 GB/s per direction

### Latency Estimates
- **P2P**: ~1-2 µs
- **RDMA**: ~1-5 µs
- **MPI** (for comparison): ~10-50 µs

### Optimization Priority
1. Use P2P for intra-node (highest bandwidth)
2. Use RDMA for inter-node (lower latency than TCP/MPI)
3. Batch transfers to amortize overhead
4. Overlap communication with computation

## Testing Strategy

### Unit Tests
- Symmetric memory allocation/deallocation
- P2P handle exchange
- RDMA key exchange
- Basic dispatch/combine correctness

### Integration Tests
- Multi-GPU same node
- Multi-GPU across nodes
- Various token distributions
- Different hidden dimensions

### Performance Tests
- Bandwidth benchmarks
- Latency measurements
- Scaling tests (weak and strong)
- Comparison with baseline (MPI, NCCL)

## Future Enhancements

1. **Dynamic Load Balancing**: Adapt to imbalanced expert assignments
2. **Pipelining**: Overlap dispatch and expert computation
3. **Compression**: Reduce data volume for inter-node transfers
4. **Quality of Service**: Priority queuing for latency-sensitive tokens
5. **Fault Tolerance**: Handle failed nodes or GPUs
6. **Multi-Stream**: Use multiple HIP streams for concurrency
7. **Persistent Kernels**: Keep kernels running to reduce launch overhead

## References

- HIP Programming Guide: https://rocm.docs.amd.com/
- RDMA/InfiniBand: https://www.openfabrics.org/
- GPU Direct: https://docs.nvidia.com/cuda/gpudirect-rdma/
- MPI Standard: https://www.mpi-forum.org/docs/
- MORI Project: https://github.com/heagoo/mori
