# Technical Design Document: P2P AllReduce

## Overview

This document describes the implementation details, algorithms, and optimizations used in the P2P AllReduce library.

## Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────┐
│        Application Layer                │
│  (test_allreduce, benchmark_allreduce)  │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│         AllReduce Interface             │
│  - Algorithm Selection                  │
│  - Workspace Management                 │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│    Symmetric Memory Manager             │
│  - Memory Registration                  │
│  - P2P Handle Management                │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│         Bootstrap (MPI)                 │
│  - Initialization                       │
│  - Collective Operations                │
└─────────────────────────────────────────┘
```

## Symmetric Memory Management

### Concept

Symmetric memory is a key abstraction where each rank allocates memory of the same size, and all ranks exchange pointers to enable P2P access.

### Implementation

```
Rank 0          Rank 1          Rank 2
  │               │               │
  ├─ Malloc      ├─ Malloc      ├─ Malloc
  │  (size N)     │  (size N)     │  (size N)
  │               │               │
  ├─ Get IPC     ├─ Get IPC     ├─ Get IPC
  │  Handle       │  Handle       │  Handle
  │               │               │
  └──────────── Allgather ────────────┘
        Exchange IPC Handles
  ┌──────────────────────────────────┐
  │                                  │
  ├─ Open Peer  ├─ Open Peer  ├─ Open Peer
  │  Handles     │  Handles     │  Handles
  │               │               │
  ▼               ▼               ▼
[Local Ptr]    [Local Ptr]    [Local Ptr]
[Peer 0 Ptr]   [Peer 0 Ptr]   [Peer 0 Ptr]
[Peer 1 Ptr]   [Peer 1 Ptr]   [Peer 1 Ptr]
[Peer 2 Ptr]   [Peer 2 Ptr]   [Peer 2 Ptr]
```

### Key Functions

1. **hipIpcGetMemHandle**: Obtains IPC handle for local GPU memory
2. **hipIpcOpenMemHandle**: Opens remote GPU memory using IPC handle
3. **Direct Memory Access**: GPU can directly read/write remote GPU memory

## AllReduce Algorithms

### 1. Ring Algorithm (for large messages)

The ring algorithm is bandwidth-optimal for large messages, achieving near-theoretical peak bandwidth.

#### Phase 1: Reduce-Scatter

Data is divided into P chunks (where P = world size). In each step, each rank:
1. Reduces one chunk from its left neighbor
2. The result stays local for that chunk

```
Initial:     Rank 0: [A0, B0, C0, D0]
             Rank 1: [A1, B1, C1, D1]
             Rank 2: [A2, B2, C2, D2]
             Rank 3: [A3, B3, C3, D3]

Step 1:      Rank 0: [A0, B0, C0, D0+D3]
             Rank 1: [A1+A0, B1, C1, D1]
             Rank 2: [A2, B2+B1, C2, D2]
             Rank 3: [A3, B3, C3+C2, D3]

Step 2:      Rank 0: [A0, B0, C0+C3, D0+D3]
             Rank 1: [A1+A0, B1+B0, C1, D1]
             Rank 2: [A2, B2+B1, C2+C1, D2]
             Rank 3: [A3+A2, B3, C3+C2, D3]

Step 3:      Rank 0: [A0, B0+B3, C0+C3, D0+D3]
             Rank 1: [A1+A0, B1+B0, C1+C0, D1]
             Rank 2: [A2+A1, B2+B1, C2+C1, D2]
             Rank 3: [A3+A2, B3+B2, C3+C2, D3]

After P-1 steps, each rank has one fully reduced chunk.
```

#### Phase 2: AllGather

Each rank gathers the reduced chunks from all other ranks.

```
After Allgather:
All ranks: [A_sum, B_sum, C_sum, D_sum]
```

#### Complexity Analysis

- **Bandwidth**: 2 × (P-1)/P × N ≈ 2N for large P
- **Latency**: 2 × (P-1) × α where α is per-step latency
- **Optimal for**: Large messages where bandwidth dominates

### 2. Recursive Doubling Algorithm (for small messages)

Recursive doubling minimizes latency by reducing the number of steps.

#### Algorithm

In each step, ranks exchange data with a partner at distance 2^step:

```
Step 0: Distance = 1
  Rank 0 ↔ Rank 1
  Rank 2 ↔ Rank 3
  Rank 4 ↔ Rank 5
  ...

Step 1: Distance = 2
  Rank 0 ↔ Rank 2
  Rank 1 ↔ Rank 3
  Rank 4 ↔ Rank 6
  ...

Step 2: Distance = 4
  Rank 0 ↔ Rank 4
  Rank 1 ↔ Rank 5
  ...
```

#### Example (4 ranks)

```
Initial:  R0: [A0]    R1: [A1]    R2: [A2]    R3: [A3]

Step 0:   R0: [A0+A1] R1: [A0+A1] R2: [A2+A3] R3: [A2+A3]
          (0↔1, 2↔3)

Step 1:   R0: [SUM]   R1: [SUM]   R2: [SUM]   R3: [SUM]
          (0↔2, 1↔3)
```

#### Complexity Analysis

- **Bandwidth**: N × log(P)
- **Latency**: log(P) × α
- **Optimal for**: Small messages where latency dominates

### Algorithm Selection

The implementation automatically selects the algorithm based on message size:

```cpp
const size_t THRESHOLD = 32 * 1024;  // 32KB

if (message_size <= THRESHOLD) {
    RecursiveDoubling();  // Lower latency
} else {
    Ring();               // Higher bandwidth
}
```

This threshold is tunable based on hardware characteristics.

## Optimizations

### 1. Direct P2P Memory Access

**Benefit**: Zero-copy data transfer without CPU/host involvement

**Implementation**:
```cpp
// GPU kernel directly accesses peer memory
__global__ void kernel(SymmMemObj* workspace, int peer_rank) {
  T* peer_data = workspace->GetPeerAs<T>(peer_rank);
  // Direct access to peer GPU memory
  T value = peer_data[idx];
}
```

### 2. Vectorized Memory Operations

**Benefit**: Increased memory bandwidth utilization

**Implementation**:
```cpp
// Load/store 128-bit chunks instead of scalar
float4 vec = reinterpret_cast<float4*>(src)[i];
reinterpret_cast<float4*>(dst)[i] = vec;
```

This can provide up to 4× improvement for float data.

### 3. Memory Coalescing

**Benefit**: Efficient use of memory bandwidth

**Strategy**: 
- Threads in a warp access consecutive memory addresses
- Enables coalesced memory transactions
- Reduces memory latency

### 4. Overlapping Computation and Communication

For future optimization, computation can overlap with data transfer:

```
Pipeline Stage 1: Transfer chunk 0, Compute chunk N-1
Pipeline Stage 2: Transfer chunk 1, Compute chunk 0
Pipeline Stage 3: Transfer chunk 2, Compute chunk 1
...
```

### 5. Workspace Reuse

**Benefit**: Reduced memory allocation overhead

The AllReduce class maintains a workspace that is reused across calls:

```cpp
void EnsureWorkspace(size_t required_size) {
  if (workspace_size >= required_size) {
    return;  // Reuse existing
  }
  // Allocate new larger workspace
}
```

## Performance Characteristics

### Expected Performance

#### Ring Algorithm (8 GPUs, MI300X)

| Message Size | Bandwidth | Latency |
|--------------|-----------|---------|
| 1 MB         | ~200 GB/s | ~5 μs   |
| 16 MB        | ~300 GB/s | ~50 μs  |
| 256 MB       | ~350 GB/s | ~700 μs |

#### Recursive Doubling (8 GPUs)

| Message Size | Bandwidth | Latency |
|--------------|-----------|---------|
| 4 KB         | ~20 GB/s  | ~2 μs   |
| 16 KB        | ~40 GB/s  | ~3 μs   |
| 64 KB        | ~80 GB/s  | ~5 μs   |

### Bottlenecks and Limitations

1. **P2P Bandwidth**: Limited by interconnect (XGMI, PCIe)
2. **Synchronization**: Barrier synchronization adds latency
3. **Memory Alignment**: Misaligned access reduces performance
4. **GPU Topology**: Non-uniform topology affects performance

## Future Optimizations

### 1. Hierarchical Algorithms

For multi-node systems, combine intra-node and inter-node algorithms:

```
Level 1: P2P within node (fast)
Level 2: RDMA between nodes (slower)
```

### 2. Pipelining

Overlap data transfer with computation for better utilization.

### 3. Double Buffering

Use double buffering to hide memory allocation latency:

```
Buffer A: Processing current operation
Buffer B: Preparing for next operation
```

### 4. GPU Kernel Fusion

Fuse multiple small kernels to reduce launch overhead:

```
Single kernel:
  - Copy data
  - Reduce
  - Copy result
```

### 5. Compression

For low-precision workloads, compress data before transfer:

```
FP32 → FP16: 2× bandwidth improvement
FP32 → INT8: 4× bandwidth improvement
```

## Comparison with NCCL

| Feature              | P2P AllReduce | NCCL |
|----------------------|---------------|------|
| P2P Support          | Yes           | Yes  |
| Ring Algorithm       | Yes           | Yes  |
| Tree Algorithm       | No            | Yes  |
| Multi-node           | MPI-based     | Built-in |
| Compression          | No            | No   |
| Topology-aware       | Basic         | Advanced |

## Testing and Validation

### Correctness Tests

1. **Sum reduction**: Verify sum of rank values
2. **Max reduction**: Verify maximum rank value
3. **Min reduction**: Verify minimum rank value
4. **Average**: Verify average of rank values

### Performance Tests

1. **Latency benchmark**: Small message sizes
2. **Bandwidth benchmark**: Large message sizes
3. **Scaling test**: Verify performance scales with GPU count

### Example Test

```cpp
// Test: Sum reduction with 4 ranks
// Rank i sends value (i+1)
// Expected result: 1+2+3+4 = 10

float expected = 10.0f;
assert(result == expected);
```

## References

1. Thakur, R., Rabenseifner, R., & Gropp, W. (2005). "Optimization of Collective Communication Operations in MPICH"
2. AMD ROCm Documentation: https://rocm.docs.amd.com/
3. MPI Standard: https://www.mpi-forum.org/docs/

## Glossary

- **P2P**: Peer-to-Peer, direct GPU-to-GPU communication
- **IPC**: Inter-Process Communication
- **XGMI**: AMD's GPU interconnect technology
- **Ring Algorithm**: Bandwidth-optimal AllReduce algorithm
- **Recursive Doubling**: Latency-optimal AllReduce algorithm
- **Symmetric Memory**: Memory allocated and registered symmetrically across all ranks
