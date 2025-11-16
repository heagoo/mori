# MORI Code Structure Overview

This document provides an understanding of the current MORI codebase structure and key concepts for GPU-to-GPU data interchange.

## Architecture Overview

MORI (Modular RDMA Interface) is a framework for building high-performance GPU-to-GPU communication applications using RDMA (Remote Direct Memory Access). The architecture is designed around several key layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Bindings (Optional)               │
│                  (src/pybind, python/)                      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              High-Level Libraries & Operations              │
│  - MORI-EP: MoE dispatch/combine (src/ops)                 │
│  - MORI-IO: P2P communication (src/io)                     │
│  - MORI-CCL: Collective communication (src/shmem)          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│            Application Layer (src/application)              │
│  - RdmaContext: Device management                          │
│  - RdmaDevice: Physical device abstraction                 │
│  - RdmaEndpoint: Communication endpoints                   │
│  - Memory management: Registration & regions               │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              Core Layer (src/core, include/mori/core)       │
│  - Low-level RDMA operations (PostWrite, PostRecv, etc.)   │
│  - Device-side functions (GPU kernels)                     │
│  - Queue management (WQ, CQ handling)                      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│           Hardware Abstraction Layer                        │
│  - DirectVerbs (ibverbs): InfiniBand/RoCE                  │
│  - Provider-specific support (MLX5, BNXT)                  │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
mori/
├── include/mori/           # Public API headers
│   ├── application/        # High-level application APIs
│   ├── core/              # Core RDMA operations
│   ├── shmem/             # SHMEM-style programming interface
│   ├── ops/               # Specialized operations (dispatch/combine)
│   ├── io/                # I/O communication primitives
│   └── utils/             # Utility functions
│
├── src/                   # Implementation files
│   ├── application/       # Application layer implementation
│   ├── shmem/            # SHMEM implementation
│   ├── ops/              # Operation kernels
│   ├── io/               # I/O library implementation
│   └── pybind/           # Python bindings
│
├── examples/             # Example programs
│   ├── local_rdma_ops/   # Local RDMA operations
│   ├── dist_rdma_ops/    # Distributed RDMA operations
│   ├── shmem/            # SHMEM examples
│   ├── gpu_interchange/  # Simple GPU interchange (NEW)
│   └── benchmarks/       # Performance benchmarks
│
└── tests/                # Test suites
    ├── cpp/              # C++ tests
    └── python/           # Python tests
```

## Key Components

### 1. RdmaContext (Application Layer)

**Purpose**: Entry point for RDMA operations. Manages device discovery and initialization.

**Usage**:
```cpp
RdmaContext rdmaContext(RdmaBackendType::DirectVerbs);
RdmaDeviceList devices = rdmaContext.GetRdmaDeviceList();
```

**Key Methods**:
- `GetRdmaDeviceList()`: Enumerate available RDMA devices
- `nums_device`: Number of available RDMA devices

### 2. RdmaDevice (Application Layer)

**Purpose**: Represents a physical RDMA-capable device (e.g., InfiniBand HCA).

**Usage**:
```cpp
RdmaDevice* device = devicePort.first;
RdmaDeviceContext* ctx = device->CreateRdmaDeviceContext();
```

**Key Methods**:
- `CreateRdmaDeviceContext()`: Create a device context for operations

### 3. RdmaDeviceContext (Application Layer)

**Purpose**: Per-device context that manages endpoints and memory regions.

**Usage**:
```cpp
RdmaEndpoint ep = deviceContext->CreateRdmaEndpoint(config);
RdmaMemoryRegion mr = deviceContext->RegisterRdmaMemoryRegion(buf, size, flags);
```

**Key Methods**:
- `CreateRdmaEndpoint()`: Create communication endpoint
- `ConnectEndpoint()`: Establish connection between endpoints
- `RegisterRdmaMemoryRegion()`: Register GPU/CPU memory for RDMA
- `DeregisterRdmaMemoryRegion()`: Unregister memory

### 4. RdmaEndpoint (Application Layer)

**Purpose**: Communication endpoint with queue pairs (QP) and completion queues (CQ).

**Structure**:
```cpp
struct RdmaEndpoint {
    RdmaEndpointHandle handle;  // QP number, GID, etc.
    WqHandle wqHandle;          // Work Queue handle
    CqHandle cqHandle;          // Completion Queue handle
    // ...
};
```

**Components**:
- **QP (Queue Pair)**: Work queue for posting send/recv operations
- **CQ (Completion Queue)**: Queue for operation completions
- **Doorbell**: Hardware notification mechanism

### 5. RdmaMemoryRegion (Core Layer)

**Purpose**: Represents a registered memory region accessible by RDMA.

**Structure**:
```cpp
struct RdmaMemoryRegion {
    void* addr;      // Virtual address
    uint32_t lkey;   // Local key (for local access)
    uint32_t rkey;   // Remote key (for remote access)
    // ...
};
```

**Access Flags** (MR_ACCESS_FLAG):
- `IBV_ACCESS_LOCAL_WRITE`: Local write access
- `IBV_ACCESS_REMOTE_WRITE`: Remote write access
- `IBV_ACCESS_REMOTE_READ`: Remote read access
- `IBV_ACCESS_REMOTE_ATOMIC`: Remote atomic operations

## GPU Data Interchange Flow

### Step-by-Step Process

1. **Initialization**
   - Create RdmaContext
   - Discover RDMA devices and active ports
   - Create device contexts for sender and receiver

2. **Endpoint Setup**
   - Configure endpoint (max messages, CQ entries, GPU mode)
   - Create endpoints for sender and receiver
   - Connect endpoints (exchange QP info)

3. **Memory Registration**
   - Allocate GPU buffers using `hipMalloc` or `cudaMalloc`
   - Register buffers with RDMA subsystem
   - Get memory keys (lkey, rkey) for access control

4. **Data Transfer (GPU-Initiated)**
   - From GPU kernel, call RDMA primitives:
     ```cpp
     PostWrite<P>(epSend.wqHandle, qpn, local_addr, lkey, 
                  remote_addr, rkey, size);
     UpdateSendDbrRecord<P>(dbrRecAddr, postIdx);
     RingDoorbell<P>(dbrAddr, dbr_val);
     ```
   - Poll for completion:
     ```cpp
     PollCq<P>(cqAddr, cqeNum, &consIdx);
     UpdateCqDbrRecord<P>(dbrRecAddr, consIdx, cqeNum);
     ```

5. **Cleanup**
   - Deregister memory regions
   - Free GPU buffers
   - Clean up endpoints and contexts

### GPU-Initiated RDMA (IBGDA - InfiniBand GPUDirect Async)

**Key Feature**: GPU threads can directly post RDMA operations without CPU involvement.

**Requirements**:
- Set `config.onGpu = true` when creating endpoint
- Use device-side RDMA functions (PostWrite, PostRecv, etc.)
- Proper memory fences (`__threadfence_system()`)

**Benefits**:
- Ultra-low latency (no CPU in data path)
- High throughput for GPU-to-GPU transfers
- Efficient for small message sizes

## RDMA Operations

### Core Operations (include/mori/core/core.hpp)

1. **PostWrite**: RDMA Write operation
   ```cpp
   template <ProviderType P>
   __device__ uint64_t PostWrite(WqHandle& wqHandle, uint32_t qpn,
                                  void* localAddr, uint32_t lkey,
                                  void* remoteAddr, uint32_t rkey,
                                  size_t size);
   ```

2. **PostSend**: RDMA Send operation (requires matching Recv)
   ```cpp
   template <ProviderType P>
   __device__ uint64_t PostSend(WqHandle& wqHandle, uint32_t qpn,
                                 void* addr, uint32_t lkey, size_t size);
   ```

3. **PostRecv**: Post receive buffer
   ```cpp
   template <ProviderType P>
   __device__ uint64_t PostRecv(WqHandle& wqHandle, uint32_t qpn,
                                 void* addr, uint32_t lkey, size_t size);
   ```

4. **PollCq**: Poll completion queue
   ```cpp
   template <ProviderType P>
   __device__ int PollCq(void* cqAddr, uint32_t cqeNum, 
                         uint32_t* consIdx, uint16_t* wqeIdx);
   ```

### Higher-Level APIs (include/mori/shmem/)

MORI also provides SHMEM-style APIs that abstract lower-level RDMA operations:

```cpp
// SHMEM put operation (similar to MPI_Put)
ShmemPutMemNbi(dest, destOffset, source, sourceOffset, bytes, pe, qpId);

// Atomic operations
ShmemAtomicFetchAdd(dest, val, pe);
```

## Provider Support

MORI supports multiple RDMA providers:

1. **MLX5** (Mellanox/NVIDIA ConnectX-5/6/7)
   - Most common provider
   - Full feature support

2. **BNXT** (Broadcom Thor)
   - Compile with `-DUSE_BNXT=ON`
   - Requires bnxt_re library

**Provider Selection**: At runtime, the correct provider is selected based on the RDMA device.

## Transport Types

MORI supports different transport mechanisms:

1. **IBGDA** (InfiniBand GPUDirect Async)
   - GPU-initiated RDMA operations
   - Lowest latency

2. **P2P** (Peer-to-Peer)
   - Direct GPU-to-GPU memory access
   - Used for local GPU communication

3. **Standard RDMA**
   - CPU-initiated RDMA operations
   - Broader compatibility

## Build System

**CMake Configuration** (CMakeLists.txt):
```cmake
option(USE_ROCM "Whether to use ROCm" ON)      # AMD GPU support
option(USE_BNXT "Whether to use BNXT NIC" OFF) # Broadcom NIC
option(BUILD_EXAMPLES "Whether to build examples" ON)
```

**Building**:
```bash
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=ON
make
```

## Memory Synchronization

**Critical**: GPU-initiated RDMA requires proper memory fences:

```cpp
// Before RDMA operation
__threadfence_system();  // Flush writes to global memory

// After RDMA operation
__threadfence_system();  // Ensure RDMA completion is visible
```

## Common Patterns

### Pattern 1: Simple Write Transfer
1. Allocate and register GPU buffers
2. Fill send buffer on GPU
3. PostWrite from GPU
4. Poll for completion
5. Verify data on receiver

### Pattern 2: Send/Recv Transfer
1. Receiver posts Recv buffer
2. Sender posts Send
3. Both poll for completion
4. Data is matched by RDMA stack

### Pattern 3: Distributed Communication
1. Use MPI/GLOO for process coordination
2. Exchange endpoint information (QPN, GID, etc.)
3. Each process creates local endpoints
4. Connect endpoints across processes
5. Perform RDMA operations

## Performance Considerations

1. **Message Size**: Small messages benefit from IBGDA, large messages may prefer CPU-initiated
2. **Batching**: Batch multiple operations to amortize doorbell cost
3. **CU Count**: More CUs can hide RDMA latency
4. **Memory Alignment**: Align buffers to 4K boundaries for best performance

## Related Files for Reference

- **Core API**: `include/mori/core/core.hpp`
- **Application API**: `include/mori/application/application.hpp`
- **IBGDA Kernels**: `include/mori/shmem/shmem_ibgda_kernels.hpp`
- **P2P Kernels**: `include/mori/shmem/shmem_p2p_kernels.hpp`
- **Examples**: `examples/local_rdma_ops/`, `examples/shmem/`

## Next Steps

For developers wanting to use MORI:

1. **Start Simple**: Use the `examples/gpu_interchange/` example
2. **Understand RDMA Basics**: Read about Queue Pairs, Completion Queues, Memory Registration
3. **Explore Higher-Level APIs**: Look at SHMEM examples for easier programming
4. **Performance Tuning**: Use benchmarks to optimize for your use case

## Summary

MORI provides a modular framework for GPU-to-GPU communication via RDMA. The key innovation is **GPU-initiated RDMA** (IBGDA), which allows GPU threads to directly post RDMA operations without CPU involvement, achieving ultra-low latency. The framework is organized in layers from low-level RDMA primitives to high-level application APIs, making it accessible for both simple and complex use cases.
