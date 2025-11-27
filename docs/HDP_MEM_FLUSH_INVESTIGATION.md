# HDP Memory Flush Code Investigation

## Summary

This document details the investigation into the following code pattern that was mentioned as existing in a "history version":

```cpp
HIP_RUNTIME_CHECK(hipDeviceGetAttribute((init*)&endpoint.gpuHdpReg, hipDeviceAttributeHdpMemFlushCntl, deviceId));
```

> **Note**: The cast `(init*)` in the above code appears to be a typo - it should likely be `(int*)` to properly cast the pointer for the device attribute query.

## Investigation Results

**Finding: This code pattern does not exist in the git history of the MORI repository.**

A comprehensive search was performed:
1. Unshallowed the repository to access the full git history (195+ commits)
2. Searched all branches for patterns including:
   - `hipDeviceAttributeHdpMemFlushCntl`
   - `gpuHdpReg`
   - `HdpMemFlush`
   - `Hdp` (any case)
3. Searched through all commit diffs and file contents

None of these patterns were found in the repository history.

## Current RDMA Endpoint Implementation

The MORI project uses `RdmaEndpoint` structures for RDMA communication. The current implementation is located in:

- **Header**: `include/mori/application/transport/rdma/rdma.hpp`
- **Core Types**: `include/mori/core/transport/rdma/primitives.hpp`
- **MLX5 Provider**: `src/application/transport/rdma/providers/mlx5/mlx5.cpp`
- **BNXT Provider**: `src/application/transport/rdma/providers/bnxt/bnxt.cpp`

The `RdmaEndpoint` structure contains:
- `RdmaEndpointHandle handle` - QPN, PSN, port information
- `WorkQueueHandle wqHandle` - Send/Receive queue addresses
- `CompletionQueueHandle cqHandle` - Completion queue addresses  
- `IBVerbsHandle ibvHandle` - IB Verbs structures
- `IbufHandle atomicIbuf` - Atomic operations internal buffer

**There is no `gpuHdpReg` field in the current `RdmaEndpoint` structure.**

## About HDP Memory Flush

### What is HDP?

HDP (Host Data Path) is an AMD GPU hardware component that manages memory coherence between the CPU and GPU. The HDP memory flush mechanism ensures that GPU writes are visible to the CPU and vice versa.

### hipDeviceAttributeHdpMemFlushCntl

`hipDeviceAttributeHdpMemFlushCntl` is a HIP device attribute that retrieves the address of the GPU's HDP memory flush control register. This would typically be used for:

1. **Low-level memory coherence control** - Manually triggering HDP flushes for fine-grained memory synchronization
2. **GPU-initiated RDMA operations** - When GPU kernels need to ensure memory visibility before or after RDMA operations
3. **Performance optimization** - Avoiding full memory barriers by using targeted HDP flushes

### Typical Usage Pattern

```cpp
// Hypothetical usage (not in this codebase)
int* hdpFlushReg;
hipDeviceGetAttribute((int*)&hdpFlushReg, hipDeviceAttributeHdpMemFlushCntl, deviceId);

// In a GPU kernel, trigger HDP flush by writing to this register
*hdpFlushReg = 1;
```

## Memory Coherence in MORI

The MORI project handles GPU-CPU memory coherence through:

1. **HIP Memory Registration**: Using `hipHostRegister` with appropriate flags
2. **Device Memory Allocation**: Using `hipMalloc` for GPU-side buffers
3. **UMEM Registration**: Using `mlx5dv_devx_umem_reg` for RDMA memory registration
4. **UAR (User Access Region)**: Hardware doorbell mechanisms for RDMA operations

Recent ROCm 7.0.2 compatibility fix (commit `ac88ae5`) modified the host registration flags:
```cpp
// Changed from:
uint32_t flag = hipHostRegisterPortable | hipHostRegisterMapped | hipHostRegisterIoMemory;
// To:
uint32_t flag = hipHostRegisterPortable | hipHostRegisterMapped;
```

## Conclusion

The code pattern `HIP_RUNTIME_CHECK(hipDeviceGetAttribute((init*)&endpoint.gpuHdpReg, hipDeviceAttributeHdpMemFlushCntl, deviceId))` was likely from:

1. A different project or codebase
2. An experimental branch that was never merged
3. Internal development code that was not committed to this repository

The MORI project does not currently use HDP memory flush control registers directly. Memory coherence is managed through standard HIP memory management APIs and RDMA-specific mechanisms.

## References

- [HIP Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [ROCm RDMA Documentation](https://rocm.docs.amd.com/projects/RDMA/en/latest/)
- MORI source code: `src/application/transport/rdma/providers/`
