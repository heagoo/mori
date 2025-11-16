# GPU Data Interchange Example

This directory contains a minimal example demonstrating how to interchange data between GPUs using the MORI framework.

## Overview

This example (`simple_gpu_interchange.cpp`) shows the essential steps for setting up GPU-to-GPU data transfer using RDMA (Remote Direct Memory Access) with MORI. The example demonstrates:

1. **RDMA Context Initialization** - Setting up the RDMA backend
2. **Device Discovery** - Finding available RDMA devices and ports
3. **Endpoint Creation** - Creating communication endpoints for sender and receiver
4. **Connection Setup** - Establishing connections between endpoints
5. **Memory Registration** - Registering GPU buffers for RDMA operations
6. **GPU-Initiated Transfer** - Performing data transfer directly from GPU kernel
7. **Verification** - Validating the transferred data on the GPU

## Key Concepts

### RDMA (Remote Direct Memory Access)
RDMA allows direct memory access from the memory of one computer into that of another without involving either's operating system. This enables:
- **High throughput** with low CPU overhead
- **Low latency** for inter-GPU communication
- **GPU-initiated operations** (GPUDirect ASYNC - IBGDA)

### MORI Components Used

- **RdmaContext**: Manages the RDMA backend and device enumeration
- **RdmaDevice**: Represents a physical RDMA-capable device
- **RdmaDeviceContext**: Per-device context for RDMA operations
- **RdmaEndpoint**: Communication endpoint with queue pairs (QP) and completion queues (CQ)
- **RdmaMemoryRegion**: Registered memory region accessible by RDMA

### Data Flow

```
GPU1 (Sender)                                    GPU2 (Receiver)
┌──────────────┐                                ┌──────────────┐
│  Send Buffer │                                │  Recv Buffer │
│  (GPU mem)   │                                │  (GPU mem)   │
└──────┬───────┘                                └──────▲───────┘
       │                                               │
       │  1. Fill buffer with data                    │
       │     (GPU kernel)                             │
       │                                               │
       │  2. PostWrite (RDMA Write)                   │
       ├───────────────────────────────────────────────┤
       │  3. UpdateDoorbell & RingDoorbell            │
       │                                               │
       │  4. RDMA NIC transfers data directly         │
       │     to GPU2 memory (bypass CPU)              │
       │                                               │
       │  5. PollCq (wait for completion)             │
       │                                               │
       │  6. Verify data on GPU2                      │
       └───────────────────────────────────────────────┘
```

## Code Structure

### Essential Steps

1. **Initialize RDMA Context**
   ```cpp
   RdmaContext rdmaContext(RdmaBackendType::DirectVerbs);
   ```

2. **Get RDMA Devices and Ports**
   ```cpp
   RdmaDeviceList rdmaDevices = rdmaContext.GetRdmaDeviceList();
   ActiveDevicePortList activeDevicePortList = GetActiveDevicePortList(rdmaDevices);
   ```

3. **Create Device Contexts**
   ```cpp
   RdmaDeviceContext* deviceContextSend = device->CreateRdmaDeviceContext();
   RdmaDeviceContext* deviceContextRecv = device->CreateRdmaDeviceContext();
   ```

4. **Create and Configure Endpoints**
   ```cpp
   RdmaEndpointConfig config;
   config.onGpu = true;  // Enable GPU-initiated RDMA
   RdmaEndpoint epSend = deviceContextSend->CreateRdmaEndpoint(config);
   RdmaEndpoint epRecv = deviceContextRecv->CreateRdmaEndpoint(config);
   ```

5. **Connect Endpoints**
   ```cpp
   deviceContextSend->ConnectEndpoint(epSend.handle, epRecv.handle);
   deviceContextRecv->ConnectEndpoint(epRecv.handle, epSend.handle);
   ```

6. **Register GPU Buffers**
   ```cpp
   void* sendBuf;
   hipMalloc(&sendBuf, msgSize);
   RdmaMemoryRegion mrSend = deviceContextSend->RegisterRdmaMemoryRegion(
       sendBuf, msgSize, MR_ACCESS_FLAG);
   ```

7. **Perform GPU-Initiated Transfer**
   ```cpp
   // In GPU kernel:
   PostWrite<P>(epSend.wqHandle, epSend.handle.qpn, 
                sendMr.addr, sendMr.lkey, 
                recvMr.addr, recvMr.rkey, msgSize);
   ```

### GPU Kernel Operations

The example demonstrates GPU-initiated RDMA operations:

- **`PostWrite`**: Post an RDMA write operation
- **`UpdateSendDbrRecord`**: Update the doorbell record
- **`RingDoorbell`**: Ring the doorbell to notify hardware
- **`PollCq`**: Poll the completion queue
- **`UpdateCqDbrRecord`**: Update completion queue doorbell

These operations allow the GPU to directly initiate and manage RDMA transfers without CPU involvement.

## Building

This example is integrated into the MORI build system. To build:

```bash
cd /path/to/mori
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=ON
make simple_gpu_interchange
```

## Running

**Note**: This example requires:
- ROCm-capable AMD GPUs (or CUDA-capable NVIDIA GPUs)
- RDMA-capable network adapters (e.g., Mellanox ConnectX or Broadcom Thor)
- Proper RDMA drivers and libraries installed

To run the example:

```bash
./simple_gpu_interchange
```

Expected output:
```
=== Simple GPU Data Interchange Example ===
Transferring 1024 bytes with value 42

Step 1: Initializing RDMA context...
  Found X RDMA device(s)
Step 2: Getting RDMA devices...
  Found X active device port(s)
Step 3: Creating device contexts...
  Device contexts created
Step 4: Creating RDMA endpoints...
  Endpoints created (QPN: send=XXX, recv=XXX)
Step 5: Connecting endpoints...
  Endpoints connected
Step 6: Allocating and registering GPU buffers...
  Buffers allocated and registered
Step 7: Launching GPU kernel for data interchange...
Success! All 1024 bytes verified with value 42
  Kernel execution completed
Step 8: Cleaning up resources...
  Resources cleaned up

=== Example completed successfully! ===
```

## Understanding the Code

### Why Two Endpoints?

In this example, we create two endpoints (`epSend` and `epRecv`) to simulate communication between two different GPUs or processes. In a real distributed system:
- Each GPU/process would have its own endpoint
- Endpoints would be on different physical machines
- The RDMA network would connect them

### Memory Registration

GPU buffers must be registered with the RDMA subsystem before use:
```cpp
RdmaMemoryRegion mrSend = deviceContextSend->RegisterRdmaMemoryRegion(
    sendBuf, msgSize, MR_ACCESS_FLAG);
```

This registration:
- Pins the memory in physical address space
- Creates memory keys (lkey, rkey) for access control
- Enables the RDMA NIC to directly access GPU memory

### GPU-Initiated RDMA (GPUDirect ASYNC)

The key feature demonstrated here is **GPU-initiated RDMA**:
- RDMA operations are posted directly from GPU kernel code
- No CPU involvement in the data path
- Minimal latency for GPU-to-GPU transfers

This is achieved by setting `config.onGpu = true` and using device-side RDMA functions.

## Related Examples

For more complex scenarios, see:
- `examples/local_rdma_ops/` - Various RDMA operations (send/recv, write, atomic)
- `examples/shmem/` - SHMEM-style programming interface
- `examples/dist_rdma_ops/` - Distributed RDMA operations

## References

- MORI Documentation: See main README.md
- GPUDirect RDMA: https://docs.nvidia.com/cuda/gpudirect-rdma/
- InfiniBand Architecture: https://www.infinibandta.org/
