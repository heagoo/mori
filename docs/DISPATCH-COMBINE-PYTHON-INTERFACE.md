# MORI Dispatch/Combine Python Interface - Calling Stacks and Architecture

This document describes the calling stacks when using the Python interface for dispatch and combine operations in MORI, including buffer preparation, RDMA setup, and the complete data flow.

## Table of Contents

1. [Overview](#overview)
2. [Python Interface Calling Stack](#python-interface-calling-stack)
3. [Buffer Preparation](#buffer-preparation)
4. [RDMA Setup and Initialization](#rdma-setup-and-initialization)
5. [Dispatch Operation Flow](#dispatch-operation-flow)
6. [Combine Operation Flow](#combine-operation-flow)
7. [Memory Management](#memory-management)

## Overview

MORI (Modular RDMA Interface) provides high-performance dispatch and combine operations for MoE (Mixture of Experts) models. The Python interface wraps C++ implementations that leverage RDMA and GPU kernels for efficient inter-node and intra-node communication.

### Key Components

- **Python Layer**: `python/mori/ops/dispatch_combine.py`
- **PyBind11 Bindings**: `src/pybind/mori.cpp`
- **C++ Implementation**: `src/ops/dispatch_combine/dispatch_combine.cpp`
- **GPU Kernels**: `src/ops/dispatch_combine/intranode.hpp`, `src/ops/dispatch_combine/internode_v1.cpp`

## Python Interface Calling Stack

### 1. Initialization Flow

```
Python User Code
    ↓
mori.ops.EpDispatchCombineOp.__init__(config)
    ↓
mori.cpp._cpp_dispatch_combine_factory("EpDispatchCombineHandle")
    ↓
[C++] EpDispatchCombineHandle::EpDispatchCombineHandle(config)
    ↓
├── InitializeShmemBuf()           # Allocate symmetric memory buffers
├── InitializeTokenNumSignalBuf()  # Allocate signal buffers for token counts
├── InitializeOrderMapBuf()        # Allocate mapping buffers
└── InitializeBarrier()            # Allocate synchronization barriers
```

**Detailed Initialization Steps:**

1. **Python Layer** (`python/mori/ops/dispatch_combine.py`):
   ```python
   class EpDispatchCombineOp:
       def __init__(self, config):
           handle_class = _cpp_dispatch_combine_factory("EpDispatchCombineHandle")
           self._handle = handle_class(mori_cpp.EpDispatchCombineConfig(...))
   ```

2. **PyBind11 Layer** (`src/pybind/mori.cpp`):
   ```cpp
   pybind11::class_<mori::moe::EpDispatchCombineHandle>(m, "EpDispatchCombineHandle")
       .def(pybind11::init<mori::moe::EpDispatchCombineConfig>(), ...)
   ```

3. **C++ Constructor** (`src/ops/dispatch_combine/dispatch_combine.cpp`):
   ```cpp
   EpDispatchCombineHandle::EpDispatchCombineHandle(EpDispatchCombineConfig config)
   ```

### 2. Dispatch Operation Calling Stack

```
Python User Code
    ↓
EpDispatchCombineOp.dispatch(input, weights, scales, indices)
    ↓
[Python] mori.cpp.launch_dispatch(handle, kernel_type, input, weights, scales, topkIds)
    ↓
[C++/PyBind] LaunchDispatch() in src/pybind/mori.cpp
    ↓
├── handle.PrepareInference()      # Set input/output pointers and metadata
    ↓
└── handle.LaunchDispatch(kernelType, blockNum, warpPerBlock, stream)
    ↓
    [C++] EpDispatchCombineHandle::LaunchDispatch()
        ↓
        ├── GetEpDispatchCombineArgsByInputType()  # Create kernel arguments
        ↓
        └── Launch appropriate kernel based on KernelType:
            ├── EpDispatchIntraNodeKernel<<<grid, block, sharedMem, stream>>>(args)
            ├── EpDispatchInterNodeKernel<<<grid, block, sharedMem, stream>>>(args)
            ├── EpDispatchInterNodeV1Kernel<<<grid, block, sharedMem, stream>>>(args)
            └── EpDispatchInterNodeV1KernelLowLatency<<<grid, block, sharedMem, stream>>>(args)
```

**Return Path:**

```
[GPU Kernel Execution]
    ↓
[C++] Return torch::Tensor objects wrapping shmem buffers:
    ├── dispatch_output (from shmemDispatchOutTokMemObj)
    ├── dispatch_weights (from shmemDispatchOutWeightsMemObj)
    ├── dispatch_scales (from shmemOutScalesMemObj)
    ├── dispatch_indices (from shmemOutIndicesMemObj)
    └── totalRecvTokenNum (from totalRecvTokenNum)
    ↓
[Python] Receive tuple of tensors
```

### 3. Combine Operation Calling Stack

```
Python User Code
    ↓
EpDispatchCombineOp.combine(input, weights, indices)
    ↓
[Python] mori.cpp.launch_combine(handle, kernel_type, input, weights, topkIds)
    ↓
[C++/PyBind] LaunchCombine() in src/pybind/mori.cpp
    ↓
├── handle.PrepareInference()      # Set input/output pointers
    ↓
└── handle.LaunchCombine(kernelType, blockNum, warpPerBlock, stream)
    ↓
    [C++] EpDispatchCombineHandle::LaunchCombine()
        ↓
        ├── GetEpDispatchCombineArgsByInputType()  # Create kernel arguments
        ↓
        └── Launch appropriate kernel based on KernelType:
            ├── EpCombineIntraNodeKernel<<<grid, block, sharedMem, stream>>>(args)
            ├── EpCombineInterNodeKernel<<<grid, block, sharedMem, stream>>>(args)
            └── EpCombineInterNodeV1Kernel<<<grid, block, sharedMem, stream>>>(args)
```

**Return Path:**

```
[GPU Kernel Execution]
    ↓
[C++] Return torch::Tensor objects wrapping shmem buffers:
    ├── combine_output (from shmemCombineOutTokMemObj)
    └── combine_weights (from shmemCombineOutWeightsMemObj)
    ↓
[Python] Receive tuple of tensors
```

## Buffer Preparation

### Buffer Initialization During Handle Construction

The `EpDispatchCombineHandle` constructor allocates all necessary buffers upfront:

#### 1. Token Buffers (InitializeShmemBuf)

```cpp
void EpDispatchCombineHandle::InitializeShmemBuf() {
    // Calculate buffer sizes
    size_t maxTokenSize = MaxNumTokensToRecv() * hiddenDim * maxTokenTypeSize;
    size_t maxStagingTokSize = MaxNumTokensToRecv() * 
        (hiddenDim * maxTokenTypeSize + 
         (sizeof(float) + sizeof(index_t)) * numExpertPerToken +
         scaleDim * scaleTypeSize);
    
    // Allocate symmetric memory buffers using ShmemExtMallocWithFlags
    shmemDispatchInpTokMemObj = ShmemMallocAndReturnMemObjPtr(maxStagingTokSize, hipDeviceMallocUncached);
    shmemCombineInpTokMemObj = ShmemMallocAndReturnMemObjPtr(maxStagingTokSize, hipDeviceMallocUncached);
    shmemDispatchOutTokMemObj = ShmemMallocAndReturnMemObjPtr(maxTokenSize, hipDeviceMallocUncached);
    shmemCombineOutTokMemObj = ShmemMallocAndReturnMemObjPtr(maxTokenSize, hipDeviceMallocUncached);
    shmemStagingTokMemObj = ShmemMallocAndReturnMemObjPtr(maxStagingTokSize, hipDeviceMallocUncached);
}
```

**Buffer Types:**

- **shmemDispatchInpTokMemObj**: Input buffer for dispatch staging
- **shmemCombineInpTokMemObj**: Input buffer for combine staging
- **shmemDispatchOutTokMemObj**: Output buffer for dispatched tokens
- **shmemCombineOutTokMemObj**: Output buffer for combined tokens
- **shmemStagingTokMemObj**: Temporary staging buffer for transfers

#### 2. Weight and Scale Buffers

```cpp
// Weight buffers
size_t maxWeightSize = MaxNumTokensToRecv() * numExpertPerToken * sizeof(float);
shmemInpWeightsMemObj = ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);
shmemDispatchOutWeightsMemObj = ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);
shmemCombineOutWeightsMemObj = ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);

// Scale buffers (if enabled)
if (scaleDim > 0 && scaleTypeSize > 0) {
    size_t maxScaleSize = MaxNumTokensToRecv() * scaleDim * scaleTypeSize;
    shmemInpScalesMemObj = ShmemMallocAndReturnMemObjPtr(maxScaleSize, hipDeviceMallocUncached);
    shmemOutScalesMemObj = ShmemMallocAndReturnMemObjPtr(maxScaleSize, hipDeviceMallocUncached);
}
```

#### 3. Index Buffers

```cpp
size_t maxIndicesSize = MaxNumTokensToRecv() * numExpertPerToken * sizeof(index_t);
shmemInpIndicesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndicesSize, hipDeviceMallocUncached);
shmemOutIndicesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndicesSize, hipDeviceMallocUncached);
```

#### 4. Signal and Synchronization Buffers (InitializeTokenNumSignalBuf)

```cpp
void EpDispatchCombineHandle::InitializeTokenNumSignalBuf() {
    // Token count signals for inter-PE communication
    size_t tokenNumSignalSize = worldSize * sizeof(index_t) * 2;
    recvTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
    sendTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
    
    // Atomic signal buffer for synchronization
    sendAtomicSignalMemObj = ShmemMallocAndReturnMemObjPtr(
        (worldSize * 2) * sizeof(int64_t) * 2, hipDeviceMallocUncached);
    
    // Total received token counter
    hipMalloc(&totalRecvTokenNum, sizeof(index_t));
    
    // Inter-node token count signals
    size_t nodeTokenNumSignalSize = worldSize / gpuPerNode * sizeof(index_t);
    nodeRecvTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(nodeTokenNumSignalSize, hipDeviceMallocUncached);
}
```

#### 5. Mapping and Counter Buffers (InitializeOrderMapBuf)

```cpp
void EpDispatchCombineHandle::InitializeOrderMapBuf() {
    size_t maxNumOutToken = worldSize * maxNumInpTokenPerRank * numExpertPerRank;
    
    // Mapping buffers for token routing
    hipMalloc(&dispReceiverIdxMap, maxNumOutToken * sizeof(index_t));
    hipMalloc(&dispSenderIdxMap, maxNumOutToken * sizeof(index_t));
    hipMalloc(&destPeTokenIdxMap, maxNumOutToken * sizeof(index_t));
    hipMalloc(&srcPeTokenIdxMap, maxNumOutToken * sizeof(index_t));
    
    // Counter buffers
    hipMalloc(&destPeTokenCounter, worldSize * sizeof(index_t));
    hipMalloc(&destNodeTokenCounter, (worldSize / gpuPerNode) * sizeof(index_t));
    hipMalloc(&localPeTokenCounter, worldSize * sizeof(index_t));
    
    // Intra-node specific buffers
    dispTokOffsetMemObj = ShmemMallocAndReturnMemObjPtr(sizeof(index_t), hipDeviceMallocUncached);
    dispTokIdToSrcTokIdMemObj = ShmemMallocAndReturnMemObjPtr(maxNumOutToken * sizeof(index_t), hipDeviceMallocUncached);
    hipMalloc(&dispDestTokIdMap, maxNumOutToken * sizeof(index_t));
    
    // Inter-node specific buffers
    size_t maxNumInterNodeToken = (worldSize / gpuPerNode) * maxNumInpTokenPerRank * numExpertPerToken;
    hipMalloc(&interNodeDispDestTokIdMap, maxNumInterNodeToken * sizeof(index_t));
    hipMalloc(&blockFlagCounter, (worldSize / gpuPerNode) * sizeof(index_t));
}
```

#### 6. Barrier Buffers (InitializeBarrier)

```cpp
void EpDispatchCombineHandle::InitializeBarrier() {
    size_t barrierSize = worldSize * sizeof(uint32_t);
    
    // Grid-level barriers for kernel synchronization
    hipMalloc(&dispatchGridBarrier, barrierSize);
    hipMalloc(&combineGridBarrier, barrierSize);
    
    // Cross-device barrier
    hipMalloc(&crossDeviceBarrierFlag, sizeof(uint32_t));
    crossDeviceBarrierMemObj = ShmemMallocAndReturnMemObjPtr(
        barrierSize * 2 * sizeof(uint64_t) / sizeof(uint32_t), hipDeviceMallocUncached);
    
    // Inter-node chunk flags for RDMA transfer tracking
    size_t interNodeChunkFlagSize = (worldSize / gpuPerNode) * MaxNumTokensToRecvPerRank() * sizeof(uint64_t);
    interNodeChunkFlagMemObj = ShmemMallocAndReturnMemObjPtr(interNodeChunkFlagSize, hipDeviceMallocUncached);
    hipMalloc(&interNodeChunkFlagCombine, interNodeChunkFlagSize);
    
    // Barrier for RDMA blocks
    hipMalloc(&interNodeBlocksBarrier, sizeof(index_t));
}
```

### Runtime Buffer Setup (PrepareInference)

Before each dispatch or combine operation, `PrepareInference` sets up the input/output pointers:

```cpp
void PrepareInference(hipDataType inputType, void* input, void* output, 
                     float* weights, uint8_t* scales, index_t* tokenIndices, 
                     index_t numToken) {
    this->inputType = inputType;
    this->inpTokenBuf = input;         // User-provided input buffer
    this->outTokenBuf = output;         // User-provided output buffer (or nullptr)
    this->weightsBuf = weights;         // Weight buffer
    this->scalesBuf = scales;           // Scale buffer (optional)
    this->tokenIndices = tokenIndices;  // Expert routing indices
    this->curRankNumToken = numToken;   // Number of tokens on this rank
}
```

## RDMA Setup and Initialization

RDMA setup occurs during the shmem initialization phase, which happens before creating the dispatch/combine handle.

### RDMA Initialization Flow

```
Python: mori.shmem.shmem_torch_process_group_init("default")
    ↓
[C++] ShmemTorchProcessGroupInit(groupName)
    ↓
ShmemInit(new TorchBootstrapNetwork(groupName))
    ↓
├── BootstrapNetwork::Initialize()     # Exchange rank information
    ↓
├── RdmaStatesInit()
    │   └── Create application::Context
    │       └── Discover and initialize RDMA devices
    ↓
├── MemoryStatesInit()
    │   ├── Create SymmMemManager (manages symmetric memory)
    │   └── Create RdmaMemoryRegionManager (manages RDMA MRs)
    ↓
└── GpuStateInit()
        ├── Copy transport types to GPU constant memory
        ├── Copy RDMA endpoints to GPU
        └── Initialize endpoint locks
```

### Detailed RDMA Setup Steps

#### 1. Bootstrap Network Initialization

```cpp
// src/shmem/init.cpp
int ShmemTorchProcessGroupInit(const std::string& groupName) {
    return ShmemInit(new application::TorchBootstrapNetwork(groupName));
}

int ShmemInit(application::BootstrapNetwork* bootNet) {
    ShmemStates* states = ShmemStatesSingleton::GetInstance();
    
    states->bootStates = new BootStates();
    states->bootStates->bootNet = bootNet;
    states->bootStates->bootNet->Initialize();  // Exchange rank/size info via Torch ProcessGroup
    states->bootStates->rank = bootNet->GetLocalRank();
    states->bootStates->worldSize = bootNet->GetWorldSize();
    
    RdmaStatesInit();      // Initialize RDMA devices and contexts
    MemoryStatesInit();    // Initialize memory management
    GpuStateInit();        // Copy RDMA info to GPU
}
```

#### 2. RDMA Device Discovery and Context Creation

```cpp
void RdmaStatesInit() {
    ShmemStates* states = ShmemStatesSingleton::GetInstance();
    states->rdmaStates = new RdmaStates();
    
    // Create communication context - this discovers RDMA devices
    states->rdmaStates->commContext = new application::Context(*states->bootStates->bootNet);
    
    // Inside Context constructor:
    // 1. Query available RDMA devices (filtered by MORI_RDMA_DEVICES env var)
    // 2. Create queue pairs (QPs) for each peer
    // 3. Exchange QP information via bootstrap network
    // 4. Connect QPs to enable RDMA operations
}
```

**RDMA Device Selection:**

The `MORI_RDMA_DEVICES` environment variable controls which devices are used:
- Not set: Use all available RDMA devices
- `MORI_RDMA_DEVICES=mlx5_0,mlx5_1`: Use specific devices
- `MORI_RDMA_DEVICES=^mlx5_2,mlx5_3`: Exclude specific devices (use `^` prefix)

#### 3. Memory Region Management Setup

```cpp
void MemoryStatesInit() {
    ShmemStates* states = ShmemStatesSingleton::GetInstance();
    application::Context* context = states->rdmaStates->commContext;
    
    // Create symmetric memory manager
    states->memoryStates->symmMemMgr = 
        new application::SymmMemManager(*states->bootStates->bootNet, *context);
    
    // Create RDMA memory region manager
    states->memoryStates->mrMgr = 
        new application::RdmaMemoryRegionManager(*context->GetRdmaDeviceContext());
}
```

**Symmetric Memory Manager:**
- Manages memory that is accessible from all PEs (processing elements)
- Coordinates memory allocation and registration across nodes
- Ensures consistent addressing across the cluster

**Memory Region Manager:**
- Registers GPU memory with RDMA devices
- Creates Memory Regions (MRs) for RDMA access
- Manages lkey/rkey for local and remote access

#### 4. GPU State Initialization

```cpp
void GpuStateInit() {
    ShmemStates* states = ShmemStatesSingleton::GetInstance();
    RdmaStates* rdmaStates = states->rdmaStates;
    
    GpuStates gpuStates;
    gpuStates.rank = states->bootStates->rank;
    gpuStates.worldSize = states->bootStates->worldSize;
    gpuStates.numQpPerPe = rdmaStates->commContext->GetNumQpPerPe();
    
    // Copy transport types (XGMI vs RDMA) to GPU
    hipMalloc(&gpuStates.transportTypes, sizeof(application::TransportType) * worldSize);
    hipMemcpy(gpuStates.transportTypes, rdmaStates->commContext->GetTransportTypes().data(),
              sizeof(application::TransportType) * worldSize, hipMemcpyHostToDevice);
    
    // Copy RDMA endpoints to GPU constant memory
    if (rdmaStates->commContext->RdmaTransportEnabled()) {
        size_t numEndpoints = gpuStates.worldSize * gpuStates.numQpPerPe;
        hipMalloc(&gpuStates.rdmaEndpoints, sizeof(application::RdmaEndpoint) * numEndpoints);
        hipMemcpy(gpuStates.rdmaEndpoints, rdmaStates->commContext->GetRdmaEndpoints().data(),
                  sizeof(application::RdmaEndpoint) * numEndpoints, hipMemcpyHostToDevice);
        
        // Create locks for endpoint access
        hipMalloc(&gpuStates.endpointLock, numEndpoints * sizeof(uint32_t));
        hipMemset(gpuStates.endpointLock, 0, numEndpoints * sizeof(uint32_t));
    }
    
    // Copy to GPU constant memory
    hipMemcpyToSymbol(globalGpuStates, &gpuStates, sizeof(GpuStates), 0, hipMemcpyHostToDevice);
}
```

### Symmetric Memory Allocation

When buffers are allocated using `ShmemMallocAndReturnMemObjPtr`:

```cpp
mori::application::SymmMemObjPtr ShmemMallocAndReturnMemObjPtr(size_t size, unsigned int flags) {
    // 1. Allocate GPU memory with specified flags (e.g., hipDeviceMallocUncached)
    void* buf = ShmemExtMallocWithFlags(size, flags);
    
    // 2. Zero-initialize the buffer
    hipMemset(buf, 0, size);
    
    // 3. Register with symmetric memory manager
    mori::application::SymmMemObjPtr obj = ShmemQueryMemObjPtr(buf);
    
    // The SymmMemObjPtr contains:
    // - localPtr: local GPU pointer
    // - Get(): method to get remote pointer for specified PE
    // - Memory region information for RDMA access
    
    return obj;
}
```

**SymmMemObjPtr Structure:**
- Contains local and remote addressing information
- Allows kernel to access memory on remote PEs
- Encapsulates RDMA memory region keys

## Dispatch Operation Flow

### High-Level Dispatch Flow

```
1. Python calls op.dispatch(input, weights, scales, indices)
2. PyBind layer wraps tensors and calls C++ LaunchDispatch
3. PrepareInference sets up buffer pointers
4. Appropriate dispatch kernel is launched based on kernel_type
5. Kernel performs:
   a. Token routing computation
   b. Data transfer (XGMI for intra-node, RDMA for inter-node)
   c. Synchronization
6. Output tensors are created from shmem buffers
7. Returns to Python
```

### Dispatch Kernel Selection

Based on `kernel_type`:

- **IntraNode**: Uses XGMI for single-node communication
  - Kernel: `EpDispatchIntraNodeKernel`
  - Uses GPU-GPU peer access within node
  
- **InterNode**: Basic inter-node version
  - Kernel: `EpDispatchInterNodeKernel`
  - Uses RDMA for cross-node transfers
  
- **InterNodeV1**: Optimized inter-node version
  - Kernel: `EpDispatchInterNodeV1Kernel`
  - Better RDMA utilization
  
- **InterNodeV1LL**: Low-latency inter-node version
  - Kernel: `EpDispatchInterNodeV1KernelLowLatency`
  - Optimized for small batch sizes

### Dispatch Kernel Execution (Conceptual)

```cpp
template <typename T>
__global__ void EpDispatchKernel(EpDispatchCombineArgs<T> args) {
    // Phase 1: Compute token routing
    //   - Each thread block processes tokens
    //   - Determine destination PE for each token based on expert indices
    //   - Update routing maps and counters
    
    // Phase 2: Signal token counts
    //   - Notify destination PEs about incoming token counts
    //   - Use shmem_put or RDMA operations for inter-node
    
    // Phase 3: Transfer tokens
    //   - Copy tokens to destination PE buffers
    //   - For intra-node: direct GPU memory copy via XGMI
    //   - For inter-node: RDMA write operations
    
    // Phase 4: Synchronization
    //   - Wait for all transfers to complete
    //   - Use barriers or flags for synchronization
}
```

## Combine Operation Flow

### High-Level Combine Flow

```
1. Python calls op.combine(input, weights, indices)
2. PyBind layer wraps tensors and calls C++ LaunchCombine
3. PrepareInference sets up buffer pointers
4. Appropriate combine kernel is launched based on kernel_type
5. Kernel performs:
   a. Gather tokens from dispatch output
   b. Accumulate tokens belonging to same source token
   c. Apply weights
6. Output tensors are created from shmem buffers
7. Returns to Python
```

### Combine Kernel Selection

Based on `kernel_type`:

- **IntraNode**: `EpCombineIntraNodeKernel`
- **InterNode**: `EpCombineInterNodeKernel`
- **InterNodeV1/V1LL**: `EpCombineInterNodeV1Kernel`

**Note**: Unlike dispatch kernels where InterNodeV1 and InterNodeV1LL use different implementations (`EpDispatchInterNodeV1Kernel` vs `EpDispatchInterNodeV1KernelLowLatency`), the combine operation uses the same kernel (`EpCombineInterNodeV1Kernel`) for both V1 and V1LL. This is because the combine phase has less performance variation between configurations, and the low-latency optimizations are primarily needed in the dispatch phase.

### Combine Kernel Execution (Conceptual)

```cpp
template <typename T>
__global__ void EpCombineKernel(EpDispatchCombineArgs<T> args) {
    // Phase 1: Identify source tokens
    //   - Map output tokens back to original input tokens
    //   - Group tokens by source token ID
    
    // Phase 2: Accumulate tokens
    //   - For each source token, accumulate all expert outputs
    //   - Apply routing weights
    
    // Phase 3: Write results
    //   - Write accumulated results to output buffer
}
```

## Memory Management

### Buffer Lifecycle

1. **Allocation**: During `EpDispatchCombineHandle` constructor
   - All buffers allocated upfront
   - Registered with RDMA devices
   - Symmetric memory coordination across nodes

2. **Usage**: During dispatch/combine operations
   - Buffers reused across multiple operations
   - No per-operation allocation overhead

3. **Deallocation**: During `EpDispatchCombineHandle` destructor
   - All buffers freed
   - Memory regions deregistered from RDMA

### Zero-Copy Design

The Python interface returns tensors that directly wrap C++ symmetric memory buffers:

```cpp
torch::Tensor out = torch::from_blob(
    handle.shmemDispatchOutTokMemObj->Get(),
    {handle.config.MaxNumTokensToRecv(), handle.config.hiddenDim},
    torch::TensorOptions().dtype(input.scalar_type()).device(torch::kCUDA)
);
```

**Benefits:**
- No data copy between C++ and Python
- Tensors can be directly used in PyTorch operations
- Memory remains on GPU throughout

**Important Note:**
These tensors reference memory owned by the handle. The handle must remain alive while tensors are in use.

## Example Usage

### Complete Example

```python
import mori
import torch
import torch.distributed as dist

# 1. Initialize distributed environment
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
torch.cuda.set_device(rank)
dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)

# 2. Initialize MORI shmem (sets up RDMA)
world_group = torch.distributed.group.WORLD
torch._C._distributed_c10d._register_process_group("default", world_group)
mori.shmem.shmem_torch_process_group_init("default")

# 3. Configure dispatch/combine operation
config = mori.ops.EpDispatchCombineConfig(
    data_type=torch.bfloat16,
    rank=rank,
    world_size=world_size,
    hidden_dim=7168,
    scale_dim=0,
    scale_type_size=1,
    max_token_type_size=4,
    max_num_inp_token_per_rank=4096,
    num_experts_per_rank=32,
    num_experts_per_token=8,
    warp_num_per_block=8,
    block_num=80,
    use_external_inp_buf=False,
    kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode
)

# 4. Create dispatch/combine operator (allocates buffers)
op = mori.ops.EpDispatchCombineOp(config)

# 5. Prepare input data
num_tokens = 128
input_tokens = torch.randn(num_tokens, config.hidden_dim, dtype=torch.bfloat16, device=f"cuda:{rank}")
weights = torch.rand(num_tokens, config.num_experts_per_token, dtype=torch.float32, device=f"cuda:{rank}")
indices = torch.randint(0, config.num_experts_per_rank * config.world_size, 
                       (num_tokens, config.num_experts_per_token), dtype=torch.int32, device=f"cuda:{rank}")

# 6. Dispatch: route tokens to appropriate experts
dispatch_output, dispatch_weights, dispatch_scales, dispatch_indices, num_recv_tokens = \
    op.dispatch(input_tokens, weights, None, indices, block_num=80, warp_per_block=8)

# 7. Process tokens at experts (not shown)
# ...

# 8. Combine: aggregate expert outputs back to original tokens
combine_output, combine_weights = op.combine(dispatch_output, dispatch_weights, dispatch_indices, 
                                            block_num=80, warp_per_block=8, call_reset=False)

# 9. Cleanup
del op
mori.shmem.shmem_finalize()
dist.destroy_process_group()
```

## Performance Considerations

### Buffer Allocation Strategy

- **Pre-allocation**: All buffers allocated during initialization
  - Avoids per-operation allocation overhead
  - Enables buffer reuse across operations
  
- **Symmetric Memory**: Enables efficient RDMA access
  - Memory registered once at allocation
  - No per-transfer registration cost

### RDMA Optimization

- **Queue Pair per PE**: Multiple QPs for parallel transfers
- **Batched Transfers**: Group small transfers to reduce overhead
- **GPUDirect RDMA**: Direct GPU-to-GPU transfers without CPU involvement
- **Device Selection**: Filter devices via `MORI_RDMA_DEVICES` for optimal topology

### Kernel Configuration

- **Block Number**: Controls parallelism
  - Higher values: more parallelism, but more synchronization overhead
  - Default: 80 (tuned for MI300X)
  
- **Warps per Block**: Controls thread block size
  - Affects shared memory usage and occupancy
  - Default: 8 warps (256 threads)

## Debugging and Troubleshooting

### Common Issues

1. **RDMA Device Not Found**
   - Check `ibv_devices` output
   - Verify RDMA drivers are loaded
   - Check `MORI_RDMA_DEVICES` environment variable

2. **Memory Registration Failures**
   - Ensure sufficient locked memory limit (`ulimit -l`)
   - Check GPU memory availability
   - Verify RDMA device supports GPUDirect

3. **Synchronization Hangs**
   - Ensure all ranks call operations in same order
   - Check for mismatched world_size configurations
   - Verify network connectivity between nodes

### Environment Variables

- `MORI_RDMA_DEVICES`: Control RDMA device selection
- `GLOO_SOCKET_IFNAME`: Specify network interface for Gloo backend
- `HIP_VISIBLE_DEVICES`: Control GPU visibility

## References

- Main implementation: `src/ops/dispatch_combine/dispatch_combine.cpp`
- Python bindings: `src/pybind/mori.cpp`
- SHMEM initialization: `src/shmem/init.cpp`
- Kernel implementations: `src/ops/dispatch_combine/intranode.hpp`, `internode_v1.cpp`
