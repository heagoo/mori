// Simple standalone dispatch-combine implementation with P2P and RDMA
// Copyright (c) 2025. MIT License.

#include "dispatch_combine.hpp"
#include <cassert>
#include <cstring>
#include <iostream>

namespace simple_dispatch_combine {

// Helper macros for error checking
#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define MPI_CHECK(cmd) \
    do { \
        int error = (cmd); \
        if (error != MPI_SUCCESS) { \
            std::cerr << "MPI error: " << error \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

//==============================================================================
// DispatchCombineContext Implementation
//==============================================================================

DispatchCombineContext::DispatchCombineContext(const Config& cfg)
    : config_(cfg), initialized_(false), comm_(MPI_COMM_WORLD) {
}

DispatchCombineContext::~DispatchCombineContext() {
    if (initialized_) {
        Finalize();
    }
}

void DispatchCombineContext::Initialize() {
    if (initialized_) return;
    
    InitializeMPI();
    // InitializeRDMA(); // Simplified - would initialize ibverbs in full implementation
    
    initialized_ = true;
}

void DispatchCombineContext::InitializeMPI() {
    // Note: MPI_Init should be called once by the application before creating context
    // We just verify it's initialized
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        std::cerr << "Error: MPI must be initialized before creating DispatchCombineContext" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Get rank and size from MPI
    int rank, size;
    MPI_CHECK(MPI_Comm_rank(comm_, &rank));
    MPI_CHECK(MPI_Comm_size(comm_, &size));
    
    if (config_.rank != rank || config_.world_size != size) {
        std::cerr << "Warning: Config rank/size doesn't match MPI rank/size" << std::endl;
        config_.rank = rank;
        config_.world_size = size;
    }
    
    // Set GPU device based on local rank
    int local_rank = rank % config_.gpu_per_node;
    HIP_CHECK(hipSetDevice(local_rank));
}

void DispatchCombineContext::InitializeRDMA() {
    // In a full implementation, this would:
    // 1. Open RDMA device
    // 2. Create protection domain
    // 3. Create completion queues
    // 4. Create queue pairs for each peer
    // 5. Exchange QP information via MPI
    // 
    // For simplicity, we'll simulate this with placeholder structures
    rdma_ctx_.pd = nullptr;
    rdma_ctx_.cq = nullptr;
    rdma_ctx_.qps.resize(config_.world_size, nullptr);
}

void DispatchCombineContext::Finalize() {
    // Free all symmetric memory allocations
    for (auto* mem : allocations_) {
        if (mem) {
            if (mem->local_ptr) {
                HIP_CHECK(hipFree(mem->local_ptr));
            }
            delete mem;
        }
    }
    allocations_.clear();
    
    // Note: We don't call MPI_Finalize here - that's application's responsibility
    initialized_ = false;
}

bool DispatchCombineContext::IsSameNode(int peer_rank) const {
    // Two ranks are on same node if they share the same node ID
    int my_node = config_.rank / config_.gpu_per_node;
    int peer_node = peer_rank / config_.gpu_per_node;
    return my_node == peer_node;
}

SymmetricMemory* DispatchCombineContext::AllocateSymmetricMemory(size_t size) {
    SymmetricMemory* mem = new SymmetricMemory();
    mem->size = size;
    mem->peer_ptrs.resize(config_.world_size);
    mem->rkeys.resize(config_.world_size, 0);
    
    // Allocate GPU memory
    HIP_CHECK(hipMalloc(&mem->local_ptr, size));
    HIP_CHECK(hipMemset(mem->local_ptr, 0, size));
    
    // Exchange P2P handles with peers on same node
    ExchangeP2PHandles(mem);
    
    // Exchange RDMA keys with peers on different nodes
    ExchangeRDMAKeys(mem);
    
    allocations_.push_back(mem);
    return mem;
}

void DispatchCombineContext::ExchangeP2PHandles(SymmetricMemory* mem) {
    // First, exchange raw pointers via MPI Allgather
    uintptr_t local_ptr_val = reinterpret_cast<uintptr_t>(mem->local_ptr);
    MPI_CHECK(MPI_Allgather(&local_ptr_val, sizeof(uintptr_t), MPI_BYTE,
                           mem->peer_ptrs.data(), sizeof(uintptr_t), MPI_BYTE,
                           comm_));
    
    // For peers on the same node, open IPC memory handles
    hipIpcMemHandle_t local_handle;
    HIP_CHECK(hipIpcGetMemHandle(&local_handle, mem->local_ptr));
    
    // Exchange IPC handles
    std::vector<hipIpcMemHandle_t> all_handles(config_.world_size);
    MPI_CHECK(MPI_Allgather(&local_handle, sizeof(hipIpcMemHandle_t), MPI_BYTE,
                           all_handles.data(), sizeof(hipIpcMemHandle_t), MPI_BYTE,
                           comm_));
    
    // Open handles for peers on same node
    for (int i = 0; i < config_.world_size; i++) {
        if (i == config_.rank) continue;
        if (IsSameNode(i)) {
            void* peer_ptr = nullptr;
            HIP_CHECK(hipIpcOpenMemHandle(&peer_ptr, all_handles[i],
                                         hipIpcMemLazyEnablePeerAccess));
            mem->peer_ptrs[i] = reinterpret_cast<uintptr_t>(peer_ptr);
        }
    }
}

void DispatchCombineContext::ExchangeRDMAKeys(SymmetricMemory* mem) {
    // In a full implementation with ibverbs:
    // 1. Register memory with RDMA: ibv_reg_mr(pd, ptr, size, access_flags)
    // 2. Get lkey and rkey from the memory region
    // 3. Exchange rkeys via MPI Allgather
    //
    // For simplicity, we'll use placeholder values
    mem->lkey = config_.rank + 1; // Placeholder
    mem->rkeys[config_.rank] = mem->lkey;
    
    // Exchange rkeys
    MPI_CHECK(MPI_Allgather(&mem->lkey, sizeof(uint32_t), MPI_BYTE,
                           mem->rkeys.data(), sizeof(uint32_t), MPI_BYTE,
                           comm_));
}

void DispatchCombineContext::FreeSymmetricMemory(SymmetricMemory* mem) {
    if (!mem) return;
    
    // Close IPC handles for peers on same node
    for (int i = 0; i < config_.world_size; i++) {
        if (i == config_.rank) continue;
        if (IsSameNode(i) && mem->peer_ptrs[i]) {
            // Note: hipIpcCloseMemHandle is not always necessary with lazy peer access
            // but we include it for completeness
            void* peer_ptr = reinterpret_cast<void*>(mem->peer_ptrs[i]);
            HIP_CHECK(hipIpcCloseMemHandle(peer_ptr));
        }
    }
    
    // Free local memory
    if (mem->local_ptr) {
        HIP_CHECK(hipFree(mem->local_ptr));
    }
    
    // Remove from allocations list
    auto it = std::find(allocations_.begin(), allocations_.end(), mem);
    if (it != allocations_.end()) {
        allocations_.erase(it);
    }
    
    delete mem;
}

//==============================================================================
// DispatchCombineHandle Implementation
//==============================================================================

DispatchCombineHandle::DispatchCombineHandle(DispatchCombineContext* ctx)
    : ctx_(ctx), config_(ctx->GetConfig()),
      input_tokens_(nullptr), expert_ids_(nullptr), weights_(nullptr), num_tokens_(0),
      dispatch_buffer_(nullptr), output_buffer_(nullptr), token_counts_(nullptr),
      send_offsets_(nullptr), recv_offsets_(nullptr) {
    
    AllocateBuffers();
}

DispatchCombineHandle::~DispatchCombineHandle() {
    FreeBuffers();
}

void DispatchCombineHandle::AllocateBuffers() {
    int max_tokens = config_.max_tokens_per_rank;
    int hidden_dim = config_.hidden_dim;
    int world_size = config_.world_size;
    
    // Allocate symmetric buffers for dispatched tokens
    // Each rank can receive up to max_tokens from each other rank
    size_t dispatch_size = world_size * max_tokens * hidden_dim * sizeof(float);
    dispatch_buffer_ = ctx_->AllocateSymmetricMemory(dispatch_size);
    
    // Allocate symmetric buffer for combined output
    size_t output_size = max_tokens * hidden_dim * sizeof(float);
    output_buffer_ = ctx_->AllocateSymmetricMemory(output_size);
    
    // Allocate symmetric buffer for token counts (how many tokens each rank sends/receives)
    size_t counts_size = world_size * sizeof(int32_t);
    token_counts_ = ctx_->AllocateSymmetricMemory(counts_size);
    
    // Allocate local GPU buffers for offsets
    HIP_CHECK(hipMalloc(&send_offsets_, world_size * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&recv_offsets_, world_size * sizeof(int32_t)));
}

void DispatchCombineHandle::FreeBuffers() {
    if (dispatch_buffer_) {
        ctx_->FreeSymmetricMemory(dispatch_buffer_);
        dispatch_buffer_ = nullptr;
    }
    if (output_buffer_) {
        ctx_->FreeSymmetricMemory(output_buffer_);
        output_buffer_ = nullptr;
    }
    if (token_counts_) {
        ctx_->FreeSymmetricMemory(token_counts_);
        token_counts_ = nullptr;
    }
    if (send_offsets_) {
        HIP_CHECK(hipFree(send_offsets_));
        send_offsets_ = nullptr;
    }
    if (recv_offsets_) {
        HIP_CHECK(hipFree(recv_offsets_));
        recv_offsets_ = nullptr;
    }
}

void DispatchCombineHandle::PrepareInference(void* tokens, int32_t* expert_ids,
                                            float* weights, int num_tokens) {
    input_tokens_ = tokens;
    expert_ids_ = expert_ids;
    weights_ = weights;
    num_tokens_ = num_tokens;
}

void DispatchCombineHandle::ComputeSendRecvCounts() {
    // This will be implemented in GPU kernel
    // For now, placeholder - should count how many tokens go to each rank
}

void DispatchCombineHandle::LaunchDispatch(hipStream_t stream) {
    // Dispatch phase implementation will be in GPU kernels
    // This is a placeholder that shows the general flow:
    
    // 1. Compute send counts for each destination rank
    ComputeSendRecvCounts();
    
    // 2. Launch GPU kernel to:
    //    - For each token, determine which expert(s) it should go to
    //    - Write tokens directly to peer GPU memory using P2P or RDMA
    //    - For P2P: direct GPU memory copy to peer_ptrs[peer_rank]
    //    - For RDMA: use GPU-initiated RDMA writes
    
    // 3. Synchronize to ensure all transfers complete
    HIP_CHECK(hipStreamSynchronize(stream));
}

void DispatchCombineHandle::LaunchCombine(hipStream_t stream) {
    // Combine phase implementation will be in GPU kernels
    // This is a placeholder that shows the general flow:
    
    // 1. Launch GPU kernel to:
    //    - Read expert outputs from dispatch_buffer_
    //    - For each original token, gather outputs from all assigned experts
    //    - Combine using weighted sum with weights_
    //    - Write final output to output_buffer_
    
    // 2. Synchronize to ensure all operations complete
    HIP_CHECK(hipStreamSynchronize(stream));
}

} // namespace simple_dispatch_combine
