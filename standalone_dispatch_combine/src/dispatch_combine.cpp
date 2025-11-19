// Simple standalone dispatch-combine implementation with P2P and RDMA
// Copyright (c) 2025. MIT License.

#include "dispatch_combine.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <algorithm>

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
    
    // Initialize RDMA for inter-node communication
    // Note: This is a simplified implementation
    // In production, would only initialize if inter-node communication is needed
    InitializeRDMA();
    
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
    // Initialize RDMA context for inter-node communication
    // This is a simplified implementation that sets up the basic structure
    // A full implementation would use ibverbs or vendor-specific APIs
    
    // In a full implementation, this would:
    // 1. Open RDMA device (e.g., using ibv_open_device)
    //    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    //    struct ibv_context *ctx = ibv_open_device(dev_list[0]);
    //
    // 2. Create protection domain
    //    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    //
    // 3. Create completion queues
    //    struct ibv_cq *cq = ibv_create_cq(ctx, cq_size, NULL, NULL, 0);
    //
    // 4. Create queue pairs for each peer
    //    for each peer:
    //        struct ibv_qp_init_attr qp_attr;
    //        qp_attr.send_cq = cq;
    //        qp_attr.recv_cq = cq;
    //        qp_attr.qp_type = IBV_QPT_RC;
    //        struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);
    //
    // 5. Exchange QP information via MPI (QPN, LID, GID)
    //    struct qp_info { uint32_t qpn; uint16_t lid; uint8_t gid[16]; };
    //    MPI_Allgather(local_info, ..., all_qp_info, ...);
    //
    // 6. Transition QP states: RESET -> INIT -> RTR -> RTS
    //    ibv_modify_qp(qp, &attr, IBV_QP_STATE | ...);
    
    // For this simplified version, we'll just initialize the structure
    rdma_ctx_.pd = nullptr;  // Would be actual protection domain
    rdma_ctx_.cq = nullptr;  // Would be actual completion queue
    rdma_ctx_.qps.resize(config_.world_size, nullptr);  // Would be actual queue pairs
    
    // Note: In production, RDMA initialization must happen before symmetric memory
    // allocation, as memory needs to be registered with RDMA (ibv_reg_mr)
    // and the remote keys (rkeys) exchanged between ranks
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
    // Exchange RDMA memory registration keys for inter-node communication
    // In a full implementation with ibverbs:
    // 1. Register memory with RDMA protection domain
    //    struct ibv_mr *mr = ibv_reg_mr(pd, ptr, size,
    //                                   IBV_ACCESS_LOCAL_WRITE |
    //                                   IBV_ACCESS_REMOTE_WRITE |
    //                                   IBV_ACCESS_REMOTE_READ);
    //
    // 2. Extract local key (lkey) and remote key (rkey) from memory region
    //    mem->lkey = mr->lkey;
    //    uint32_t local_rkey = mr->rkey;
    //
    // 3. Exchange remote keys via MPI Allgather so all ranks know each other's rkeys
    //    MPI_CHECK(MPI_Allgather(&local_rkey, sizeof(uint32_t), MPI_BYTE,
    //                            mem->rkeys.data(), sizeof(uint32_t), MPI_BYTE,
    //                            comm_));
    //
    // 4. Store the memory region pointer for later cleanup
    //    mem->rdma_mr = mr;
    //
    // The rkey is used by remote ranks to access this memory via RDMA:
    //    ibv_post_send(qp, &wr, ...) where wr.wr.rdma.rkey = mem->rkeys[dest_rank]
    
    // For this simplified version, use placeholder values
    // In production, these would be actual RDMA memory region keys
    mem->lkey = (config_.rank + 1) * 1000;  // Simulate unique lkey per rank
    
    // Exchange rkeys with all other ranks
    uint32_t local_rkey = mem->lkey;  // In ibverbs, lkey and rkey are usually same for local
    MPI_CHECK(MPI_Allgather(&local_rkey, sizeof(uint32_t), MPI_BYTE,
                           mem->rkeys.data(), sizeof(uint32_t), MPI_BYTE,
                           comm_));
    
    // Note: For GPU Direct RDMA, the memory must be:
    // 1. Allocated on GPU (hipMalloc)
    // 2. Registered with RDMA (ibv_reg_mr with GPU pointer)
    // 3. Accessible via both GPU kernels and RDMA operations
    // This requires GPU Direct RDMA kernel module (e.g., nvidia_peermem, amd_peermem)
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
      hidden_states_(nullptr), expert_ids_(nullptr), weights_(nullptr), num_tokens_(0),
      dispatch_buffer_(nullptr), output_buffer_(nullptr), token_counts_(nullptr),
      send_offsets_(nullptr), recv_offsets_(nullptr), send_counts_(nullptr), recv_counts_(nullptr),
      dispatch_map_(nullptr), dest_token_counter_(nullptr),
      dispatch_barrier_(nullptr), combine_barrier_(nullptr), barrier_flag_(0) {
    
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
    
    // Allocate local GPU buffers for offsets and counts
    HIP_CHECK(hipMalloc(&send_offsets_, world_size * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&recv_offsets_, world_size * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&send_counts_, world_size * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&recv_counts_, world_size * sizeof(int32_t)));
    
    // Initialize to zero
    HIP_CHECK(hipMemset(send_offsets_, 0, world_size * sizeof(int32_t)));
    HIP_CHECK(hipMemset(recv_offsets_, 0, world_size * sizeof(int32_t)));
    HIP_CHECK(hipMemset(send_counts_, 0, world_size * sizeof(int32_t)));
    HIP_CHECK(hipMemset(recv_counts_, 0, world_size * sizeof(int32_t)));
    
    // Allocate mapping information buffers
    // dispatch_map stores destination (rank * max_tokens + offset) for each token-expert pair
    size_t map_size = max_tokens * config_.num_experts_per_token * sizeof(int32_t);
    HIP_CHECK(hipMalloc(&dispatch_map_, map_size));
    HIP_CHECK(hipMemset(dispatch_map_, 0, map_size));
    
    // Allocate symmetric buffer for destination token counters
    size_t counter_size = world_size * sizeof(int32_t);
    HIP_CHECK(hipMalloc(&dest_token_counter_, counter_size));
    HIP_CHECK(hipMemset(dest_token_counter_, 0, counter_size));
    
    // Allocate barrier buffers (symmetric for cross-device synchronization)
    HIP_CHECK(hipMalloc(&dispatch_barrier_, sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&combine_barrier_, sizeof(uint32_t)));
    HIP_CHECK(hipMemset(dispatch_barrier_, 0, sizeof(uint32_t)));
    HIP_CHECK(hipMemset(combine_barrier_, 0, sizeof(uint32_t)));
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
    if (send_counts_) {
        HIP_CHECK(hipFree(send_counts_));
        send_counts_ = nullptr;
    }
    if (recv_counts_) {
        HIP_CHECK(hipFree(recv_counts_));
        recv_counts_ = nullptr;
    }
    if (dispatch_map_) {
        HIP_CHECK(hipFree(dispatch_map_));
        dispatch_map_ = nullptr;
    }
    if (dest_token_counter_) {
        HIP_CHECK(hipFree(dest_token_counter_));
        dest_token_counter_ = nullptr;
    }
    if (dispatch_barrier_) {
        HIP_CHECK(hipFree(dispatch_barrier_));
        dispatch_barrier_ = nullptr;
    }
    if (combine_barrier_) {
        HIP_CHECK(hipFree(combine_barrier_));
        combine_barrier_ = nullptr;
    }
}

void DispatchCombineHandle::PrepareInference(void* states, int32_t* expert_ids,
                                            float* weights, int num_tokens) {
    hidden_states_ = states;
    expert_ids_ = expert_ids;
    weights_ = weights;
    num_tokens_ = num_tokens;
}

void DispatchCombineHandle::LaunchDispatch(hipStream_t stream) {
    // Reset counters for this dispatch operation
    HIP_CHECK(hipMemsetAsync(send_offsets_, 0, config_.world_size * sizeof(int32_t), stream));
    HIP_CHECK(hipMemsetAsync(dest_token_counter_, 0, config_.world_size * sizeof(int32_t), stream));
    HIP_CHECK(hipMemsetAsync(dispatch_barrier_, 0, sizeof(uint32_t), stream));
    
    // Increment barrier flag for this iteration
    barrier_flag_++;
    
    // Prepare peer pointers array for kernel access
    std::vector<uintptr_t> peer_ptrs = dispatch_buffer_->peer_ptrs;
    uintptr_t* d_peer_ptrs;
    HIP_CHECK(hipMalloc(&d_peer_ptrs, config_.world_size * sizeof(uintptr_t)));
    HIP_CHECK(hipMemcpyAsync(d_peer_ptrs, peer_ptrs.data(), 
                             config_.world_size * sizeof(uintptr_t),
                             hipMemcpyHostToDevice, stream));
    
    size_t max_tokens_to_send = config_.world_size * config_.max_tokens_per_rank;
    
    // Launch dispatch kernel based on node topology
    // For simplicity, we'll use intra-node dispatch kernel for all cases
    // In production, would check topology and use appropriate kernel
    
    // Each warp processes token-expert pairs
    int num_dispatches = num_tokens_ * config_.num_experts_per_token;
    int num_warps = (num_dispatches + 31) / 32;  // Round up to warps
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    
    // Call external kernel launcher (defined in kernels.hip.cpp)
    extern void LaunchDispatchKernels(
        const float* input_tokens,
        const int32_t* expert_ids,
        const uintptr_t* peer_ptrs,
        int num_tokens,
        int num_experts_per_token,
        int num_experts_per_rank,
        int hidden_dim,
        int rank,
        int world_size,
        int gpu_per_node,
        int32_t* send_offsets,
        int32_t* dispatch_map,
        int32_t* dest_token_counter,
        size_t max_tokens_to_send,
        hipStream_t stream);
    
    LaunchDispatchKernels(
        static_cast<const float*>(hidden_states_),
        expert_ids_,
        d_peer_ptrs,
        num_tokens_,
        config_.num_experts_per_token,
        config_.num_experts_per_rank,
        config_.hidden_dim,
        config_.rank,
        config_.world_size,
        config_.gpu_per_node,
        send_offsets_,
        dispatch_map_,
        dest_token_counter_,
        max_tokens_to_send,
        stream);
    
    // Synchronize to ensure all transfers complete
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Cleanup temporary buffer
    HIP_CHECK(hipFree(d_peer_ptrs));
    
    // Optional: MPI barrier for inter-node synchronization if using RDMA
    // MPI_Barrier(MPI_COMM_WORLD);
}

void DispatchCombineHandle::LaunchCombine(hipStream_t stream) {
    // Reset combine barrier
    HIP_CHECK(hipMemsetAsync(combine_barrier_, 0, sizeof(uint32_t), stream));
    
    // Prepare peer pointers for reading dispatch buffer
    std::vector<uintptr_t> peer_ptrs = dispatch_buffer_->peer_ptrs;
    uintptr_t* d_peer_ptrs;
    HIP_CHECK(hipMalloc(&d_peer_ptrs, config_.world_size * sizeof(uintptr_t)));
    HIP_CHECK(hipMemcpyAsync(d_peer_ptrs, peer_ptrs.data(),
                             config_.world_size * sizeof(uintptr_t),
                             hipMemcpyHostToDevice, stream));
    
    size_t max_tokens_to_send = config_.world_size * config_.max_tokens_per_rank;
    
    // Call external kernel launcher (defined in kernels.hip.cpp)
    extern void LaunchCombineKernel(
        const float* dispatch_buffer,
        const float* weights,
        const int32_t* expert_ids,
        float* output,
        const int32_t* dispatch_map,
        int num_tokens,
        int num_experts_per_token,
        int num_experts_per_rank,
        int hidden_dim,
        int rank,
        size_t max_tokens_to_send,
        hipStream_t stream);
    
    LaunchCombineKernel(
        static_cast<const float*>(dispatch_buffer_->local_ptr),
        weights_,
        expert_ids_,
        static_cast<float*>(output_buffer_->local_ptr),
        dispatch_map_,
        num_tokens_,
        config_.num_experts_per_token,
        config_.num_experts_per_rank,
        config_.hidden_dim,
        config_.rank,
        max_tokens_to_send,
        stream);
    
    // Synchronize to ensure all operations complete
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Cleanup temporary buffer
    HIP_CHECK(hipFree(d_peer_ptrs));
}

} // namespace simple_dispatch_combine
