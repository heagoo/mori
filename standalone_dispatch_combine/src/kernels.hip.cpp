// GPU kernels for dispatch-combine operations
// Copyright (c) 2025. MIT License.
//
// ============================================================================
// DISPATCH-COMBINE ALGORITHM FOR MIXTURE-OF-EXPERTS (MoE)
// ============================================================================
//
// This implementation provides GPU kernels for efficient token routing in
// Mixture-of-Experts models. The algorithm has two main phases:
//
// 1. DISPATCH PHASE:
//    - Each rank has input tokens to process
//    - Based on expert assignments (top-k routing), tokens are sent to ranks
//      that own the required experts
//    - Uses direct GPU-to-GPU transfers (P2P within node, RDMA across nodes)
//    - Records mapping information for later combining
//
// 2. COMBINE PHASE:
//    - After expert computation, outputs need to be aggregated back
//    - Uses mapping info from dispatch to locate expert outputs
//    - Performs weighted combination based on routing weights
//    - Outputs final aggregated results
//
// KEY FEATURES:
// - Deduplication: If token needs multiple experts on same rank, only sent once
// - Warp-level operations: All transfers use warp-cooperative primitives
// - Zero-copy: Direct GPU memory access, no intermediate CPU buffers
// - Symmetric memory: All ranks can access each other's GPU memory
//
// SYNCHRONIZATION:
// - P2P writes are visible after __threadfence() within same node
// - RDMA writes require explicit synchronization (barrier or signaling)
// - Mapping information (dispatch_map) coordinates dispatch and combine
//
// Based on the MORI project's EpDispatchIntraNodeKernel and related kernels.
// ============================================================================

#include "dispatch_combine.hpp"
#include <hip/hip_runtime.h>

namespace simple_dispatch_combine {

// Device function to compute which rank owns a given expert
__device__ inline int GetExpertRank(int expert_id, int num_experts_per_rank) {
    return expert_id / num_experts_per_rank;
}

// Device function to write data to peer GPU memory (P2P)
// This directly accesses peer GPU memory through IPC handles
__device__ void WriteToP2PPeer(float* dest, const float* src, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < size; i += stride) {
        dest[i] = src[i];
    }
}

// Kernel to compute send/receive counts per rank
__global__ void ComputeSendRecvCountsKernel(
    const int32_t* expert_ids,      // [num_tokens, num_experts_per_token]
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int world_size,
    int32_t* send_counts,            // [world_size]
    int32_t* send_offsets)           // [world_size]
{
    // Use shared memory to accumulate counts
    extern __shared__ int32_t shared_counts[];
    
    int tid = threadIdx.x;
    
    // Initialize shared memory
    if (tid < world_size) {
        shared_counts[tid] = 0;
    }
    __syncthreads();
    
    // Count tokens going to each rank
    int total_dispatches = num_tokens * num_experts_per_token;
    for (int idx = tid; idx < total_dispatches; idx += blockDim.x) {
        int expert_id = expert_ids[idx];
        int dest_rank = GetExpertRank(expert_id, num_experts_per_rank);
        atomicAdd(&shared_counts[dest_rank], 1);
    }
    __syncthreads();
    
    // Write results to global memory
    if (tid < world_size) {
        send_counts[tid] = shared_counts[tid];
    }
}

// Dispatch kernel using P2P for intra-node communication
// This kernel directly writes to peer GPU memory
// Based on EpDispatchIntraNodeKernel from the parent mori project
__global__ void DispatchIntraNodeKernel(
    const float* input_tokens,       // [num_tokens, hidden_dim]
    const int32_t* expert_ids,       // [num_tokens, num_experts_per_token]
    const uintptr_t* peer_ptrs,      // [world_size] - peer GPU pointers
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int hidden_dim,
    int rank,
    int32_t* send_offsets,           // [world_size] - current write offset for each peer
    int32_t* dispatch_map,           // [num_tokens * num_experts_per_token] - mapping info
    int32_t* dest_token_counter,     // [world_size] - atomic counters for destination indices
    size_t max_tokens_to_send)       // Maximum tokens that can be sent
{
    int laneId = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x / warpSize;
    int warpNum = blockDim.x / warpSize;
    int globalWarpId = blockIdx.x * warpNum + warpId;
    int globalWarpNum = gridDim.x * warpNum;
    
    // Each warp processes token-expert pairs
    for (int i = globalWarpId; i < num_tokens * num_experts_per_token; i += globalWarpNum) {
        int token_idx = i / num_experts_per_token;
        int expert_slot = i % num_experts_per_token;
        
        int expert_id = expert_ids[i];
        int dest_rank = expert_id / num_experts_per_rank;
        
        // Deduplication: check if this token already being sent to this destination
        // by another expert slot (avoids sending same token multiple times)
        bool is_duplicate = false;
        if (laneId < expert_slot) {
            int other_expert = expert_ids[token_idx * num_experts_per_token + laneId];
            int other_rank = other_expert / num_experts_per_rank;
            if (other_rank == dest_rank) {
                is_duplicate = true;
            }
        }
        
        // Use warp vote to check if any lane found a duplicate
        unsigned int dup_mask = __ballot(is_duplicate);
        if (dup_mask != 0) {
            // Mark this as duplicate in dispatch_map (use overflow value)
            if (laneId == 0) {
                dispatch_map[i] = max_tokens_to_send; // Overflow marker
            }
            continue;
        }
        
        // Lane 0 of warp handles atomic operations
        int dest_token_idx = 0;
        if (laneId == 0) {
            // Atomically get next available slot in destination buffer
            dest_token_idx = atomicAdd(&dest_token_counter[dest_rank], 1);
            
            // Record mapping: global position = dest_rank * max_per_rank + local_idx
            dispatch_map[i] = dest_rank * max_tokens_to_send + dest_token_idx;
        }
        
        // Broadcast dest_token_idx to all lanes in warp
        dest_token_idx = __shfl(dest_token_idx, 0);
        
        // Get destination pointer (peer GPU memory via P2P)
        float* dest_buffer = reinterpret_cast<float*>(peer_ptrs[dest_rank]);
        float* dest_token = dest_buffer + dest_token_idx * hidden_dim;
        
        // Get source pointer
        const float* src_token = input_tokens + token_idx * hidden_dim;
        
        // Warp-level copy: all threads in warp cooperatively copy the token
        for (int dim = laneId; dim < hidden_dim; dim += warpSize) {
            dest_token[dim] = src_token[dim];
        }
    }
}

// Dispatch kernel using RDMA for inter-node communication
// In a full implementation, this would use GPU-initiated RDMA
// For now, we'll simulate with staging buffer + synchronization
// Based on EpDispatchInterNodeKernel from the parent mori project
__global__ void DispatchInterNodeKernel(
    const float* input_tokens,       // [num_tokens, hidden_dim]
    const int32_t* expert_ids,       // [num_tokens, num_experts_per_token]
    float* staging_buffer,           // Staging buffer for RDMA transfers
    const uintptr_t* peer_ptrs,      // [world_size] - destination pointers
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int hidden_dim,
    int rank,
    int world_size,
    int gpu_per_node,
    int32_t* send_offsets,           // [world_size]
    int32_t* dispatch_map,           // Mapping information
    int32_t* dest_token_counter,     // Atomic counters
    size_t max_tokens_to_send)
{
    int laneId = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x / warpSize;
    int warpNum = blockDim.x / warpSize;
    int globalWarpId = blockIdx.x * warpNum + warpId;
    int globalWarpNum = gridDim.x * warpNum;
    
    int my_node = rank / gpu_per_node;
    
    // Phase 1: Build dispatch map and compute destinations
    // Similar to intra-node, but tracks inter-node destinations
    for (int i = globalWarpId; i < num_tokens * num_experts_per_token; i += globalWarpNum) {
        int token_idx = i / num_experts_per_token;
        int expert_slot = i % num_experts_per_token;
        
        int expert_id = expert_ids[i];
        int dest_rank = expert_id / num_experts_per_rank;
        int dest_node = dest_rank / gpu_per_node;
        
        // Skip if same node (handled by intra-node kernel)
        if (dest_node == my_node) continue;
        
        // Deduplication check
        bool is_duplicate = false;
        if (laneId < expert_slot) {
            int other_expert = expert_ids[token_idx * num_experts_per_token + laneId];
            int other_rank = other_expert / num_experts_per_rank;
            if (other_rank == dest_rank) {
                is_duplicate = true;
            }
        }
        
        unsigned int dup_mask = __ballot(is_duplicate);
        if (dup_mask != 0) {
            if (laneId == 0) {
                dispatch_map[i] = max_tokens_to_send; // Mark as duplicate
            }
            continue;
        }
        
        // Allocate destination slot
        int dest_token_idx = 0;
        if (laneId == 0) {
            dest_token_idx = atomicAdd(&dest_token_counter[dest_rank], 1);
            dispatch_map[i] = dest_rank * max_tokens_to_send + dest_token_idx;
            __threadfence();  // Ensure visibility for RDMA operations
        }
        dest_token_idx = __shfl(dest_token_idx, 0);
        
        // Copy to staging buffer first
        // In production: this would directly initiate GPU-RDMA transfer
        const float* src_token = input_tokens + token_idx * hidden_dim;
        float* stage_token = staging_buffer + (dest_rank * max_tokens_to_send + dest_token_idx) * hidden_dim;
        
        for (int dim = laneId; dim < hidden_dim; dim += warpSize) {
            stage_token[dim] = src_token[dim];
        }
    }
    
    // Phase 2: RDMA transfer (simplified)
    // In a full implementation with GPU Direct RDMA:
    // - Would use vendor-specific APIs (e.g., MLX5 for Mellanox NICs)
    // - Post RDMA write work requests from GPU kernel
    // - Example: mlx5_post_rdma_write_wqe(qp, local_addr, remote_addr, size, rkey)
    // - Signal completion and handle acknowledgments
    //
    // For this simplified version, we rely on explicit synchronization
    // between dispatch and combine phases via MPI barriers
}

// Combine kernel - aggregate expert outputs with weighted sum
// Uses the dispatch_map to locate expert outputs
__global__ void CombineKernel(
    const float* dispatch_buffer,    // [world_size * max_tokens, hidden_dim]
    const float* weights,            // [num_tokens, num_experts_per_token]
    const int32_t* expert_ids,       // [num_tokens, num_experts_per_token]
    float* output,                   // [num_tokens, hidden_dim]
    const int32_t* dispatch_map,     // [num_tokens * num_experts_per_token] - mapping info
    int num_tokens,
    int num_experts_per_token,
    int num_experts_per_rank,
    int hidden_dim,
    int rank,
    size_t max_tokens_to_send)       // Maximum tokens per rank
{
    int token_idx = blockIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int warpNum = blockDim.x / warpSize;
    
    if (token_idx >= num_tokens) return;
    
    // Calculate dimensions per warp
    int warps_per_token = 1;  // Simple case: one warp per token
    int dims_per_warp = (hidden_dim + warpSize - 1) / warpSize;
    
    // Process each dimension assigned to this thread
    for (int dim = laneId; dim < hidden_dim; dim += warpSize) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        // Aggregate outputs from all experts assigned to this token
        for (int expert_slot = 0; expert_slot < num_experts_per_token; expert_slot++) {
            int map_idx = token_idx * num_experts_per_token + expert_slot;
            int dest_location = dispatch_map[map_idx];
            
            // Check if this was a duplicate (marked with overflow value)
            if (dest_location >= max_tokens_to_send) {
                continue;  // Skip duplicates
            }
            
            // Extract rank and local token index from mapping
            int dest_rank = dest_location / max_tokens_to_send;
            int local_token_idx = dest_location % max_tokens_to_send;
            
            // Read expert output from dispatch buffer
            // Note: In real implementation, would read from peer memory using dispatch_map
            // For now, assume data is in local dispatch_buffer after synchronization
            float expert_output = dispatch_buffer[dest_location * hidden_dim + dim];
            
            // Get routing weight for this expert
            float weight = weights[token_idx * num_experts_per_token + expert_slot];
            
            // Accumulate weighted output
            sum += weight * expert_output;
            weight_sum += weight;
        }
        
        // Normalize and write output
        if (weight_sum > 0.0f) {
            output[token_idx * hidden_dim + dim] = sum / weight_sum;
        } else {
            output[token_idx * hidden_dim + dim] = 0.0f;
        }
    }
}

// Host function to launch dispatch (chooses P2P or RDMA based on topology)
extern "C" void LaunchDispatchKernels(
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
    hipStream_t stream)
{
    // Calculate grid dimensions
    // Each warp processes multiple token-expert pairs
    int num_dispatches = num_tokens * num_experts_per_token;
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int num_warps = (num_dispatches + 31) / 32;  // Round up
    int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    
    dim3 grid(num_blocks);
    dim3 block(threads_per_block);
    
    // Launch intra-node dispatch kernel
    // In production, would check topology and use appropriate kernel for inter-node
    hipLaunchKernelGGL(DispatchIntraNodeKernel, grid, block, 0, stream,
                       input_tokens, expert_ids, peer_ptrs,
                       num_tokens, num_experts_per_token, num_experts_per_rank,
                       hidden_dim, rank, send_offsets, dispatch_map,
                       dest_token_counter, max_tokens_to_send);
}

// Host function to launch combine
extern "C" void LaunchCombineKernel(
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
    hipStream_t stream)
{
    // One block per token, multiple threads per block
    dim3 grid(num_tokens);
    dim3 block(256);  // Use 256 threads (8 warps) per block
    
    hipLaunchKernelGGL(CombineKernel, grid, block, 0, stream,
                       dispatch_buffer, weights, expert_ids, output, dispatch_map,
                       num_tokens, num_experts_per_token, num_experts_per_rank,
                       hidden_dim, rank, max_tokens_to_send);
}

} // namespace simple_dispatch_combine
