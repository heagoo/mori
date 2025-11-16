// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// This is a minimal example demonstrating GPU-to-GPU data interchange using MORI.
// It shows the essential steps for setting up RDMA communication between GPUs.

#include <hip/hip_runtime.h>
#include <stdio.h>

#include "mori/application/application.hpp"
#include "mori/core/core.hpp"

using namespace mori;
using namespace mori::application;
using namespace mori::core;

// Memory region access flags for RDMA operations
#define MR_ACCESS_FLAG                                                        \
  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | \
      IBV_ACCESS_REMOTE_ATOMIC

// Simple kernel to write data on GPU
template <ProviderType P>
__device__ void WriteDataKernel(RdmaEndpoint& epSend, RdmaMemoryRegion sendMr,
                                 RdmaEndpoint& epRecv, RdmaMemoryRegion recvMr,
                                 int msgSize, uint8_t value) {
  // Fill send buffer with the value
  for (int j = 0; j < msgSize; j++) {
    reinterpret_cast<char*>(sendMr.addr)[j] = value;
  }

  // Ensure all writes are visible before RDMA operation
  __threadfence_system();

  // Post RDMA write operation
  uint64_t dbr_val = PostWrite<P>(epSend.wqHandle, epSend.handle.qpn,
                                   sendMr.addr, sendMr.lkey, recvMr.addr,
                                   recvMr.rkey, msgSize);

  // Update send doorbell record
  __threadfence_system();
  UpdateSendDbrRecord<P>(epSend.wqHandle.dbrRecAddr, epSend.wqHandle.postIdx);

  // Ring the doorbell to notify the hardware
  __threadfence_system();
  RingDoorbell<P>(epSend.wqHandle.dbrAddr, dbr_val);

  // Poll completion queue to wait for operation completion
  __threadfence_system();
  uint16_t wqeIdx;
  int snd_opcode = PollCq<P>(epSend.cqHandle.cqAddr, epSend.cqHandle.cqeNum,
                              &epSend.cqHandle.consIdx, &wqeIdx);
  epSend.cqHandle.consIdx += 1;

  // Update completion queue doorbell record
  UpdateCqDbrRecord<P>(epSend.cqHandle.dbrRecAddr, epSend.cqHandle.consIdx,
                       epSend.cqHandle.cqeNum);
}

// Simple kernel to verify received data on GPU
template <ProviderType P>
__device__ void VerifyDataKernel(RdmaMemoryRegion recvMr, int msgSize,
                                  uint8_t expectedValue) {
  for (int j = 0; j < msgSize; j++) {
    uint8_t recvVal = reinterpret_cast<char*>(recvMr.addr)[j];
    if (recvVal != expectedValue) {
      printf("Verification failed at position %d: expected %d, got %d\n", j,
             expectedValue, recvVal);
      return;
    }
  }
  printf("Success! All %d bytes verified with value %d\n", msgSize,
         expectedValue);
}

// GPU kernel that performs write and verification
__global__ void GpuInterchangeKernel(RdmaEndpoint& epSend, RdmaEndpoint& epRecv,
                                      RdmaMemoryRegion mrSend,
                                      RdmaMemoryRegion mrRecv, int msgSize,
                                      uint8_t value) {
  // Write data from GPU1 to GPU2
  switch (epSend.GetProviderType()) {
    case ProviderType::MLX5:
      WriteDataKernel<ProviderType::MLX5>(epSend, mrSend, epRecv, mrRecv,
                                           msgSize, value);
      break;
#ifdef ENABLE_BNXT
    case ProviderType::BNXT:
      WriteDataKernel<ProviderType::BNXT>(epSend, mrSend, epRecv, mrRecv,
                                           msgSize, value);
      break;
#endif
    default:
      printf("Unsupported provider type\n");
      return;
  }

  // Verify the received data
  switch (epRecv.GetProviderType()) {
    case ProviderType::MLX5:
      VerifyDataKernel<ProviderType::MLX5>(mrRecv, msgSize, value);
      break;
#ifdef ENABLE_BNXT
    case ProviderType::BNXT:
      VerifyDataKernel<ProviderType::BNXT>(mrRecv, msgSize, value);
      break;
#endif
    default:
      printf("Unsupported provider type\n");
      return;
  }
}

void SimpleGpuInterchange() {
  int msgSize = 1024;  // Size of data to transfer
  uint8_t testValue = 42;  // Test value to write

  printf("=== Simple GPU Data Interchange Example ===\n");
  printf("Transferring %d bytes with value %d\n\n", msgSize, testValue);

  // Step 1: Initialize RDMA context
  printf("Step 1: Initializing RDMA context...\n");
  RdmaContext rdmaContext(RdmaBackendType::DirectVerbs);
  printf("  Found %d RDMA device(s)\n", rdmaContext.nums_device);

  // Step 2: Get available RDMA devices
  printf("Step 2: Getting RDMA devices...\n");
  RdmaDeviceList rdmaDevices = rdmaContext.GetRdmaDeviceList();
  if (rdmaDevices.empty()) {
    printf("Error: No RDMA devices found!\n");
    return;
  }

  ActiveDevicePortList activeDevicePortList = GetActiveDevicePortList(rdmaDevices);
  if (activeDevicePortList.empty()) {
    printf("Error: No active device ports found!\n");
    return;
  }
  printf("  Found %zu active device port(s)\n", activeDevicePortList.size());

  ActiveDevicePort devicePort = activeDevicePortList[0];
  RdmaDevice* device = devicePort.first;

  // Step 3: Create device contexts (one for sender, one for receiver)
  printf("Step 3: Creating device contexts...\n");
  RdmaDeviceContext* deviceContextSend = device->CreateRdmaDeviceContext();
  RdmaDeviceContext* deviceContextRecv = device->CreateRdmaDeviceContext();
  printf("  Device contexts created\n");

  // Step 4: Configure and create RDMA endpoints
  printf("Step 4: Creating RDMA endpoints...\n");
  RdmaEndpointConfig config;
  config.portId = devicePort.second;
  config.gidIdx = 3;
  config.maxMsgsNum = 256;
  config.maxCqeNum = 256;
  config.alignment = 4096;
  config.onGpu = true;  // Enable GPU-initiated RDMA

  RdmaEndpoint epSend = deviceContextSend->CreateRdmaEndpoint(config);
  RdmaEndpoint epRecv = deviceContextRecv->CreateRdmaEndpoint(config);
  printf("  Endpoints created (QPN: send=%d, recv=%d)\n", epSend.handle.qpn,
         epRecv.handle.qpn);

  // Step 5: Connect endpoints
  printf("Step 5: Connecting endpoints...\n");
  deviceContextSend->ConnectEndpoint(epSend.handle, epRecv.handle);
  deviceContextRecv->ConnectEndpoint(epRecv.handle, epSend.handle);
  printf("  Endpoints connected\n");

  // Step 6: Allocate GPU buffers and register with RDMA
  printf("Step 6: Allocating and registering GPU buffers...\n");

  // Copy endpoints to GPU
  RdmaEndpoint* devEpSend;
  HIP_RUNTIME_CHECK(hipMalloc(&devEpSend, sizeof(RdmaEndpoint)));
  HIP_RUNTIME_CHECK(
      hipMemcpy(devEpSend, &epSend, sizeof(RdmaEndpoint), hipMemcpyHostToDevice));

  RdmaEndpoint* devEpRecv;
  HIP_RUNTIME_CHECK(hipMalloc(&devEpRecv, sizeof(RdmaEndpoint)));
  HIP_RUNTIME_CHECK(
      hipMemcpy(devEpRecv, &epRecv, sizeof(RdmaEndpoint), hipMemcpyHostToDevice));

  // Allocate send buffer on GPU
  void* sendBuf;
  HIP_RUNTIME_CHECK(hipMalloc(&sendBuf, msgSize));
  RdmaMemoryRegion mrSend = deviceContextSend->RegisterRdmaMemoryRegion(
      sendBuf, msgSize, MR_ACCESS_FLAG);

  // Allocate receive buffer on GPU
  void* recvBuf;
  HIP_RUNTIME_CHECK(hipMalloc(&recvBuf, msgSize));
  RdmaMemoryRegion mrRecv = deviceContextRecv->RegisterRdmaMemoryRegion(
      recvBuf, msgSize, MR_ACCESS_FLAG);

  printf("  Buffers allocated and registered\n");

  // Step 7: Launch GPU kernel to perform data interchange
  printf("Step 7: Launching GPU kernel for data interchange...\n");
  GpuInterchangeKernel<<<1, 1>>>(*devEpSend, *devEpRecv, mrSend, mrRecv,
                                  msgSize, testValue);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  printf("  Kernel execution completed\n");

  // Step 8: Cleanup
  printf("Step 8: Cleaning up resources...\n");
  deviceContextSend->DeregisterRdmaMemoryRegion(sendBuf);
  deviceContextRecv->DeregisterRdmaMemoryRegion(recvBuf);
  HIP_RUNTIME_CHECK(hipFree(devEpSend));
  HIP_RUNTIME_CHECK(hipFree(devEpRecv));
  HIP_RUNTIME_CHECK(hipFree(sendBuf));
  HIP_RUNTIME_CHECK(hipFree(recvBuf));
  printf("  Resources cleaned up\n");

  printf("\n=== Example completed successfully! ===\n");
}

int main() {
  SimpleGpuInterchange();
  return 0;
}
