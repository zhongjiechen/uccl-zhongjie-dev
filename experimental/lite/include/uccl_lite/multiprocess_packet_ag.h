#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace uccl_lite::multiprocess {
inline constexpr int kPacketRanks = 8;
inline constexpr size_t kAgPacketBytes = 1UL << 20;
inline constexpr size_t kAgPacketElements = kAgPacketBytes / sizeof(uint16_t);
inline constexpr size_t kTransportPacketBytes = 1UL << 20;
inline constexpr size_t kRsStripeElements =
    kTransportPacketBytes / sizeof(uint16_t) / kPacketRanks;
static_assert(kAgPacketBytes == kTransportPacketBytes);

struct alignas(128) PacketGeneration {
  uint64_t value = 0;
};

struct alignas(128) CrewState {
  uint32_t arrivals = 0;
  uint32_t generation = 0;
  uint64_t ready = 0;
};

struct alignas(4096) PacketShared {
  PacketGeneration uploaded[2][kPacketRanks];
  PacketGeneration uploadedDest[2][kPacketRanks][kPacketRanks];
  PacketGeneration inputReadDone[2][kPacketRanks];
  PacketGeneration localReady[2][kPacketRanks];
  PacketGeneration partialReadDone[2][kPacketRanks];
  PacketGeneration routed[2][kPacketRanks];
  PacketGeneration consumed[2][kPacketRanks];
  alignas(4096)
      uint16_t input[2][kPacketRanks][kTransportPacketBytes / sizeof(uint16_t)];
  alignas(4096) uint16_t result[2][2][kPacketRanks][kRsStripeElements];
  alignas(4096) uint16_t partial[2][kPacketRanks][2][kRsStripeElements];
};

cudaError_t launchAgPacket(PacketShared* mapped, uint16_t* agInput,
                           void const* input, void* output, size_t count,
                           int rank, uint64_t baseGeneration,
                           uint64_t previousSlot0, uint64_t previousSlot1,
                           cudaStream_t stream);
cudaError_t launchAgPacketPipelined(PacketShared* mapped, uint16_t* agInput,
                                    CrewState* crews, void const* input,
                                    void* output, size_t count, int rank,
                                    uint64_t baseGeneration,
                                    uint64_t previousSlot0,
                                    uint64_t previousSlot1,
                                    uint32_t initialPhase, cudaStream_t stream);
cudaError_t launchRsPacket(PacketShared* mapped, void const* input,
                           void* output, size_t shardCount, int rank,
                           uint64_t baseGeneration, uint64_t previousSlot0,
                           uint64_t previousSlot1, cudaStream_t stream);
cudaError_t launchRsPacketPipelined(PacketShared* mapped, CrewState* crews,
                                    void const* input, void* output,
                                    size_t shardCount, int rank,
                                    uint64_t baseGeneration,
                                    uint64_t previousSlot0,
                                    uint64_t previousSlot1,
                                    uint32_t initialPhase, cudaStream_t stream);
cudaError_t launchRsPacketCpu(PacketShared* mapped, CrewState* crews,
                              void const* input, void* output,
                              size_t shardCount, int rank,
                              uint64_t baseGeneration, uint64_t previousSlot0,
                              uint64_t previousSlot1, uint32_t initialPhase,
                              uint32_t previousCpuMask, bool splitCredits,
                              bool retainOwn, cudaStream_t stream);
cudaError_t launchRsPacketDestReady(PacketShared* mapped, CrewState* crews,
                                    void const* input, void* output,
                                    size_t shardCount, int rank,
                                    uint64_t baseGeneration,
                                    uint64_t previousSlot0,
                                    uint64_t previousSlot1,
                                    uint32_t initialPhase, cudaStream_t stream);
cudaError_t launchArPacketCpu(PacketShared* mapped, CrewState* crews,
                              void const* input, void* output, size_t count,
                              int rank, uint64_t baseGeneration,
                              uint64_t previousSlot0, uint64_t previousSlot1,
                              uint32_t initialPhase, bool replicateResults,
                              uint32_t previousCpuMask, bool splitCredits,
                              cudaStream_t stream);
cudaError_t launchArPacketSmall(PacketShared* mapped, void const* input,
                                void* output, size_t count, int rank,
                                uint64_t generation, uint64_t previousSlot0,
                                cudaStream_t stream);
}  // namespace uccl_lite::multiprocess
