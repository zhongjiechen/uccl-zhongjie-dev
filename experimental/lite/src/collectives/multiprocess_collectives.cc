#include "uccl_lite/multiprocess_collectives.h"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
constexpr int kMaxRanks = 32;
constexpr uint64_t kMagic = 0x5543434c4d503031ULL;
constexpr auto kTimeout = std::chrono::seconds(60);
thread_local std::string lastError;

void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

// Shared POD is initialized once by rank 0. Aligned 64-bit atomics are
// lock-free on the supported x86-64 hosts and operate on the shared mapping.
struct alignas(64) Shared {
  std::atomic<uint64_t> magic;
  int ranks;
  size_t maxBytes;
  std::atomic<int> failed;
  std::atomic<int> openedCount;
  std::atomic<uint64_t> arrived[kMaxRanks];
  std::atomic<uint64_t> reduced[kMaxRanks];
  std::atomic<uint64_t> finished[kMaxRanks];
  int operation[kMaxRanks];
  size_t count[kMaxRanks];
};
static_assert(std::atomic<uint64_t>::is_always_lock_free);

size_t alignPage(size_t n) { return (n + 4095) & ~size_t(4095); }

struct Handle {
  std::string name;
  int fd = -1;
  void* mapping = nullptr;
  size_t mappedBytes = 0;
  Shared* shared = nullptr;
  char* ownInput = nullptr;
  char* ownOutput = nullptr;
  int rank = -1;
  int ranks = 0;
  int device = -1;
  size_t maxBytes = 0;
  uint64_t nextSequence = 1;
  cudaStream_t lastStream = nullptr;

  char* inputs() const {
    return static_cast<char*>(mapping) + alignPage(sizeof(Shared));
  }
  char* outputs() const { return inputs() + size_t(ranks) * maxBytes; }
  char* input(int r) const { return inputs() + size_t(r) * maxBytes; }
  char* output(int r) const { return outputs() + size_t(r) * maxBytes; }

  ~Handle() {
    int previousDevice = -1;
    cudaGetDevice(&previousDevice);
    if (device >= 0) cudaSetDevice(device);
    // The CUDA stream may already have been destroyed by the caller. Drain
    // outstanding staging copies before unregistering their host buffers.
    cudaDeviceSynchronize();
    if (ownOutput) cudaHostUnregister(ownOutput);
    if (ownInput) cudaHostUnregister(ownInput);
    if (mapping) munmap(mapping, mappedBytes);
    if (fd >= 0) close(fd);
    if (previousDevice >= 0 && previousDevice != device)
      cudaSetDevice(previousDevice);
    // The final successful opener unlinks the shared-memory name. Existing
    // mappings remain valid until every rank exits, including abnormal exit.
  }
};

void waitFor(std::atomic<uint64_t>& value, uint64_t sequence, Shared* shared) {
  auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (value.load(std::memory_order_acquire) < sequence) {
    if (shared->failed.load(std::memory_order_acquire) ||
        std::chrono::steady_clock::now() >= deadline) {
      shared->failed.store(1, std::memory_order_release);
      throw std::runtime_error("collective rank timeout or failure");
    }
    std::this_thread::yield();
  }
}

float bf16ToFloat(uint16_t value) {
  uint32_t bits = uint32_t(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
uint16_t floatToBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  // Round to nearest, ties to even. This is the BF16 conversion used by the
  // CPU reference and may differ by one ulp from a NCCL reduction tree.
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return uint16_t(bits >> 16);
}

struct Call {
  Handle* handle;
  uint64_t sequence;
  int operation;
  size_t count;
};

void CUDART_CB route(void* opaque) {
  std::unique_ptr<Call> call(static_cast<Call*>(opaque));
  auto* h = call->handle;
  try {
    Shared* s = h->shared;
    s->operation[h->rank] = call->operation;
    s->count[h->rank] = call->count;
    s->arrived[h->rank].store(call->sequence, std::memory_order_release);
    for (int r = 0; r < h->ranks; ++r)
      waitFor(s->arrived[r], call->sequence, s);
    for (int r = 0; r < h->ranks; ++r)
      if (s->operation[r] != call->operation || s->count[r] != call->count)
        throw std::runtime_error("collective metadata mismatch");

    if (call->operation == 1) {
      size_t bytes = call->count * sizeof(uint16_t);
      for (int r = 0; r < h->ranks; ++r)
        std::memcpy(h->ownOutput + size_t(r) * bytes, h->input(r), bytes);
    } else if (call->operation == 2) {
      auto* out = reinterpret_cast<uint16_t*>(h->ownOutput);
      for (size_t i = 0; i < call->count; ++i) {
        float sum = 0;
        for (int r = 0; r < h->ranks; ++r) {
          auto* in = reinterpret_cast<const uint16_t*>(h->input(r));
          sum += bf16ToFloat(in[size_t(h->rank) * call->count + i]);
        }
        out[i] = floatToBf16(sum);
      }
    } else {
      // Each rank reduces a different range. Its source range remains in its
      // own output buffer while peers copy it, so no extra staging is needed.
      size_t begin = call->count * size_t(h->rank) / size_t(h->ranks);
      size_t end = call->count * size_t(h->rank + 1) / size_t(h->ranks);
      auto* out = reinterpret_cast<uint16_t*>(h->ownOutput);
      for (size_t i = begin; i < end; ++i) {
        float sum = 0;
        for (int r = 0; r < h->ranks; ++r) {
          auto* in = reinterpret_cast<const uint16_t*>(h->input(r));
          sum += bf16ToFloat(in[i]);
        }
        out[i] = floatToBf16(sum);
      }
      s->reduced[h->rank].store(call->sequence, std::memory_order_release);
      for (int r = 0; r < h->ranks; ++r)
        waitFor(s->reduced[r], call->sequence, s);
      for (int r = 0; r < h->ranks; ++r) {
        if (r == h->rank) continue;
        size_t sourceBegin = call->count * size_t(r) / size_t(h->ranks);
        size_t sourceEnd = call->count * size_t(r + 1) / size_t(h->ranks);
        size_t offset = sourceBegin * sizeof(uint16_t);
        size_t bytes = (sourceEnd - sourceBegin) * sizeof(uint16_t);
        std::memcpy(h->ownOutput + offset, h->output(r) + offset, bytes);
      }
    }
    s->finished[h->rank].store(call->sequence, std::memory_order_release);
    for (int r = 0; r < h->ranks; ++r)
      waitFor(s->finished[r], call->sequence, s);
  } catch (const std::exception& e) {
    h->shared->failed.store(1, std::memory_order_release);
    std::fprintf(stderr, "UCCL Lite multiprocess collective failed: %s\n", e.what());
    // CUDA host callbacks have no error return. Never permit a following H2D
    // copy to publish stale output after a failed collective.
    std::abort();
  }
}

std::unique_ptr<Handle> create(const char* identifier, int ranks, int rank,
                               size_t maxBytes) {
  if (!identifier || !*identifier || ranks < 2 || ranks > kMaxRanks ||
      rank < 0 || rank >= ranks || maxBytes == 0)
    throw std::invalid_argument("invalid collective configuration");
  auto h = std::make_unique<Handle>();
  h->name = std::string("/uccl-lite-mp-") + identifier;
  h->rank = rank;
  h->ranks = ranks;
  check(cudaGetDevice(&h->device));
  h->maxBytes = alignPage(maxBytes);
  h->mappedBytes = alignPage(sizeof(Shared)) +
                   2 * size_t(ranks) * h->maxBytes;
  if (rank == 0) {
    h->fd = shm_open(h->name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (h->fd < 0) throw std::runtime_error("shm_open creator failed");
    if (ftruncate(h->fd, h->mappedBytes) != 0)
      throw std::runtime_error("ftruncate shared collective failed");
  } else {
    auto deadline = std::chrono::steady_clock::now() + kTimeout;
    while ((h->fd = shm_open(h->name.c_str(), O_RDWR, 0600)) < 0) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("waiting for shared collective timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    struct stat status {};
    while (fstat(h->fd, &status) != 0 ||
           size_t(status.st_size) < h->mappedBytes) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("shared collective size timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  h->mapping = mmap(nullptr, h->mappedBytes, PROT_READ | PROT_WRITE,
                    MAP_SHARED, h->fd, 0);
  if (h->mapping == MAP_FAILED) {
    h->mapping = nullptr;
    throw std::runtime_error("mmap collective failed");
  }
  h->shared = static_cast<Shared*>(h->mapping);
  if (rank == 0) {
    new (h->mapping) Shared{};
    h->shared->ranks = ranks;
    h->shared->maxBytes = h->maxBytes;
    h->shared->magic.store(kMagic, std::memory_order_release);
  } else {
    auto deadline = std::chrono::steady_clock::now() + kTimeout;
    while (h->shared->magic.load(std::memory_order_acquire) != kMagic) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("shared collective initialization timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (h->shared->ranks != ranks || h->shared->maxBytes != h->maxBytes)
      throw std::runtime_error("shared collective layout mismatch");
  }
  h->ownInput = h->input(rank);
  check(cudaHostRegister(h->ownInput, h->maxBytes, cudaHostRegisterPortable));
  h->ownOutput = h->output(rank);
  check(cudaHostRegister(h->ownOutput, h->maxBytes, cudaHostRegisterPortable));
  if (h->shared->openedCount.fetch_add(1, std::memory_order_acq_rel) + 1 ==
      ranks)
    shm_unlink(h->name.c_str());
  return h;
}
}  // namespace

extern "C" void* uccl_lite_mp_create(const char* identifier, int ranks,
                                      int rank, size_t max_bytes) {
  try {
    return create(identifier, ranks, rank, max_bytes).release();
  } catch (const std::exception& e) {
    lastError = e.what();
    return nullptr;
  }
}
extern "C" int uccl_lite_mp_enqueue(void* handle, int operation,
                                     const void* input, void* output,
                                     size_t count, void* stream) {
  try {
    auto* h = static_cast<Handle*>(handle);
    if (!h || !input || !output ||
        (operation != 1 && operation != 2 && operation != 3) ||
        count > SIZE_MAX / (sizeof(uint16_t) * size_t(h->ranks)))
      throw std::invalid_argument("invalid collective call");
    size_t inputBytes = count * sizeof(uint16_t) *
                        (operation == 2 ? size_t(h->ranks) : 1);
    size_t outputBytes = count * sizeof(uint16_t) *
                         (operation == 1 ? size_t(h->ranks) : 1);
    if (inputBytes > h->maxBytes || outputBytes > h->maxBytes)
      throw std::invalid_argument("collective exceeds prepared capacity");
    auto cudaStream = static_cast<cudaStream_t>(stream);
    // Subsequent calls on this communicator must use the same stream. This
    // preserves ordering of the single shared staging slot.
    if (h->lastStream && h->lastStream != cudaStream)
      throw std::invalid_argument("communicator stream changed");
    h->lastStream = cudaStream;
    check(cudaMemcpyAsync(h->ownInput, input, inputBytes,
                          cudaMemcpyDeviceToHost, cudaStream));
    auto call = std::make_unique<Call>(Call{h, h->nextSequence, operation, count});
    check(cudaLaunchHostFunc(cudaStream, route, call.get()));
    call.release();
    check(cudaMemcpyAsync(output, h->ownOutput, outputBytes,
                          cudaMemcpyHostToDevice, cudaStream));
    ++h->nextSequence;
    return 0;
  } catch (const std::exception& e) {
    lastError = e.what();
    return -1;
  }
}
extern "C" int uccl_lite_mp_destroy(void* handle) {
  delete static_cast<Handle*>(handle);
  return 0;
}
extern "C" const char* uccl_lite_mp_last_error(void) {
  return lastError.c_str();
}
