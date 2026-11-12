#include "uccl_lite/multiprocess_collectives.h"

#include <nccl.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#ifndef UCCL_LITE_REAL_NCCL
#define UCCL_LITE_REAL_NCCL "libnccl.so.2"
#endif

namespace {
void* realLibrary() {
  static void* library = [] {
    const char* path = std::getenv("UCCL_LITE_REAL_NCCL");
    if (!path || !*path) path = UCCL_LITE_REAL_NCCL;
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      std::fprintf(stderr, "UCCL Lite NCCL proxy: dlopen %s: %s\n", path,
                   dlerror());
      std::abort();
    }
    return handle;
  }();
  return library;
}
template <typename Fn> Fn real(const char* name) {
  auto* symbol = dlsym(realLibrary(), name);
  if (!symbol) {
    std::fprintf(stderr, "UCCL Lite NCCL proxy: missing %s\n", name);
    std::abort();
  }
  return reinterpret_cast<Fn>(symbol);
}

struct State {
  std::string id;
  int ranks;
  int rank;
  void* collective = nullptr;
  std::mutex lock;
  uint64_t agCalls = 0;
  uint64_t rsCalls = 0;
  ~State() {
    if (collective) {
      uccl_lite_mp_destroy(collective);
      std::fprintf(stderr,
                   "UCCL Lite NCCL proxy: rank=%d completed AG=%llu RS=%llu\n",
                   rank, static_cast<unsigned long long>(agCalls),
                   static_cast<unsigned long long>(rsCalls));
    }
  }
};
std::mutex statesLock;
std::unordered_map<ncclComm_t, std::shared_ptr<State>> states;
thread_local int groupDepth = 0;
bool enabled();

std::string hashId(ncclUniqueId id) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : id.internal) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  char result[17];
  std::snprintf(result, sizeof(result), "%016llx",
                static_cast<unsigned long long>(hash));
  return result;
}
std::shared_ptr<State> stateFor(ncclComm_t comm) {
  std::lock_guard<std::mutex> guard(statesLock);
  auto it = states.find(comm);
  return it == states.end() ? nullptr : it->second;
}
void remember(ncclComm_t comm, int ranks, ncclUniqueId id, int rank) {
  if (!enabled()) return;
  auto state = std::make_shared<State>();
  state->id = hashId(id);
  state->ranks = ranks;
  state->rank = rank;
  std::lock_guard<std::mutex> guard(statesLock);
  states[comm] = state;
  std::fprintf(stderr, "UCCL Lite NCCL proxy: communicator rank=%d/%d id=%s\n",
               rank, ranks, state->id.c_str());
}
bool enabled() {
  const char* value = std::getenv("UCCL_LITE_NCCL_ENABLE");
  return value && std::strcmp(value, "1") == 0;
}
size_t maxBytes() {
  const char* value = std::getenv("UCCL_LITE_NCCL_MAX_BYTES");
  if (!value || !*value) return 32UL << 20;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (!end || *end || !parsed || parsed > (1ULL << 30)) return 0;
  return static_cast<size_t>(parsed);
}
ncclResult_t dispatch(int operation, const void* send, void* recv, size_t count,
                      ncclDataType_t dtype, ncclRedOp_t reduction,
                      ncclComm_t comm, cudaStream_t stream) {
  auto s = stateFor(comm);
  size_t capacity = maxBytes();
  size_t bytes = count * sizeof(uint16_t);
  if (!enabled() || !s || groupDepth || s->ranks != 8 ||
      dtype != ncclBfloat16 || (operation == 2 && reduction != ncclSum) ||
      !capacity || count > SIZE_MAX / (sizeof(uint16_t) * 8) ||
      bytes * size_t(s->ranks) > capacity || !send || !recv)
    return ncclInvalidUsage;
  std::lock_guard<std::mutex> guard(s->lock);
  if (!s->collective) {
    s->collective = uccl_lite_mp_create(s->id.c_str(), s->ranks, s->rank,
                                         capacity);
    if (!s->collective) {
      std::fprintf(stderr, "UCCL Lite NCCL proxy: create: %s\n",
                   uccl_lite_mp_last_error());
      return ncclSystemError;
    }
  }
  if (uccl_lite_mp_enqueue(s->collective, operation, send, recv, count,
                            stream) != 0) {
    std::fprintf(stderr, "UCCL Lite NCCL proxy: enqueue: %s\n",
                 uccl_lite_mp_last_error());
    return ncclSystemError;
  }
  uint64_t calls = operation == 1 ? ++s->agCalls : ++s->rsCalls;
  if (calls == 1)
    std::fprintf(stderr, "UCCL Lite NCCL proxy: rank=%d op=%s BF16 count=%zu\n",
                 s->rank, operation == 1 ? "AG" : "RS", count);
  const char* trace = std::getenv("UCCL_LITE_NCCL_TRACE");
  if (s->rank == 0 && trace && std::strcmp(trace, "1") == 0)
    std::fprintf(stderr,
                 "UCCL Lite NCCL proxy TRACE rank=0 op=%s calls=%llu "
                 "ag_calls=%llu rs_calls=%llu count=%zu comm=%s\n",
                 operation == 1 ? "AG" : "RS",
                 static_cast<unsigned long long>(calls),
                 static_cast<unsigned long long>(s->agCalls),
                 static_cast<unsigned long long>(s->rsCalls), count,
                 s->id.c_str());
  return ncclSuccess;
}
}  // namespace

extern "C" {
const char* ncclGetErrorString(ncclResult_t result) {
  return real<decltype(&ncclGetErrorString)>("ncclGetErrorString")(result);
}
ncclResult_t ncclGetVersion(int* version) {
  return real<decltype(&ncclGetVersion)>("ncclGetVersion")(version);
}
ncclResult_t ncclGetUniqueId(ncclUniqueId* id) {
  return real<decltype(&ncclGetUniqueId)>("ncclGetUniqueId")(id);
}
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ranks, ncclUniqueId id,
                              int rank) {
  auto result = real<decltype(&ncclCommInitRank)>("ncclCommInitRank")(
      comm, ranks, id, rank);
  if (result == ncclSuccess) remember(*comm, ranks, id, rank);
  return result;
}
ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int ranks,
                                    ncclUniqueId id, int rank,
                                    ncclConfig_t* config) {
  auto result = real<decltype(&ncclCommInitRankConfig)>("ncclCommInitRankConfig")(
      comm, ranks, id, rank, config);
  if (result == ncclSuccess) remember(*comm, ranks, id, rank);
  return result;
}
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  {
    std::lock_guard<std::mutex> guard(statesLock);
    states.erase(comm);
  }
  return real<decltype(&ncclCommDestroy)>("ncclCommDestroy")(comm);
}
ncclResult_t ncclAllGather(const void* send, void* recv, size_t count,
                           ncclDataType_t dtype, ncclComm_t comm,
                           cudaStream_t stream) {
  auto s = stateFor(comm);
  size_t capacity = maxBytes();
  if (!enabled() || !s || groupDepth || s->ranks != 8 ||
      dtype != ncclBfloat16 || !capacity ||
      count > capacity / (sizeof(uint16_t) * size_t(s->ranks)))
    return real<decltype(&ncclAllGather)>("ncclAllGather")(
        send, recv, count, dtype, comm, stream);
  return dispatch(1, send, recv, count, dtype, ncclSum, comm, stream);
}
ncclResult_t ncclReduceScatter(const void* send, void* recv, size_t count,
                               ncclDataType_t dtype, ncclRedOp_t op,
                               ncclComm_t comm, cudaStream_t stream) {
  auto s = stateFor(comm);
  size_t capacity = maxBytes();
  if (!enabled() || !s || groupDepth || s->ranks != 8 ||
      dtype != ncclBfloat16 || op != ncclSum || !capacity ||
      count > capacity / (sizeof(uint16_t) * size_t(s->ranks)))
    return real<decltype(&ncclReduceScatter)>("ncclReduceScatter")(
        send, recv, count, dtype, op, comm, stream);
  return dispatch(2, send, recv, count, dtype, op, comm, stream);
}
ncclResult_t ncclAllReduce(const void* send, void* recv, size_t count,
                           ncclDataType_t dtype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
  return real<decltype(&ncclAllReduce)>("ncclAllReduce")(
      send, recv, count, dtype, op, comm, stream);
}
ncclResult_t ncclReduce(const void* send, void* recv, size_t count,
                        ncclDataType_t dtype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
  return real<decltype(&ncclReduce)>("ncclReduce")(
      send, recv, count, dtype, op, root, comm, stream);
}
ncclResult_t ncclBroadcast(const void* send, void* recv, size_t count,
                           ncclDataType_t dtype, int root,
                           ncclComm_t comm, cudaStream_t stream) {
  return real<decltype(&ncclBroadcast)>("ncclBroadcast")(
      send, recv, count, dtype, root, comm, stream);
}
ncclResult_t ncclSend(const void* send, size_t count, ncclDataType_t dtype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  return real<decltype(&ncclSend)>("ncclSend")(
      send, count, dtype, peer, comm, stream);
}
ncclResult_t ncclRecv(void* recv, size_t count, ncclDataType_t dtype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  return real<decltype(&ncclRecv)>("ncclRecv")(
      recv, count, dtype, peer, comm, stream);
}
ncclResult_t ncclGroupStart() {
  ++groupDepth;
  return real<decltype(&ncclGroupStart)>("ncclGroupStart")();
}
ncclResult_t ncclGroupEnd() {
  auto result = real<decltype(&ncclGroupEnd)>("ncclGroupEnd")();
  if (groupDepth) --groupDepth;
  return result;
}
}  // extern "C"
