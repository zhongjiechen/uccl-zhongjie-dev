#include <nccl.h>
#include <cuda_runtime_api.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return uint16_t(bits >> 16);
}
void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(result));
    std::exit(2);
  }
}
void check(ncclResult_t result) {
  if (result != ncclSuccess) {
    std::fprintf(stderr, "NCCL: %s\n", ncclGetErrorString(result));
    std::exit(3);
  }
}
void rankRun(int rank, ncclUniqueId id) {
  check(cudaSetDevice(rank));
  ncclComm_t comm = nullptr;
  check(ncclCommInitRank(&comm, 8, id, rank));
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (size_t count : {size_t(8), size_t(1009), size_t(16384), size_t(131072)}) {
    std::vector<uint16_t> hostInput(count);
    for (size_t i = 0; i < count; ++i)
      hostInput[i] = bf16(float(rank + 1 + i % 4));
    std::vector<uint16_t> hostOutput(count * 8);
    void *input = nullptr, *output = nullptr;
    check(cudaMalloc(&input, count * 2));
    check(cudaMalloc(&output, count * 8 * 2));
    check(cudaMemcpyAsync(input, hostInput.data(), count * 2,
                          cudaMemcpyHostToDevice, stream));
    check(ncclAllGather(input, output, count, ncclBfloat16, comm, stream));
    check(cudaMemcpyAsync(hostOutput.data(), output, count * 8 * 2,
                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    for (int r = 0; r < 8; ++r)
      for (size_t i = 0; i < count; ++i)
        if (hostOutput[size_t(r) * count + i] !=
            bf16(float(r + 1 + i % 4))) {
          std::fprintf(stderr, "AG mismatch rank=%d count=%zu source=%d i=%zu\n",
                       rank, count, r, i);
          std::exit(4);
        }
    check(cudaFree(input));
    check(cudaFree(output));

    // Distinct exact BF16 values per destination check RS shard placement.
    hostInput.resize(count * 8);
    for (int destination = 0; destination < 8; ++destination)
      for (size_t i = 0; i < count; ++i)
        hostInput[size_t(destination) * count + i] =
            bf16(float(rank + 1 + destination * 8));
    hostOutput.assign(count, 0);
    check(cudaMalloc(&input, count * 8 * 2));
    check(cudaMalloc(&output, count * 2));
    check(cudaMemcpyAsync(input, hostInput.data(), count * 8 * 2,
                          cudaMemcpyHostToDevice, stream));
    check(ncclReduceScatter(input, output, count, ncclBfloat16, ncclSum,
                            comm, stream));
    check(cudaMemcpyAsync(hostOutput.data(), output, count * 2,
                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < count; ++i)
      if (hostOutput[i] != bf16(float(36 + rank * 64))) {
        std::fprintf(stderr, "RS mismatch rank=%d count=%zu i=%zu got=%04x\n",
                     rank, count, i, hostOutput[i]);
        std::exit(5);
      }
    check(cudaFree(input));
    check(cudaFree(output));
  }
  check(cudaStreamDestroy(stream));
  check(ncclCommDestroy(comm));
}
}  // namespace

int main() {
  ncclUniqueId id;
  check(ncclGetUniqueId(&id));
  std::vector<pid_t> children;
  for (int rank = 0; rank < 8; ++rank) {
    pid_t pid = fork();
    if (pid == 0) {
      rankRun(rank, id);
      std::_Exit(0);
    }
    if (pid < 0) return 1;
    children.push_back(pid);
  }
  bool passed = true;
  for (pid_t pid : children) {
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status)) {
      std::fprintf(stderr, "child %d status=%d signal=%d exit=%d\n", pid,
                   status, WIFSIGNALED(status) ? WTERMSIG(status) : 0,
                   WIFEXITED(status) ? WEXITSTATUS(status) : -1);
      passed = false;
    }
  }
  std::puts(passed ? "UCCL Lite NCCL AG/RS 8-process check passed" :
                     "UCCL Lite NCCL AG/RS 8-process check failed");
  return passed ? 0 : 1;
}
