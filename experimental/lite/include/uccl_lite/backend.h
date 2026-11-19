#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <cuda_runtime_api.h>

namespace uccl_lite {

enum class CollectiveOperation {
  AllGather = 1,
  ReduceScatter = 2,
  AllReduce = 3
};
enum class ScalarType { Bfloat16, Float32 };
using BackendId = std::array<unsigned char, 128>;

// Describes the explicitly linked backend, never an automatic routing rule.
// Admission additionally requires an exact, verified runtime profile.
struct BackendCapabilities {
  char const* name;
  char const* profileMagic;
  int hosts;
  int ranksPerHost;
  ScalarType dtype;
  size_t elementBytes;
  size_t maxBytes;
  size_t extentAlignment;

  int ranks() const { return hosts * ranksPerHost; }
  size_t extentScale(CollectiveOperation op) const {
    return elementBytes * (op == CollectiveOperation::AllReduce ? 1 : ranks());
  }
};

struct BackendConfig {
  BackendId id;
  int rank;
  int device;
  size_t capacity;
};

// Construction is CPU-only. prepare() owns resource initialization; enqueue()
// preserves caller-stream completion and buffer reuse semantics. Callers
// serialize host submissions. Backends own their algorithms and staging.
class CollectiveBackend {
 public:
  virtual ~CollectiveBackend() = default;
  virtual void prepare() = 0;
  virtual void enqueue(CollectiveOperation op, void const* input, void* output,
                       size_t count, cudaStream_t stream) = 0;
  virtual void synchronize() = 0;
  virtual bool initialized() const = 0;
  virtual bool idle() const = 0;
};

// Each library links exactly one implementation of this factory contract.
// Neither the interface nor the local backend depends on NCCL/MSCCL++/MPI.
BackendCapabilities const& backendCapabilities();
BackendId createBackendId();
std::unique_ptr<CollectiveBackend> createBackend(BackendConfig const& config);

}  // namespace uccl_lite
