#include "uccl_lite/backend.h"
#include "uccl_lite/backends/multinode.h"

namespace uccl_lite {
namespace {
class MultinodeBackend final : public CollectiveBackend {
 public:
  explicit MultinodeBackend(BackendConfig const& config)
      : collectives_(config.id, config.rank, config.device, config.device / 2,
                     config.capacity) {}
  void prepare() override { collectives_.prepare(); }
  void enqueue(CollectiveOperation op, void const* input, void* output,
               size_t count, cudaStream_t stream) override {
    collectives_.enqueue(static_cast<NetworkOperation>(op), input, output,
                         count, stream);
  }
  void synchronize() override { collectives_.synchronize(); }
  bool initialized() const override { return collectives_.initialized(); }
  bool idle() const override { return collectives_.idle(); }

 private:
  NetworkCollectives collectives_;
};
}  // namespace

BackendCapabilities const& backendCapabilities() {
  static constexpr BackendCapabilities capabilities{
      "multinode",
      "UCCL_LITE_MSCCLPP_STRICT_V1",
      2,
      4,
      ScalarType::Float32,
      4,
      1UL << 30,
      128};
  return capabilities;
}
BackendId createBackendId() { return NetworkCollectives::createUniqueId(); }
std::unique_ptr<CollectiveBackend> createBackend(BackendConfig const& config) {
  return std::make_unique<MultinodeBackend>(config);
}
}  // namespace uccl_lite
