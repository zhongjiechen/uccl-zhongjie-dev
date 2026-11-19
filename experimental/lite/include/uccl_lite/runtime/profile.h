#pragma once

#include "uccl_lite/backend.h"
#include <map>
#include <string>
#include <vector>

namespace uccl_lite::runtime {
struct ProfileIdentity {
  std::string hostname;
  std::string verified;
  int driverVersion;
  int runtimeVersion;
  int abiVersion;
};

struct BackendProfile {
  std::map<std::pair<int, size_t>, bool> shapes;
  std::vector<std::string> hosts, identities, uuids;
  uint64_t digest = 0;
  size_t capacity = 0;
  int node = -1;
};

// Pure profile parsing: no CUDA calls, allocations on devices, or workers.
// Invalid/stale profiles return an empty shape set and are never admitted.
BackendProfile parseBackendProfile(std::string const& data,
                                   ProfileIdentity const& identity,
                                   BackendCapabilities const& capabilities);
}  // namespace uccl_lite::runtime
