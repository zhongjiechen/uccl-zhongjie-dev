#include "uccl_lite/runtime/profile.h"
#include <algorithm>
#include <set>
#include <sstream>

namespace uccl_lite::runtime {
BackendProfile parseBackendProfile(std::string const& data,
                                   ProfileIdentity const& identity,
                                   BackendCapabilities const& caps) {
  BackendProfile result;
  if (identity.verified.empty() || caps.hosts < 1 || caps.ranksPerHost < 1 ||
      !caps.elementBytes || !caps.extentAlignment || !caps.maxBytes)
    return {};
  std::istringstream input(data);
  std::string magic;
  if (!(input >> magic) || magic != caps.profileMagic) return {};
  result.hosts.resize(caps.hosts);
  result.identities.resize(caps.hosts);
  result.uuids.resize(caps.ranks());
  std::set<std::string> hosts;
  for (int n = 0; n < caps.hosts; ++n) {
    int driver, runtime, abi;
    if (!(input >> result.hosts[n] >> result.identities[n] >> driver >>
          runtime >> abi) ||
        !hosts.insert(result.hosts[n]).second)
      return {};
    for (int i = 0; i < caps.ranksPerHost; ++i)
      if (!(input >> result.uuids[caps.ranksPerHost * n + i])) return {};
    if (result.hosts[n] == identity.hostname &&
        result.identities[n] == identity.verified &&
        driver == identity.driverVersion &&
        runtime == identity.runtimeVersion && abi == identity.abiVersion)
      result.node = n;
  }
  if (result.node < 0) return {};
  std::string line;
  std::getline(input, line);
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::istringstream row(line);
    int operation;
    size_t count;
    std::string extra;
    if (!(row >> operation >> count) || row >> extra || operation < 1 ||
        operation > 3 || !count)
      return {};
    size_t scale =
        caps.extentScale(static_cast<CollectiveOperation>(operation));
    if (count > caps.maxBytes / scale ||
        (count * scale) % caps.extentAlignment ||
        !result.shapes.emplace(std::make_pair(operation, count), true).second)
      return {};
    result.capacity = std::max(result.capacity, count * scale);
  }
  result.digest = 14695981039346656037ULL;
  for (unsigned char c : data)
    result.digest = (result.digest ^ c) * 1099511628211ULL;
  return result;
}
}  // namespace uccl_lite::runtime
