#include "uccl_lite/runtime/profile.h"
#include <cassert>
#include <iostream>
#include <sstream>

using namespace uccl_lite;
using namespace uccl_lite::runtime;

std::string descriptor(BackendCapabilities const& caps,
                       bool duplicateHost = false) {
  std::ostringstream text;
  text << caps.profileMagic << '\n';
  for (int host = 0; host < caps.hosts; ++host) {
    text << "host" << (duplicateHost ? 0 : host) << " identity" << host
         << " 13000 12090 22809";
    for (int rank = 0; rank < caps.ranksPerHost; ++rank)
      text << " uuid" << host << '-' << rank;
    text << '\n';
  }
  return text.str();
}

int main() {
  BackendCapabilities local{"local",
                            "UCCL_LITE_LOCAL8_STRICT_V1",
                            1,
                            8,
                            ScalarType::Bfloat16,
                            2,
                            1UL << 30,
                            128};
  BackendCapabilities network{"multinode",
                              "UCCL_LITE_MSCCLPP_STRICT_V1",
                              2,
                              4,
                              ScalarType::Float32,
                              4,
                              1UL << 30,
                              128};
  BackendCapabilities local4{"local4",
                             "UCCL_LITE_MSCCLPP_LOCAL_STRICT_V1",
                             1,
                             4,
                             ScalarType::Float32,
                             4,
                             1UL << 30,
                             128};
  int checks = 0;
  for (auto const& caps : {local, network, local4}) {
    std::ostringstream shapes;
    for (int op = 1; op <= 3; ++op)
      for (int power = 10; power <= 30; ++power)
        shapes << op << ' '
               << (size_t(1) << power) /
                      caps.extentScale(static_cast<CollectiveOperation>(op))
               << '\n';
    std::string prefix = descriptor(caps), valid = prefix + shapes.str();
    for (int node = 0; node < caps.hosts; ++node) {
      ProfileIdentity identity{"host" + std::to_string(node),
                               "identity" + std::to_string(node), 13000, 12090,
                               22809};
      auto parse = [&](std::string const& data) {
        return parseBackendProfile(data, identity, caps);
      };
      auto profile = parse(valid);
      assert(profile.node == node && profile.shapes.size() == 63);
      assert(profile.capacity == (1UL << 30) &&
             profile.uuids.size() == size_t(caps.ranks()));
      assert(profile.digest == parse(valid).digest);
      ++checks;
      for (auto const& invalid :
           {std::string{}, prefix, prefix + "0 64\n", prefix + "4 64\n",
            prefix + "1 0\n", prefix + "1 1\n", prefix + "1 -1\n",
            prefix + "3 18446744073709551615\n", prefix + "1 64 extra\n",
            valid + "1 64\n", valid + "garbage\n",
            descriptor(caps, true) +
                (caps.hosts == 1 ? "invalid\n" : shapes.str())}) {
        assert(parse(invalid).shapes.empty());
        ++checks;
      }
      for (int field = 0; field < 5; ++field) {
        auto stale = identity;
        if (field == 0) stale.hostname = "other";
        if (field == 1) stale.verified = "";
        if (field == 2) ++stale.driverVersion;
        if (field == 3) ++stale.runtimeVersion;
        if (field == 4) ++stale.abiVersion;
        assert(parseBackendProfile(valid, stale, caps).shapes.empty());
        ++checks;
      }
      for (auto const& other : {local, network, local4}) {
        if (std::string(other.name) == caps.name) continue;
        assert(parseBackendProfile(valid, identity, other).shapes.empty());
        ++checks;
      }
    }
  }
  std::cout << "PASS backend profile contract: " << checks << " checks\n";
}
