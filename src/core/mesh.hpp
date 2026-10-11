#pragma once

#include <cstddef>
#include <expected>
#include <vector>

#include "tessera/types.hpp"

namespace tessera::core {

// Canonical device mesh for hybrid parallelism: `dp_groups`
// data-parallel groups of `mp_ranks` model-parallel ranks each. Group `g`
// owns `devices[g * mp_ranks .. (g + 1) * mp_ranks)`. Every multi-device
// route reads this one map, so a mesh means the same thing everywhere.
struct DeviceMesh {
  std::size_t dp_groups = 0;
  std::size_t mp_ranks = 0;
  std::vector<int> devices;  // dp_groups * mp_ranks entries, group-major
};

// The mesh is usable: at least one group and rank, one device index per
// slot, no negative indices. Device availability is the caller's check
// (engines fail clearly on a bad index).
[[nodiscard]] inline std::expected<void, StatusCode> ValidateMesh(
    const DeviceMesh& mesh) {
  if (mesh.dp_groups == 0 || mesh.mp_ranks == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (mesh.devices.size() != mesh.dp_groups * mesh.mp_ranks) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (const int device : mesh.devices) {
    if (device < 0) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
  }
  return {};
}

// Device index of model-parallel rank `rank` in data-parallel group
// `group`. The mesh must pass ValidateMesh first.
[[nodiscard]] inline int MeshDevice(const DeviceMesh& mesh, std::size_t group,
                                    std::size_t rank) {
  return mesh.devices[group * mesh.mp_ranks + rank];
}

}  // namespace tessera::core
