#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

#include "tessera/calibrate.hpp"

// Concurrent-session calibration helpers: the largest session count whose
// KV caches fit a context, and the throughput aggregation for one benchmark
// row. Pure arithmetic, so the report and the command share one canonical
// implementation (AGENTS.md rule 2).

namespace tessera {

std::size_t MaxConcurrentSessions(std::uint64_t available_bytes,
                                  std::uint64_t kv_bytes_per_token,
                                  std::size_t context) {
  if (available_bytes == 0 || kv_bytes_per_token == 0 || context == 0) {
    return 0;
  }
  const std::uint64_t per_session =
      kv_bytes_per_token * static_cast<std::uint64_t>(context);
  if (per_session == 0) {
    return 0;
  }
  return static_cast<std::size_t>(available_bytes / per_session);
}

ConcurrentRate SummarizeConcurrentSessions(
    std::size_t sessions, std::span<const std::size_t> tokens,
    std::span<const double> decode_ms) {
  ConcurrentRate rate;
  rate.sessions = sessions;
  if (sessions == 0 || tokens.size() != sessions ||
      decode_ms.size() != sessions) {
    return rate;
  }
  std::size_t produced = 0;
  double wall_ms = 0.0;
  for (std::size_t i = 0; i < sessions; ++i) {
    produced += tokens[i];
    wall_ms = std::max(wall_ms, decode_ms[i]);
  }
  if (wall_ms <= 0.0) {
    return rate;
  }
  rate.total_tps = static_cast<double>(produced) * 1000.0 / wall_ms;
  rate.per_session_tps = rate.total_tps / static_cast<double>(sessions);
  return rate;
}

}  // namespace tessera
