#pragma once

#include "tessera/log.hpp"

#include <chrono>
#include <cstddef>
#include <string>
#include <string_view>

namespace tessera::core {

// Monotonic clock for load and setup phase timing.
using PhaseClock = std::chrono::steady_clock;

// Milliseconds between two points of PhaseClock.
[[nodiscard]] inline long long ElapsedMs(PhaseClock::time_point start,
                                         PhaseClock::time_point end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

// One-line summary of a completed bulk transfer for diagnostics, shared by
// the backends so both report the same shape:
// "batched H2D upload of N bytes in M copies took T ms (R MiB/s)".
[[nodiscard]] inline std::string FormatTransferSummary(
    unsigned long long bytes, std::size_t copies, long long ms) {
  const unsigned long long mib_per_s =
      ms > 0 ? bytes * 1000 / (1024 * 1024) /
                   static_cast<unsigned long long>(ms)
             : 0;
  return "batched H2D upload of " + std::to_string(bytes) + " bytes in " +
         std::to_string(copies) + " copies took " + std::to_string(ms) +
         " ms (" + std::to_string(mib_per_s) + " MiB/s)";
}

// RAII wall-clock timer for one named load or setup phase. It logs
// "phase '<name>' done in N ms" once, on Stop or scope exit, with an
// optional detail; with a null `log` it is a no-op.
class PhaseTimer {
 public:
  PhaseTimer(const log::Diagnostics* log, std::string_view prefix,
             std::string_view phase)
      : log_(log), prefix_(prefix), phase_(phase),
        started_(PhaseClock::now()) {}

  PhaseTimer(const PhaseTimer&) = delete;
  PhaseTimer& operator=(const PhaseTimer&) = delete;

  // Logs the phase result now; a later call or the destructor does nothing.
  void Stop(std::string detail = {}) {
    if (!active_) {
      return;
    }
    active_ = false;
    if (log_ == nullptr) {
      return;
    }
    std::string message =
        "phase '" + std::string(phase_) + "' done in " +
        std::to_string(ElapsedMs(started_, PhaseClock::now())) + " ms";
    if (!detail.empty()) {
      message += " (" + detail + ")";
    }
    log_->Info(prefix_, message);
  }

  ~PhaseTimer() { Stop(); }

 private:
  const log::Diagnostics* log_;
  std::string prefix_;
  std::string phase_;
  PhaseClock::time_point started_;
  bool active_ = true;
};

}  // namespace tessera::core
