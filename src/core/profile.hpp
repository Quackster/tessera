#pragma once

#include "tessera/log.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <string_view>

namespace tessera::core {

// Named decode phases of the opt-in decoder profiler. Coarse phases are
// wall-clock inclusive (their calls end in a host readback, so the time is
// real GPU time); the projector sub-phases synchronize when profiling is
// on, so each one is attributed to the GPU work it launched.
enum class Phase : std::size_t {
  // One DecodeLogits of the just-emitted anchor token.
  kAnchor,
  // Every target trunk forward, including the ones inside a verify.
  kForward,
  // Every target output head (the vocab projection).
  kHead,
  // Every target logits download to the host.
  kDownload,
  // The whole strategy Draft call (sum of the four draft sub-phases).
  kDraft,
  kDraftPrep,
  kDraftBlock,
  kDraftHead,
  kDraftArgmax,
  // The whole VerifyDraft call.
  kVerify,
  // One full-attention layer (RunFullBlock, including its FFN) and one
  // linear-attention layer (RunLinearBlock, including its FFN), plus the
  // FFN alone (nested inside both). Splits the trunk so a slow phase is
  // attributable.
  kFullLayer,
  kLinearLayer,
  kFfn,
  kCount,
};

// Total decode phases (one past the last), for loop bounds.
inline constexpr std::size_t kPhaseCount = static_cast<std::size_t>(Phase::kCount);

// Human name of a phase, for the profile report.
[[nodiscard]] inline std::string_view PhaseName(Phase phase) {
  switch (phase) {
  case Phase::kAnchor:
    return "anchor";
  case Phase::kForward:
    return "target forward";
  case Phase::kHead:
    return "target head";
  case Phase::kDownload:
    return "target logits download";
  case Phase::kDraft:
    return "draft";
  case Phase::kDraftPrep:
    return "draft prep";
  case Phase::kDraftBlock:
    return "draft block";
  case Phase::kDraftHead:
    return "draft head";
  case Phase::kDraftArgmax:
    return "draft argmax";
  case Phase::kVerify:
    return "verify";
  case Phase::kFullLayer:
    return "full-attn layer";
  case Phase::kLinearLayer:
    return "linear-attn layer";
  case Phase::kFfn:
    return "ffn";
  case Phase::kCount:
    break;
  }
  return "unknown";
}

// Wall-clock accumulator for the decode phases. One instance per
// generation, owned by the engine and referenced from the decode cache; a
// null pointer disables profiling at no cost (PhaseScope checks). Not
// thread-safe: called from the generation thread only.
class Profile {
public:
  // Deep mode synchronizes at the fine-grained (per-layer, per-op)
  // boundaries so those phases are attributable; it perturbs the pipeline
  // and is off by default. Coarse phases synchronize at readbacks anyway.
  explicit Profile(bool deep = false) : deep_(deep) {}

  [[nodiscard]] bool Deep() const { return deep_; }

  void Add(Phase phase, long long nanoseconds) {
    const std::size_t index = static_cast<std::size_t>(phase);
    if (index >= kPhaseCount) {
      return;
    }
    ns_[index] += nanoseconds;
    count_[index] += 1;
  }

  [[nodiscard]] long long TotalNs(Phase phase) const {
    return ns_[static_cast<std::size_t>(phase)];
  }

  [[nodiscard]] std::size_t Count(Phase phase) const {
    return count_[static_cast<std::size_t>(phase)];
  }

  // One line per observed phase: total, call count and mean. The prefix
  // identifies the engine. Deliberately separate from the decode summary
  // so a reader can diff phase by phase.
  void Report(const log::Diagnostics& log, std::string_view prefix,
              std::size_t steps) const {
    for (std::size_t i = 0; i < kPhaseCount; ++i) {
      if (count_[i] == 0) {
        continue;
      }
      const long long ms = ns_[i] / 1'000'000;
      const long long avg = ms / static_cast<long long>(count_[i]);
      const long long per_step =
          steps > 0 ? ms / static_cast<long long>(steps) : 0;
      const std::string message =
          "profile: " + std::string(PhaseName(static_cast<Phase>(i))) + " " +
          std::to_string(ms) + " ms over " + std::to_string(count_[i]) +
          " call(s), " + std::to_string(avg) + " ms avg, " +
          std::to_string(per_step) + " ms/step";
      log.Info(prefix, message);
    }
  }

 private:
  bool deep_ = false;
  std::array<long long, kPhaseCount> ns_{};
  std::array<std::size_t, kPhaseCount> count_{};
};

// RAII timer for one phase; null-safe and cheap.
class PhaseScope {
 public:
  PhaseScope(Profile* profile, Phase phase)
      : profile_(profile), phase_(phase) {
    if (profile_ != nullptr) {
      started_ = std::chrono::steady_clock::now();
    }
  }

  PhaseScope(const PhaseScope&) = delete;
  PhaseScope& operator=(const PhaseScope&) = delete;

  ~PhaseScope() {
    if (profile_ != nullptr) {
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started_)
              .count();
      profile_->Add(phase_, elapsed);
    }
  }

 private:
  Profile* profile_;
  Phase phase_;
  std::chrono::steady_clock::time_point started_{};
};

// Env switch. TESSERA_PROFILE=1 (any non-empty value except "0") turns on
// the decode phase profiler; TESSERA_PROFILE=2 (or "deep") adds the
// fine-grained layers with a synchronization per phase. Unset keeps the
// decode loop at zero overhead.
[[nodiscard]] inline bool DecodeProfilingEnabled() {
  const char* value = std::getenv("TESSERA_PROFILE");
  return value != nullptr && *value != '\0' && std::string_view(value) != "0";
}

// True when the caller asked for the deep profile (TESSERA_PROFILE=2).
[[nodiscard]] inline bool DecodeProfilingDeep() {
  const char* value = std::getenv("TESSERA_PROFILE");
  if (value == nullptr) {
    return false;
  }
  const std::string_view view(value);
  return view == "2" || view == "deep";
}

}  // namespace tessera::core
