#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"

namespace tessera {

class Model;
namespace core {
class DecodeCache;
}

// Configuration for a speculative-decoding strategy.
struct StrategyOptions {
  // Directory holding the draft checkpoint (config.json + weights).
  std::string draft_path;
  // Tokens per draft block; 0 uses the checkpoint's configured block size.
  std::size_t draft_block_size = 0;
};

// A speculative-decoding strategy plugged into the engine's single decode
// loop. The engine owns the target, the KV/recurrent state and the accept
// rule; the strategy owns the draft model, the draft policy and the target
// hidden capture buffers it reads. With no strategy attached the same loop
// is plain greedy. This mirrors hipfire's Speculator/SpecTarget seam: one
// acceptance window per `Draft`, one greedy accept rule (`VerifyDraft`)
// shared by every drafter.
//
// Thread-safety: a strategy is owned by one Engine and called from the
// generation thread only; externally synchronized.
class SpeculativeStrategy {
 public:
  virtual ~SpeculativeStrategy() = default;

  // "dflash2", "mtp", ...
  [[nodiscard]] virtual std::string_view Name() const = 0;

  // Validate the draft checkpoint layout and prepare for use.
  // Idempotent: a second call returns Ok.
  virtual std::expected<void, StatusCode> Attach(
      const StrategyOptions& options) = 0;

  // Load the draft model and allocate the capture buffers. Called once
  // before a generation; idempotent. A stateless drafter may return Ok.
  virtual std::expected<void, StatusCode> Prepare(Backend& backend,
                                                  Model& target) = 0;

  // Maximum draft tokens proposed per window (0 = strategy default).
  [[nodiscard]] virtual std::size_t DraftBlock() const = 0;

  // True when the engine folds the per-step anchor (the just-emitted token)
  // into the draft verification batch instead of running a separate
  // single-token target forward for it. The verify then processes
  // [anchor, drafts...] and the strategy reads the anchor's captured
  // features from the verify: OnAnchor is called *after* the verify, not
  // before Draft. False keeps the classic bonus-forward-then-draft order
  // (MTP). A strategy that folds must propose more than one draft per step
  // for the batch to amortize the anchor row.
  [[nodiscard]] virtual bool FoldsAnchor() const { return false; }

  // Block indices whose per-position residual hidden the target must capture
  // for the drafter (empty when the drafter does not read target hidden).
  [[nodiscard]] virtual std::span<const std::size_t> CaptureLayers() const = 0;

  // The capture buffers the engine passes to the target forward: one device
  // buffer per CaptureLayers() entry, each DraftBlock() rows x hidden. Empty
  // when CaptureLayers() is empty.
  [[nodiscard]] virtual std::span<Buffer* const> CaptureBuffers() = 0;

  // The target has advanced by `token` at absolute `position`; `hidden` is
  // that position's target residual hidden. The strategy's own capture buffers
  // hold the per-layer residual hidden for the same position. Record the
  // pre-draft state and append the anchor position to the draft context. A
  // stateless drafter only records the pre-draft state.
  virtual std::expected<void, StatusCode> OnAnchor(
      Backend& backend, Model& target, core::DecodeCache& cache,
      std::uint32_t token, std::uint64_t position,
      std::span<const float> hidden) = 0;

  // Propose up to `drafts.size()` tokens after `anchor`, given the target's
  // current logits. Returns the number of drafts written (0 = no proposal,
  // the loop falls back to the target's greedy token).
  virtual std::expected<std::size_t, StatusCode> Draft(
      Backend& backend, Model& target, core::DecodeCache& cache,
      std::span<const float> current_logits, std::uint32_t anchor,
      std::span<std::uint32_t> drafts) = 0;

  // The verifier accepted the first `accepted` drafts. The capture buffers
  // hold the accepted positions' residual hidden (`accepted` rows each);
  // advance the drafter state (append context, truncate the draft KV).
  virtual std::expected<void, StatusCode> Commit(
      Backend& backend, Model& target, core::DecodeCache& cache,
      std::size_t accepted) = 0;
};

// Factories (implemented in src/spec/).
[[nodiscard]] std::unique_ptr<SpeculativeStrategy> CreateDFlash2Strategy();
[[nodiscard]] std::unique_ptr<SpeculativeStrategy> CreateMtpStrategy();

}  // namespace tessera
