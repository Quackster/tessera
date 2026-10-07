#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <string>
#include <string_view>

#include "tessera/types.hpp"

namespace tessera {

// Configuration for a speculative-decoding strategy.
struct StrategyOptions {
  // Directory holding the draft checkpoint (config.json + weights).
  std::string draft_path;
  // Tokens per draft block; a tunable, not a format constant.
  std::size_t draft_block_size = 4;
};

// Strategy interface for speculative decoding (DFlash2 today; DFlash,
// DSpark, others later). Strategies are off by default; the
// non-speculative path is the reference baseline and must stay stable
// within per-backend tolerance.
class SpeculativeStrategy {
 public:
  virtual ~SpeculativeStrategy() = default;

  // "dflash2" etc.
  [[nodiscard]] virtual std::string_view Name() const = 0;

  // Validate the draft checkpoint layout and prepare for use.
  // Idempotent: a second call returns Ok.
  virtual std::expected<void, StatusCode> Attach(
      const StrategyOptions& options) = 0;
};

// Factory for the DFlash2 strategy (implemented in src/spec/).
//
// Usage:
//   auto strategy = CreateDFlash2Strategy();
//   strategy->Attach(StrategyOptions{draft_dir, 4});
//   engine.AttachSpeculative(std::move(strategy));
[[nodiscard]] std::unique_ptr<SpeculativeStrategy> CreateDFlash2Strategy();

}  // namespace tessera
