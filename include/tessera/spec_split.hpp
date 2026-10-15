#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "tessera/defaults.hpp"
#include "tessera/types.hpp"

namespace tessera {

class Tokenizer;

// Options for speculative-split generation: the target runs on
// devices[0] and the MTP draft head on devices[1]. The draft device
// holds only the head shard (embedding, MTP block, output head); the
// anchor hidden travels as host floats, so no per-step device copy is
// needed and the verify runs wholly on the target.
struct SpecSplitOptions {
  std::string model_path;
  std::vector<int> devices;  // [target, draft], in that order
  std::size_t context = kDefaultContext;
  KvCacheType kv_type = KvCacheType::F32;
  // Tokens per draft block; 0 uses the strategy default.
  std::size_t draft_block = 0;
  // Prefill tokens per forward; 0 selects the automatic default.
  std::size_t prefill_chunk = 0;
  // Directory of a separately fetched MTP head (tessera-cli fetch-mtp).
  // The draft device reads it; the target does not.
  std::string mtp_path;
  // Layers kept in VRAM on the target device; 0 uploads all. A model too
  // large for one device pages its trunk blocks from system RAM.
  std::size_t gpu_layers = 0;
};

// MTP speculation with the draft head on a second device. The target
// forward, the verify batch and the accept rule run on the target; the
// MTP draft steps run against the head shard on the draft device. Greedy
// output equals the single-device speculative run, which equals plain
// greedy. DFlash2 split is a follow-up; this driver only speaks MTP.
//
// Usage:
//   SpecSplitOptions options;
//   options.model_path = path;
//   options.devices = {0, 1};
//   auto split = SpecSplit::Create(options);
//   if (split) { auto ids = (*split)->Generate(prompt, 32); }
class SpecSplit {
 public:
  static std::expected<std::unique_ptr<SpecSplit>, StatusCode> Create(
      const SpecSplitOptions& options);

  ~SpecSplit();
  SpecSplit(SpecSplit&& other) noexcept;
  SpecSplit& operator=(SpecSplit&& other) noexcept;
  SpecSplit(const SpecSplit&) = delete;
  SpecSplit& operator=(const SpecSplit&) = delete;

  // Greedy MTP generation with a split draft head. `prompt` seeds; up to
  // `max_tokens` (0 fills the context). Stops at a declared stop token.
  std::expected<std::vector<std::uint32_t>, StatusCode> Generate(
      std::span<const std::uint32_t> prompt, std::size_t max_tokens);

  [[nodiscard]] const Tokenizer* GetTokenizer() const;
  [[nodiscard]] std::expected<std::string, StatusCode> ChatPrompt(
      std::string_view user_text, bool enable_thinking = false,
      std::string_view reasoning_effort = "") const;

 private:
  struct Impl;
  explicit SpecSplit(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace tessera
