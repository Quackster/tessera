#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "tessera/defaults.hpp"
#include "tessera/engine.hpp"
#include "tessera/types.hpp"

namespace tessera {

class Engine;
class Model;
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

  // Streaming variant: call `on_token` for each produced token; stop early
  // when it returns false. `cancelled`, when set, is polled between tokens
  // and stops the turn (Aborted). Each call starts from a fresh cache, so
  // the same split serves sequential requests.
  //
  // Usage:
  //   split->GenerateStreaming(prompt, options,
  //                            [](std::uint32_t id) { return keep(id); });
  [[nodiscard]] std::expected<GenerateOutcome, StatusCode> GenerateStreaming(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled = {});

  [[nodiscard]] const Tokenizer* GetTokenizer() const;
  // The target engine and model, for measurement and serve metadata (the
  // tokenizer, chat rendering and the context length). The draft engine
  // is internal.
  [[nodiscard]] Engine& TargetEngine();
  [[nodiscard]] Model& TargetModel();
  [[nodiscard]] std::expected<std::string, StatusCode> ChatPrompt(
      std::string_view user_text, bool enable_thinking = false,
      std::string_view reasoning_effort = "") const;

 private:
  struct Impl;
  explicit SpecSplit(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace tessera
