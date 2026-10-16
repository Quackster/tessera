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

// Model-parallel route inside one hybrid data-parallel group.
enum class HybridMpKind {
  Tensor,   // TensorParallel team (sharded projections, rank reduce)
  Context,  // ContextParallel team (split key range, partial combine)
};

// Options for a hybrid engine: `dp_groups` data-parallel groups, each an
// independent model-parallel team of `devices.size() / dp_groups` ranks
// running `mp_kind`. Every group loads its own weights; groups run
// different prompts concurrently.
struct HybridOptions {
  std::string model_path;
  std::vector<int> devices;  // dp_groups * mp_ranks, group-major
  std::size_t dp_groups = 1;
  HybridMpKind mp_kind = HybridMpKind::Tensor;
  std::size_t context = kDefaultContext;
  KvCacheType kv_type = KvCacheType::F32;
};

// One data-parallel group of a hybrid layout: its index and its devices
// in rank order, computed from the same mesh validation Hybrid::Create
// applies. A caller that runs each group separately (for example a serve
// pool of one runner per group) uses HybridGroups instead of building the
// whole Hybrid, so the group math lives in one place.
struct HybridGroupDevices {
  std::size_t index = 0;
  std::vector<int> devices;
};

// The groups of a hybrid layout. InvalidArgument for an empty layout or a
// device count that is not dp_groups * mp_ranks; the device rules are the
// mesh validator's.
//
// Usage:
//   auto groups = HybridGroups(options);
//   for (const HybridGroupDevices& group : *groups) { ... }
[[nodiscard]] std::expected<std::vector<HybridGroupDevices>, StatusCode>
HybridGroups(const HybridOptions& options);

// Data-parallel groups of model-parallel teams behind one canonical
// device mesh. One thread per group runs its team's forward; groups are
// independent, so two prompts decode at once. Each group's output equals
// the matching single-route run (Engine, TensorParallel, ContextParallel).
//
// Usage:
//   HybridOptions options;
//   options.model_path = path;
//   options.devices = {0, 1};
//   options.dp_groups = 2;
//   auto hybrid = Hybrid::Create(options);
//   if (hybrid) { auto out = (*hybrid)->GenerateMany(prompts, 32); }
class Hybrid {
 public:
  static std::expected<std::unique_ptr<Hybrid>, StatusCode> Create(
      const HybridOptions& options);

  ~Hybrid();
  Hybrid(Hybrid&& other) noexcept;
  Hybrid& operator=(Hybrid&& other) noexcept;
  Hybrid(const Hybrid&) = delete;
  Hybrid& operator=(const Hybrid&) = delete;

  // Greedy generation, one prompt per data-parallel group. Returns one
  // token vector per group, in group order. `max_tokens` (0 fills the
  // context) applies per group. Each group stops at a declared stop
  // token.
  std::expected<std::vector<std::vector<std::uint32_t>>, StatusCode>
  GenerateMany(std::span<const std::vector<std::uint32_t>> prompts,
               std::size_t max_tokens);

  [[nodiscard]] std::size_t GroupCount() const;
  [[nodiscard]] const Tokenizer* GetTokenizer() const;
  [[nodiscard]] std::expected<std::string, StatusCode> ChatPrompt(
      std::string_view user_text, bool enable_thinking = false,
      std::string_view reasoning_effort = "") const;

 private:
  struct Impl;
  explicit Hybrid(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace tessera
