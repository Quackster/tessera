#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

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
  std::size_t context = 4096;
  KvCacheType kv_type = KvCacheType::F32;
};

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
      std::string_view user_text, bool enable_thinking = false) const;

 private:
  struct Impl;
  explicit Hybrid(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace tessera
