#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "tessera/types.hpp"

// The pluggable architecture interface. Each architecture is one module
// under src/models/<arch>/ that owns the architecture specific decode
// behavior; the generic engine, scheduler and kernels stay in src/core.
// Modules register in src/models/registry.cpp, selected at model load
// time by the file's architecture name (general.architecture). Generic
// code calls the interface and never names a model. See AGENTS.md.
//
// Usage:
//   auto* arch = model.Arch();
//   if (arch != nullptr) arch->Logits(backend, model, cache, token);

namespace tessera {

class Backend;
class Buffer;
class Model;

namespace core {
struct DecodeCache;
}  // namespace core

// The outcome of verifying a greedy draft against the target model.
// `accepted` leading draft tokens match the target's greedy distribution
// and are already in the cache; `next_token` is the target's greedy token
// after them (the bonus token); `logits` is the distribution after the
// accepted prefix, ready to verify the next draft.
struct DraftVerification {
  std::size_t accepted = 0;
  std::uint32_t next_token = 0;
  std::vector<float> logits;
};

class Architecture {
 public:
  virtual ~Architecture() = default;

  // The architecture name (the model file's general.architecture).
  [[nodiscard]] virtual std::string_view Name() const = 0;

  // One decode step without the output head: embed the token and run the
  // blocks, leaving the cache at the new position. `capture_layers` names
  // the block indices whose residual hidden is copied into the matching
  // `capture` buffer. `embedding`, when non-null, replaces the gathered
  // token row (an image token).
  [[nodiscard]] virtual std::expected<void, StatusCode> Forward(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::uint32_t token, std::vector<float>* hidden_out,
      const std::vector<std::size_t>* capture_layers,
      std::vector<Buffer*>* capture, const Buffer* embedding) const = 0;

  // One decode step returning the vocab logits (same cache effect as
  // Forward).
  [[nodiscard]] virtual std::expected<std::vector<float>, StatusCode> Logits(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::uint32_t token, std::vector<float>* hidden_out,
      const std::vector<std::size_t>* capture_layers,
      std::vector<Buffer*>* capture, const Buffer* embedding) const = 0;

  // Several tokens in one batched forward, advancing the cache by all of
  // them. `logits_out` receives one vocab row per token when all_logits is
  // true, otherwise only the last row; `embedding` supplies one embedding
  // per row when non-null.
  [[nodiscard]] virtual std::expected<void, StatusCode> ForwardBatch(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const std::uint32_t> tokens, std::vector<float>* logits_out,
      std::vector<float>* hidden_out, bool all_logits,
      const Buffer* embeddings) const = 0;

  // Score a draft against the target with rollback: accept the matching
  // prefix, leaving the cache at it, and return the bonus token. The
  // generic sequential verifier is in src/core; architectures override
  // this to batch the scoring.
  [[nodiscard]] virtual std::expected<DraftVerification, StatusCode> Verify(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const std::uint32_t> draft,
      std::span<const float> prefix_logits,
      std::vector<float>* hidden_out) const = 0;

  // One multi-token-prediction draft step: fuse the token embedding and
  // the backbone hidden, run the draft block and return the drafted
  // token. `mtp_hidden_out`, when non-null, receives the draft hidden to
  // chain the next draft step.
  [[nodiscard]] virtual std::expected<std::uint32_t, StatusCode> Draft(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const float> hidden, std::uint32_t token, std::uint64_t pos,
      std::vector<float>* mtp_hidden_out) const = 0;
};

// Build the architecture module for `arch`, or nullptr when no module is
// registered (the caller then falls back to the generic path).
[[nodiscard]] std::unique_ptr<Architecture> CreateArchitecture(
    std::string_view arch);

}  // namespace tessera
