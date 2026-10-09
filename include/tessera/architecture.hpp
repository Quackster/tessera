#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
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

// A value-layout conversion for a checkpoint whose tensor layout differs
// from the internal layout (for example the MXFP4 value-head order). `src`
// maps internal index i to checkpoint index src[i] along the value
// dimension (empty means no permutation); `inner` selects the contiguous
// dimension instead of the outer one. `exp_negate` turns log-space values
// (A_log) into the internal -exp form with F32 output. `add_one` adds 1 to
// each value, for checkpoints that store a unit-offset norm weight (the
// Gemma RMSNorm convention: the effective gain is 1 + weight).
struct WeightConversion {
  std::vector<std::size_t> src;
  bool inner = false;
  bool exp_negate = false;
  bool add_one = false;
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
  // per row when non-null. `capture_layers` names the block indices whose
  // per-row residual hidden is copied into the matching `capture` buffer
  // (each sized rows*hidden), for a drafter that conditions on every
  // position; nullptr skips capture.
  [[nodiscard]] virtual std::expected<void, StatusCode> ForwardBatch(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const std::uint32_t> tokens, std::vector<float>* logits_out,
      std::vector<float>* hidden_out, bool all_logits,
      const Buffer* embeddings,
      const std::vector<std::size_t>* capture_layers = nullptr,
      std::vector<Buffer*>* capture = nullptr) const = 0;

  // Score a draft against the target with rollback: accept the matching
  // prefix, leaving the cache at it, and return the bonus token. The
  // generic sequential verifier is in src/core; architectures override
  // this to batch the scoring. `capture` mirrors ForwardBatch and receives
  // the per-row residual hidden at `capture_layers` for every scored row,
  // so a caller can keep the hidden of each accepted position as context.
  [[nodiscard]] virtual std::expected<DraftVerification, StatusCode> Verify(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const std::uint32_t> draft,
      std::span<const float> prefix_logits,
      std::vector<float>* hidden_out,
      const std::vector<std::size_t>* capture_layers = nullptr,
      std::vector<Buffer*>* capture = nullptr) const = 0;

  // One multi-token-prediction draft step: fuse the token embedding and
  // the backbone hidden, run the draft block and return the drafted
  // token. `mtp_hidden_out`, when non-null, receives the draft hidden to
  // chain the next draft step.
  [[nodiscard]] virtual std::expected<std::uint32_t, StatusCode> Draft(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const float> hidden, std::uint32_t token, std::uint64_t pos,
      std::vector<float>* mtp_hidden_out) const = 0;

  // Rows currently committed in the draft key/value cache (0 when there is
  // none). Draft chaining appends a row per draft; after a partial accept
  // the caller shrinks it back with DraftTruncate.
  [[nodiscard]] virtual std::size_t DraftRows(
      const core::DecodeCache& cache) const = 0;
  virtual void DraftTruncate(core::DecodeCache& cache,
                             std::size_t rows) const = 0;

  // Parse a JSON model definition (an MXFP4 directory's config.json) into
  // the transformer hyper-parameters. The GGUF path parses GGUF metadata
  // instead and never calls this. The default returns UnsupportedFeature;
  // a module overrides it when the directory format carries its config.
  [[nodiscard]] virtual std::expected<TransformerConfig, StatusCode>
  ParseConfigJson(std::string_view json) const {
    (void)json;
    return std::unexpected(StatusCode::UnsupportedFeature);
  }

  // Map a checkpoint tensor name to the internal name this module looks
  // up (for example a HuggingFace safetensors name to a "blk.N.*" name),
  // or nullopt to ignore the tensor. `config` is the parsed model config,
  // so the module can place draft blocks after the trunk layers. The
  // default returns the name unchanged, for formats that already use the
  // internal names (GGUF).
  [[nodiscard]] virtual std::optional<std::string> MapWeightName(
      std::string_view name, const TransformerConfig& config) const {
    (void)config;
    return std::string(name);
  }

  // Value-layout conversion for one internal tensor, or nullopt when the
  // bytes are already internal. The loader applies the permutation at the
  // element level (blob and scale together for MXFP4) and the exp_negate
  // transform to F32. The default returns nullopt.
  [[nodiscard]] virtual std::optional<WeightConversion> ConvertWeight(
      std::string_view internal_name, const TransformerConfig& config) const {
    (void)internal_name;
    (void)config;
    return std::nullopt;
  }
};

// Build the architecture module for `arch`, or nullptr when no module is
// registered (the caller then falls back to the generic path).
[[nodiscard]] std::unique_ptr<Architecture> CreateArchitecture(
    std::string_view arch);

}  // namespace tessera
