#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <random>
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

struct SamplingOptions;

// The sampling state a sampled verification needs. Null in a greedy
// verification. A verifier that receives it draws the target's token at
// each scored row from `sampling` with `rng` (history plus the rows
// already accepted supply the penalty history) and accepts the draft
// while the drawn token matches it, so the emitted sequence equals plain
// sampling for the same seed. The verifier advances `rng` once per scored
// row up to and including the first mismatch, then once more for the
// bonus row when every draft matched.
struct SampledVerify {
  const SamplingOptions* sampling = nullptr;
  std::mt19937_64* rng = nullptr;
  std::span<const std::uint32_t> history;
  // The draft's sampling distribution, one dense row per proposed token
  // (rows x vocab), when the drafter sampled. Null for a greedy draft; the
  // verifier then matches the drawn target token instead of using rejection.
  const std::vector<double>* draft_probs = nullptr;
  std::size_t draft_vocab = 0;
};

// Sampling state for a sampled draft. When null the drafter proposes
// greedily and records no distribution. `history` is the penalty history
// before the draft window; the drafter extends it with the drafts it draws,
// so each draft distribution matches the target's scale.
struct DraftSampling {
  const SamplingOptions* sampling = nullptr;
  std::mt19937_64* rng = nullptr;
  std::span<const std::uint32_t> history;
};

// The draft's sampling distribution, one dense row per proposed token
// (rows x vocab), row-major. The drafter fills it when it sampled the
// drafts; a greedy draft leaves it empty.
struct DraftDistribution {
  std::vector<double> probs;
  std::size_t rows = 0;
  std::size_t vocab = 0;
};

// The outcome of verifying a draft against the target model. In the greedy
// mode `accepted` leading draft tokens match the target's argmax and are
// already in the cache; `next_token` is the target's argmax token after
// them. In the sampled mode (see SampledVerify) `accepted` tokens match
// the target's drawn token and `next_token` is the drawn token at the
// first mismatch (or the bonus draw when all matched).
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

// The optional weight groups a load materializes. The MTP
// (next-token-prediction) head is the only group today: only MTP
// speculation reads it, so a run with another strategy can skip it and
// the device memory it would hold. A 27B head is about 1.6 GiB.
struct WeightLoadOptions {
  bool include_mtp_head = true;
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

  // One decode step returning the greedy (argmax) token id (same cache
  // effect as Forward). The default implementation calls Logits and scans
  // the host copy; an architecture overrides it to argmax on the device,
  // so the hot greedy decode path avoids downloading the whole vocabulary
  // and the synchronization that readback costs.
  [[nodiscard]] virtual std::expected<std::uint32_t, StatusCode> GreedyToken(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::uint32_t token, std::vector<float>* hidden_out,
      const std::vector<std::size_t>* capture_layers,
      std::vector<Buffer*>* capture, const Buffer* embedding) const;

  // Batched greedy decode of several independent sequences on one device:
  // each cache holds one sequence's state and `tokens[i]` is its next
  // token. The projections and FFN run over all sequences at once and the
  // attention runs per sequence (row_base). Returns the greedy token per
  // sequence, or the fallback status when the architecture cannot batch
  // (the caller then advances the sequences one by one). The default
  // returns UnsupportedFeature.
  [[nodiscard]] virtual std::expected<std::vector<std::uint32_t>, StatusCode>
  GreedyBatchMulti(Backend& backend, const Model& model,
                   std::span<core::DecodeCache* const> caches,
                   std::span<const std::uint32_t> tokens) const {
    (void)backend;
    (void)model;
    (void)caches;
    (void)tokens;
    return std::unexpected(StatusCode::UnsupportedFeature);
  }

  // Several tokens in one batched forward, advancing the cache by all of
  // them. `logits_out` receives one vocab row per token when all_logits is
  // true, otherwise only the last row; `embedding` supplies one embedding
  // per row when non-null. `capture_layers` names the block indices whose
  // per-row residual hidden is copied into the matching `capture` buffer
  // (each sized rows*hidden), for a drafter that conditions on every
  // position; nullptr skips capture.
  // `snapshot_state` forces the per-layer linear/recurrent state snapshots
  // a rollback needs even when `all_logits` is false (a pipeline verify runs
  // every stage with snapshots but only the last stage scores logits).
  [[nodiscard]] virtual std::expected<void, StatusCode> ForwardBatch(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const std::uint32_t> tokens, std::vector<float>* logits_out,
      std::vector<float>* hidden_out, bool all_logits,
      const Buffer* embeddings,
      const std::vector<std::size_t>* capture_layers = nullptr,
      std::vector<Buffer*>* capture = nullptr,
      bool snapshot_state = false) const = 0;

  // Roll a cache advanced by a batched forward back to `keep_position`
  // committed target rows (the accepted prefix, an absolute position):
  // truncate the full-attention KV rows, restore the linear/recurrent state
  // and conv history, and set the position. The default returns
  // UnsupportedFeature; the pipeline verify uses it on every stage. A
  // sharded stage touches only its own layers.
  [[nodiscard]] virtual std::expected<void, StatusCode> TruncateTarget(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::size_t keep_position) const {
    (void)backend;
    (void)model;
    (void)cache;
    (void)keep_position;
    return std::unexpected(StatusCode::UnsupportedFeature);
  }

  // Score a draft against the target with rollback: accept the matching
  // prefix, leaving the cache at it, and return the bonus token. The
  // generic sequential verifier is in src/core; architectures override
  // this to batch the scoring. `capture` mirrors ForwardBatch and receives
  // the per-row residual hidden at `capture_layers` for every scored row,
  // so a caller can keep the hidden of each accepted position as context.
  // When `anchor` is set the anchor token is processed as the batch's first
  // row (so its logits score `draft[0]` and its captured features are row 0
  // of `capture`); the caller then commits the anchor plus the accepted
  // drafts. `prefix_logits` is ignored in that case. When `sampled` is
  // non-null the accept rule draws the target's token per row and matches
  // it (see SampledVerify); the rollback keeps exactly the matched prefix.
  [[nodiscard]] virtual std::expected<DraftVerification, StatusCode> Verify(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const std::uint32_t> draft,
      std::span<const float> prefix_logits,
      std::optional<std::uint32_t> anchor,
      std::vector<float>* hidden_out,
      const std::vector<std::size_t>* capture_layers = nullptr,
      std::vector<Buffer*>* capture = nullptr,
      const SampledVerify* sampled = nullptr) const = 0;

  // One multi-token-prediction draft step: fuse the token embedding and
  // the backbone hidden, run the draft block and return the drafted
  // token. `mtp_hidden_out`, when non-null, receives the draft hidden to
  // chain the next draft step. `logits_out`, when non-null, receives the
  // draft's vocab-wide logits so a caller can sample the draft and keep its
  // distribution.
  [[nodiscard]] virtual std::expected<std::uint32_t, StatusCode> Draft(
      Backend& backend, const Model& model, core::DecodeCache& cache,
      std::span<const float> hidden, std::uint32_t token, std::uint64_t pos,
      std::vector<float>* mtp_hidden_out,
      std::vector<float>* logits_out = nullptr) const = 0;

  // Rows currently committed in the draft key/value cache (0 when there is
  // none). Draft chaining appends a row per draft; after a partial accept
  // the caller shrinks it back with DraftTruncate.
  [[nodiscard]] virtual std::size_t DraftRows(
      const core::DecodeCache& cache) const = 0;
  virtual void DraftTruncate(core::DecodeCache& cache,
                             std::size_t rows) const = 0;

  // Allocate the decode state (kernels, scratch) for `cache` without
  // running a forward, so a device that only drafts (never runs the
  // trunk) can still use the draft head. The default returns
  // UnsupportedFeature; a module with a draft head overrides it.
  [[nodiscard]] virtual std::expected<void, StatusCode> EnsureReady(
      Backend& backend, const Model& model,
      core::DecodeCache& cache) const {
    (void)backend;
    (void)model;
    (void)cache;
    return std::unexpected(StatusCode::UnsupportedFeature);
  }

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
  // so the module can place draft blocks after the trunk layers. `options`
  // selects the optional weight groups to materialize: with
  // include_mtp_head false the module ignores the next-token-prediction
  // head, which only MTP speculation reads. The default returns the name
  // unchanged, for formats that already use the internal names (GGUF).
  [[nodiscard]] virtual std::optional<std::string> MapWeightName(
      std::string_view name, const TransformerConfig& config,
      const WeightLoadOptions& options) const {
    (void)config;
    (void)options;
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

  // How this internal tensor is split across tensor-parallel ranks. The
  // loader slices a Column tensor's output rows or a Row tensor's input
  // columns at block boundaries; the forward all-reduces a Row projection
  // across the ranks. The default is Replicated, so a model with no
  // tensor shard and an architecture without the override load unsharded.
  [[nodiscard]] virtual ParallelKind TensorParallelKind(
      std::string_view internal_name) const {
    (void)internal_name;
    return ParallelKind::Replicated;
  }
};

// Build the architecture module for `arch`, or nullptr when no module is
// registered (the caller then falls back to the generic path).
[[nodiscard]] std::unique_ptr<Architecture> CreateArchitecture(
    std::string_view arch);

}  // namespace tessera
