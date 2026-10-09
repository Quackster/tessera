#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "tessera/backend.hpp"
#include "tessera/architecture.hpp"
#include "tessera/model.hpp"
#include "tessera/types.hpp"

namespace tessera::core {

// Base for the architecture-specific decode state. Each architecture
// module defines its own subclass (in src/models/<arch>/) and stores it in
// DecodeCache::arch; src/core never names the fields.
struct ArchState {
  virtual ~ArchState() = default;
};

// Per-step state for single-token vanilla decoding: the host key/value
// cache per layer plus the loaded kernels. The cache grows one row per
// step; kernels load once and are reused.
struct DecodeCache {
  struct LayerCache {
    std::vector<float> k;
    std::vector<float> v;
  };
  std::vector<LayerCache> layers;
  std::unique_ptr<Kernel> gemm_kernel;
  std::unique_ptr<Kernel> rope_kernel;
  std::unique_ptr<Kernel> attention_kernel;
  // Constant F32 weights (norms) downloaded once and reused.
  std::unordered_map<std::string, std::vector<float>> host_weights;
  // Storage type of the full-attention KV cache (default fp32). Set
  // before the first step.
  KvCacheType kv_type = KvCacheType::F32;
  // Architecture-specific state (built by the model's Architecture
  // module); null until the first step, and null for the generic path.
  std::unique_ptr<ArchState> arch;
  // Device-resident vanilla state, created on the first vanilla step.
  std::unique_ptr<struct DeviceDecodeState> device;
};

// Per-step device-resident state for the vanilla decode: kernels,
// scratch device buffers, and a device KV cache. Activations stay on
// the device; only the logits download for the argmax.
struct DeviceDecodeState {
  struct Kv {
    std::unique_ptr<Buffer> k;
    std::unique_ptr<Buffer> v;
    // One fp32 scale per row when the type is Q8.
    std::unique_ptr<Buffer> k_scale;
    std::unique_ptr<Buffer> v_scale;
    std::size_t rows = 0;
    // Allocated row capacity; grows geometrically so a decode step appends
    // one row instead of reallocating the whole cache.
    std::size_t capacity = 0;
    KvCacheType type = KvCacheType::F32;
  };
  std::unique_ptr<Kernel> rmsnorm_kernel;
  std::unique_ptr<Kernel> add_kernel;
  std::unique_ptr<Kernel> silu_mul_kernel;
  std::unique_ptr<Kernel> rope_kernel;
  std::unique_ptr<Kernel> attention_kernel;
  std::unique_ptr<Kernel> cast_kernel;
  std::unique_ptr<Kernel> quant_kernel;
  std::unique_ptr<Buffer> kv_scratch;
  std::unique_ptr<Buffer> scale_scratch;
  std::unordered_map<int, std::unique_ptr<Kernel>> gemms;
  std::unique_ptr<Buffer> x;
  std::unique_ptr<Buffer> xn;
  std::unique_ptr<Buffer> proj;
  std::unique_ptr<Buffer> q;
  std::unique_ptr<Buffer> k;
  std::unique_ptr<Buffer> v;
  std::unique_ptr<Buffer> attn;
  std::unique_ptr<Buffer> gate;
  std::unique_ptr<Buffer> up;
  std::unique_ptr<Buffer> mlp;
  std::unique_ptr<Buffer> logits;
  std::vector<Kv> kv;
  bool ready = false;
};

// One vanilla decoder step: embed, block forward, greedy argmax.
// Q4_K projections and F32 vectors only. When `hidden` is non-null it
// receives the final hidden state (hidden_dim floats) before the output
// norm, for the MTP head.
[[nodiscard]] std::expected<std::uint32_t, StatusCode> DecodeStep(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out = nullptr);

// One device-resident vanilla decoder step: activations stay on the
// device and chain through the elementwise/projection kernels; only the
// logits download for the argmax. Dispatched from DecodeStep for a
// non-hybrid config.
[[nodiscard]] std::expected<std::uint32_t, StatusCode> DecodeStepDevice(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out = nullptr);

// One multi-token-prediction draft step. Dispatched to the model's
// architecture module, which fuses the token embedding and the backbone
// hidden (`hidden`, hidden_dim floats), runs its draft block and returns
// the drafted token. `pos` is the sequence position for the draft block's
// RoPE. UnsupportedFeature when the architecture has no draft head.
[[nodiscard]] std::expected<std::uint32_t, StatusCode> MtpDraftStep(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::span<const float> hidden, std::uint32_t token, std::uint64_t pos,
    std::vector<float>* mtp_hidden_out = nullptr);

// The full vocab logits for one device-resident vanilla step.
[[nodiscard]] std::expected<std::vector<float>, StatusCode>
DecodeStepDeviceLogits(Backend& backend, const Model& model,
                       DecodeCache& cache, std::uint32_t token,
                       std::vector<float>* hidden_out = nullptr);

// The block forward of one device-resident vanilla step without the head.
[[nodiscard]] std::expected<void, StatusCode> DecodeStepDeviceForward(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out = nullptr);

// One decoder step without the output head: embed and run the blocks,
// leaving the cache at the new position. Dispatches to the hybrid or
// device path. Use for prompt tokens that do not need logits.
[[nodiscard]] std::expected<void, StatusCode> DecodeForward(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out = nullptr,
    const Buffer* embedding = nullptr,
    const std::vector<std::size_t>* capture_layers = nullptr,
    std::vector<Buffer*>* capture = nullptr);

// One decoder step returning the vocab logits instead of the argmax
// token. Dispatches to the hybrid or device path, preserving the cache
// state exactly as DecodeStep does. This is the primitive the
// speculative verifier scores candidates with.
[[nodiscard]] std::expected<std::vector<float>, StatusCode> DecodeLogits(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out = nullptr,
    const std::vector<std::size_t>* capture_layers = nullptr,
    std::vector<Buffer*>* capture = nullptr,
    const Buffer* embedding = nullptr);

// Score a token sequence with a fresh cache: run each token in order and
// return the logits at every position (row i is the distribution after
// tokens[0..i]). The cache is local, so the caller's decode state is not
// touched. This is the verifier vocabulary for speculative decoding.
[[nodiscard]] std::expected<std::vector<std::vector<float>>, StatusCode>
ScoreTokens(Backend& backend, const Model& model,
            std::span<const std::uint32_t> tokens);

// Score a token sequence in one batched forward: run tokens[0..k-1] in
// order and return one logits row per token (row i is the distribution
// after tokens[0..i]), advancing `cache` by all of them. The hybrid path
// batches the GEMMs and attention; other models fall back to a
// sequential loop. Used by speculative verification.
[[nodiscard]] std::expected<std::vector<std::vector<float>>, StatusCode>
DecodeLogitsBatch(Backend& backend, const Model& model, DecodeCache& cache,
                  std::span<const std::uint32_t> tokens);

// Prefill a prompt in one batched forward: commit every token to the
// cache and return the logits after the last token (and its hidden).
// The hybrid path runs one batched trunk forward with the output head on
// the last row only; other models fall back to a sequential loop.
[[nodiscard]] std::expected<std::vector<float>, StatusCode> PrefillTokens(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::span<const std::uint32_t> tokens,
    std::vector<float>* hidden_out = nullptr,
    const Buffer* embeddings = nullptr);

// Greedy speculative verification. Given the target's distribution at the
// current prefix (`prefix_logits`) and a draft, feed each draft token only
// while it equals the target's greedy token, then stop. The cache is left
// at the accepted prefix, so no state is ever rolled back. The output
// (accepted draft tokens followed by `next_token`) is identical to plain
// greedy decoding for any draft. `prefix_logits` must be the logits from
// the step that produced the current cache position; empty is invalid.
// When `hidden_out` is non-null it receives the final hidden state of the
// last accepted token (unchanged when nothing is accepted), so an MTP
// drafter can chain on it.
[[nodiscard]] std::expected<DraftVerification, StatusCode> VerifyDraft(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::span<const std::uint32_t> draft,
    std::span<const float> prefix_logits,
    std::vector<float>* hidden_out = nullptr,
    const std::vector<std::size_t>* capture_layers = nullptr,
    std::vector<Buffer*>* capture = nullptr);

}  // namespace tessera::core
