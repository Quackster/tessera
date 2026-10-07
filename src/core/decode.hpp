#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "tessera/backend.hpp"
#include "tessera/model.hpp"
#include "tessera/types.hpp"

namespace tessera::core {

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
  // Hybrid state, created on the first hybrid step (null for vanilla).
  std::unique_ptr<struct HybridDecodeCache> hybrid;
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
    std::size_t rows = 0;
  };
  std::unique_ptr<Kernel> rmsnorm_kernel;
  std::unique_ptr<Kernel> add_kernel;
  std::unique_ptr<Kernel> silu_mul_kernel;
  std::unique_ptr<Kernel> rope_kernel;
  std::unique_ptr<Kernel> attention_kernel;
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

// Per-step device-resident state for a hybrid model: the loaded
// kernels, scratch device buffers, and per-layer state (full-attention
// KV caches and linear-attention conv/recurrent state) on the device.
struct HybridDecodeCache {
  std::unique_ptr<Kernel> rmsnorm_kernel;
  std::unique_ptr<Kernel> add_kernel;
  std::unique_ptr<Kernel> silu_mul_kernel;
  std::unique_ptr<Kernel> sigmoid_gate_kernel;
  std::unique_ptr<Kernel> qgate_split_kernel;
  std::unique_ptr<Kernel> mrope_kernel;
  std::unique_ptr<Kernel> attention_kernel;
  std::unique_ptr<Kernel> conv1d_step_kernel;
  std::unique_ptr<Kernel> repeat_heads_kernel;
  std::unique_ptr<Kernel> l2norm_kernel;
  std::unique_ptr<Kernel> ssm_gate_kernel;
  std::unique_ptr<Kernel> delta_step_heads_kernel;
  std::unique_ptr<Kernel> rmsnorm_gated_kernel;
  std::unordered_map<int, std::unique_ptr<Kernel>> gemms;
  // Shared scratch buffers (sized from the config).
  std::unique_ptr<Buffer> x, xn, proj, logits, pos;
  std::unique_ptr<Buffer> fused, q, gate, kf, vf, attn;
  std::unique_ptr<Buffer> fgate, fup, fmlp;
  std::unique_ptr<Buffer> qkv, z, alpha_raw, beta_raw, alpha, beta;
  std::unique_ptr<Buffer> conv_mixed;
  std::unique_ptr<Buffer> q_l, k_l;
  std::unique_ptr<Buffer> q_exp, k_exp, v_l, core, out;
  // MTP head scratch (fused embedding+hidden, hidden norms).
  std::unique_ptr<Buffer> mtp_fused, mtp_h;
  // Per linear layer, the conv input history on the host (oldest first).
  std::vector<std::vector<float>> conv_hist;
  struct FullKv {
    std::unique_ptr<Buffer> k;
    std::unique_ptr<Buffer> v;
    std::size_t rows = 0;
  };
  struct LinearState {
    std::unique_ptr<Buffer> conv_hist;
    std::unique_ptr<Buffer> state;
  };
  std::vector<FullKv> full;
  std::vector<LinearState> linear;
  FullKv mtp_kv;
  // Host-side cache of constant weights (conv kernels) and the sequence
  // position (full-layer KV rows equal the position).
  std::unordered_map<std::string, std::vector<float>> host_weights;
  std::size_t position = 0;  // full-layer KV rows equal the position
  bool ready = false;
};

// One vanilla decoder step: embed, block forward, greedy argmax.
// Q4_K projections and F32 vectors only. When `hidden` is non-null it
// receives the final hidden state (hidden_dim floats) before the output
// norm, for the MTP head.
[[nodiscard]] std::expected<std::uint32_t, StatusCode> DecodeStep(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out = nullptr);

// One hybrid decoder step (interleaved full-attention and linear
// attention). Dispatched from DecodeStep when the config is hybrid. The
// full-attention gated path is implemented; a linear layer reports
// UnsupportedFeature until the recurrent path lands.
[[nodiscard]] std::expected<std::uint32_t, StatusCode> HybridDecodeStep(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out = nullptr);

// One device-resident vanilla decoder step: activations stay on the
// device and chain through the elementwise/projection kernels; only the
// logits download for the argmax. Dispatched from DecodeStep for a
// non-hybrid config.
[[nodiscard]] std::expected<std::uint32_t, StatusCode> DecodeStepDevice(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out = nullptr);

// One multi-token-prediction draft step (Qwen3.5 nextn head): fuse the
// token embedding and the backbone hidden (`hidden`, hidden_dim floats),
// run the MTP full-attention block, and return the drafted token. `pos`
// is the sequence position for the MTP block's RoPE.
[[nodiscard]] std::expected<std::uint32_t, StatusCode> MtpDraftStep(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::span<const float> hidden, std::uint32_t token, std::uint64_t pos);

// The full vocab logits for one hybrid decoder step (same forward as
// HybridDecodeStep, but the whole row comes back). Dispatched from
// DecodeLogits.
[[nodiscard]] std::expected<std::vector<float>, StatusCode>
HybridDecodeLogits(Backend& backend, const Model& model, DecodeCache& cache,
                   std::uint32_t token,
                   std::vector<float>* hidden_out = nullptr);

// The full vocab logits for one device-resident vanilla step.
[[nodiscard]] std::expected<std::vector<float>, StatusCode>
DecodeStepDeviceLogits(Backend& backend, const Model& model,
                       DecodeCache& cache, std::uint32_t token,
                       std::vector<float>* hidden_out = nullptr);

// One decoder step returning the vocab logits instead of the argmax
// token. Dispatches to the hybrid or device path, preserving the cache
// state exactly as DecodeStep does. This is the primitive the
// speculative verifier scores candidates with.
[[nodiscard]] std::expected<std::vector<float>, StatusCode> DecodeLogits(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token, std::vector<float>* hidden_out = nullptr);

// Score a token sequence with a fresh cache: run each token in order and
// return the logits at every position (row i is the distribution after
// tokens[0..i]). The cache is local, so the caller's decode state is not
// touched. This is the verifier vocabulary for speculative decoding.
[[nodiscard]] std::expected<std::vector<std::vector<float>>, StatusCode>
ScoreTokens(Backend& backend, const Model& model,
            std::span<const std::uint32_t> tokens);

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
    std::vector<float>* hidden_out = nullptr);

}  // namespace tessera::core
