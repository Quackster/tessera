#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
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
  // Host-side cache of constant weights (conv kernels) and the sequence
  // position (full-layer KV rows equal the position).
  std::unordered_map<std::string, std::vector<float>> host_weights;
  std::size_t position = 0;  // full-layer KV rows equal the position
  bool ready = false;
};

// One vanilla decoder step: embed, block forward, greedy argmax.
// Q4_K projections and F32 vectors only.
[[nodiscard]] std::expected<std::uint32_t, StatusCode> DecodeStep(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token);

// One hybrid decoder step (interleaved full-attention and linear
// attention). Dispatched from DecodeStep when the config is hybrid. The
// full-attention gated path is implemented; a linear layer reports
// UnsupportedFeature until the recurrent path lands.
[[nodiscard]] std::expected<std::uint32_t, StatusCode> HybridDecodeStep(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token);

// One device-resident vanilla decoder step: activations stay on the
// device and chain through the elementwise/projection kernels; only the
// logits download for the argmax. Dispatched from DecodeStep for a
// non-hybrid config.
[[nodiscard]] std::expected<std::uint32_t, StatusCode> DecodeStepDevice(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token);

}  // namespace tessera::core
