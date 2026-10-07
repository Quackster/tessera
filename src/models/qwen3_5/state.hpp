#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/decode.hpp"
#include "tessera/backend.hpp"

namespace tessera::models::qwen3_5 {

// Scratch for the multi-token (batched) hybrid forward used by
// speculative verification. Buffers hold `capacity` rows. The
// recurrent-state history keeps one state per processed token so a
// verification can roll back to the accepted prefix.
struct Qwen35BatchScratch {
  std::size_t capacity = 0;
  std::unique_ptr<Buffer> x, xn, proj, logits, pos;
  std::unique_ptr<Buffer> fused, q, gate, kf, vf, attn;
  std::unique_ptr<Buffer> fgate, fup, fmlp;
  std::unique_ptr<Buffer> qkv, z, alpha_raw, beta_raw, alpha, beta, out;
  // Per linear layer: capacity device-state snapshots (capacity x state)
  // and the matching host conv histories.
  std::vector<std::unique_ptr<Buffer>> state_hist;
  std::vector<std::vector<float>> conv_hist_hist;
};

// The Qwen3.5 decode state: the loaded kernels, scratch device buffers,
// per-layer full-attention KV caches, the linear-attention conv/recurrent
// state and the nextn MTP KV. Stored in core::DecodeCache::arch.
struct Qwen35State final : core::ArchState {
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
  std::unique_ptr<Kernel> cast_kernel;
  std::unique_ptr<Kernel> quant_kernel;
  std::unique_ptr<Buffer> kv_scratch;
  std::unique_ptr<Buffer> scale_scratch;
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
    // One fp32 scale per row when the type is Q8.
    std::unique_ptr<Buffer> k_scale;
    std::unique_ptr<Buffer> v_scale;
    std::size_t rows = 0;
    // Allocated row capacity; grows geometrically (see DeviceDecodeState).
    std::size_t capacity = 0;
    KvCacheType type = KvCacheType::F32;
  };
  struct LinearState {
    std::unique_ptr<Buffer> conv_hist;
    std::unique_ptr<Buffer> state;
  };
  std::vector<FullKv> full;
  std::vector<LinearState> linear;
  // Batched-forward scratch, created on the first batched step.
  std::unique_ptr<Qwen35BatchScratch> batch;
  FullKv mtp_kv;
  // Host-side cache of constant weights (conv kernels) and the sequence
  // position (full-layer KV rows equal the position).
  std::unordered_map<std::string, std::vector<float>> host_weights;
  std::size_t position = 0;  // full-layer KV rows equal the position
  bool ready = false;
};

// The state for `cache`, created on first use. One architecture module per
// cache.
inline Qwen35State& State(core::DecodeCache& cache) {
  if (!cache.arch) {
    cache.arch = std::make_unique<Qwen35State>();
  }
  return *static_cast<Qwen35State*>(cache.arch.get());
}

}  // namespace tessera::models::qwen3_5
