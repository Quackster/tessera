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
  // Rows the logits scratch holds. Prefill scores one row (the last token)
  // and batch verify scores at most the draft block, so this stays small
  // even when `capacity` covers a long prompt.
  std::size_t logits_capacity = 0;
  // Rows the per-layer linear state snapshots hold, and whether this batch
  // writes them. Only a verification (rollback) needs them.
  std::size_t hist_rows = 0;
  bool snapshot_states = false;
  std::unique_ptr<Buffer> x, xn, proj, logits, pos;
  std::unique_ptr<Buffer> fused, q, gate, kf, vf, attn;
  std::unique_ptr<Buffer> fgate, fup, fmlp;
  std::unique_ptr<Buffer> qkv, z, alpha_raw, beta_raw, alpha, beta, out;
  // Row-batched linear-attention scratch (prefill only): the conv output
  // (q_all/k_all/v_all), the repeated q/k heads, and the delta output.
  std::unique_ptr<Buffer> q_all, k_all, v_all, q_exp_all, k_exp_all, core_all;
  // Per linear layer: capacity device-state snapshots (capacity x state)
  // and the matching device conv-history snapshots.
  std::vector<std::unique_ptr<Buffer>> state_hist;
  std::vector<std::unique_ptr<Buffer>> conv_hist_hist;
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
  std::unique_ptr<Kernel> conv1d_state_kernel;
  std::unique_ptr<Kernel> repeat_heads_kernel;
  std::unique_ptr<Kernel> l2norm_kernel;
  std::unique_ptr<Kernel> ssm_gate_kernel;
  std::unique_ptr<Kernel> delta_step_heads_kernel;
  std::unique_ptr<Kernel> rmsnorm_gated_kernel;
  std::unique_ptr<Kernel> cast_kernel;
  std::unique_ptr<Kernel> quant_kernel;
  // Per-token FP8 E4M3 QDQ of an MXFP4 linear's activation (the served
  // W4A8 contract) and its per-row scale scratch, grown on demand.
  std::unique_ptr<Kernel> fp8_quant_kernel;
  std::unique_ptr<Buffer> fp8_scale;
  // bf16 activation rounding of a projection output (the served target's
  // fused epilogue), loaded on demand when TESSERA_TARGET_BF16 is set.
  std::unique_ptr<Kernel> bf16_kernel;
  // fp8 tensor-core MXFP4 GEMM path (TESSERA_MXFP4_WMMA): the activation
  // packing, per-row weight reference-exponent and GEMM kernels, the A
  // scratch, and the per-weight Wref cache.
  std::unique_ptr<Kernel> fp8_pack_kernel;
  std::unique_ptr<Kernel> mxfp4_rowref_kernel;
  std::unique_ptr<Kernel> mxfp4_wmma_kernel;
  std::unique_ptr<Kernel> mxfp4_reduce_kernel;
  std::unique_ptr<Buffer> wmma_a;
  std::unique_ptr<Buffer> wmma_as;
  std::unique_ptr<Buffer> wmma_part;
  // The activation last packed into wmma_a/wmma_as. Several projections in a
  // layer share one input (q/k/v/gate); the pack is reused while the source
  // buffer and shape are unchanged. Invalidated whenever a norm rewrites a
  // projection input or wmma_a is reallocated.
  const void* wmma_pack_src = nullptr;
  std::size_t wmma_pack_rows = 0;
  std::size_t wmma_pack_cols = 0;
  bool wmma_pack_valid = false;
  std::unordered_map<const void*, std::unique_ptr<Buffer>> mxfp4_wref;
  std::unique_ptr<Buffer> kv_scratch;
  std::unique_ptr<Buffer> scale_scratch;
  std::unordered_map<int, std::unique_ptr<Kernel>> gemms;
  // Device embedding-gather kernel, keyed by the embedding weight dtype.
  std::unordered_map<int, std::unique_ptr<Kernel>> embed_kernels;
  // Token ids for the embedding-gather launch, grown on demand.
  std::unique_ptr<Buffer> embed_ids;
  // Tiled batched GEMM kernels, keyed by dtype (used when rows > 1).
  std::unordered_map<int, std::unique_ptr<Kernel>> gemm_tiled;
  // Shared scratch buffers (sized from the config).
  std::unique_ptr<Buffer> x, xn, proj, logits, pos;
  std::unique_ptr<Buffer> fused, q, gate, kf, vf, attn;
  std::unique_ptr<Buffer> fgate, fup, fmlp;
  std::unique_ptr<Buffer> qkv, z, alpha_raw, beta_raw, alpha, beta;
  std::unique_ptr<Buffer> q_l, k_l;
  std::unique_ptr<Buffer> q_exp, k_exp, v_l, core, out;
  // MTP head scratch (fused embedding+hidden, hidden norms).
  std::unique_ptr<Buffer> mtp_fused, mtp_h;
  // MTP draft argmax over the shared-head logits, computed on the device so a
  // draft does not download the whole vocabulary.
  std::unique_ptr<Kernel> mtp_top_k_kernel;
  std::unique_ptr<Buffer> mtp_argmax_ids, mtp_argmax_vals;
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
