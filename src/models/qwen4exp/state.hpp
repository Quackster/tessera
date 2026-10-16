#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "core/decode.hpp"
#include "models/qwen4exp/mtp_weights.hpp"
#include "models/qwen4exp/ple_table.hpp"
#include "tessera/backend.hpp"
#include "tessera/log.hpp"

namespace tessera::models::qwen4exp {

// The qwen4exp decode state: the loaded kernels, the shared scratch, the
// per-layer recurrent (gated-delta) and paged (QSA) state, and the PLE
// history. Held in core::DecodeCache::arch. One token advances every
// sequence through all blocks without any allocation.
struct Qwen4ExpState final : core::ArchState {
  // Shared elementwise and shape ops.
  std::unique_ptr<Kernel> rmsnorm, rmsnorm_rows, add, silu_mul, sigmoid_gate,
      qgate_split, rope, attention, cast_f16_f32;
  // Gated residual stack (hyper-connection).
  std::unique_ptr<Kernel> gr_norm, gr_down, gr_gate, gr_mean, gr_inject,
      gr_write;
  // Gated-delta linear layer.
  std::unique_ptr<Kernel> conv1d_state, l2norm, ssm_gate, repeat_heads,
      delta_step_heads, rmsnorm_gated, gdn_out_norm;
  // Sparse QSA attention.
  std::unique_ptr<Kernel> kv_append, indexer_key_append, qsa_index, qsa_topk,
      kv_gather, qsa_attend, qsa_gate_apply;
  // PLE post-ops.
  std::unique_ptr<Kernel> ple_gate, ple_broadcast, ple_conv, ple_hist_advance;
  // Router / expert feed-forward.
  std::unique_ptr<Kernel> moe_gate, moe_scale_add, moe_topk;
  // Fused routed-expert kernels, one per expert format the pack uses.
  std::unique_ptr<Kernel> moe_gate_up_iq2s, moe_gate_up_iq3xxs,
      moe_gate_up_iq3s, moe_gate_up_iq4xs, moe_down_iq4nl, moe_down_q20;
  core::KernelCache gemms, embed_kernels;
  std::unique_ptr<Buffer> embed_ids;

  // Shared scratch (f32 unless noted).
  std::unique_ptr<Buffer> x, xn, proj, logits, pos;
  // Residual stack R (hc x hidden) and the sub-block I/O.
  std::unique_ptr<Buffer> R, mixed, block_out, inject, inject2;
  // Hyper-connection workspace: xn (hc*hidden), xq (hc*hidden bf16-as-f32),
  // lq (hc_lr), gated (hc*hidden).
  std::unique_ptr<Buffer> hc_xn, hc_xq, hc_lq, hc_gated;
  // Gated-delta scratch.
  std::unique_ptr<Buffer> qkv, z, alpha_raw, beta_raw, alpha, beta, q_l, k_l,
      q_exp, k_exp, v_l, core, out;
  // QSA scratch.
  std::unique_ptr<Buffer> q_full, q_attn, gate_buf, kcur, vcur, idx_raw, q_idx,
      cell_scores, ids, k_scratch, v_scratch, idx_bias, attn_f16,
      attn_in;
  // Router / expert scratch.
  std::unique_ptr<Buffer> moe_logits, moe_ids, moe_vals, moe_wts, moe_acc,
      moe_sig, moe_gate_out, moe_up_out, moe_inter, moe_shared, moe_phi,
      moe_cpu_acc;
  // Host scratch for the CPU+GPU hybrid MoE: the activation and the weighted
  // sum of the CPU-computed experts.
  std::vector<float> cpu_x, cpu_acc;
  // Per-layer fp32 copy of the shared-expert gate weight, converted from the
  // checkpoint's bf16 once and reused (indexed by trunk layer).
  std::vector<std::unique_ptr<Buffer>> moe_shexp_w;
  // PLE scratch and history.
  std::shared_ptr<PleTable> ple_table;
  std::unique_ptr<Buffer> ple_hist, ple_key, ple_query, ple_value, ple_gated,
      ple_normalized, ple_conv_out, ple_proj_key, ple_norm_key, ple_norm_query,
      ple_norm_conv, ple_rows, ple_emb;
  std::vector<float> ple_host_value;
  std::vector<std::uint32_t> ple_host_rows;
  std::vector<std::int32_t> ple_prev;
  std::vector<std::int32_t> ple_tokens;
  std::uint64_t ple_position = 0;

  struct GdnLayer {
    std::unique_ptr<Buffer> conv_hist;
    std::unique_ptr<Buffer> state;
  };
  struct QsaLayer {
    std::unique_ptr<Buffer> k_pool;
    std::unique_ptr<Buffer> v_pool;
    std::unique_ptr<Buffer> page_table;
    std::unique_ptr<Buffer> idx_tail;
    std::unique_ptr<Buffer> idx_dead;
    std::unique_ptr<Buffer> idx_pooled;
    std::unique_ptr<Buffer> idx_cos;
    std::unique_ptr<Buffer> idx_sin;
    std::unique_ptr<Buffer> block_pos;
  };
  std::vector<GdnLayer> gdn;
  std::vector<QsaLayer> qsa;
  std::size_t position = 0;
  bool ready = false;

  // Batched-forward scratch (prefill and the multi-row verify). The residual
  // stack is held per row (rows * hc * hidden) so a layer can run every row
  // before its experts stage once; the embedding gather writes one row each.
  // The FFN input, the per-stream injection and the router selection are
  // held per row so a layer can run every row's router, read the ids back
  // once, stage the union and then run the experts.
  std::unique_ptr<Buffer> batch_r, batch_x, batch_mixed, batch_inject2,
      batch_ids, batch_wts, batch_sig;
  std::size_t batch_rows = 0;

  // Per-layer gated-delta state snapshots for the cross-stage verify: one
  // slot per state (rows + 1, slot 0 is the pre-batch state), indexed by the
  // global layer number. Only allocated when a verify requests snapshots.
  std::vector<std::unique_ptr<Buffer>> batch_gdn_state, batch_gdn_conv;
  std::size_t batch_snapshot_rows = 0;
  // Absolute position of snapshot slot 0, so a rollback maps a committed
  // position to its slot.
  std::uint64_t batch_snapshot_start = 0;

  // MTP draft head: the shared read-only weights (owned by the model's
  // module, loaded once), this turn's paged FP16 QSA cache, dense-attention
  // scratch and expert scratch.
  const MtpWeights* mtp_weights = nullptr;
  std::unique_ptr<Buffer> mtp_k_pool, mtp_v_pool, mtp_page_table, mtp_ids,
      mtp_k_scratch, mtp_v_scratch, mtp_expert_scratch, mtp_hidden_branch,
      mtp_streams;
  std::uint64_t mtp_rows = 0;
  std::uint64_t mtp_position = 0;
  bool mtp_ready = false;
  // Opt-in decode phase profiler (TESSERA_PROFILE), mirrored from the cache
  // each forward. Null disables timing.
  core::Profile* profile = nullptr;

  void ReportCaches(const log::Diagnostics& log,
                    std::string_view prefix) const override {
    gemms.Report(log, prefix, "target gemm");
    embed_kernels.Report(log, prefix, "target embedding");
  }
};

// The state for `cache`, created on first use.
[[nodiscard]] inline Qwen4ExpState& State(core::DecodeCache& cache) {
  if (!cache.arch) {
    cache.arch = std::make_unique<Qwen4ExpState>();
  }
  return *static_cast<Qwen4ExpState*>(cache.arch.get());
}

}  // namespace tessera::models::qwen4exp
