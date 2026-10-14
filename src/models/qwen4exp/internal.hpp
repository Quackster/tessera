#pragma once

#include <cstddef>
#include <expected>
#include <string_view>

#include "core/decode_internal.hpp"
#include "models/qwen4exp/state.hpp"
#include "tessera/backend.hpp"
#include "tessera/model.hpp"
#include "tessera/types.hpp"

namespace tessera::models::qwen4exp {

// The derived geometry of a qwen4exp trunk: the attention widths, the
// hyper-connection widths and the gated-delta (recurrent) dimensions. All
// values come from the model config; a mismatch is MalformedFile.
struct Qwen4ExpGeometry {
  std::size_t hidden = 0;
  std::size_t heads = 0;
  std::size_t kv_heads = 0;
  std::size_t head_dim = 0;
  std::size_t q_dim = 0;   // heads * head_dim
  std::size_t kv_dim = 0;  // kv_heads * head_dim
  std::size_t rope_dim = 0;
  double rope_theta = 0.0;
  std::size_t hc = 0;
  std::size_t hc_lr = 0;
  std::size_t hc_dim = 0;  // hc * hidden
  // Gated-delta dimensions (only the recurrent layers use these).
  std::size_t gdn_conv_dim = 0;
  std::size_t gdn_key_dim = 0;
  std::size_t gdn_value_dim = 0;
  std::size_t gdn_num_k_heads = 0;
  std::size_t gdn_num_v_heads = 0;
  std::size_t gdn_head_k_dim = 0;
  std::size_t gdn_head_v_dim = 0;
  std::size_t gdn_width = 0;
  std::size_t gdn_factor = 0;
  // Sparse indexer dimensions (only the QSA layers use these).
  std::size_t idx_heads = 0;
  std::size_t idx_dim = 0;
  std::size_t idx_top_k = 0;
};

// Derive the geometry or fail when a required field is absent.
[[nodiscard]] std::expected<Qwen4ExpGeometry, StatusCode> DeriveGeometry(
    const TransformerConfig& cfg);

// C = A @ dequant(W)^T for one projection, through the GEMM kernel that
// matches the weight dtype. `a` is (m x k), `w` is (n x k), `out` is
// (m x n). All buffers are device-resident fp32 activations.
[[nodiscard]] std::expected<void, StatusCode> Project(
    Backend& backend, Qwen4ExpState& h, DType dtype, const Buffer& a,
    const Buffer& w, Buffer& out, std::size_t m, std::size_t n,
    std::size_t k);

// One gated residual read: normalize the stack into the sub-block input
// `mixed`, and (when `w_inject` is non-null) the per-stream injection.
// `w_norm` is the f32 gamma, `w_down`/`w_up` the bf16 low-rank pair and
// `w_inject` the bf16 per-stream injection, or null. Used by the trunk and
// the MTP head.
[[nodiscard]] std::expected<void, StatusCode> HyperRead(
    Backend& backend, Qwen4ExpState& h, const TransformerConfig& cfg,
    const Qwen4ExpGeometry& g, const Buffer& w_norm, const Buffer& w_down,
    const Buffer& w_up, const Buffer* w_inject, Buffer& r, Buffer& mixed,
    Buffer* inject);

// One gated residual write: fold `block_out` back into the stack with the
// per-stream injection.
[[nodiscard]] std::expected<void, StatusCode> HyperWrite(
    Backend& backend, Qwen4ExpState& h, const Qwen4ExpGeometry& g, Buffer& r,
    const Buffer& block_out, const Buffer& inject);

// The sparse mixture-of-experts FFN for one layer: route, run `top_k`
// routed experts and the always-on shared expert, and write the block
// output to `h.block_out`. `mixed` is the normed activation from the FFN
// hyper-connection read. The caller folds it into the residual stack with
// the hyper-connection write.
[[nodiscard]] std::expected<void, StatusCode> RunMoeFfn(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen4ExpState& h, std::size_t layer, const Buffer& mixed);

}  // namespace tessera::models::qwen4exp
