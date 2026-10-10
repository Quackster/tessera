#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "models/qwen3_5/state.hpp"
#include "tessera/backend.hpp"
#include "tessera/model.hpp"
#include "tessera/types.hpp"

// Internal hybrid helpers shared by the hybrid decoder and the MTP head
// (both run full-attention Qwen3.5 blocks on the device).

namespace tessera::models::qwen3_5 {

// Derived recurrent-layer geometry from the SSM config.
struct LinearGeometry {
  std::size_t key_dim = 0;
  std::size_t value_dim = 0;
  std::size_t num_v_heads = 0;
  std::size_t head_v_dim = 0;
  std::size_t num_k_heads = 0;
  std::size_t head_k_dim = 0;
  std::size_t conv_dim = 0;
  std::size_t width = 0;
  std::size_t factor = 0;
};

inline std::expected<LinearGeometry, StatusCode> DeriveGeometry(
    const TransformerConfig& cfg) {
  LinearGeometry g;
  g.head_k_dim = cfg.ssm.state_size;
  g.num_k_heads = cfg.ssm.group_count;
  g.value_dim = cfg.ssm.inner_size;
  g.num_v_heads = cfg.ssm.time_step_rank;
  g.width = cfg.ssm.conv_kernel;
  if (g.head_k_dim == 0 || g.num_k_heads == 0 || g.value_dim == 0 ||
      g.num_v_heads == 0 || g.width == 0 ||
      g.value_dim % g.num_v_heads != 0 ||
      g.num_v_heads % g.num_k_heads != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  g.key_dim = g.head_k_dim * g.num_k_heads;
  g.head_v_dim = g.value_dim / g.num_v_heads;
  g.factor = g.num_v_heads / g.num_k_heads;
  g.conv_dim = g.key_dim * 2 + g.value_dim;
  return g;
}

// Loads the hybrid kernels and allocates the per-step scratch on first
// use; idempotent. Requires the derived linear geometry.
[[nodiscard]] std::expected<void, StatusCode> EnsureHybridReady(
    Backend& backend, const TransformerConfig& cfg, core::DecodeCache& cache,
    const LinearGeometry& g);

// One projection C = A x W^T for `m` rows of `a` (m x k fp32). For an
// MXFP4 weight (F4E2M1) the activation is first quantize-dequantized per
// token to FP8 E4M3 in place, the W4A8 contract the served Qwen3.8 target
// runs; other dtypes pass through unchanged. `a` is read and, for MXFP4,
// rewritten in place. `w` is n x k, `out` is m x n.
[[nodiscard]] std::expected<void, StatusCode> ProjectBatch(
    Backend& backend, Qwen35State& h, DType dtype, Buffer& a,
    const Buffer& w, Buffer& out, std::size_t m, std::size_t n,
    std::size_t k);

// Drop the cached fp8 activation pack (see ProjectBatch). Call after a
// kernel rewrites a projection input, so the next projection re-packs.
void InvalidateActivationPack(Qwen35State& h);

// The per-token FP8 QDQ used by ProjectBatch for an MXFP4 weight; a no-op
// for other dtypes. Exposed for the MTP head. `data` is rows x cols fp32.
[[nodiscard]] std::expected<void, StatusCode> QuantizeMxFp4Input(
    Backend& backend, Qwen35State& h, DType dtype, Buffer& data,
    std::size_t rows, std::size_t cols);

// C = A x dequant(W)^T through the warp-per-output GEMV family. Small
// output counts leave the device under-filled (a 5120-column projection is
// 640 workgroups, under one wave), so the k range splits across grid_y
// workgroups and a second pass sums the partials in fixed order; a large
// projection keeps the plain one-workgroup-per-output grid. Deterministic:
// no atomics. Use only with a "_vec" kernel.
[[nodiscard]] inline std::expected<void, StatusCode> ProjectGemvDevice(
    Backend& backend, Qwen35State& h, const Kernel& gemm, const Buffer& a,
    const Buffer& w, Buffer& out, std::size_t m, std::size_t n,
    std::size_t k) {
  const std::size_t total = m * n;
  const std::size_t split = core::detail::GemmVecSplit(total, k);
  if (split <= 1) {
    return core::detail::ProjectDevice(backend, gemm, a, w, out, m, n, k);
  }
  if (h.gemv_part == nullptr || h.gemv_part->Size() < split * total * 4) {
    auto buffer =
        backend.AllocateBuffer(split * total * 4, MemoryKind::Device);
    if (!buffer) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    h.gemv_part = std::move(*buffer);
  }
  if (h.gemv_reduce_kernel == nullptr) {
    auto kernel = backend.LoadKernel("gemm_vec_reduce", {});
    if (!kernel) {
      return std::unexpected(kernel.error());
    }
    h.gemv_reduce_kernel = std::move(*kernel);
  }
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>(
      (total + core::detail::kGemmVecOutputsPerBlock - 1) /
      core::detail::kGemmVecOutputsPerBlock);
  launch.grid_y = static_cast<std::uint32_t>(split);
  launch.block_x = 256;
  launch.buffers = {&a, &w, h.gemv_part.get()};
  launch.scalars = {m, n, k, split};
  if (!backend.LaunchKernel(gemm, launch)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  KernelLaunch reduce;
  reduce.grid_x = static_cast<std::uint32_t>((total + 255) / 256);
  reduce.block_x = 256;
  reduce.buffers = {h.gemv_part.get(), &out};
  reduce.scalars = {total, split};
  return backend.LaunchKernel(*h.gemv_reduce_kernel, reduce);
}

// Gather `tokens` embedding rows of the model's token_embd.weight into
// `out` (rows x hidden fp32) on the device. Uses a device gather kernel
// for the formats it covers (f32, bf16, Q4_K) and a host dequantize plus
// upload otherwise. `out` must hold rows * hidden floats.
[[nodiscard]] std::expected<void, StatusCode> GatherEmbeddingRows(
    Backend& backend, const Model& model, Qwen35State& h,
    std::span<const std::uint32_t> tokens, std::size_t hidden, Buffer& out);

// Runs the block's gated MLP (post-attention norm, gate/up/down, residual
// add) on h.x in place. The attention output must already be added to h.x.
[[nodiscard]] std::expected<void, StatusCode> RunFfn(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen35State& h, std::size_t layer);

// Runs one full-attention block (attn_norm, gated attention, gated MLP) on
// h.x in place, using `kv` for the key/value cache at position `pos`.
[[nodiscard]] std::expected<void, StatusCode> RunFullBlock(
    Backend& backend, const Model& model, const TransformerConfig& cfg,
    Qwen35State& h, std::size_t layer, std::uint64_t pos,
    Qwen35State::FullKv& kv);

// Per-row top-1 of `logits` (rows x vocab) computed on the device, so a
// caller reads back `rows` ids instead of the whole vocabulary. Loads the
// top_k_rows kernel and grows the shared argmax scratch on demand.
[[nodiscard]] inline std::expected<std::vector<std::uint32_t>, StatusCode>
DeviceRowArgMax(Backend& backend, Qwen35State& h, const Buffer& logits,
                std::size_t rows, std::size_t vocab) {
  if (h.top_k_kernel == nullptr) {
    auto kernel = backend.LoadKernel("top_k_rows", {});
    if (!kernel) {
      return std::unexpected(kernel.error());
    }
    h.top_k_kernel = std::move(*kernel);
  }
  if (h.argmax_ids == nullptr || h.argmax_ids->Size() < rows * 4) {
    auto ids = backend.AllocateBuffer(rows * 4, MemoryKind::Device);
    if (!ids) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    h.argmax_ids = std::move(*ids);
  }
  if (h.argmax_vals == nullptr || h.argmax_vals->Size() < rows * 4) {
    auto vals = backend.AllocateBuffer(rows * 4, MemoryKind::Device);
    if (!vals) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    h.argmax_vals = std::move(*vals);
  }
  auto top = core::detail::TopKRowsDevice(
      backend, *h.top_k_kernel, logits, *h.argmax_ids, *h.argmax_vals, rows,
      vocab, 1);
  if (!top) {
    return std::unexpected(top.error());
  }
  std::vector<std::uint32_t> out(rows, 0);
  if (!backend.CopyD2H(*h.argmax_ids,
                       reinterpret_cast<std::byte*>(out.data()), rows * 4)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return out;
}

}  // namespace tessera::models::qwen3_5
