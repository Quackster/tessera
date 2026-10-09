#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <vector>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"

namespace tessera::spec {

// A read-only view of one draft layer's cached context K/V, ctx rows of
// kv_heads*head_dim each. Null means the layer has no context.
struct DraftContextKvView {
  const Buffer* k = nullptr;
  const Buffer* v = nullptr;
};

// One draft layer's owned context K/V buffers.
struct DraftContextKvBuffers {
  std::unique_ptr<Buffer> k;
  std::unique_ptr<Buffer> v;
};

// The device-resident draft context: one growing K/V pair per draft layer,
// row t holding the RoPE-rotated context key/value for absolute position
// `base + t`. The draft block attends over rows `0..ctx-1`.
struct DraftContextCache {
  std::vector<DraftContextKvBuffers> layers;
  std::size_t rows = 0;      // committed context rows
  std::size_t capacity = 0;  // allocated rows
  std::size_t base = 0;      // absolute position of row 0
};

// Per-layer views of the cache K/V, for the draft block/stack/layer.
[[nodiscard]] inline std::vector<DraftContextKvView> DraftContextViews(
    const DraftContextCache& cache) {
  std::vector<DraftContextKvView> views;
  views.reserve(cache.layers.size());
  for (const DraftContextKvBuffers& layer : cache.layers) {
    views.push_back({layer.k.get(), layer.v.get()});
  }
  return views;
}

// The DFlash2 context key/value projection: the fused target hidden
// states become the draft attention context. They are normalized by
// hidden_norm, projected by the layer's k_proj/v_proj, K-normalized and
// RoPE-rotated. `context` is ctx x hidden_dim; `k_proj`/`v_proj` are
// (kv_heads*head_dim) x hidden_dim; `k_norm` is head_dim; `out_k` and
// `out_v` are ctx x (kv_heads*head_dim).
[[nodiscard]] std::expected<void, StatusCode> DraftContextKvRef(
    std::span<const float> context, std::span<const float> hidden_norm,
    std::span<const float> k_proj, std::span<const float> v_proj,
    std::span<const float> k_norm, std::span<float> out_k,
    std::span<float> out_v, std::size_t ctx, std::size_t hidden_dim,
    std::size_t kv_heads, std::size_t head_dim, std::uint64_t pos_base,
    double theta, float eps);

// Device version. `rmsnorm`, `gemm` (gemm_f32) and `rope` are built-ins;
// `normed` is a ctx x hidden_dim scratch buffer.
[[nodiscard]] std::expected<void, StatusCode> DraftContextKvDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& rope, Buffer& normed, const Buffer& context,
    const Buffer& hidden_norm, const Buffer& k_proj, const Buffer& v_proj,
    const Buffer& k_norm, Buffer& out_k, Buffer& out_v, std::size_t ctx,
    std::size_t hidden_dim, std::size_t kv_heads, std::size_t head_dim,
    std::uint64_t pos_base, double theta, float eps);

// Append `new_rows` committed target positions to the draft context cache.
// `aux` holds one device buffer per capture layer (each new_rows x
// features); `fc_w` is hidden_dim x (n*features); the per-layer
// `hidden_norm`, `k_w`, `v_w` and `k_norm` are the context projection
// weights. Each new row is fused (`fc` of the concatenated aux), normalized,
// projected per layer, K-normalized and RoPE-rotated at its absolute
// position (`cache.rows + row` before the limit), then stored. With
// `limit > 0` the cache keeps only the most recent `limit` rows and advances
// `cache.base`. The kernels are the context's `rmsnorm`, `gemm` (gemm_f32),
// `rope`, `concat` (concat_features) and `quantize` (quantize_fp8, the
// per-token QDQ the drafter was trained on); scratch is allocated inside.
[[nodiscard]] std::expected<void, StatusCode> DraftContextAppendDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& rope, const Kernel& concat, const Kernel& quantize,
    DraftContextCache& cache, const std::vector<const Buffer*>& aux,
    const Buffer& fc_w, const std::vector<const Buffer*>& hidden_norm,
    const std::vector<const Buffer*>& k_w,
    const std::vector<const Buffer*>& v_w,
    const std::vector<const Buffer*>& k_norm, std::size_t n,
    std::size_t features, std::size_t row_offset, std::size_t new_rows,
    std::size_t hidden_dim, std::size_t kv_heads, std::size_t head_dim,
    double theta, float eps, std::size_t limit = 0);

}  // namespace tessera::spec
