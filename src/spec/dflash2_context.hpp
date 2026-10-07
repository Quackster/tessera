#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"

namespace tessera::spec {

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

}  // namespace tessera::spec
