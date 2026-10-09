#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"
#include "spec/dflash2_layer.hpp"

namespace tessera::spec {

// A stack of DFlash2 draft layers. Runs the layers in order, carrying the
// residual, then the final RMSNorm over (last output + residual). `embed`
// and `out` are rows x hidden_dim.
[[nodiscard]] std::expected<void, StatusCode> DraftStackRef(
    std::span<const float> embed,
    const std::vector<DraftLayerWeights>& layers,
    std::span<const float> final_norm, std::span<float> out,
    std::size_t rows, std::size_t hidden_dim, std::size_t heads,
    std::size_t kv_heads, std::size_t head_dim, std::size_t ffn,
    std::size_t taps, std::size_t group_size, std::size_t block_size,
    std::size_t window, std::uint64_t pos_base, double theta, float eps,
    std::span<const float> context_hidden = {}, bool causal = true);

// Device version. `layers` are the per-layer weight buffers in order; the
// per-call layer scratch is allocated inside each layer call.
[[nodiscard]] std::expected<void, StatusCode> DraftStackDevice(
    Backend& backend, const Kernel& rmsnorm, const Kernel& gemm,
    const Kernel& conv, const Kernel& rope, const Kernel& attention,
    const Kernel& silu, const Kernel& add, const Buffer& embed,
    const std::vector<DraftLayerBuffers>& layers, const Buffer& final_norm,
    Buffer& out, std::size_t rows, std::size_t hidden_dim, std::size_t heads,
    std::size_t kv_heads, std::size_t head_dim, std::size_t ffn,
    std::size_t taps, std::size_t group_size, std::size_t block_size,
    std::size_t window, std::uint64_t pos_base, double theta, float eps,
    const Buffer* context_hidden = nullptr, std::size_t ctx = 0,
    bool causal = true,
    const std::vector<DraftContextKvView>* context_kv = nullptr);

}  // namespace tessera::spec
