#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"

namespace tessera::spec {

// The DFlash2 target-hidden fusion: concatenate `n` target hidden tensors
// (each rows x features) along the feature axis, then project with `fc`
// to the draft hidden size. `aux` is n x rows x features (layer-major),
// `fc_w` is hidden x (n*features) and `out` is rows x hidden.
[[nodiscard]] std::expected<void, StatusCode> DraftFuseRef(
    std::span<const float> aux, std::span<const float> fc_w,
    std::span<float> out, std::size_t n, std::size_t rows,
    std::size_t features, std::size_t hidden);

// Device version. `gemm` is gemm_f32, `concat` the concat_features
// built-in, and `scratch` a rows x (n*features) buffer.
[[nodiscard]] std::expected<void, StatusCode> DraftFuseDevice(
    Backend& backend, const Kernel& gemm, const Kernel& concat,
    Buffer& scratch, const Buffer& aux, const Buffer& fc_w, Buffer& out,
    std::size_t n, std::size_t rows, std::size_t features,
    std::size_t hidden);

}  // namespace tessera::spec
