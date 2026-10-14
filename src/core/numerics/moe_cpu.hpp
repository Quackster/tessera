#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "tessera/types.hpp"

namespace tessera::core {

// One routed expert's forward on the host: the gate and up projections, the
// silu product and the down projection, through the block-quantized host GEMM
// references. `gate` and `up` are (inter x hidden) in the GGUF layout (the
// output row length first), `down` is (hidden x inter), and `x` and `out` are
// (hidden). A shape mismatch or an unsupported dtype is InvalidArgument or
// UnsupportedFeature. This is the CPU side of the CPU+GPU hybrid MoE: it
// computes the experts the device cache does not hold.
//
// Usage:
//   auto status = CpuMoeExpertForward(x, gate_dt, gate, up_dt, up, down_dt,
//                                     down, hidden, inter, out);
[[nodiscard]] std::expected<void, StatusCode> CpuMoeExpertForward(
    std::span<const float> x, DType gate_dtype,
    std::span<const std::byte> gate, DType up_dtype,
    std::span<const std::byte> up, DType down_dtype,
    std::span<const std::byte> down, std::size_t hidden, std::size_t inter,
    std::span<float> out);

}  // namespace tessera::core
