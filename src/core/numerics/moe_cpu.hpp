#pragma once

#include <cstddef>
#include <cstdint>
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

// Compute several routed experts on the host across threads and accumulate
// their weighted down outputs into `acc`: acc[j] += sum_c wts[c] * out_c[j].
// `gate`, `up` and `down` are the whole paged tensors (every expert), each
// with its per-expert byte stride; `experts` indexes into them and `wts`
// holds one weight per expert. Threads split the experts and the partial sums
// combine in thread order, so the result is deterministic. The activation `x`
// and `acc` are (hidden). This is the CPU side of the CPU+GPU hybrid MoE.
//
// Usage:
//   CpuMoeExpertsForward(x, cpu_ids, cpu_wts, gate_dt, gate, gate_per, ...,
//                        hidden, inter, acc);
[[nodiscard]] std::expected<void, StatusCode> CpuMoeExpertsForward(
    std::span<const float> x, std::span<const std::uint32_t> experts,
    std::span<const float> wts, DType gate_dtype,
    std::span<const std::byte> gate, std::size_t gate_per, DType up_dtype,
    std::span<const std::byte> up, std::size_t up_per, DType down_dtype,
    std::span<const std::byte> down, std::size_t down_per, std::size_t hidden,
    std::size_t inter, std::span<float> acc);

}  // namespace tessera::core
