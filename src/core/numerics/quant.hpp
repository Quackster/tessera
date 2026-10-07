#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "tessera/types.hpp"

namespace tessera::core {

// Q4_K block layout (ggml 2026 numbering): kQ4KBlockElements (256)
// elements in 144 bytes: d fp16, dmin fp16, 12 scale/min bytes, 128
// nibble bytes.
constexpr std::size_t kQ4KBlockBytes = 144;
constexpr std::size_t kQ4KScaleBytes = 12;

// Decode an IEEE 754 binary16 (little-endian) to fp32. The conversion
// is exact (no rounding loss).
//
// Usage:
//   std::uint16_t half = 0x3C00;  // 1.0
//   float value = Fp16ToFloat(half);
[[nodiscard]] float Fp16ToFloat(std::uint16_t half);

// Encode an fp32 value to IEEE 754 binary16 bits (round to nearest,
// ties to even; overflow becomes inf).
[[nodiscard]] std::uint16_t Fp32ToHalfBits(float value);

// Dequantize one Q4_K block into 256 fp32 values.
// `block` must hold kQ4KBlockBytes bytes; `out` kQ4KBlockElements
// floats. The block bytes are the ggml layout (d, dmin, scales, qs).
void DequantizeQ4K(std::span<const std::byte> block, std::span<float> out);

// Deterministically quantize 256 fp32 values into one Q4_K block
// (test fixtures and future weight conversion). Per element, the error
// against the dequantized value is bounded by step/2 + dm/2 where step
// is the sub-block's d*scale step and dm the block's dmin; the all
// zero input produces the all zero block.
void QuantizeQ4K(std::span<const float> values, std::byte* block);

}  // namespace tessera::core
