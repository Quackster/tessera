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

// Decode an OCP FP8 E4M3 byte to fp32 (bias 8; only the all-ones
// mantissa is NaN, no infinities, max 448). Exact for every finite
// value.
//
// Usage:
//   float value = Fp8E4M3ToFloat(0x40);  // 1.0
[[nodiscard]] float Fp8E4M3ToFloat(std::uint8_t bits);

// Decode an OCP MX E8M0 scale byte to its fp32 multiplier (2^(s - 127)
// for finite s; NaN only for 0xFF, which yields fp32 infinity here and
// never appears in checked files).
//
// Usage:
//   float scale = E8M0ToFloat(127);  // 1.0
[[nodiscard]] float E8M0ToFloat(std::uint8_t scale);

// Decode one OCP MX E2M1 nibble (0..15) to fp32, unscaled (bias 1,
// max 4.0; 0x7 is NaN and never appears in checked files).
//
// Usage:
//   float value = F4E2M1ToFloat(3);  // 1.5
[[nodiscard]] float F4E2M1ToFloat(std::uint8_t nibble);

// Encode fp32 to OCP FP8 E4M3 bits (round to nearest, ties to even;
// overflow and NaN become 0x7F). Used for fixtures and conversion.
//
// Usage:
//   std::uint8_t bits = Fp32ToFp8E4M3Bits(1.0f);  // 0x40
[[nodiscard]] std::uint8_t Fp32ToFp8E4M3Bits(float value);

// Encode fp32 to the nearest OCP MX E2M1 nibble (0..15, ties away
// from the grid midpoint order; NaN becomes 0x7, overflow clamps to
// 4.0). Used for fixtures and conversion.
//
// Usage:
//   std::uint8_t nibble = Fp32ToF4E2M1Nibble(1.5f);  // 3
[[nodiscard]] std::uint8_t Fp32ToF4E2M1Nibble(float value);

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
