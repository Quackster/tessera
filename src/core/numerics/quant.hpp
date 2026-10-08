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

// Encode fp32 to IEEE 754 binary16 (round to nearest, ties to even;
// overflow saturates to infinity). Lossy; the inverse of Fp16ToFloat for
// representable values.
[[nodiscard]] std::uint16_t Fp16FromFloat(float value);

// Host reference for the "quantize_q8" built-in: symmetric int8 over
// rows x cols fp32 (per-row absmax scale amax/127, round half away from
// zero, clamped to [-127, 127]). `out` is rows*cols signed bytes and
// `scale` has one fp32 per row. cols must be a multiple of 4.
[[nodiscard]] std::expected<void, StatusCode> QuantizeQ8Ref(
    std::span<const float> in, std::span<std::byte> out,
    std::span<float> scale, std::size_t rows, std::size_t cols);

// Host reference for the "quantize_q4" built-in: symmetric 4-bit over
// rows x cols fp32 (per-row absmax scale amax/7, round half away from
// zero, clamped to [-7, 7]). `out` is rows*cols/2 bytes (element 2i in the
// low nibble, 2i+1 in the high) and `scale` has one fp32 per row. cols
// must be a multiple of 8.
[[nodiscard]] std::expected<void, StatusCode> QuantizeQ4Ref(
    std::span<const float> in, std::span<std::byte> out,
    std::span<float> scale, std::size_t rows, std::size_t cols);

// Host reference for the "cast_f32_f16" built-in: encode `in` (an even
// number of fp32) to fp16, little-endian, into `out` (in.size()*2 bytes).
[[nodiscard]] std::expected<void, StatusCode> CastF32F16Ref(
    std::span<const float> in, std::span<std::byte> out);

// The OCP FP8 E4M3 maximum magnitude, the scale target vLLM's dynamic
// per-token fp8 quantization divides the row absmax by.
inline constexpr float kFp8E4M3Max = 448.0f;

// The smallest dynamic per-token fp8 scale (1 / (448 * 512)); a row whose
// absmax is below this still uses it, matching vLLM scaled_fp8_quant.
inline constexpr float kFp8E4M3MinScale = 1.0f / (kFp8E4M3Max * 512.0f);

// Host reference for the "quantize_fp8" built-in: quantize-dequantize
// each row of `data` in place to OCP FP8 E4M3 with a dynamic per-token
// scale max(amax/448, 1/(448*512)) and round-to-nearest-even, so the
// result reproduces the W4A8 activation the served MXFP4 target feeds its
// linear layers. `scale` receives one fp32 per row. rows and cols must be
// non-zero and `data`/`scale` sized rows*cols / rows.
[[nodiscard]] std::expected<void, StatusCode> QuantizeFp8Ref(
    std::span<float> data, std::span<float> scale, std::size_t rows,
    std::size_t cols);

// Decode a bfloat16 (little-endian) to fp32. bf16 shares the fp32
// exponent field, so the conversion is exact.
[[nodiscard]] float Bf16ToFloat(std::uint16_t bits);

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

// IQ4 codebook shared by the IQ4_NL and IQ4_XS dequants (llama.cpp
// kvalues_iq4nl; see CREDITS.md).
inline constexpr std::int8_t kIq4NlValues[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89,
    113};

// IQ3_S 3-bit grid: 512 entries of 4 magnitude bytes (llama.cpp
// iq3s_grid; see CREDITS.md). Index with 9 bits; byte j of the word
// is grid value j, little-endian.
inline constexpr std::uint32_t kIq3sGrid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301,
    0x01010303, 0x01010305, 0x01010309, 0x0101030d, 0x01010501, 0x01010503,
    0x0101050b, 0x01010707, 0x01010901, 0x01010905, 0x0101090b, 0x0101090f,
    0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301,
    0x01030303, 0x0103030b, 0x01030501, 0x01030507, 0x0103050f, 0x01030703,
    0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b, 0x01030f05, 0x01050101,
    0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b,
    0x0105090f, 0x01050b03, 0x01050b07, 0x01050f01, 0x01050f07, 0x01070107,
    0x01070303, 0x0107030b, 0x01070501, 0x01070505, 0x01070703, 0x01070707,
    0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509,
    0x01090705, 0x01090901, 0x01090907, 0x01090b03, 0x01090f01, 0x010b0105,
    0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d, 0x010b0707, 0x010b0903,
    0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105,
    0x010f0109, 0x010f0501, 0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01,
    0x010f0b09, 0x03010101, 0x03010103, 0x03010105, 0x03010109, 0x03010301,
    0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03,
    0x03010f05, 0x03030101, 0x03030103, 0x03030107, 0x0303010d, 0x03030301,
    0x03030309, 0x03030503, 0x03030701, 0x03030707, 0x03030903, 0x03030b01,
    0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907,
    0x03050b0b, 0x03050d01, 0x03050f05, 0x03070103, 0x03070109, 0x0307010f,
    0x03070301, 0x03070307, 0x03070503, 0x0307050f, 0x03070701, 0x03070709,
    0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01,
    0x03090b09, 0x030b0103, 0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701,
    0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509, 0x030d050f, 0x030d0909,
    0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103,
    0x05010107, 0x0501010b, 0x0501010f, 0x05010301, 0x05010305, 0x05010309,
    0x0501030d, 0x05010503, 0x05010507, 0x0501050f, 0x05010701, 0x05010705,
    0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301,
    0x05030307, 0x0503030f, 0x05030505, 0x0503050b, 0x05030703, 0x05030709,
    0x05030905, 0x05030b03, 0x05050103, 0x05050109, 0x0505010f, 0x05050503,
    0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303,
    0x05070505, 0x05070509, 0x05070703, 0x05070707, 0x05070905, 0x05070b01,
    0x05070d0d, 0x05090103, 0x0509010f, 0x05090501, 0x05090507, 0x05090705,
    0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101,
    0x050d0105, 0x050d010f, 0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b,
    0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907, 0x050f0b01, 0x07010105,
    0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03,
    0x07010d07, 0x07010f03, 0x07030103, 0x07030107, 0x0703010b, 0x07030309,
    0x07030503, 0x07030507, 0x07030901, 0x07030d01, 0x07030f05, 0x07030f0d,
    0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f,
    0x07070701, 0x07070903, 0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07,
    0x07090107, 0x07090303, 0x0709030d, 0x07090505, 0x07090703, 0x07090b05,
    0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903,
    0x070f0103, 0x070f0107, 0x070f0501, 0x070f0505, 0x070f070b, 0x09010101,
    0x09010109, 0x09010305, 0x09010501, 0x09010509, 0x0901050f, 0x09010705,
    0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03,
    0x09030b0b, 0x09050103, 0x09050107, 0x09050301, 0x0905030b, 0x09050503,
    0x09050707, 0x09050901, 0x09050b0f, 0x09050d05, 0x09050f01, 0x09070109,
    0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03,
    0x090b010b, 0x090b010f, 0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709,
    0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701, 0x090f0907, 0x090f0b03,
    0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107,
    0x0b03010b, 0x0b030305, 0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101,
    0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d, 0x0b050b07, 0x0b070105,
    0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d,
    0x0b0b0305, 0x0b0b050d, 0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105,
    0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307, 0x0d01030b, 0x0d010703,
    0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01,
    0x0d070101, 0x0d070309, 0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907,
    0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709, 0x0d0b0d01, 0x0d0d010b,
    0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05,
    0x0f030105, 0x0f030303, 0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103,
    0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503, 0x0f050701, 0x0f050b03,
    0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105,
    0x0f0d0703, 0x0f0f0101,
};

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

// Dequantize one Q5_K block (176 bytes) into 256 fp32 values.
// Layout (llama.cpp block_q5_K): d, dmin, 12 scale/min bytes, 32 high
// bits, 128 low nibbles. Ports dequantize_row_q5_K.
void DequantizeQ5K(std::span<const std::byte> block, std::span<float> out);

// Dequantize one Q6_K block (210 bytes) into 256 fp32 values.
// Layout: 128 low nibbles, 64 2-bit highs, 16 int8 scales, d.
// Ports dequantize_row_q6_K.
void DequantizeQ6K(std::span<const std::byte> block, std::span<float> out);

// Dequantize one Q3_K block (110 bytes) into 256 fp32 values.
// Layout: 32 high-bit bytes, 64 low 2-bit bytes, 12 scale bytes, d.
// Ports dequantize_row_q3_K.
void DequantizeQ3K(std::span<const std::byte> block, std::span<float> out);

// Dequantize one IQ4_NL block (18 bytes) into 32 fp32 values (d plus
// 16 codebook nibbles through the 16-entry table below).
void DequantizeIQ4NL(std::span<const std::byte> block, std::span<float> out);

// Dequantize one IQ4_XS block (136 bytes) into 256 fp32 values (d,
// packed 6-bit scales, codebook nibbles). Ports dequantize_row_iq4_xs.
void DequantizeIQ4XS(std::span<const std::byte> block, std::span<float> out);

// Dequantize one IQ3_S block (110 bytes) into 256 fp32 values (d, 3-bit
// grid quants, signs, scales). Ports dequantize_row_iq3_s.
void DequantizeIQ3S(std::span<const std::byte> block, std::span<float> out);

// Dequantize one Q8_0 block (34 bytes) into 32 fp32 values (d fp16 plus
// 32 signed bytes). The GGUF Q8_0 layout.
void DequantizeQ80(std::span<const std::byte> block, std::span<float> out);

// Block layout (elements and bytes per block) of a supported quant
// dtype. UnsupportedFeature for plain or unsized dtypes. One canonical
// table shared by the GEMM and embedding paths.
struct QuantBlockLayout {
  std::size_t elements = 0;
  std::size_t bytes = 0;
};
[[nodiscard]] std::expected<QuantBlockLayout, StatusCode> BlockLayout(
    DType dtype);

// Dequantize a whole number of `dtype` blocks from `bytes` into out
// (bytes must hold blocks*BlockLayout.bytes; out a matching element
// count). InvalidArgument on a size mismatch; UnsupportedFeature for a
// plain or unsized dtype.
// Decode a whole bf16 tensor (2 bytes per element) to fp32.
[[nodiscard]] std::expected<void, StatusCode> DequantizeBf16(
    std::span<const std::byte> bytes, std::span<float> out);

// Decode a block-scaled FP8 E4M3 tensor (one fp32 scale per 128 x 128
// block, the DFlash2 draft quantization) to fp32. rows and cols must be
// multiples of 128; `scales` is (rows/128) x (cols/128) row-major.
[[nodiscard]] std::expected<void, StatusCode> DequantizeFp8Block(
    std::span<const std::byte> bytes, std::span<const float> scales,
    std::span<float> out, std::size_t rows, std::size_t cols);

[[nodiscard]] std::expected<void, StatusCode> DequantizeBlocks(
    DType dtype, std::span<const std::byte> bytes, std::span<float> out);

}  // namespace tessera::core
