#include "core/numerics/quant.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace tessera::core {

namespace {

std::uint8_t ReadByte(std::span<const std::byte> data, std::size_t offset) {
  return static_cast<std::uint8_t>(data[offset]);
}

std::uint16_t ReadU16(std::span<const std::byte> data, std::size_t offset) {
  return static_cast<std::uint16_t>(
      ReadByte(data, offset) | (ReadByte(data, offset + 1) << 8));
}

void WriteU16(std::byte* data, std::size_t offset, std::uint16_t value) {
  data[offset] = static_cast<std::byte>(value & 0xFF);
  data[offset + 1] = static_cast<std::byte>((value >> 8) & 0xFF);
}

// Round to nearest, ties to even (the default FP rounding mode).
std::uint32_t RoundHalfEven(float value) {
  const long long rounded = std::lrint(static_cast<double>(value));
  if (rounded < 0) {
    return 0;
  }
  return static_cast<std::uint32_t>(rounded);
}

std::uint32_t ClampTo6(std::uint32_t value) {
  return value > 63 ? 63 : value;
}

std::uint32_t ClampTo15(std::uint32_t value) {
  return value > 15 ? 15 : value;
}

// Shift right with round-to-nearest-even (1 <= shift <= 31).
std::uint32_t ShiftRNE(std::uint32_t value, int shift) {
  const std::uint32_t mask = (1u << shift) - 1;
  const std::uint32_t rem = value & mask;
  const std::uint32_t half = 1u << (shift - 1);
  std::uint32_t result = value >> shift;
  if (rem > half || (rem == half && (result & 1))) {
    ++result;
  }
  return result;
}

// The 6-bit scale/min pair of sub-block j from the 12-byte packing.
// Byte-wise port of ggml's get_scale_min_k4 (the kernels in
// backends/vulkan and backends/rocm are ports of this same formula).
void GetScaleMin(std::size_t j, const std::byte* scales, std::uint8_t* scale,
                 std::uint8_t* min) {
  const auto byte = [scales](std::size_t i) {
    return static_cast<std::uint8_t>(scales[i]);
  };
  if (j < 4) {
    *scale = byte(j) & 63;
    *min = byte(j + 4) & 63;
  } else {
    *scale = (byte(j + 4) & 15) | ((byte(j - 4) >> 6) << 4);
    *min = (byte(j + 4) >> 4) | ((byte(j) >> 6) << 4);
  }
}

}  // namespace

float Fp16ToFloat(std::uint16_t half) {
  const std::uint32_t sign = half >> 15;
  const std::uint32_t exp = (half >> 10) & 0x1F;
  const std::uint32_t mant = half & 0x3FF;
  float value;
  if (exp == 0) {
    // Subnormal: mant * 2^-24 is exactly representable in fp32.
    value = static_cast<float>(mant) * 0x1p-24f;
  } else if (exp == 31) {
    value = mant != 0 ? std::nanf("") : std::numeric_limits<float>::infinity();
  } else {
    // (1 + mant/1024) * 2^(exp-15): the power-of-two factor keeps the
    // product exact.
    value = (1.0f + static_cast<float>(mant) / 1024.0f) *
            std::ldexp(1.0f, static_cast<int>(exp) - 15);
  }
  return sign != 0 ? -value : value;
}

float Fp8E4M3ToFloat(std::uint8_t bits) {
  const std::uint32_t sign = bits >> 7;
  const std::uint32_t exp = (bits >> 3) & 0xF;
  const std::uint32_t mant = bits & 0x7;
  float value;
  if (exp == 0) {
    value = static_cast<float>(mant) * 0x1p-10f;
  } else if (exp == 15) {
    // Only the all-ones mantissa is NaN; the rest encode 2^8 scale.
    if (mant == 7) {
      return std::nanf("");
    }
    value = (1.0f + static_cast<float>(mant) / 8.0f) * 256.0f;
  } else {
    value = (1.0f + static_cast<float>(mant) / 8.0f) *
            std::ldexp(1.0f, static_cast<int>(exp) - 8);
  }
  return sign != 0 ? -value : value;
}

float E8M0ToFloat(std::uint8_t scale) {
  return std::ldexp(1.0f, static_cast<int>(scale) - 127);
}

float F4E2M1ToFloat(std::uint8_t nibble) {
  const std::uint32_t sign = (nibble >> 3) & 1;
  const std::uint32_t exp = (nibble >> 1) & 0x3;
  const std::uint32_t mant = nibble & 1;
  float value;
  if (exp == 0) {
    value = static_cast<float>(mant) * 0.5f;
  } else if (exp == 3 && mant == 1) {
    value = std::nanf("");
  } else {
    value = (1.0f + static_cast<float>(mant) / 2.0f) *
            std::ldexp(1.0f, static_cast<int>(exp) - 1);
  }
  return sign != 0 ? -value : value;
}

std::uint8_t Fp32ToFp8E4M3Bits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = bits >> 31;
  if ((bits & 0x7FFFFFFF) > 0x7F800000 || !(value == value)) {
    return static_cast<std::uint8_t>((sign << 7) | 0x7F);
  }
  float abs = value < 0 ? -value : value;
  if (abs == 0.0f) {
    return static_cast<std::uint8_t>(sign << 7);
  }
  if (abs > 448.0f) {
    return static_cast<std::uint8_t>((sign << 7) | 0x7F);
  }
  if (abs > 240.0f) {
    // Top bin: 2^8 scale, mantissas 0..6 (448 max).
    long long mant = std::llrint(abs / 256.0f * 8.0f - 8.0f);
    if (mant < 0) {
      mant = 0;
    }
    if (mant > 6) {
      return static_cast<std::uint8_t>((sign << 7) | 0x7F);
    }
    return static_cast<std::uint8_t>((sign << 7) | (15 << 3) |
                                     static_cast<std::uint32_t>(mant));
  }
  int exp = 0;
  const float frac = std::frexp(abs, &exp);  // abs = frac * 2^exp
  int field = exp + 7;  // 2^(field-8) brackets abs with frac in [1, 2)
  if (field < 1) {
    const long long mant = std::llrint(abs * 1024.0f);
    if (mant >= 8) {
      return static_cast<std::uint8_t>((sign << 7) | (1 << 3));
    }
    return static_cast<std::uint8_t>((sign << 7) |
                                     static_cast<std::uint32_t>(mant));
  }
  long long mant = std::llrint((frac * 2.0f - 1.0f) * 8.0f);
  if (mant == 8) {
    if (field == 14) {
      mant = 7;  // 120 beats 256 below the 188 midpoint
    } else {
      ++field;
      mant = 0;
    }
  }
  if (field > 14) {
    // Between the E=14 max (120) and the E=15 base (256).
    if (abs < 188.0f) {
      field = 14;
      mant = 7;
    } else {
      field = 15;
      mant = 0;
    }
  }
  return static_cast<std::uint8_t>(
      (sign << 7) | (static_cast<std::uint32_t>(field) << 3) |
      static_cast<std::uint32_t>(mant));
}

std::uint8_t Fp32ToF4E2M1Nibble(float value) {
  if (!(value == value)) {
    return 0x7;
  }
  float abs = value < 0 ? -value : value;
  if (abs > 4.0f) {
    abs = 4.0f;
  }
  constexpr float kGrid[7] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};
  std::uint32_t best = 0;
  float best_dist = abs;
  for (std::uint32_t i = 1; i < 7; ++i) {
    const float dist = abs > kGrid[i] ? abs - kGrid[i] : kGrid[i] - abs;
    if (dist < best_dist) {
      best_dist = dist;
      best = i;
    }
  }
  const std::uint32_t sign = value < 0 ? 8 : 0;
  std::uint32_t exp = 0;
  std::uint32_t mant = 0;
  switch (best) {
    case 0: break;
    case 1: mant = 1; break;
    case 2: exp = 1; break;
    case 3:
      exp = 1;
      mant = 1;
      break;
    case 4: exp = 2; break;
    case 5:
      exp = 2;
      mant = 1;
      break;
    default: exp = 3; break;  // 4.0
  }
  return static_cast<std::uint8_t>(sign | (exp << 1) | mant);
}

std::uint16_t Fp32ToHalfBits(float value) {
  std::uint32_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = bits >> 31;
  const std::uint32_t exp_field = (bits >> 23) & 0xFF;
  const std::uint32_t mant = bits & 0x7FFFFF;
  const auto inf = static_cast<std::uint16_t>((sign << 15) | 0x7C00);
  if (exp_field == 0xFF) {
    return static_cast<std::uint16_t>(inf | (mant != 0 ? 0x200 : 0));
  }
  const int32_t exp = static_cast<int32_t>(exp_field) - 127;
  if (exp_field == 0) {
    // The largest fp32 subnormal (2^-149) underflows to 0 in fp16.
    return static_cast<std::uint16_t>(sign << 15);
  }
  if (exp > 15) {
    return inf;
  }
  if (exp < -14) {
    // fp16 subnormal: value / 2^-24 = (mant + 2^23) * 2^(exp + 1).
    const std::uint32_t scaled = mant + (1u << 23);
    const int shift = -exp - 1;  // 14..127
    if (shift >= 32) {
      return static_cast<std::uint16_t>(sign << 15);
    }
    const std::uint32_t result = ShiftRNE(scaled, shift);
    if (result > 1023) {
      // Rounding carried into the normal range (2^-14, the smallest
      // normal fp16).
      return static_cast<std::uint16_t>((sign << 15) | 0x0400);
    }
    return static_cast<std::uint16_t>((sign << 15) | result);
  }
  // Normal: reduce the 24-bit significand to 10 mantissa bits.
  const std::uint32_t significand = mant + (1u << 23);
  const std::uint32_t mant10 = ShiftRNE(significand, 13);
  if (mant10 > 1023) {
    if (exp + 1 > 15) {
      return inf;
    }
    return static_cast<std::uint16_t>(
        (sign << 15) | (static_cast<std::uint32_t>(exp + 1 + 15) << 10));
  }
  return static_cast<std::uint16_t>((sign << 15) |
                                    (static_cast<std::uint32_t>(exp + 15) << 10) |
                                    mant10);
}

void DequantizeQ4K(std::span<const std::byte> block, std::span<float> out) {
  const float d = Fp16ToFloat(ReadU16(block, 0));
  const float dmin = Fp16ToFloat(ReadU16(block, 2));
  const std::byte* scales = block.data() + 4;
  const std::byte* qs = block.data() + 16;
  for (std::size_t group = 0; group < 4; ++group) {
    std::uint8_t sc0, mn0, sc1, mn1;
    GetScaleMin(2 * group, scales, &sc0, &mn0);
    GetScaleMin(2 * group + 1, scales, &sc1, &mn1);
    const float scale0 = d * static_cast<float>(sc0);
    const float offset0 = dmin * static_cast<float>(mn0);
    const float scale1 = d * static_cast<float>(sc1);
    const float offset1 = dmin * static_cast<float>(mn1);
    for (std::size_t l = 0; l < 32; ++l) {
      const std::uint8_t q = ReadByte(block, 16 + 32 * group + l);
      out[group * 64 + l] = scale0 * static_cast<float>(q & 15) - offset0;
      out[group * 64 + 32 + l] = scale1 * static_cast<float>(q >> 4) - offset1;
    }
  }
}

void QuantizeQ4K(std::span<const float> values, std::byte* block) {
  std::memset(block, 0, kQ4KBlockBytes);
  float max_abs = 0.0f;
  for (float value : values) {
    max_abs = std::max(max_abs, std::fabs(value));
  }
  if (max_abs == 0.0f) {
    return;  // the all zero block represents the all zero values
  }
  // d = max_abs / 32 keeps every sub-block scale and offset within the
  // 6-bit range, so the clamps below never clip.
  const std::uint16_t d_bits = Fp32ToHalfBits(max_abs / 32.0f);
  const float d = Fp16ToFloat(d_bits);
  const float dm = d;  // the offset uses the same scale as the values
  WriteU16(block, 0, d_bits);
  WriteU16(block, 2, d_bits);
  std::uint8_t scales[kQ4KScaleBytes];
  std::memset(scales, 0, sizeof(scales));
  std::uint8_t qs[kQ4KBlockBytes - 16];
  std::memset(qs, 0, sizeof(qs));
  for (std::size_t j = 0; j < 8; ++j) {
    const float* sub = values.data() + 32 * j;
    float submin = sub[0];
    float submax = sub[0];
    for (std::size_t i = 1; i < 32; ++i) {
      submin = std::min(submin, sub[i]);
      submax = std::max(submax, sub[i]);
    }
    const float effective_min = std::min(submin, 0.0f);
    const float step = (submax - effective_min) / 15.0f;
    const std::uint32_t sc =
        ClampTo6(RoundHalfEven(step / d));
    const std::uint32_t m = submin < 0.0f
                               ? ClampTo6(RoundHalfEven(-submin / dm))
                               : 0;
    if (j < 4) {
      scales[j] = static_cast<std::uint8_t>(sc);
      scales[j + 4] = static_cast<std::uint8_t>(m);
    } else {
      scales[j + 4] = static_cast<std::uint8_t>(
          (sc & 0xF) | ((m & 0xF) << 4));
      scales[j - 4] |= static_cast<std::uint8_t>((sc >> 4) << 6);
      scales[j] |= static_cast<std::uint8_t>((m >> 4) << 6);
    }
    const float offset = dm * static_cast<float>(m);
    const float scale = d * static_cast<float>(sc);
    for (std::size_t i = 0; i < 32; ++i) {
      std::uint32_t q = 0;
      if (sc != 0) {
        q = ClampTo15(RoundHalfEven((sub[i] + offset) / scale));
      }
      const std::size_t element = 32 * j + i;
      const std::size_t byte_index = 32 * (element / 64) + element % 32;
      if (element / 32 % 2 == 1) {
        qs[byte_index] |= static_cast<std::uint8_t>(q << 4);
      } else {
        qs[byte_index] |= static_cast<std::uint8_t>(q & 0xF);
      }
    }
  }
  std::memcpy(block + 4, scales, sizeof(scales));
  std::memcpy(block + 16, qs, sizeof(qs));
}

}  // namespace tessera::core
