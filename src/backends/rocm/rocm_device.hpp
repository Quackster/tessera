#pragma once

// Shared gfx11/gfx12 device primitives, ported from libr4d v0.5.0
// r4d_common.h (see CREDITS.md). Every helper is inline so each kernel
// translation unit carries its own copy, and no device call crosses a
// translation unit boundary. Only this backend directory includes HIP
// headers.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace tessera::backends::rocm {

// exp2 through the hardware transcendental where a target has one. The
// scalar form keeps the header usable from the host.
inline __host__ __device__ float Ex2(float value) {
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_exp2f)
  return __builtin_amdgcn_exp2f(value);
#else
  return exp2f(value);
#endif
}

// fp32 -> bf16 round-to-nearest-even. gfx11 and gfx12 have no pack
// instruction for bf16, so this is the software form the reference uses.
inline __host__ __device__ std::uint16_t F32ToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(
      (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

// bf16 -> fp32: shift the 16 bits into the fp32 high half.
inline __host__ __device__ float Bf16ToF32(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

// Exchange a value between lane L and lane L^16, the two halves a wave32
// WMMA fragment splits a row across. __shfl_xor lowers to the LDS-pipe
// ds_bpermute; the reference keeps it as the correctness baseline.
inline __device__ float Swap16(float value) {
  return __shfl_xor(value, 16, 32);
}

inline __device__ uint2 Swap16U2(uint2 value) {
  return make_uint2(__builtin_bit_cast(std::uint32_t, Swap16(
                                                 __builtin_bit_cast(float,
                                                                    value.x))),
                    __builtin_bit_cast(std::uint32_t, Swap16(
                                                 __builtin_bit_cast(float,
                                                                    value.y))));
}

// Gather the low (select 0) or high (select 1) byte of each of the four
// 16-bit lanes of {w0, w1} into one dword. Written so the compiler can
// contract each form to a single v_perm_b32.
inline __host__ __device__ std::uint32_t ByteGather(std::uint32_t w0,
                                                    std::uint32_t w1,
                                                    int select) {
  const std::uint32_t shift = static_cast<std::uint32_t>(select) * 8u;
  return ((w0 >> shift) & 0xFFu) | (((w0 >> (16u + shift)) & 0xFFu) << 8) |
         (((w1 >> shift) & 0xFFu) << 16) |
         (((w1 >> (16u + shift)) & 0xFFu) << 24);
}

// Stop the scheduler from moving work across this point. Place it between
// a register prefetch group and the WMMA consumers of that group.
inline __device__ void SchedBarrier() { __builtin_amdgcn_sched_barrier(0); }

// A barrier that orders LDS traffic only. __syncthreads() is a
// workgroup-scope acquire-release fence over every address space, so on
// gfx12 it invalidates the vector caches and the next tile misses its KV
// lines. Scoping the fence to the local address space removes that.
inline __device__ void LdsBarrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

}  // namespace tessera::backends::rocm
