#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/attention.hpp"
#include "core/numerics/conv.hpp"
#include "core/numerics/vision.hpp"
#include "core/vision_block.hpp"
#include "core/vision_stack.hpp"
#include "core/vision_merger.hpp"
#include "core/numerics/selector.hpp"
#include "spec/dflash2_conv.hpp"
#include "spec/dflash2_mlp.hpp"
#include "spec/dflash2_attention.hpp"
#include "spec/dflash2_layer.hpp"
#include "spec/dflash2_stack.hpp"
#include "spec/dflash2_selector.hpp"
#include "spec/dflash2_fuse.hpp"
#include "spec/dflash2_context.hpp"
#include "spec/dflash2_block.hpp"
#include "spec/dflash2_candidates.hpp"
#include "core/numerics/gemm.hpp"
#include "core/numerics/norm.hpp"
#include "core/numerics/quant.hpp"
#include "models/qwen3_5/internal.hpp"
#include "models/qwen3_5/state.hpp"
#include "test_helpers.hpp"
#include "tessera/backend.hpp"
#include "tessera/types.hpp"

using tessera::Backend;
using tessera::CreateBackend;
using tessera::DType;
using tessera::MemoryKind;
using tessera::StatusCode;
using tessera::kQ4KBlockElements;
using tessera::testing::BlockU16;
using tessera::testing::DrawValue;
using tessera::testing::GemmTolerance;
using tessera::testing::QuantizeRows;
using tessera::testing::TestGetScaleMin;
using tessera::testing::TestDeviceIndex;
using tessera::testing::ToleranceFor;
namespace core = tessera::core;

namespace {

// CPU rasterizer marker: it must never appear in the GPU list.
constexpr const char* kLlvmpipeDeviceName = "llvmpipe";

// A device is required for these tests; skip cleanly without one.
void MakeBackendOrSkip(std::unique_ptr<Backend>& backend) {
  backend = CreateBackend();
  if (!backend) {
    GTEST_SKIP() << "no backend in this build";
  }
  backend->SetDeviceIndex(TestDeviceIndex());
  auto init = backend->Init();
  if (!init) {
    GTEST_SKIP() << "no device available: "
                 << tessera::ToString(init.error());
  }
}

}  // namespace

TEST(BackendTest, CreateAndInit) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  EXPECT_TRUE(backend->Name() == "vulkan" || backend->Name() == "rocm");
  EXPECT_NE(backend->DeviceName().size(), 0u);
  // Init is idempotent.
  auto second = backend->Init();
  ASSERT_TRUE(second.has_value()) << tessera::ToString(second.error());
}

// GPU-only enumeration: a CPU rasterizer must never appear in the GPU
// list, so no test run can silently execute on software rendering.

// TESSERA_TEST_GPU parsing: unset means GPU 0, a set value is honored.
// The previous value is restored, so later tests see an intact env.


// Batched upload: several buffers land in one call, each with its own
// bytes, and an oversized source is rejected. This is the weight-upload
// path, so it must match CopyH2D exactly.


// Per backend tolerance for the fp GEMM kernels.
struct FpTolerance {
  float abs = 0.0f;
  float rel = 0.0f;
};
FpTolerance FpToleranceFor(std::string_view backend) {
  if (backend == "vulkan") {
    return {2.0e-4f, 2.0e-4f};
  }
  if (backend == "rocm") {
    return {2.0e-4f, 2.0e-4f};
  }
  return {1.0e-3f, 1.0e-3f};
}

// Host: the fp8 and mxfp4 codecs hit exact known patterns.
TEST(BackendTest, FpCodecsHitKnownPatterns) {
  EXPECT_FLOAT_EQ(core::Fp8E4M3ToFloat(0x00), 0.0f);
  EXPECT_FLOAT_EQ(core::Fp8E4M3ToFloat(0x38), 1.0f);
  EXPECT_FLOAT_EQ(core::Fp8E4M3ToFloat(0x40), 2.0f);
  EXPECT_FLOAT_EQ(core::Fp8E4M3ToFloat(0xBC), -1.5f);
  EXPECT_FLOAT_EQ(core::Fp8E4M3ToFloat(0xC4), -3.0f);
  EXPECT_FLOAT_EQ(core::Fp8E4M3ToFloat(0x76), 224.0f);
  EXPECT_FLOAT_EQ(core::Fp8E4M3ToFloat(0x7E), 448.0f);
  EXPECT_TRUE(std::isnan(core::Fp8E4M3ToFloat(0x7F)));
  EXPECT_TRUE(std::isnan(core::Fp8E4M3ToFloat(0xFF)));
  EXPECT_EQ(core::Fp32ToFp8E4M3Bits(1.0f), 0x38);
  EXPECT_EQ(core::Fp32ToFp8E4M3Bits(-0.0f), 0x80);
  EXPECT_EQ(core::Fp32ToFp8E4M3Bits(448.0f), 0x7E);
  EXPECT_EQ(core::Fp32ToFp8E4M3Bits(1e30f), 0x7F);
  // OCP MX E2M1 has no NaN/inf: 0x7/0xF are +-6.0.
  const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                         2.0f, 3.0f, 4.0f, 6.0f};
  for (std::uint32_t i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(core::F4E2M1ToFloat(static_cast<std::uint8_t>(i)),
                                       grid[i])
        << "nibble " << i;
    EXPECT_FLOAT_EQ(core::F4E2M1ToFloat(static_cast<std::uint8_t>(8 | i)),
                                       -grid[i])
        << "nibble " << (8 | i);
    EXPECT_EQ(core::Fp32ToF4E2M1Nibble(grid[i]),
              static_cast<std::uint8_t>(i));
  }
  EXPECT_FLOAT_EQ(core::Fp32ToF4E2M1Nibble(6.0f), 0x7);
  EXPECT_FLOAT_EQ(core::Fp32ToF4E2M1Nibble(-6.0f), 0xF);
  EXPECT_FLOAT_EQ(core::E8M0ToFloat(126), 0.5f);
  EXPECT_FLOAT_EQ(core::E8M0ToFloat(127), 1.0f);
  EXPECT_FLOAT_EQ(core::E8M0ToFloat(128), 2.0f);
}

// Device: FP8 GEMM matches the host reference (2 x 32 outputs).
TEST(BackendTest, GemmFp8DeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(2024);
  constexpr std::size_t kM = 2;
  constexpr std::size_t kN = 32;
  constexpr std::size_t kK = 64;
  std::vector<float> a(kM * kK);
  std::vector<std::byte> w(kN * kK);
  std::vector<float> s(kN);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  for (std::size_t i = 0; i < w.size(); ++i) {
    w[i] = static_cast<std::byte>(
        core::Fp32ToFp8E4M3Bits(DrawValue(rng) * 2.0f));
  }
  for (std::size_t j = 0; j < kN; ++j) {
    s[j] = 0.25f * static_cast<float>(1 + (j % 4));
  }
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  auto s_buf = backend->AllocateBuffer(s.size() * 4, MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && w_buf.has_value() &&
              s_buf.has_value() && c_buf.has_value());
  const auto upload = [&backend](auto& buf, const auto& data,
                                 std::size_t elem) {
    return backend->CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data.data()),
                   data.size() * elem));
  };
  ASSERT_TRUE(upload(a_buf, a, 4).has_value());
  ASSERT_TRUE(upload(w_buf, w, 1).has_value());
  ASSERT_TRUE(upload(s_buf, s, 4).has_value());

  auto kernel = backend->LoadKernel("gemm_fp8", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kM * kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*s_buf).get(),
                    (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();

  std::vector<std::byte> readback(kM * kN * 4);
  auto download =
      backend->CopyD2H(**c_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(kM * kN);
  auto ref_status = core::GemmFp8Ref(
      std::span<const float>(a), std::span<const std::byte>(w),
      std::span<const float>(s), std::span<float>(ref), kM, kN, kK);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const FpTolerance tol = FpToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float e = std::abs(device_out[i] - ref[i]);
    const float r =
        std::abs(device_out[i] - ref[i]) / std::max(1.0f, std::abs(ref[i]));
    max_abs = std::max(max_abs, e);
    max_rel = std::max(max_rel, r);
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(max_rel, tol.rel)
      << "backend " << backend->Name() << " max_rel " << max_rel;
}

// Device: MXFP4 GEMM matches the host reference (2 x 32 outputs).
TEST(BackendTest, GemmMxFp4DeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(2025);
  constexpr std::size_t kM = 2;
  constexpr std::size_t kN = 32;
  constexpr std::size_t kK = 64;
  constexpr float kGrid[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                              2.0f, 3.0f, 4.0f, 6.0f};
  std::vector<float> a(kM * kK);
  std::vector<std::byte> w(kN * kK / 2, std::byte{0});
  std::vector<std::byte> s(kN * kK / 32);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  for (std::size_t j = 0; j < kN; ++j) {
    for (std::size_t b = 0; b < kK / 32; ++b) {
      s[j * (kK / 32) + b] = static_cast<std::byte>(125 + (j + b) % 5);
      for (std::size_t l = 0; l < 32; ++l) {
        const std::size_t t = b * 32 + l;
        const std::uint8_t nib = core::Fp32ToF4E2M1Nibble(
            kGrid[(j + t) % 8] * (t % 3 == 0 ? -1.0f : 1.0f));
        const std::size_t at = (j * kK + t) / 2;
        std::uint8_t packed = static_cast<std::uint8_t>(w[at]);
        if (t % 2 == 0) {
          packed = static_cast<std::uint8_t>((packed & 0xF0) | nib);
        } else {
          packed = static_cast<std::uint8_t>((packed & 0x0F) |
                                             (nib << 4));
        }
        w[at] = static_cast<std::byte>(packed);
      }
    }
  }
  // The device kernel takes the packed weight: blob bytes then scales.
  std::vector<std::byte> packed = w;
  packed.insert(packed.end(), s.begin(), s.end());
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(packed.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && w_buf.has_value() && c_buf.has_value());
  const auto upload = [&backend](auto& buf, const auto& data,
                                 std::size_t elem) {
    return backend->CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data.data()),
                   data.size() * elem));
  };
  ASSERT_TRUE(upload(a_buf, a, 4).has_value());
  ASSERT_TRUE(upload(w_buf, packed, 1).has_value());

  auto kernel = backend->LoadKernel("gemm_mxfp4", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = tessera::core::detail::GemmGridFor(**kernel, kM, kN);
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();

  std::vector<std::byte> readback(kM * kN * 4);
  auto download =
      backend->CopyD2H(**c_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(kM * kN);
  auto ref_status = core::GemmMxFp4Ref(
      std::span<const float>(a), std::span<const std::byte>(w),
      std::span<const std::byte>(s), std::span<float>(ref), kM, kN, kK);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const FpTolerance tol = FpToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float e = std::abs(device_out[i] - ref[i]);
    const float r =
        std::abs(device_out[i] - ref[i]) / std::max(1.0f, std::abs(ref[i]));
    max_abs = std::max(max_abs, e);
    max_rel = std::max(max_rel, r);
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(max_rel, tol.rel)
      << "backend " << backend->Name() << " max_rel " << max_rel;
}

// Device: the small-batch rows GEMV (one warp per weight column, all m rows)
// matches the host reference for m > 1, on both backends.
TEST(BackendTest, GemmMxFp4RowsDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(2026);
  constexpr std::size_t kM = 3;
  constexpr std::size_t kN = 24;
  constexpr std::size_t kK = 64;
  constexpr float kGrid[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                              2.0f, 3.0f, 4.0f, 6.0f};
  std::vector<float> a(kM * kK);
  std::vector<std::byte> w(kN * kK / 2, std::byte{0});
  std::vector<std::byte> s(kN * kK / 32);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  for (std::size_t j = 0; j < kN; ++j) {
    for (std::size_t b = 0; b < kK / 32; ++b) {
      s[j * (kK / 32) + b] = static_cast<std::byte>(125 + (j + b) % 5);
      for (std::size_t l = 0; l < 32; ++l) {
        const std::size_t t = b * 32 + l;
        const std::uint8_t nib = core::Fp32ToF4E2M1Nibble(
            kGrid[(j + t) % 8] * (t % 3 == 0 ? -1.0f : 1.0f));
        const std::size_t at = (j * kK + t) / 2;
        std::uint8_t packed = static_cast<std::uint8_t>(w[at]);
        if (t % 2 == 0) {
          packed = static_cast<std::uint8_t>((packed & 0xF0) | nib);
        } else {
          packed = static_cast<std::uint8_t>((packed & 0x0F) | (nib << 4));
        }
        w[at] = static_cast<std::byte>(packed);
      }
    }
  }
  std::vector<std::byte> packed = w;
  packed.insert(packed.end(), s.begin(), s.end());
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(packed.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && w_buf.has_value() && c_buf.has_value());
  ASSERT_TRUE(backend->CopyH2D(
                  **a_buf,
                  std::span<const std::byte>(
                      reinterpret_cast<const std::byte*>(a.data()),
                      a.size() * 4))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(**w_buf, std::span<const std::byte>(packed))
                  .has_value());
  auto kernel = backend->LoadKernel("gemm_mxfp4_rows", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = tessera::core::detail::GemmGridFor(**kernel, kM, kN);
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kN * 4);
  ASSERT_TRUE(backend->CopyD2H(**c_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(kM * kN);
  ASSERT_TRUE(core::GemmMxFp4Ref(std::span<const float>(a),
                                 std::span<const std::byte>(w),
                                 std::span<const std::byte>(s),
                                 std::span<float>(ref), kM, kN, kK)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const FpTolerance tol = FpToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float e = std::abs(got[i] - ref[i]);
    max_abs = std::max(max_abs, e);
    max_rel = std::max(max_rel, e / std::max(1.0f, std::abs(ref[i])));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(max_rel, tol.rel)
      << "backend " << backend->Name() << " max_rel " << max_rel;
}

// Host: the fp GEMM references reject malformed shapes.

// Host: the MXFP4 -> fp8 fold is exact. With A = 1.0 (e4m3 0x38) and As = 1,
// the W4A8 reference must equal the plain MXFP4 dequant sum per row.
TEST(BackendTest, PermuteMxFp4ToWmmaMatchesFragmentOrder) {
  constexpr std::size_t kN = 32, kK = 64;  // nt = 2, ks = 4
  std::vector<std::byte> packed(kN * kK / 2);
  for (std::size_t i = 0; i < packed.size(); ++i) {
    packed[i] = static_cast<std::byte>(i & 0xFFu);
  }
  std::vector<std::byte> out(packed.size());
  ASSERT_TRUE(core::PermuteMxFp4ToWmma(packed, out, kN, kK).has_value());
  const std::size_t nt = kN / 16, ks = kK / 16;
  // Fragment layout is [nt][ks][half][row][4]: the byte at
  // (((t*ks+s)*2+h)*16+r)*4+b equals the checkpoint byte
  // (((t*16+r)*ks+s)*2+h)*4+b.
  for (std::size_t t = 0; t < nt; ++t) {
    for (std::size_t s = 0; s < ks; ++s) {
      for (std::size_t h = 0; h < 2; ++h) {
        for (std::size_t r = 0; r < 16; ++r) {
          for (std::size_t b = 0; b < 4; ++b) {
            const std::size_t in = (((t * 16 + r) * ks + s) * 2 + h) * 4 + b;
            const std::size_t o = (((t * ks + s) * 2 + h) * 16 + r) * 4 + b;
            ASSERT_EQ(out[o], packed[in]);
          }
        }
      }
    }
  }
  // A permutation: the byte multiset is unchanged.
  std::vector<std::byte> sorted_out = out;
  std::vector<std::byte> sorted_in = packed;
  std::sort(sorted_out.begin(), sorted_out.end());
  std::sort(sorted_in.begin(), sorted_in.end());
  EXPECT_EQ(sorted_out, sorted_in);
  // Non-multiples of 16 and short buffers are rejected.
  EXPECT_FALSE(core::PermuteMxFp4ToWmma(packed, out, kN + 1, kK).has_value());
  std::vector<std::byte> small(3);
  EXPECT_FALSE(core::PermuteMxFp4ToWmma(small, small, 16, 16).has_value());
}

TEST(BackendTest, FoldMxFp4ToFp8MatchesDequant) {
  std::mt19937 rng(4242);
  constexpr std::size_t kRows = 3;
  constexpr std::size_t kCols = 64;
  constexpr std::size_t kBlocks = kCols / 32;
  constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  std::vector<std::byte> w(kRows * kCols / 2);
  for (auto& b : w) {
    b = static_cast<std::byte>(rng() & 0xFF);
  }
  std::vector<std::byte> scales(kRows * kBlocks);
  for (std::size_t r = 0; r < kRows; ++r) {
    for (std::size_t b = 0; b < kBlocks; ++b) {
      scales[r * kBlocks + b] = static_cast<std::byte>(120 + ((r + b) % 8));
    }
  }
  std::vector<std::byte> wfp8(kRows * kCols);
  std::vector<std::byte> wref(kRows);
  auto folded = core::FoldMxFp4ToFp8(w, scales, wfp8, wref, kRows, kCols);
  ASSERT_TRUE(folded.has_value()) << tessera::ToString(folded.error());
  std::vector<std::byte> a(kCols, std::byte{0x38});
  std::vector<float> as(1, 1.0f);
  std::vector<float> c(kRows);
  auto gemm = core::GemmMxFp4Fp8Ref(a, as, wfp8, wref, c, 1, kRows, kCols);
  ASSERT_TRUE(gemm.has_value()) << tessera::ToString(gemm.error());
  for (std::size_t r = 0; r < kRows; ++r) {
    float ref = 0.0f;
    for (std::size_t col = 0; col < kCols; ++col) {
      const std::uint8_t packed =
          static_cast<std::uint8_t>(w[(r * kCols + col) / 2]);
      const std::uint8_t nib = (col % 2 == 0) ? (packed & 0xF) : (packed >> 4);
      const float scale = std::exp2(
          static_cast<float>(static_cast<std::uint8_t>(
                                 scales[r * kBlocks + col / 32])) -
          127.0f);
      ref += kMag[nib & 7] * ((nib & 8) ? -1.0f : 1.0f) * scale;
    }
    EXPECT_NEAR(c[r], ref, 1e-3f * std::max(1.0f, std::abs(ref)))
        << "row " << r;
  }
}

// Device: the fp8 tensor-core MXFP4 GEMM matches the W4A8 host reference at
// the small-m verify shape.
TEST(BackendTest, GemmMxFp4WmmaDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(99);
  constexpr std::size_t kM = 3;
  constexpr std::size_t kN = 16;
  constexpr std::size_t kK = 512;
  constexpr std::size_t kBlocks = kK / 32;
  std::vector<std::byte> w(kN * kK / 2);
  for (auto& b : w) {
    b = static_cast<std::byte>(rng() & 0xFF);
  }
  std::vector<std::byte> scales(kN * kBlocks);
  for (std::size_t r = 0; r < kN; ++r) {
    for (std::size_t b = 0; b < kBlocks; ++b) {
      scales[r * kBlocks + b] = static_cast<std::byte>(125 + ((r + b) % 5));
    }
  }
  std::vector<std::byte> wfp8(kN * kK);
  std::vector<std::byte> wref(kN);
  ASSERT_TRUE(core::FoldMxFp4ToFp8(w, scales, wfp8, wref, kN, kK).has_value());
  // The device kernel folds MXFP4 in-kernel, so upload the packed nibbles
  // followed by the E8M0 scales.
  std::vector<std::byte> wpacked = w;
  wpacked.insert(wpacked.end(), scales.begin(), scales.end());
  std::vector<std::byte> a(kM * kK);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<std::byte>(0x30 + (rng() % 8));  // 0.5..1.875
  }
  std::vector<float> as(kM);
  for (auto& v : as) {
    v = 1.0f + 0.25f * static_cast<float>(rng() % 3);
  }
  auto a_buf = backend->AllocateBuffer(a.size(), MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(wpacked.size(), MemoryKind::Device);
  auto as_buf = backend->AllocateBuffer(kM * 4, MemoryKind::Device);
  auto wref_buf = backend->AllocateBuffer(kN, MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && w_buf.has_value() && as_buf.has_value() &&
              wref_buf.has_value() && c_buf.has_value());
  ASSERT_TRUE(backend->CopyH2D(**a_buf, std::span<const std::byte>(a))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(**w_buf, std::span<const std::byte>(wpacked))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(**as_buf,
                               std::span<const std::byte>(
                                   reinterpret_cast<const std::byte*>(as.data()),
                                   as.size() * 4))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(**wref_buf, std::span<const std::byte>(wref))
                  .has_value());
  auto kernel = backend->LoadKernel("gemm_mxfp4_wmma", {});
  if (!kernel) {
    GTEST_SKIP() << "backend has no fp8 tensor-core GEMM";
  }
  std::vector<float> ref(kM * kN);
  ASSERT_TRUE(core::GemmMxFp4Fp8Ref(std::span<const std::byte>(a),
                                    std::span<const float>(as),
                                    std::span<const std::byte>(wfp8),
                                    std::span<const std::byte>(wref),
                                    std::span<float>(ref), kM, kN, kK)
                  .has_value());
  const FpTolerance tol = FpToleranceFor(backend->Name());
  auto reduce = backend->LoadKernel("gemm_mxfp4_wmma_reduce", {});
  if (!reduce) GTEST_SKIP() << "backend has no split-K reduce";
  auto part_buf = backend->AllocateBuffer(kM * kN * 4 * 4, MemoryKind::Device);
  ASSERT_TRUE(part_buf.has_value());
  for (std::uint64_t split : {1ull, 4ull}) {
    tessera::KernelLaunch launch;
    launch.grid_x = static_cast<std::uint32_t>((kN + 63) / 64);
    launch.grid_y = static_cast<std::uint32_t>(split);
    launch.block_x = 128;
    tessera::Buffer* dst = split > 1 ? part_buf->get() : c_buf->get();
    launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*as_buf).get(),
                      (*wref_buf).get(), dst};
    launch.scalars = {kM, kN, kK, split, 0};
    ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
    if (split > 1) {
      tessera::KernelLaunch rl;
      rl.grid_x = static_cast<std::uint32_t>((kM * kN + 255) / 256);
      rl.block_x = 256;
      rl.buffers = {part_buf->get(), (*c_buf).get()};
      rl.scalars = {kM * kN, split};
      ASSERT_TRUE(backend->LaunchKernel(**reduce, rl).has_value());
    }
    backend->Synchronize();
    std::vector<std::byte> readback(kM * kN * 4);
    ASSERT_TRUE(backend->CopyD2H(**c_buf, readback.data(), readback.size())
                    .has_value());
    const auto* got = reinterpret_cast<const float*>(readback.data());
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      const float e = std::abs(got[i] - ref[i]);
      max_abs = std::max(max_abs, e);
      max_rel = std::max(max_rel, e / std::max(1.0f, std::abs(ref[i])));
    }
    EXPECT_LE(max_abs, tol.abs)
        << "backend " << backend->Name() << " split " << split << " max_abs "
        << max_abs;
    EXPECT_LE(max_rel, tol.rel)
        << "backend " << backend->Name() << " split " << split << " max_rel "
        << max_rel;
  }
  // The fragment-order (WPERM) weight read must give the same result.
  std::vector<std::byte> wperm_nibbles(kN * kK / 2);
  ASSERT_TRUE(core::PermuteMxFp4ToWmma(w, wperm_nibbles, kN, kK).has_value());
  std::vector<std::byte> wperm_packed = wperm_nibbles;
  wperm_packed.insert(wperm_packed.end(), scales.begin(), scales.end());
  auto wperm_buf =
      backend->AllocateBuffer(wperm_packed.size(), MemoryKind::Device);
  ASSERT_TRUE(wperm_buf.has_value());
  ASSERT_TRUE(backend->CopyH2D(**wperm_buf,
                               std::span<const std::byte>(wperm_packed))
                  .has_value());
  tessera::KernelLaunch wl;
  wl.grid_x = static_cast<std::uint32_t>((kN + 63) / 64);
  wl.block_x = 128;
  wl.buffers = {(*a_buf).get(), (*wperm_buf).get(), (*as_buf).get(),
                (*wref_buf).get(), (*c_buf).get()};
  wl.scalars = {kM, kN, kK, 1ull, 1ull};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, wl).has_value());
  backend->Synchronize();
  std::vector<std::byte> wperm_readback(kM * kN * 4);
  ASSERT_TRUE(backend->CopyD2H(**c_buf, wperm_readback.data(),
                               wperm_readback.size())
                  .has_value());
  const auto* wperm_got =
      reinterpret_cast<const float*>(wperm_readback.data());
  float wperm_abs = 0.0f, wperm_rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float e = std::abs(wperm_got[i] - ref[i]);
    wperm_abs = std::max(wperm_abs, e);
    wperm_rel = std::max(wperm_rel, e / std::max(1.0f, std::abs(ref[i])));
  }
  EXPECT_LE(wperm_abs, tol.abs) << "wperm max_abs " << wperm_abs;
  EXPECT_LE(wperm_rel, tol.rel) << "wperm max_rel " << wperm_rel;
}

TEST(BackendTest, GemmBf16WmmaDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(1234);
  constexpr std::size_t kM = 3, kN = 16, kK = 512;
  std::vector<float> a(kM * kK);
  for (auto& v : a) v = -1.0f + 2.0f * static_cast<float>(rng() % 1000) / 1000.0f;
  std::vector<unsigned short> w(kN * kK);
  for (auto& v : w) {
    const float f = -1.0f + 2.0f * static_cast<float>(rng() % 1000) / 1000.0f;
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    v = static_cast<unsigned short>(bits >> 16);
  }
  // The kernel rounds A to bf16 while staging, so round it here too.
  auto round_bf16 = [](float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint32_t rounding = (bits >> 16) & 1u;
    bits += 0x7FFFu + rounding;
    return static_cast<unsigned short>(bits >> 16);
  };
  std::vector<float> a_bf16(kM * kK);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::uint32_t high =
        static_cast<std::uint32_t>(round_bf16(a[i])) << 16;
    std::memcpy(&a_bf16[i], &high, sizeof(high));
  }
  std::vector<float> ref(kM * kN);
  ASSERT_TRUE(core::GemmBf16Ref(
                  std::span<const float>(a_bf16),
                  std::span<const std::byte>(
                      reinterpret_cast<const std::byte*>(w.data()),
                      w.size() * 2),
                  std::span<float>(ref), kM, kN, kK)
                  .has_value());
  auto kernel = backend->LoadKernel("gemm_bf16_wmma", {});
  if (!kernel) GTEST_SKIP() << "backend has no bf16 tensor-core GEMM";
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size() * 2, MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf && w_buf && c_buf);
  ASSERT_TRUE(backend->CopyH2D(**a_buf, std::span<const std::byte>(
                                        reinterpret_cast<const std::byte*>(a.data()),
                                        a.size() * 4)).has_value());
  ASSERT_TRUE(backend->CopyH2D(**w_buf, std::span<const std::byte>(
                                        reinterpret_cast<const std::byte*>(w.data()),
                                        w.size() * 2)).has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((kN + 63) / 64);
  launch.block_x = 128;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kN * 4);
  ASSERT_TRUE(backend->CopyD2H(**c_buf, readback.data(), readback.size()).has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const FpTolerance tol = FpToleranceFor(backend->Name());
  float max_abs = 0.0f, max_rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float e = std::abs(got[i] - ref[i]);
    max_abs = std::max(max_abs, e);
    max_rel = std::max(max_rel, e / std::max(1.0f, std::abs(ref[i])));
  }
  EXPECT_LE(max_abs, tol.abs) << "max_abs " << max_abs;
  EXPECT_LE(max_rel, tol.rel) << "max_rel " << max_rel;
}


// Cold (weights far larger than the last-level cache) comparison of the fp8
// tensor-core GEMM against the m=1 GEMV. Both stream the same weight bytes,
// so the ratio isolates the weight-read access pattern from occupancy.


// Random weight bytes with small exact scales patched into every
// block (d=2^-7, dmin=0.5 where present). Small magnitudes keep fma
// order differences inside the tolerance; the reference and the
// kernel decode the same bytes, so the test pins their agreement,
// and exact formulas have unit tests below.
static std::vector<std::byte> RandomBlocks(std::mt19937& rng,
                                           std::size_t rows,
                                           std::size_t bytes_per_row,
                                           std::size_t d_off) {
  std::vector<std::byte> w(rows * bytes_per_row);
  for (auto& b : w) {
    b = static_cast<std::byte>(rng() & 0xFF);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    w[r * bytes_per_row + d_off] = std::byte{0};
    w[r * bytes_per_row + d_off + 1] = std::byte{0x20};
  }
  return w;
}

// Device: Q5_K GEMM matches the host reference (2 x 8 outputs).
TEST(BackendTest, GemmQ5KDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(55);
  constexpr std::size_t kM = 2;
  constexpr std::size_t kN = 8;
  constexpr std::size_t kK = 256;
  std::vector<float> a(kM * kK);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  std::vector<std::byte> w = RandomBlocks(rng, kN, 176, 0);
  for (std::size_t r = 0; r < kN; ++r) {  // small dmin, keeps the min term
    w[r * 176 + 2] = std::byte{0};
    w[r * 176 + 3] = std::byte{0x20};
  }
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && w_buf.has_value() && c_buf.has_value());
  auto up_a = backend->CopyH2D(**a_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(a.data()), a.size() * 4));
  auto up_w = backend->CopyH2D(**w_buf, std::span<const std::byte>(w));
  ASSERT_TRUE(up_a.has_value() && up_w.has_value());
  auto kernel = backend->LoadKernel("gemm_q5k", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kM * kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kN * 4);
  auto download =
      backend->CopyD2H(**c_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(kM * kN);
  auto ref_status = core::GemmQ5KRef(
      std::span<const float>(a), std::span<const std::byte>(w),
      std::span<float>(ref), kM, kN, kK);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const FpTolerance tol = FpToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(device_out[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the tiled batched Q5_K GEMM (weight blocks dequantized once per
// 8-row tile) matches the reference, including a non-multiple-of-8 row
// count.

// Device: the tiled batched Q6_K GEMM (weight blocks dequantized once per
// 8-row tile) matches the reference, including a non-multiple-of-8 row
// count.

// Device: Q6_K GEMM matches the host reference (2 x 8 outputs).
TEST(BackendTest, GemmQ6KDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(56);
  constexpr std::size_t kM = 2;
  constexpr std::size_t kN = 8;
  constexpr std::size_t kK = 256;
  std::vector<float> a(kM * kK);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  std::vector<std::byte> w = RandomBlocks(rng, kN, 210, 208);
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && w_buf.has_value() && c_buf.has_value());
  auto up_a = backend->CopyH2D(**a_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(a.data()), a.size() * 4));
  auto up_w = backend->CopyH2D(**w_buf, std::span<const std::byte>(w));
  ASSERT_TRUE(up_a.has_value() && up_w.has_value());
  auto kernel = backend->LoadKernel("gemm_q6k", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kM * kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kN * 4);
  auto download =
      backend->CopyD2H(**c_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(kM * kN);
  auto ref_status = core::GemmQ6KRef(
      std::span<const float>(a), std::span<const std::byte>(w),
      std::span<float>(ref), kM, kN, kK);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const FpTolerance tol = FpToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(device_out[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: Q3_K GEMM matches the host reference (2 x 8 outputs).
TEST(BackendTest, GemmQ3KDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(57);
  constexpr std::size_t kM = 2;
  constexpr std::size_t kN = 8;
  constexpr std::size_t kK = 256;
  std::vector<float> a(kM * kK);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  std::vector<std::byte> w = RandomBlocks(rng, kN, 110, 108);
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && w_buf.has_value() && c_buf.has_value());
  auto up_a = backend->CopyH2D(**a_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(a.data()), a.size() * 4));
  auto up_w = backend->CopyH2D(**w_buf, std::span<const std::byte>(w));
  ASSERT_TRUE(up_a.has_value() && up_w.has_value());
  auto kernel = backend->LoadKernel("gemm_q3k", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kM * kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kN * 4);
  auto download =
      backend->CopyD2H(**c_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(kM * kN);
  auto ref_status = core::GemmQ3KRef(
      std::span<const float>(a), std::span<const std::byte>(w),
      std::span<float>(ref), kM, kN, kK);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const FpTolerance tol = FpToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(device_out[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: IQ4_NL GEMM matches the host reference (2 x 8 outputs).
TEST(BackendTest, GemmIq4NlDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(58);
  constexpr std::size_t kM = 2;
  constexpr std::size_t kN = 8;
  constexpr std::size_t kK = 32;
  std::vector<float> a(kM * kK);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  std::vector<std::byte> w = RandomBlocks(rng, kN, 18, 0);
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && w_buf.has_value() && c_buf.has_value());
  auto up_a = backend->CopyH2D(**a_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(a.data()), a.size() * 4));
  auto up_w = backend->CopyH2D(**w_buf, std::span<const std::byte>(w));
  ASSERT_TRUE(up_a.has_value() && up_w.has_value());
  auto kernel = backend->LoadKernel("gemm_iq4nl", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kM * kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kN * 4);
  auto download =
      backend->CopyD2H(**c_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(kM * kN);
  auto ref_status = core::GemmIq4NlRef(
      std::span<const float>(a), std::span<const std::byte>(w),
      std::span<float>(ref), kM, kN, kK);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const FpTolerance tol = FpToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(device_out[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: IQ4_XS GEMM matches the host reference (2 x 8 outputs).
TEST(BackendTest, GemmIq4XsDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(59);
  constexpr std::size_t kM = 2;
  constexpr std::size_t kN = 8;
  constexpr std::size_t kK = 256;
  std::vector<float> a(kM * kK);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  std::vector<std::byte> w = RandomBlocks(rng, kN, 136, 0);
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && w_buf.has_value() && c_buf.has_value());
  auto up_a = backend->CopyH2D(**a_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(a.data()), a.size() * 4));
  auto up_w = backend->CopyH2D(**w_buf, std::span<const std::byte>(w));
  ASSERT_TRUE(up_a.has_value() && up_w.has_value());
  auto kernel = backend->LoadKernel("gemm_iq4xs", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kM * kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kN * 4);
  auto download =
      backend->CopyD2H(**c_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(kM * kN);
  auto ref_status = core::GemmIq4XsRef(
      std::span<const float>(a), std::span<const std::byte>(w),
      std::span<float>(ref), kM, kN, kK);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const FpTolerance tol = FpToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(device_out[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the tiled batched IQ4_XS GEMM (weight blocks dequantized once
// per 8-row tile) matches the reference, including a non-multiple-of-8 row
// count.

// Device: IQ3_S GEMM matches the host reference (2 x 8 outputs).

// Device: Q8_0 GEMM matches the host reference (2 x 8 outputs, one
// 32-element block per row).
TEST(BackendTest, GemmQ80DeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(61);
  constexpr std::size_t kM = 2;
  constexpr std::size_t kN = 8;
  constexpr std::size_t kK = 32;
  std::vector<float> a(kM * kK);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  std::vector<std::byte> w = RandomBlocks(rng, kN, 34, 0);
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && w_buf.has_value() && c_buf.has_value());
  auto up_a = backend->CopyH2D(**a_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(a.data()), a.size() * 4));
  auto up_w = backend->CopyH2D(**w_buf, std::span<const std::byte>(w));
  ASSERT_TRUE(up_a.has_value() && up_w.has_value());
  auto kernel = backend->LoadKernel("gemm_q80", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kM * kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kN * 4);
  auto download =
      backend->CopyD2H(**c_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(kM * kN);
  auto ref_status = core::GemmQ80Ref(
      std::span<const float>(a), std::span<const std::byte>(w),
      std::span<float>(ref), kM, kN, kK);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const FpTolerance tol = FpToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(device_out[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the K-quant and IQ GEMM contracts reject bad shapes.

// Device: the fp GEMM contracts reject bad shapes.

// Host: the K-quant and IQ dequants hit hand-computed values.
TEST(BackendTest, QuantDequantsExact) {
  // Q5_K: d=1, s[0]=1 (sub-block 0 scale 1), qs[0]=0x12, qh[0]=0x01.
  std::vector<std::byte> q5(176, std::byte{0});
  q5[1] = std::byte{0x3C};
  q5[4] = std::byte{1};
  q5[48] = std::byte{0x12};
  q5[16] = std::byte{0x01};
  std::vector<float> out5(256);
  core::DequantizeQ5K(std::span<const std::byte>(q5), std::span<float>(out5));
  EXPECT_FLOAT_EQ(out5[0], 18.0f);
  EXPECT_FLOAT_EQ(out5[32], 0.0f);
  // Q6_K: d=1, scales[0]=1, ql[0]=0x12, qh[0]=0.
  std::vector<std::byte> q6(210, std::byte{0});
  q6[209] = std::byte{0x3C};
  q6[192] = std::byte{1};
  q6[0] = std::byte{0x12};
  std::vector<float> out6(256);
  core::DequantizeQ6K(std::span<const std::byte>(q6), std::span<float>(out6));
  EXPECT_FLOAT_EQ(out6[0], -30.0f);
  // Q3_K: d=1, everything else zero (dl=-32, qv-hb=-4).
  std::vector<std::byte> q3(110, std::byte{0});
  q3[109] = std::byte{0x3C};
  std::vector<float> out3(256);
  core::DequantizeQ3K(std::span<const std::byte>(q3), std::span<float>(out3));
  EXPECT_FLOAT_EQ(out3[0], 128.0f);
  EXPECT_FLOAT_EQ(out3[255], 128.0f);
  // Q3_K scales 0x21 everywhere unpack to 17 (dl=-15); qs[32]=0xFF
  // with full hmask gives qv=3, hb=0 on the first 16 elements.
  for (std::size_t i = 96; i < 108; ++i) {
    q3[i] = std::byte{0x21};
  }
  for (std::size_t i = 32; i < 48; ++i) {
    q3[i] = std::byte{0xFF};
  }
  for (std::size_t i = 0; i < 16; ++i) {
    q3[i] = std::byte{0xFF};
  }
  core::DequantizeQ3K(std::span<const std::byte>(q3), std::span<float>(out3));
  for (std::size_t i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(out3[i], -45.0f) << "element " << i;
  }
  // IQ4_NL: d=1, qs[0]=0x21 (codebook -104 and -83).
  std::vector<std::byte> nl(18, std::byte{0});
  nl[1] = std::byte{0x3C};
  nl[2] = std::byte{0x21};
  std::vector<float> outnl(32);
  core::DequantizeIQ4NL(std::span<const std::byte>(nl),
                        std::span<float>(outnl));
  EXPECT_FLOAT_EQ(outnl[0], -104.0f);
  EXPECT_FLOAT_EQ(outnl[16], -83.0f);
  // IQ4_XS: d=1, scales_l[0]=0x21 (ls=1), qs[0]=0x21.
  std::vector<std::byte> xs(136, std::byte{0});
  xs[1] = std::byte{0x3C};
  xs[4] = std::byte{0x21};
  xs[8] = std::byte{0x21};
  std::vector<float> outxs(256);
  core::DequantizeIQ4XS(std::span<const std::byte>(xs),
                        std::span<float>(outxs));
  EXPECT_FLOAT_EQ(outxs[0], 3224.0f);
  EXPECT_FLOAT_EQ(outxs[16], 2573.0f);
  // IQ3_S: d=1, scales[0]=0 (db=1), zero quants/signs (grid 1s).
  std::vector<std::byte> s3(110, std::byte{0});
  s3[1] = std::byte{0x3C};
  std::vector<float> outs3(256);
  core::DequantizeIQ3S(std::span<const std::byte>(s3),
                       std::span<float>(outs3));
  EXPECT_FLOAT_EQ(outs3[0], 1.0f);
  EXPECT_FLOAT_EQ(outs3[7], 1.0f);
  // Q8_0: d=1, q[0]=-128, q[1]=127.
  std::vector<std::byte> q8(34, std::byte{0});
  q8[1] = std::byte{0x3C};
  q8[2] = std::byte{0x80};
  q8[3] = std::byte{0x7F};
  std::vector<float> out8(32);
  core::DequantizeQ80(std::span<const std::byte>(q8),
                      std::span<float>(out8));
  EXPECT_FLOAT_EQ(out8[0], -128.0f);
  EXPECT_FLOAT_EQ(out8[1], 127.0f);
}

// Host: the canonical block-layout table and whole-buffer dequantizer
// agree with the per-block dequants (relied on by embedding gather).

// Host: the K-quant and IQ GEMM references reject malformed shapes.







TEST(BackendTest, LoadAndLaunchFill) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  constexpr std::size_t kElements = 512;
  auto buffer = backend->AllocateBuffer(kElements * 4, MemoryKind::Device);
  ASSERT_TRUE(buffer.has_value()) << tessera::ToString(buffer.error());
  auto kernel = backend->LoadKernel("fill", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());

  tessera::KernelLaunch launch;
  launch.grid_x = (kElements + 255) / 256;
  launch.block_x = 256;
  launch.buffers.push_back((*buffer).get());
  launch.scalars = {0x11223344ULL, static_cast<std::uint64_t>(kElements)};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();

  std::vector<std::byte> readback(kElements * 4);
  auto download =
      backend->CopyD2H(**buffer, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  const auto* values =
      reinterpret_cast<const std::uint32_t*>(readback.data());
  for (std::size_t i = 0; i < kElements; ++i) {
    EXPECT_EQ(values[i], 0x11223344u);
  }
}



// The "gemm_q4k" contract (backend.hpp) checked on the device: k must
// be a positive multiple of 256; n and m must be positive.

// Host: the quantized output of QuantizeQ4K round-trips through
// DequantizeQ4K within the documented per-sub-block error bound
// (step/2 + dm/2, plus codebook-range clamping at q = 15).
TEST(BackendTest, Q4KRoundTripWithinBound) {
  std::mt19937 rng(42);
  for (int trial = 0; trial < 3; ++trial) {
    std::vector<float> values(kQ4KBlockElements);
    for (auto& v : values) {
      v = DrawValue(rng);
    }
    if (trial == 1) {
      std::fill(values.begin(), values.end(), 5.0f);
    }
    if (trial == 2) {
      std::fill(values.begin(), values.end(), 0.0f);
    }
    std::vector<std::byte> block(core::kQ4KBlockBytes);
    core::QuantizeQ4K(std::span<const float>(values), block.data());
    if (trial == 2) {
      // All-zero input must produce an all-zero block.
      for (auto b : block) {
        EXPECT_EQ(b, std::byte{0});
      }
      continue;
    }
    const float d = core::Fp16ToFloat(BlockU16(block.data(), 0));
    const float dm = core::Fp16ToFloat(BlockU16(block.data(), 2));
    std::vector<float> out(kQ4KBlockElements);
    core::DequantizeQ4K(std::span<const std::byte>(block),
                       std::span<float>(out));
    for (std::size_t j = 0; j < 8; ++j) {
      float submin = values[32 * j];
      float submax = values[32 * j];
      for (std::size_t i = 1; i < 32; ++i) {
        submin = std::min(submin, values[32 * j + i]);
        submax = std::max(submax, values[32 * j + i]);
      }
      const float eff_min = std::min(submin, 0.0f);
      std::uint8_t sc, mn;
      TestGetScaleMin(j, block.data() + core::kQ4KScaleBytes - 8, &sc, &mn);
      const float bound = d * static_cast<float>(sc) / 2.0f + dm / 2.0f +
                         std::max(0.0f, (submax - eff_min) -
                                          15.0f * d * static_cast<float>(sc));
      for (std::size_t i = 0; i < 32; ++i) {
        const float err = std::abs(out[32 * j + i] - values[32 * j + i]);
        EXPECT_LE(err, bound + 1.0e-6f)
            << "trial " << trial << " element " << (32 * j + i);
      }
      (void)mn;
    }
  }
}

// Host: a hand-built block dequantizes to the exact expected values
// (d = 1, dmin = 0, all scales 1/min 0, nibbles 0x42).
TEST(BackendTest, Q4KDequantKnownBlock) {
  std::vector<std::byte> block(core::kQ4KBlockBytes, std::byte{0});
  auto put = [&block](std::size_t i, std::uint8_t v) {
    block[i] = static_cast<std::byte>(v);
  };
  // d = 1.0 (fp16 0x3C00, little-endian), dmin = 0.
  put(1, 0x3C);
  // s[0..3] = 1, s[4..7] = 0, s[8..11] = 1 (sc = 1, min = 0 for all
  // eight sub-blocks).
  for (std::size_t i = 0; i < 4; ++i) {
    put(4 + i, 1);
    put(12 + i, 1);
  }
  // qs = 0x42: low nibble 2, high nibble 4.
  for (std::size_t i = 0; i < 128; ++i) {
    put(16 + i, 0x42);
  }
  std::vector<float> out(kQ4KBlockElements);
  core::DequantizeQ4K(std::span<const std::byte>(block),
                     std::span<float>(out));
  for (std::size_t i = 0; i < kQ4KBlockElements; ++i) {
    const float expected = ((i / 32) % 2) == 0 ? 2.0f : 4.0f;
    EXPECT_EQ(out[i], expected) << "element " << i;
  }
  // d = 2.0, dmin = 0.5, s[4..11] = 0x40: sub-blocks 0..3 are
  // scale 1/min 0 (value = 2 * q); sub-blocks 4..7 are scale 0 and
  // min 20 (the high bits of both halves: 16 | 4), so the value is
  // -0.5 * 20.
  put(1, 0x40);
  put(3, 0x38);
  for (std::size_t i = 4; i < 12; ++i) {
    put(4 + i, 0x40);
  }
  core::DequantizeQ4K(std::span<const std::byte>(block),
                     std::span<float>(out));
  for (std::size_t i = 0; i < 128; ++i) {
    const float expected = ((i / 32) % 2) == 0 ? 4.0f : 8.0f;
    EXPECT_EQ(out[i], expected) << "element " << i;
  }
  for (std::size_t i = 128; i < kQ4KBlockElements; ++i) {
    EXPECT_EQ(out[i], -10.0f) << "element " << i;
  }
}

// Host: GemmQ4KRef must bit-match a dense dequantize + fmaf dot (same
// op order and same fma), so any later drift is a real regression.
TEST(BackendTest, GemmQ4KRefMatchesDense) {
  std::mt19937 rng(7);
  constexpr std::size_t kM = 4;
  constexpr std::size_t kN = 8;
  constexpr std::size_t kK = 512;
  std::vector<float> a(kM * kK);
  std::vector<float> w_raw(kN * kK);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  for (auto& v : w_raw) {
    v = DrawValue(rng);
  }
  std::vector<std::byte> w = QuantizeRows(w_raw, kN, kK);
  std::vector<float> ref(kM * kN);
  auto status = core::GemmQ4KRef(std::span<const float>(a),
                                std::span<const std::byte>(w),
                                std::span<float>(ref), kM, kN, kK);
  ASSERT_TRUE(status.has_value()) << tessera::ToString(status.error());
  std::vector<float> dense(kM * kN);
  std::vector<float> row(kK);
  for (std::size_t j = 0; j < kN; ++j) {
    for (std::size_t b = 0; b < kK / kQ4KBlockElements; ++b) {
      core::DequantizeQ4K(
          std::span<const std::byte>(
              w.data() + (j * (kK / kQ4KBlockElements) + b) *
                  core::kQ4KBlockBytes,
              core::kQ4KBlockBytes),
          std::span<float>(row.data() + b * kQ4KBlockElements,
                          kQ4KBlockElements));
    }
    for (std::size_t i = 0; i < kM; ++i) {
      float acc = 0.0f;
      for (std::size_t t = 0; t < kK; ++t) {
        acc = std::fma(a[i * kK + t], row[t], acc);
      }
      dense[i * kN + j] = acc;
    }
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < kM * kN; ++i) {
    max_diff = std::max(max_diff, std::abs(ref[i] - dense[i]));
  }
  EXPECT_EQ(max_diff, 0.0f) << "reference drifted from the dense oracle";
}

// Host: GemmQ4KRef rejects malformed shapes at the boundary.

// Device: the GEMM output matches the host reference within the
// per-backend tolerance (m x n = 4 x 96 outputs, two workgroups).
TEST(BackendTest, GemmQ4KDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(1234);
  constexpr std::size_t kM = 4;
  constexpr std::size_t kN = 96;
  constexpr std::size_t kK = 512;
  std::vector<float> a(kM * kK);
  std::vector<float> w_raw(kN * kK);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  for (auto& v : w_raw) {
    v = DrawValue(rng);
  }
  std::vector<std::byte> w = QuantizeRows(w_raw, kN, kK);
  const std::size_t w_size =
      kN * (kK / kQ4KBlockElements) * core::kQ4KBlockBytes;

  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value()) << tessera::ToString(a_buf.error());
  auto w_buf = backend->AllocateBuffer(w_size, MemoryKind::Device);
  ASSERT_TRUE(w_buf.has_value()) << tessera::ToString(w_buf.error());
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(c_buf.has_value()) << tessera::ToString(c_buf.error());

  auto upload_a = backend->CopyH2D(**a_buf, std::span<const std::byte>(
                                               reinterpret_cast<const std::byte*>(a.data()),
                                               a.size() * 4));
  ASSERT_TRUE(upload_a.has_value()) << tessera::ToString(upload_a.error());
  auto upload_w = backend->CopyH2D(**w_buf, std::span<const std::byte>(w));
  ASSERT_TRUE(upload_w.has_value()) << tessera::ToString(upload_w.error());

  auto kernel = backend->LoadKernel("gemm_q4k", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kM * kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();

  std::vector<std::byte> readback(kM * kN * 4);
  auto download = backend->CopyD2H(**c_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  const auto* device_out =
      reinterpret_cast<const float*>(readback.data());
  std::vector<float> ref(kM * kN);
  auto ref_status = core::GemmQ4KRef(std::span<const float>(a),
                                    std::span<const std::byte>(w),
                                    std::span<float>(ref), kM, kN, kK);
  ASSERT_TRUE(ref_status.has_value()) << tessera::ToString(ref_status.error());

  const GemmTolerance tol = ToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (std::size_t i = 0; i < kM * kN; ++i) {
    const float e = std::abs(device_out[i] - ref[i]);
    const float r =
        std::abs(device_out[i] - ref[i]) /
        std::max(1.0f, std::abs(ref[i]));
    max_abs = std::max(max_abs, e);
    max_rel = std::max(max_rel, r);
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(max_rel, tol.rel)
      << "backend " << backend->Name() << " max_rel " << max_rel;
}

// Device: the tiled batched Q4_K GEMM (weight blocks dequantized once per
// 8-row tile) matches the reference, including a row count that is not a
// multiple of the tile. Skips on backends without the tiled kernel.

// Device: the tiled batched MXFP4 GEMM (weight blocks dequantized once per
// 8-row tile) matches the reference, including a row count that is not a
// multiple of the tile.
TEST(BackendTest, GemmMxFp4BatchedMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto kernel = backend->LoadKernel("gemm_mxfp4_batched", {});
  if (!kernel && kernel.error() == StatusCode::UnsupportedFeature) {
    GTEST_SKIP() << "no tiled MXFP4 kernel on " << backend->Name();
  }
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  std::mt19937 rng(2026);
  constexpr std::size_t kM = 10;
  constexpr std::size_t kN = 32;
  constexpr std::size_t kK = 64;
  constexpr float kGrid[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                              2.0f, 3.0f, 4.0f, 6.0f};
  std::vector<float> a(kM * kK);
  std::vector<std::byte> w(kN * kK / 2, std::byte{0});
  std::vector<std::byte> s(kN * kK / 32);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  for (std::size_t j = 0; j < kN; ++j) {
    for (std::size_t b = 0; b < kK / 32; ++b) {
      s[j * (kK / 32) + b] = static_cast<std::byte>(125 + (j + b) % 5);
      for (std::size_t l = 0; l < 32; ++l) {
        const std::size_t t = b * 32 + l;
        const std::uint8_t nib = core::Fp32ToF4E2M1Nibble(
            kGrid[(j + t) % 8] * (t % 3 == 0 ? -1.0f : 1.0f));
        const std::size_t at = (j * kK + t) / 2;
        std::uint8_t packed = static_cast<std::uint8_t>(w[at]);
        if (t % 2 == 0) {
          packed = static_cast<std::uint8_t>((packed & 0xF0) | nib);
        } else {
          packed = static_cast<std::uint8_t>((packed & 0x0F) | (nib << 4));
        }
        w[at] = static_cast<std::byte>(packed);
      }
    }
  }
  std::vector<std::byte> packed = w;
  packed.insert(packed.end(), s.begin(), s.end());

  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value());
  auto w_buf = backend->AllocateBuffer(packed.size(), MemoryKind::Device);
  ASSERT_TRUE(w_buf.has_value());
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(c_buf.has_value());
  ASSERT_TRUE(backend->CopyH2D(**a_buf, std::span<const std::byte>(
                                           reinterpret_cast<const std::byte*>(a.data()),
                                           a.size() * 4))
                  .has_value());
  ASSERT_TRUE(
      backend->CopyH2D(**w_buf, std::span<const std::byte>(packed)).has_value());

  auto launch = core::detail::ProjectTiledDevice(
      *backend, **kernel, **a_buf, **w_buf, **c_buf, kM, kN, kK);
  ASSERT_TRUE(launch.has_value()) << tessera::ToString(launch.error());
  backend->Synchronize();

  std::vector<std::byte> readback(kM * kN * 4);
  ASSERT_TRUE(
      backend->CopyD2H(**c_buf, readback.data(), readback.size()).has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  std::vector<float> ref(kM * kN);
  auto ref_status = core::GemmMxFp4Ref(
      std::span<const float>(a), std::span<const std::byte>(w),
      std::span<const std::byte>(s), std::span<float>(ref), kM, kN, kK);
  ASSERT_TRUE(ref_status.has_value());
  const FpTolerance tol = FpToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
    max_rel = std::max(max_rel,
                       std::abs(got[i] - ref[i]) / std::max(1.0f, std::abs(ref[i])));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(max_rel, tol.rel)
      << "backend " << backend->Name() << " max_rel " << max_rel;
}

// Device: the coalesced Q4_K GEMV (one workgroup per output, shared-memory
// reduction, grid selected by ProjectDevice) matches the reference. This is
// the Q4_K kernel the model selects.
TEST(BackendTest, GemmQ4KRowMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto kernel = backend->LoadKernel("gemm_q4k_row", {});
  if (!kernel && kernel.error() == StatusCode::UnsupportedFeature) {
    GTEST_SKIP() << "no coalesced Q4_K kernel on " << backend->Name();
  }
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  std::mt19937 rng(1357);
  constexpr std::size_t kM = 4;
  constexpr std::size_t kN = 96;
  constexpr std::size_t kK = 512;
  std::vector<float> a(kM * kK);
  std::vector<float> w_raw(kN * kK);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  for (auto& v : w_raw) {
    v = DrawValue(rng);
  }
  std::vector<std::byte> w = QuantizeRows(w_raw, kN, kK);

  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value());
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  ASSERT_TRUE(w_buf.has_value());
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(c_buf.has_value());
  ASSERT_TRUE(backend->CopyH2D(**a_buf, std::span<const std::byte>(
                                           reinterpret_cast<const std::byte*>(a.data()),
                                           a.size() * 4))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(**w_buf, std::span<const std::byte>(w)).has_value());

  auto launch = core::detail::ProjectDevice(
      *backend, **kernel, **a_buf, **w_buf, **c_buf, kM, kN, kK);
  ASSERT_TRUE(launch.has_value()) << tessera::ToString(launch.error());
  backend->Synchronize();

  std::vector<std::byte> readback(kM * kN * 4);
  ASSERT_TRUE(
      backend->CopyD2H(**c_buf, readback.data(), readback.size()).has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  std::vector<float> ref(kM * kN);
  auto ref_status = core::GemmQ4KRef(std::span<const float>(a),
                                     std::span<const std::byte>(w),
                                     std::span<float>(ref), kM, kN, kK);
  ASSERT_TRUE(ref_status.has_value());
  const GemmTolerance tol = ToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
    rel = std::max(rel,
                   std::abs(got[i] - ref[i]) / std::max(1.0f, std::abs(ref[i])));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(rel, tol.rel) << "backend " << backend->Name() << " max_rel " << rel;
}

// Per backend attention tolerance (same shape as the GEMM table).
struct AttentionTolerance {
  float abs = 0.0f;
  float rel = 0.0f;
};
AttentionTolerance AttentionToleranceFor(std::string_view backend) {
  if (backend == "vulkan") {
    return {2.0e-4f, 2.0e-4f};
  }
  if (backend == "rocm") {
    return {2.0e-4f, 2.0e-4f};
  }
  return {1.0e-3f, 1.0e-3f};
}

// Device: in-place RoPE matches the host reference (8 rows, 4 heads,
// head dim 32 with the first 16 rotated, positions start at 5).
TEST(BackendTest, RopeDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(99);
  constexpr std::size_t kRows = 8;
  constexpr std::size_t kHeads = 4;
  constexpr std::size_t kDim = 32;
  constexpr std::size_t kRopeDim = 16;
  constexpr std::uint64_t kPosBase = 5;
  constexpr float kTheta = 10000.0f;
  std::vector<float> io(kRows * kHeads * kDim);
  for (auto& v : io) {
    v = DrawValue(rng);
  }
  auto buffer = backend->AllocateBuffer(io.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(buffer.has_value()) << tessera::ToString(buffer.error());
  auto upload = backend->CopyH2D(**buffer, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(io.data()), io.size() * 4));
  ASSERT_TRUE(upload.has_value()) << tessera::ToString(upload.error());

  std::uint32_t theta_bits = 0;
  static_assert(sizeof(theta_bits) == sizeof(kTheta));
  std::memcpy(&theta_bits, &kTheta, sizeof(theta_bits));
  auto kernel = backend->LoadKernel("rope", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kRows * kHeads * (kRopeDim / 2) + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*buffer).get()};
  launch.scalars = {kRows, kHeads, kDim, kRopeDim, kPosBase, theta_bits};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();

  std::vector<std::byte> readback(io.size() * 4);
  auto download =
      backend->CopyD2H(**buffer, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref = io;
  auto ref_status = core::RopeRef(std::span<float>(ref), kRows, kHeads,
                                  kDim, kRopeDim, kPosBase, kTheta);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float e = std::abs(device_out[i] - ref[i]);
    const float r =
        std::abs(device_out[i] - ref[i]) / std::max(1.0f, std::abs(ref[i]));
    max_abs = std::max(max_abs, e);
    max_rel = std::max(max_rel, r);
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(max_rel, tol.rel)
      << "backend " << backend->Name() << " max_rel " << max_rel;
}

// Device: causal GQA attention matches the host reference (2 queries
// over 4 keys, 4 heads in 2 kv groups, query positions start at 3 so
// the second row clamps to the last key).
TEST(BackendTest, AttentionDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(1234);
  constexpr std::size_t kM = 2;
  constexpr std::size_t kN = 4;
  constexpr std::size_t kHeads = 4;
  constexpr std::size_t kKvHeads = 2;
  constexpr std::size_t kDim = 16;
  constexpr std::uint64_t kQBase = 3;
  std::vector<float> q(kM * kHeads * kDim);
  std::vector<float> k(kN * kKvHeads * kDim);
  std::vector<float> v(kN * kKvHeads * kDim);
  for (auto& x : q) {
    x = DrawValue(rng);
  }
  for (auto& x : k) {
    x = DrawValue(rng);
  }
  for (auto& x : v) {
    x = DrawValue(rng);
  }
  auto q_buf = backend->AllocateBuffer(q.size() * 4, MemoryKind::Device);
  auto k_buf = backend->AllocateBuffer(k.size() * 4, MemoryKind::Device);
  auto v_buf = backend->AllocateBuffer(v.size() * 4, MemoryKind::Device);
  auto out_buf =
      backend->AllocateBuffer(kM * kHeads * kDim * 4, MemoryKind::Device);
  ASSERT_TRUE(q_buf.has_value() && k_buf.has_value() &&
              v_buf.has_value() && out_buf.has_value());
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data.data()),
                   data.size() * 4));
  };
  ASSERT_TRUE(upload(q_buf, q).has_value());
  ASSERT_TRUE(upload(k_buf, k).has_value());
  ASSERT_TRUE(upload(v_buf, v).has_value());

  auto kernel = backend->LoadKernel("attention", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = kM * kHeads;
  launch.block_x = 256;
  launch.buffers = {(*q_buf).get(), (*k_buf).get(), (*v_buf).get(),
                    (*out_buf).get()};
  launch.scalars = {kM, kN, kHeads, kKvHeads, kDim, kQBase, 0, 0, 1};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();

  std::vector<std::byte> readback(kM * kHeads * kDim * 4);
  auto download =
      backend->CopyD2H(**out_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(kM * kHeads * kDim);
  auto ref_status = core::AttentionRef(
      std::span<const float>(q), std::span<const float>(k),
      std::span<const float>(v), std::span<float>(ref), kM, kN, kHeads,
      kKvHeads, kDim, kQBase);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float e = std::abs(device_out[i] - ref[i]);
    const float r =
        std::abs(device_out[i] - ref[i]) / std::max(1.0f, std::abs(ref[i]));
    max_abs = std::max(max_abs, e);
    max_rel = std::max(max_rel, r);
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(max_rel, tol.rel)
      << "backend " << backend->Name() << " max_rel " << max_rel;
}

// Host: the attention references reject malformed shapes at the
// boundary (zero dims, rope range wider than the head, size gaps,
// heads that do not split into kv groups).

// Device: the rope and attention contracts (backend.hpp) reject bad
// shapes before anything reaches the device.

// Device: row-wise RMS normalization matches the host reference
// (4 rows of 64, epsilon 1e-6, unit weights plus a scaled row).
TEST(BackendTest, RmsnormDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(41);
  constexpr std::size_t kRows = 4;
  constexpr std::size_t kCols = 64;
  constexpr float kEps = 1e-6f;
  std::vector<float> x(kRows * kCols);
  std::vector<float> w(kCols, 1.0f);
  for (auto& v : x) {
    v = DrawValue(rng);
  }
  w[0] = 0.5f;
  auto x_buf = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size() * 4, MemoryKind::Device);
  auto y_buf = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(x_buf.has_value() && w_buf.has_value() && y_buf.has_value());
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data.data()),
                   data.size() * 4));
  };
  ASSERT_TRUE(upload(x_buf, x).has_value());
  ASSERT_TRUE(upload(w_buf, w).has_value());

  std::uint32_t eps_bits = 0;
  static_assert(sizeof(eps_bits) == sizeof(kEps));
  std::memcpy(&eps_bits, &kEps, sizeof(eps_bits));
  auto kernel = backend->LoadKernel("rmsnorm", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  // One workgroup per row; the row sum reduces across the group.
  launch.grid_x = kRows;
  launch.block_x = 256;
  launch.buffers = {(*x_buf).get(), (*w_buf).get(), (*y_buf).get()};
  launch.scalars = {kRows, kCols, eps_bits};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();

  std::vector<std::byte> readback(x.size() * 4);
  auto download =
      backend->CopyD2H(**y_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(x.size());
  auto ref_status = core::RmsNormRef(std::span<const float>(x),
                                     std::span<const float>(w),
                                     std::span<float>(ref), kRows, kCols, kEps);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float e = std::abs(device_out[i] - ref[i]);
    const float r =
        std::abs(device_out[i] - ref[i]) / std::max(1.0f, std::abs(ref[i]));
    max_abs = std::max(max_abs, e);
    max_rel = std::max(max_rel, r);
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(max_rel, tol.rel)
      << "backend " << backend->Name() << " max_rel " << max_rel;
}

// Device: sigmoid-gated scale matches the host reference (256
// elements drawn from [-1, 1], gate included).
TEST(BackendTest, SigmoidGateDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(42);
  constexpr std::size_t kN = 256;
  std::vector<float> a(kN);
  std::vector<float> g(kN);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  for (auto& v : g) {
    v = DrawValue(rng);
  }
  auto a_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  auto g_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  auto o_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && g_buf.has_value() && o_buf.has_value());
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data.data()),
                   data.size() * 4));
  };
  ASSERT_TRUE(upload(a_buf, a).has_value());
  ASSERT_TRUE(upload(g_buf, g).has_value());

  auto kernel = backend->LoadKernel("sigmoid_gate", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*g_buf).get(), (*o_buf).get()};
  launch.scalars = {kN};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();

  std::vector<std::byte> readback(kN * 4);
  auto download =
      backend->CopyD2H(**o_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(kN);
  auto ref_status = core::SigmoidGateRef(std::span<const float>(a),
                                         std::span<const float>(g),
                                         std::span<float>(ref), kN);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float e = std::abs(device_out[i] - ref[i]);
    const float r =
        std::abs(device_out[i] - ref[i]) / std::max(1.0f, std::abs(ref[i]));
    max_abs = std::max(max_abs, e);
    max_rel = std::max(max_rel, r);
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(max_rel, tol.rel)
      << "backend " << backend->Name() << " max_rel " << max_rel;
}

// Host: the norm references reject malformed shapes at the boundary
// (zero dims, negative epsilon, size gaps).

// Device: the rmsnorm and sigmoid_gate contracts (backend.hpp) reject
// bad shapes before anything reaches the device.

// Device: causal depthwise conv1d matches the host reference
// (8 channels, 16 steps, width 4, so early steps see a short tail).
TEST(BackendTest, Conv1dDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(43);
  constexpr std::size_t kChannels = 8;
  constexpr std::size_t kLength = 16;
  constexpr std::size_t kWidth = 4;
  std::vector<float> x(kChannels * kLength);
  std::vector<float> w(kChannels * kWidth);
  for (auto& v : x) {
    v = DrawValue(rng);
  }
  for (auto& v : w) {
    v = DrawValue(rng);
  }
  auto x_buf = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size() * 4, MemoryKind::Device);
  auto y_buf = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(x_buf.has_value() && w_buf.has_value() && y_buf.has_value());
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data.data()),
                   data.size() * 4));
  };
  ASSERT_TRUE(upload(x_buf, x).has_value());
  ASSERT_TRUE(upload(w_buf, w).has_value());

  auto kernel = backend->LoadKernel("conv1d", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kChannels * kLength + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*x_buf).get(), (*w_buf).get(), (*y_buf).get()};
  launch.scalars = {kChannels, kLength, kWidth};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();

  std::vector<std::byte> readback(x.size() * 4);
  auto download =
      backend->CopyD2H(**y_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(x.size());
  auto ref_status = core::ConvRef(std::span<const float>(x),
                                  std::span<const float>(w),
                                  std::span<float>(ref), kChannels, kLength,
                                  kWidth);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float e = std::abs(device_out[i] - ref[i]);
    const float r =
        std::abs(device_out[i] - ref[i]) / std::max(1.0f, std::abs(ref[i]));
    max_abs = std::max(max_abs, e);
    max_rel = std::max(max_rel, r);
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(max_rel, tol.rel)
      << "backend " << backend->Name() << " max_rel " << max_rel;
}

// Host and contract: the conv1d reference and the launch contract
// reject malformed shapes at the boundary (zero dims, size gaps).

// Device: three gated delta steps match the host reference, outputs
// and the carried state alike (dk 8, dv 12, decay 0.9, rate 0.5).
TEST(BackendTest, DeltaStepDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(44);
  constexpr std::size_t kDk = 8;
  constexpr std::size_t kDv = 12;
  constexpr std::size_t kSteps = 3;
  constexpr float kAlpha = 0.9f;
  constexpr float kBeta = 0.5f;
  std::vector<float> state(kDk * kDv);
  for (auto& v : state) {
    v = DrawValue(rng);
  }
  auto s_buf = backend->AllocateBuffer(state.size() * 4, MemoryKind::Device);
  auto k_buf = backend->AllocateBuffer(kDk * 4, MemoryKind::Device);
  auto v_buf = backend->AllocateBuffer(kDv * 4, MemoryKind::Device);
  auto q_buf = backend->AllocateBuffer(kDk * 4, MemoryKind::Device);
  auto o_buf = backend->AllocateBuffer(kDv * 4, MemoryKind::Device);
  ASSERT_TRUE(s_buf.has_value() && k_buf.has_value() && v_buf.has_value() &&
              q_buf.has_value() && o_buf.has_value());
  auto upload = [&backend](auto& buf, const std::vector<float>& data) {
    return backend->CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data.data()),
                   data.size() * 4));
  };
  ASSERT_TRUE(upload(s_buf, state).has_value());

  std::uint32_t alpha_bits = 0;
  std::uint32_t beta_bits = 0;
  static_assert(sizeof(alpha_bits) == sizeof(kAlpha));
  std::memcpy(&alpha_bits, &kAlpha, sizeof(alpha_bits));
  std::memcpy(&beta_bits, &kBeta, sizeof(beta_bits));
  auto kernel = backend->LoadKernel("delta_step", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kDv + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*s_buf).get(), (*k_buf).get(), (*v_buf).get(),
                    (*q_buf).get(), (*o_buf).get()};
  launch.scalars = {kDk, kDv, alpha_bits, beta_bits};
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  std::vector<float> ref_state = state;
  for (std::size_t step = 0; step < kSteps; ++step) {
    std::vector<float> k(kDk);
    std::vector<float> v(kDv);
    std::vector<float> q(kDk);
    for (auto& x : k) {
      x = DrawValue(rng);
    }
    for (auto& x : v) {
      x = DrawValue(rng);
    }
    for (auto& x : q) {
      x = DrawValue(rng);
    }
    ASSERT_TRUE(upload(k_buf, k).has_value());
    ASSERT_TRUE(upload(v_buf, v).has_value());
    ASSERT_TRUE(upload(q_buf, q).has_value());
    auto result = backend->LaunchKernel(**kernel, launch);
    ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
    backend->Synchronize();

    std::vector<float> ref_out(kDv);
    auto ref_status = core::DeltaStepRef(
        std::span<float>(ref_state), std::span<const float>(k),
        std::span<const float>(v), std::span<const float>(q),
        std::span<float>(ref_out), kDk, kDv, kAlpha, kBeta);
    ASSERT_TRUE(ref_status.has_value())
        << tessera::ToString(ref_status.error());
    std::vector<std::byte> readback(kDv * 4);
    auto download =
        backend->CopyD2H(**o_buf, readback.data(), readback.size());
    ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
    const auto* device_out = reinterpret_cast<const float*>(readback.data());
    for (std::size_t i = 0; i < ref_out.size(); ++i) {
      EXPECT_NEAR(device_out[i], ref_out[i], tol.abs)
          << "backend " << backend->Name() << " step " << step;
    }
  }
  std::vector<std::byte> state_back(state.size() * 4);
  auto download_state =
      backend->CopyD2H(**s_buf, state_back.data(), state_back.size());
  ASSERT_TRUE(download_state.has_value())
      << tessera::ToString(download_state.error());
  const auto* device_state = reinterpret_cast<const float*>(state_back.data());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref_state.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(device_state[i] - ref_state[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Host and contract: the delta-step reference and the launch contract
// reject malformed shapes at the boundary (zero dims, size gaps).

// Device: multimodal RoPE matches the host reference (4 rows, 2 heads,
// head dim 32 with 16 rotated over 3+3+2 section pairs, distinct
// temporal/height/width ids per row).
TEST(BackendTest, MropeDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(45);
  constexpr std::size_t kRows = 4;
  constexpr std::size_t kHeads = 2;
  constexpr std::size_t kDim = 32;
  constexpr std::size_t kRopeDim = 16;
  constexpr std::size_t kSecT = 3;
  constexpr std::size_t kSecH = 3;
  constexpr std::size_t kSecW = 2;
  constexpr float kTheta = 10000.0f;
  std::vector<float> io(kRows * kHeads * kDim);
  for (auto& v : io) {
    v = DrawValue(rng);
  }
  std::vector<std::uint64_t> pos(kRows * 3);
  for (std::size_t r = 0; r < kRows; ++r) {
    pos[r * 3] = r;
    pos[r * 3 + 1] = 2 * r;
    pos[r * 3 + 2] = 3 * r;
  }
  auto io_buf = backend->AllocateBuffer(io.size() * 4, MemoryKind::Device);
  auto pos_buf = backend->AllocateBuffer(pos.size() * 8, MemoryKind::Device);
  ASSERT_TRUE(io_buf.has_value() && pos_buf.has_value());
  auto upload_io = backend->CopyH2D(**io_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(io.data()), io.size() * 4));
  ASSERT_TRUE(upload_io.has_value()) << tessera::ToString(upload_io.error());
  auto upload_pos = backend->CopyH2D(**pos_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(pos.data()), pos.size() * 8));
  ASSERT_TRUE(upload_pos.has_value())
      << tessera::ToString(upload_pos.error());

  std::uint32_t theta_bits = 0;
  static_assert(sizeof(theta_bits) == sizeof(kTheta));
  std::memcpy(&theta_bits, &kTheta, sizeof(theta_bits));
  auto kernel = backend->LoadKernel("mrope", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kRows * kHeads * (kRopeDim / 2) + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*io_buf).get(), (*pos_buf).get()};
  launch.scalars = {kRows, kHeads, kDim, kRopeDim, theta_bits, kSecT, kSecH,
                    kSecW};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();

  std::vector<std::byte> readback(io.size() * 4);
  auto download =
      backend->CopyD2H(**io_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref = io;
  auto ref_status = core::MropeRef(
      std::span<float>(ref), std::span<const std::uint64_t>(pos), kRows,
      kHeads, kDim, kRopeDim, kSecT, kSecH, kSecW, kTheta);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float e = std::abs(device_out[i] - ref[i]);
    const float r =
        std::abs(device_out[i] - ref[i]) / std::max(1.0f, std::abs(ref[i]));
    max_abs = std::max(max_abs, e);
    max_rel = std::max(max_rel, r);
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(max_rel, tol.rel)
      << "backend " << backend->Name() << " max_rel " << max_rel;
}

// Device: mRoPE with identical temporal/height/width ids matches the
// plain RoPE host reference (text-only rows reduce to 1D RoPE).
TEST(BackendTest, MropeMatchesRopeForText) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(46);
  constexpr std::size_t kRows = 4;
  constexpr std::size_t kHeads = 2;
  constexpr std::size_t kDim = 32;
  constexpr std::size_t kRopeDim = 16;
  constexpr std::uint64_t kPosBase = 5;
  constexpr float kTheta = 10000.0f;
  std::vector<float> io(kRows * kHeads * kDim);
  for (auto& v : io) {
    v = DrawValue(rng);
  }
  std::vector<std::uint64_t> pos(kRows * 3);
  for (std::size_t r = 0; r < kRows; ++r) {
    pos[r * 3] = kPosBase + r;
    pos[r * 3 + 1] = kPosBase + r;
    pos[r * 3 + 2] = kPosBase + r;
  }
  auto io_buf = backend->AllocateBuffer(io.size() * 4, MemoryKind::Device);
  auto pos_buf = backend->AllocateBuffer(pos.size() * 8, MemoryKind::Device);
  ASSERT_TRUE(io_buf.has_value() && pos_buf.has_value());
  auto upload_io = backend->CopyH2D(**io_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(io.data()), io.size() * 4));
  ASSERT_TRUE(upload_io.has_value()) << tessera::ToString(upload_io.error());
  auto upload_pos = backend->CopyH2D(**pos_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(pos.data()), pos.size() * 8));
  ASSERT_TRUE(upload_pos.has_value())
      << tessera::ToString(upload_pos.error());

  std::uint32_t theta_bits = 0;
  static_assert(sizeof(theta_bits) == sizeof(kTheta));
  std::memcpy(&theta_bits, &kTheta, sizeof(theta_bits));
  auto kernel = backend->LoadKernel("mrope", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kRows * kHeads * (kRopeDim / 2) + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*io_buf).get(), (*pos_buf).get()};
  launch.scalars = {kRows, kHeads, kDim, kRopeDim, theta_bits, 3, 3, 2};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();

  std::vector<std::byte> readback(io.size() * 4);
  auto download =
      backend->CopyD2H(**io_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref = io;
  auto ref_status = core::RopeRef(std::span<float>(ref), kRows, kHeads, kDim,
                                  kRopeDim, kPosBase, kTheta);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(device_out[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Host and contract: the mRoPE reference and the launch contract reject
// malformed shapes at the boundary (section overflow, size gaps).

// Device: the fused gated-attention split matches the host reference
// (3 heads of head dim 8, distinct values per segment).
TEST(BackendTest, QGateSplitDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(47);
  constexpr std::size_t kHeads = 3;
  constexpr std::size_t kHeadDim = 8;
  constexpr std::size_t kRows = 2;
  std::vector<float> fused(kRows * kHeads * 2 * kHeadDim);
  for (auto& v : fused) {
    v = DrawValue(rng);
  }
  auto fused_buf =
      backend->AllocateBuffer(fused.size() * 4, MemoryKind::Device);
  auto q_buf =
      backend->AllocateBuffer(kRows * kHeads * kHeadDim * 4, MemoryKind::Device);
  auto gate_buf =
      backend->AllocateBuffer(kRows * kHeads * kHeadDim * 4, MemoryKind::Device);
  ASSERT_TRUE(fused_buf.has_value() && q_buf.has_value() &&
              gate_buf.has_value());
  auto upload = backend->CopyH2D(**fused_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(fused.data()), fused.size() * 4));
  ASSERT_TRUE(upload.has_value()) << tessera::ToString(upload.error());

  auto kernel = backend->LoadKernel("qgate_split", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kRows * kHeads * kHeadDim + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*fused_buf).get(), (*q_buf).get(), (*gate_buf).get()};
  launch.scalars = {kHeads, kHeadDim, kRows};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();

  std::vector<float> ref_q(kRows * kHeads * kHeadDim);
  std::vector<float> ref_gate(kRows * kHeads * kHeadDim);
  auto ref_status = core::QGateSplitRef(
      std::span<const float>(fused), std::span<float>(ref_q),
      std::span<float>(ref_gate), kHeads, kHeadDim, kRows);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  std::vector<std::byte> readback_q(kRows * kHeads * kHeadDim * 4);
  std::vector<std::byte> readback_gate(kRows * kHeads * kHeadDim * 4);
  ASSERT_TRUE(backend->CopyD2H(**q_buf, readback_q.data(), readback_q.size())
                  .has_value());
  ASSERT_TRUE(backend->CopyD2H(**gate_buf, readback_gate.data(),
                               readback_gate.size())
                  .has_value());
  const auto* device_q = reinterpret_cast<const float*>(readback_q.data());
  const auto* device_gate =
      reinterpret_cast<const float*>(readback_gate.data());
  for (std::size_t i = 0; i < ref_q.size(); ++i) {
    EXPECT_FLOAT_EQ(device_q[i], ref_q[i]);
    EXPECT_FLOAT_EQ(device_gate[i], ref_gate[i]);
  }
}

// Host and contract: the qgate-split reference and the launch contract
// reject malformed shapes at the boundary (zero dims, size gaps).

// Device: row-wise L2 normalization matches the host reference (4 rows
// of 16, epsilon 1e-6).
TEST(BackendTest, L2NormDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(48);
  constexpr std::size_t kRows = 4;
  constexpr std::size_t kCols = 16;
  constexpr float kEps = 1e-6f;
  std::vector<float> x(kRows * kCols);
  for (auto& v : x) {
    v = DrawValue(rng);
  }
  auto x_buf = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  auto y_buf = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(x_buf.has_value() && y_buf.has_value());
  auto upload = backend->CopyH2D(**x_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(x.data()), x.size() * 4));
  ASSERT_TRUE(upload.has_value()) << tessera::ToString(upload.error());
  constexpr float kScale = 0.37f;
  std::uint32_t eps_bits = 0;
  std::uint32_t scale_bits = 0;
  static_assert(sizeof(eps_bits) == sizeof(kEps));
  std::memcpy(&eps_bits, &kEps, sizeof(eps_bits));
  std::memcpy(&scale_bits, &kScale, sizeof(scale_bits));
  auto kernel = backend->LoadKernel("l2norm", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = kRows;
  launch.block_x = 256;
  launch.buffers = {(*x_buf).get(), (*y_buf).get()};
  launch.scalars = {kRows, kCols, eps_bits, scale_bits};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> readback(x.size() * 4);
  auto download =
      backend->CopyD2H(**y_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(x.size());
  auto ref_status = core::L2NormRef(std::span<const float>(x),
                                    std::span<float>(ref), kRows, kCols, kEps,
                                    kScale);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(device_out[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: gated RMS normalization matches the host reference (4 rows
// of 16, unit weights plus a scale, random gate).
TEST(BackendTest, RmsNormGatedDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(49);
  constexpr std::size_t kRows = 4;
  constexpr std::size_t kCols = 16;
  constexpr float kEps = 1e-6f;
  std::vector<float> x(kRows * kCols);
  std::vector<float> gate(kRows * kCols);
  std::vector<float> w(kCols, 1.0f);
  for (auto& v : x) {
    v = DrawValue(rng);
  }
  for (auto& v : gate) {
    v = DrawValue(rng);
  }
  w[0] = 0.5f;
  auto x_buf = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size() * 4, MemoryKind::Device);
  auto gate_buf = backend->AllocateBuffer(gate.size() * 4, MemoryKind::Device);
  auto y_buf = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(x_buf.has_value() && w_buf.has_value() && gate_buf.has_value() &&
              y_buf.has_value());
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data.data()),
                   data.size() * 4));
  };
  ASSERT_TRUE(upload(x_buf, x).has_value());
  ASSERT_TRUE(upload(w_buf, w).has_value());
  ASSERT_TRUE(upload(gate_buf, gate).has_value());
  std::uint32_t eps_bits = 0;
  static_assert(sizeof(eps_bits) == sizeof(kEps));
  std::memcpy(&eps_bits, &kEps, sizeof(eps_bits));
  auto kernel = backend->LoadKernel("rmsnorm_gated", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = kRows;
  launch.block_x = 256;
  launch.buffers = {(*x_buf).get(), (*w_buf).get(), (*gate_buf).get(),
                    (*y_buf).get()};
  launch.scalars = {kRows, kCols, eps_bits, 0, 0};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> readback(x.size() * 4);
  auto download =
      backend->CopyD2H(**y_buf, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  std::vector<float> ref(x.size());
  auto ref_status = core::RmsNormGatedRef(
      std::span<const float>(x), std::span<const float>(w),
      std::span<const float>(gate), std::span<float>(ref), kRows, kCols, kEps);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  const auto* device_out = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(device_out[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Host and contract: the l2norm and rmsnorm_gated references and the
// launch contracts reject malformed shapes at the boundary.

// Device: elementwise add matches the host reference.
TEST(BackendTest, AddDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(72);
  constexpr std::size_t kN = 256;
  std::vector<float> a(kN);
  std::vector<float> b(kN);
  for (auto& v : a) v = DrawValue(rng);
  for (auto& v : b) v = DrawValue(rng);
  auto a_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  auto b_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  auto o_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf && b_buf && o_buf);
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(**buf, std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size() * 4));
  };
  ASSERT_TRUE(upload(a_buf, a).has_value());
  ASSERT_TRUE(upload(b_buf, b).has_value());
  auto kernel = backend->LoadKernel("add", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*b_buf).get(), (*o_buf).get()};
  launch.scalars = {kN};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kN * 4);
  ASSERT_TRUE(backend->CopyD2H(**o_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(kN);
  auto ref_status = core::AddRef(std::span<const float>(a),
                                 std::span<const float>(b),
                                 std::span<float>(ref), kN);
  ASSERT_TRUE(ref_status.has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(got[i], ref[i]);
  }
  // Contract: two buffers is invalid.
  launch.buffers = {(*a_buf).get(), (*b_buf).get()};
  auto bad = backend->LaunchKernel(**kernel, launch);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
}

// Device: silu(gate) * up matches the host reference.
TEST(BackendTest, SiluMulDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(73);
  constexpr std::size_t kN = 256;
  std::vector<float> g(kN);
  std::vector<float> u(kN);
  for (auto& v : g) v = DrawValue(rng);
  for (auto& v : u) v = DrawValue(rng);
  auto g_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  auto u_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  auto o_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  ASSERT_TRUE(g_buf && u_buf && o_buf);
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(**buf, std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size() * 4));
  };
  ASSERT_TRUE(upload(g_buf, g).has_value());
  ASSERT_TRUE(upload(u_buf, u).has_value());
  auto kernel = backend->LoadKernel("silu_mul", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*g_buf).get(), (*u_buf).get(), (*o_buf).get()};
  launch.scalars = {kN};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kN * 4);
  ASSERT_TRUE(backend->CopyD2H(**o_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(kN);
  auto ref_status = core::SiluMulRef(std::span<const float>(g),
                                     std::span<const float>(u),
                                     std::span<float>(ref), kN);
  ASSERT_TRUE(ref_status.has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(got[i], ref[i], tol.abs);
  }
}

// Device: moe_scale_add computes o = a + f[idx] * b elementwise.
TEST(BackendTest, MoeScaleAddDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(91);
  constexpr std::size_t kN = 200;
  constexpr std::size_t kF = 8;
  constexpr std::size_t kIdx = 3;
  std::vector<float> a(kN);
  std::vector<float> b(kN);
  std::vector<float> f(kF);
  for (auto& v : a) v = DrawValue(rng);
  for (auto& v : b) v = DrawValue(rng);
  for (auto& v : f) v = DrawValue(rng);
  auto a_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  auto b_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  auto f_buf = backend->AllocateBuffer(kF * 4, MemoryKind::Device);
  auto o_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf && b_buf && f_buf && o_buf);
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(**buf, std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size() * 4));
  };
  ASSERT_TRUE(upload(a_buf, a).has_value());
  ASSERT_TRUE(upload(b_buf, b).has_value());
  ASSERT_TRUE(upload(f_buf, f).has_value());
  auto kernel = backend->LoadKernel("moe_scale_add", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>((kN + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*b_buf).get(), (*f_buf).get(),
                    (*o_buf).get()};
  launch.scalars = {kN, kIdx};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kN * 4);
  ASSERT_TRUE(backend->CopyD2H(**o_buf, readback.data(), readback.size())
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(got[i], a[i] + f[kIdx] * b[i], tol.abs);
  }
}

// Device: moe_gate softmaxes the selected router logits and computes the
// shared-expert gate from the normed hidden and the shared-expert weight.
TEST(BackendTest, MoeGateDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(92);
  const std::size_t kTopK = 4;
  const std::size_t kHidden = 64;
  std::vector<float> vals(kTopK);
  for (auto& v : vals) v = DrawValue(rng);
  std::vector<float> x(kHidden);
  std::vector<float> w(kHidden);
  for (auto& v : x) v = DrawValue(rng);
  for (auto& v : w) v = DrawValue(rng);
  auto vals_buf = backend->AllocateBuffer(kTopK * 4, MemoryKind::Device);
  auto x_buf = backend->AllocateBuffer(kHidden * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(kHidden * 4, MemoryKind::Device);
  auto wts_buf = backend->AllocateBuffer(kTopK * 4, MemoryKind::Device);
  auto sig_buf = backend->AllocateBuffer(4, MemoryKind::Device);
  ASSERT_TRUE(vals_buf && x_buf && w_buf && wts_buf && sig_buf);
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(**buf, std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size() * 4));
  };
  ASSERT_TRUE(upload(vals_buf, vals).has_value());
  ASSERT_TRUE(upload(x_buf, x).has_value());
  ASSERT_TRUE(upload(w_buf, w).has_value());
  auto kernel = backend->LoadKernel("moe_gate", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = 1;
  launch.block_x = 256;
  launch.buffers = {(*vals_buf).get(), (*x_buf).get(), (*w_buf).get(),
                    (*wts_buf).get(), (*sig_buf).get()};
  launch.scalars = {kTopK, kHidden};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> wts_read(kTopK * 4);
  std::vector<std::byte> sig_read(4);
  ASSERT_TRUE(backend->CopyD2H(**wts_buf, wts_read.data(), wts_read.size())
                  .has_value());
  ASSERT_TRUE(backend->CopyD2H(**sig_buf, sig_read.data(), sig_read.size())
                  .has_value());
  const float maxv = *std::max_element(vals.begin(), vals.end());
  float denom = 0.0f;
  for (float v : vals) {
    denom += std::exp(v - maxv);
  }
  const auto* wts_got = reinterpret_cast<const float*>(wts_read.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  for (std::size_t p = 0; p < kTopK; ++p) {
    EXPECT_NEAR(wts_got[p], std::exp(vals[p] - maxv) / denom, tol.abs);
  }
  float dot = 0.0f;
  for (std::size_t t = 0; t < kHidden; ++t) {
    dot += x[t] * w[t];
  }
  const auto* sig_got = reinterpret_cast<const float*>(sig_read.data());
  EXPECT_NEAR(sig_got[0], 1.0f / (1.0f + std::exp(-dot)), tol.abs);
}

// Device: the fused MoE expert kernels reproduce a host reference. gate and
// up are Q4_K [inter, hidden] per expert, down is Q4_K [hidden, inter].
TEST(BackendTest, MoeFusedExpertsMatchRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(93);
  const std::size_t kHidden = 256;
  const std::size_t kInter = 256;
  const std::size_t kExperts = 4;
  const std::size_t kTopK = 2;
  const std::vector<std::uint32_t> ids = {1, 3};
  const std::vector<float> wts = {0.6f, 0.4f};
  const std::size_t kGatePerExpert =
      (kInter * kHidden / tessera::kQ4KBlockElements) *
      core::kQ4KBlockBytes;
  const std::size_t kDownPerExpert =
      (kHidden * kInter / tessera::kQ4KBlockElements) *
      core::kQ4KBlockBytes;

  std::vector<float> x(kHidden);
  for (auto& v : x) v = DrawValue(rng);
  std::vector<float> gate_vals(kExperts * kInter * kHidden);
  std::vector<float> up_vals(kExperts * kInter * kHidden);
  std::vector<float> down_vals(kExperts * kHidden * kInter);
  for (auto& v : gate_vals) v = DrawValue(rng);
  for (auto& v : up_vals) v = DrawValue(rng);
  for (auto& v : down_vals) v = DrawValue(rng);
  std::vector<std::byte> gate = QuantizeRows(gate_vals, kExperts * kInter,
                                             kHidden);
  std::vector<std::byte> up = QuantizeRows(up_vals, kExperts * kInter, kHidden);
  std::vector<std::byte> down = QuantizeRows(down_vals, kExperts * kHidden,
                                             kInter);

  auto x_buf = backend->AllocateBuffer(kHidden * 4, MemoryKind::Device);
  auto gate_buf = backend->AllocateBuffer(gate.size(), MemoryKind::Device);
  auto up_buf = backend->AllocateBuffer(up.size(), MemoryKind::Device);
  auto down_buf = backend->AllocateBuffer(down.size(), MemoryKind::Device);
  auto ids_buf = backend->AllocateBuffer(kTopK * 4, MemoryKind::Device);
  auto wts_buf = backend->AllocateBuffer(kTopK * 4, MemoryKind::Device);
  auto phi_buf = backend->AllocateBuffer(kTopK * kInter * 4, MemoryKind::Device);
  auto out_buf = backend->AllocateBuffer(kHidden * 4, MemoryKind::Device);
  ASSERT_TRUE(x_buf && gate_buf && up_buf && down_buf && ids_buf && wts_buf &&
              phi_buf && out_buf);
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(**buf, std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size() * 4));
  };
  ASSERT_TRUE(upload(x_buf, x).has_value());
  ASSERT_TRUE(backend->CopyH2D(**gate_buf, std::span<const std::byte>(gate))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(**up_buf, std::span<const std::byte>(up))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(**down_buf, std::span<const std::byte>(down))
                  .has_value());
  ASSERT_TRUE(upload(ids_buf, ids).has_value());
  ASSERT_TRUE(upload(wts_buf, wts).has_value());

  auto gate_up = backend->LoadKernel("moe_experts_gate_up_q4k", {});
  auto down_k = backend->LoadKernel("moe_experts_down_q4k", {});
  ASSERT_TRUE(gate_up.has_value()) << tessera::ToString(gate_up.error());
  ASSERT_TRUE(down_k.has_value()) << tessera::ToString(down_k.error());
  tessera::KernelLaunch gu;
  gu.grid_x = static_cast<std::uint32_t>((kInter * kTopK + 7) / 8);
  gu.block_x = 256;
  gu.buffers = {(*x_buf).get(), (*gate_buf).get(), (*up_buf).get(),
                (*ids_buf).get(), (*phi_buf).get()};
  gu.scalars = {kHidden, kInter, kTopK, kGatePerExpert, kGatePerExpert};
  ASSERT_TRUE(backend->LaunchKernel(**gate_up, gu).has_value());
  tessera::KernelLaunch dn;
  dn.grid_x = static_cast<std::uint32_t>((kHidden + 7) / 8);
  dn.block_x = 256;
  dn.buffers = {(*phi_buf).get(), (*down_buf).get(), (*ids_buf).get(),
                (*wts_buf).get(), (*out_buf).get()};
  dn.scalars = {kHidden, kInter, kTopK, kDownPerExpert, 0};
  ASSERT_TRUE(backend->LaunchKernel(**down_k, dn).has_value());
  backend->Synchronize();

  std::vector<std::byte> phi_read(kTopK * kInter * 4);
  std::vector<std::byte> out_read(kHidden * 4);
  ASSERT_TRUE(backend->CopyD2H(**phi_buf, phi_read.data(), phi_read.size())
                  .has_value());
  ASSERT_TRUE(backend->CopyD2H(**out_buf, out_read.data(), out_read.size())
                  .has_value());
  const auto* phi_got = reinterpret_cast<const float*>(phi_read.data());
  const auto* out_got = reinterpret_cast<const float*>(out_read.data());

  const auto tol = tessera::testing::ToleranceFor(backend->Name());
  std::vector<float> out_ref(kHidden, 0.0f);
  for (std::size_t p = 0; p < kTopK; ++p) {
    const std::size_t e = ids[p];
    std::span<const std::byte> g_slice(gate.data() + e * kGatePerExpert,
                                       kGatePerExpert);
    std::span<const std::byte> u_slice(up.data() + e * kGatePerExpert,
                                       kGatePerExpert);
    std::span<const std::byte> d_slice(down.data() + e * kDownPerExpert,
                                       kDownPerExpert);
    std::vector<float> gate_ref(kInter);
    std::vector<float> up_ref(kInter);
    ASSERT_TRUE(core::GemmQ4KRef(std::span<const float>(x), g_slice,
                                 std::span<float>(gate_ref), 1, kInter,
                                 kHidden)
                    .has_value());
    ASSERT_TRUE(core::GemmQ4KRef(std::span<const float>(x), u_slice,
                                 std::span<float>(up_ref), 1, kInter, kHidden)
                    .has_value());
    std::vector<float> phi_ref(kInter);
    for (std::size_t i = 0; i < kInter; ++i) {
      const float silu = gate_ref[i] / (1.0f + std::exp(-gate_ref[i]));
      phi_ref[i] = silu * up_ref[i];
      EXPECT_NEAR(phi_got[p * kInter + i], phi_ref[i],
                  tol.abs + tol.rel * std::abs(phi_ref[i]));
    }
    std::vector<float> down_ref(kHidden);
    ASSERT_TRUE(core::GemmQ4KRef(std::span<const float>(phi_ref), d_slice,
                                 std::span<float>(down_ref), 1, kHidden, kInter)
                    .has_value());
    for (std::size_t j = 0; j < kHidden; ++j) {
      out_ref[j] += wts[p] * down_ref[j];
    }
  }
  for (std::size_t j = 0; j < kHidden; ++j) {
    // The device reduce sums in a different order than the host reference.
    EXPECT_NEAR(out_got[j], out_ref[j],
                1e-4f + 1e-4f * std::abs(out_ref[j]));
  }
}

// Device: chained device-to-device kernels (gemm_q4k -> rmsnorm -> add)
// with no host round-trips match the host references.
TEST(BackendTest, DeviceChainMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(80);
  constexpr std::size_t kM = 1;
  constexpr std::size_t kN = 64;
  constexpr std::size_t kK = 256;
  constexpr float kEps = 1e-5f;
  std::vector<float> a(kM * kK);
  std::vector<float> w_raw(kN * kK);
  std::vector<float> norm_w(kN, 1.0f);
  std::vector<float> bias(kN);
  for (auto& v : a) v = DrawValue(rng);
  for (auto& v : w_raw) v = DrawValue(rng);
  for (auto& v : bias) v = DrawValue(rng);
  std::vector<std::byte> w = QuantizeRows(w_raw, kN, kK);
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  auto nw_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  auto y_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  auto b_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  auto z_buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf && w_buf && c_buf && nw_buf && y_buf && b_buf && z_buf);
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(**buf, std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size() * 4));
  };
  ASSERT_TRUE(upload(a_buf, a).has_value());
  ASSERT_TRUE(backend->CopyH2D(**w_buf, std::span<const std::byte>(w))
                  .has_value());
  ASSERT_TRUE(upload(nw_buf, norm_w).has_value());
  ASSERT_TRUE(upload(b_buf, bias).has_value());
  auto gemm = backend->LoadKernel("gemm_q4k", {});
  auto rmsnorm = backend->LoadKernel("rmsnorm", {});
  auto add = backend->LoadKernel("add", {});
  ASSERT_TRUE(gemm && rmsnorm && add);
  ASSERT_TRUE(core::detail::ProjectDevice(*backend, **gemm, **a_buf, **w_buf,
                                          **c_buf, kM, kN, kK)
                  .has_value());
  ASSERT_TRUE(core::detail::RmsNormDevice(*backend, **rmsnorm, **c_buf,
                                          **nw_buf, **y_buf, kM, kN, kEps)
                  .has_value());
  ASSERT_TRUE(core::detail::AddDevice(*backend, **add, **y_buf, **b_buf,
                                      **z_buf, kN)
                  .has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(kN * 4);
  ASSERT_TRUE(backend->CopyD2H(**z_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> c_ref(kM * kN);
  ASSERT_TRUE(core::GemmQ4KRef(std::span<const float>(a),
                               std::span<const std::byte>(w),
                               std::span<float>(c_ref), kM, kN, kK)
                  .has_value());
  std::vector<float> y_ref(kN);
  ASSERT_TRUE(core::RmsNormRef(std::span<const float>(c_ref),
                               std::span<const float>(norm_w),
                               std::span<float>(y_ref), kM, kN, kEps)
                  .has_value());
  std::vector<float> z_ref(kN);
  ASSERT_TRUE(core::AddRef(std::span<const float>(y_ref),
                           std::span<const float>(bias),
                           std::span<float>(z_ref), kN)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const GemmTolerance tol = ToleranceFor(backend->Name());
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(got[i], z_ref[i], tol.abs);
  }
}

// Device: device-to-device copies move byte ranges between buffers.

// Device: the DeltaStepDevice helper updates the state in place and
// matches the host reference.

// Device: head repeat expands key heads to the value heads.
TEST(BackendTest, RepeatHeadsDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(82);
  constexpr std::size_t kNumV = 4;
  constexpr std::size_t kHeadDim = 8;
  constexpr std::size_t kFactor = 2;
  const std::size_t in_n = (kNumV / kFactor) * kHeadDim;
  std::vector<float> in(in_n);
  for (auto& v : in) v = DrawValue(rng);
  auto in_buf = backend->AllocateBuffer(in_n * 4, MemoryKind::Device);
  auto out_buf =
      backend->AllocateBuffer(kNumV * kHeadDim * 4, MemoryKind::Device);
  ASSERT_TRUE(in_buf && out_buf);
  ASSERT_TRUE(backend->CopyH2D(**in_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(in.data()), in.size() * 4))
                  .has_value());
  auto kernel = backend->LoadKernel("repeat_heads", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kNumV * kHeadDim + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*in_buf).get(), (*out_buf).get()};
  launch.scalars = {kNumV, kHeadDim, kFactor, 1};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kNumV * kHeadDim * 4);
  ASSERT_TRUE(backend->CopyD2H(**out_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(kNumV * kHeadDim);
  ASSERT_TRUE(core::RepeatHeadsRef(std::span<const float>(in),
                                   std::span<float>(ref), kNumV, kHeadDim,
                                   kFactor)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  for (std::size_t i = 0; i < ref.size(); ++i) {
    EXPECT_FLOAT_EQ(got[i], ref[i]);
  }
  // Contract: heads not divisible by factor is rejected.
  launch.scalars = {kNumV, kHeadDim, 3, 1};
  auto bad = backend->LaunchKernel(**kernel, launch);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
}

// Device: the gated-delta gates match the host reference.
TEST(BackendTest, SsmGateDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(83);
  constexpr std::size_t kHeads = 8;
  constexpr std::size_t kRows = 3;
  std::vector<float> a_log(kHeads), dt(kHeads), ar(kRows * kHeads),
      br(kRows * kHeads);
  for (auto& v : a_log) v = DrawValue(rng);
  for (auto& v : dt) v = DrawValue(rng);
  for (auto& v : ar) v = DrawValue(rng);
  for (auto& v : br) v = DrawValue(rng);
  auto mk = [&](std::size_t n) {
    return backend->AllocateBuffer(n * 4, MemoryKind::Device);
  };
  auto a_buf = mk(kHeads), d_buf = mk(kHeads), ar_buf = mk(kRows * kHeads);
  auto br_buf = mk(kRows * kHeads), al_buf = mk(kRows * kHeads),
       be_buf = mk(kRows * kHeads);
  ASSERT_TRUE(a_buf && d_buf && ar_buf && br_buf && al_buf && be_buf);
  const auto upload = [&backend](auto& buf, const std::vector<float>& data) {
    return backend->CopyH2D(**buf, std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size() * 4));
  };
  ASSERT_TRUE(upload(a_buf, a_log).has_value());
  ASSERT_TRUE(upload(d_buf, dt).has_value());
  ASSERT_TRUE(upload(ar_buf, ar).has_value());
  ASSERT_TRUE(upload(br_buf, br).has_value());
  auto kernel = backend->LoadKernel("ssm_gate", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = (kRows * kHeads + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*d_buf).get(), (*ar_buf).get(),
                    (*br_buf).get(), (*al_buf).get(), (*be_buf).get()};
  launch.scalars = {kHeads, kRows};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> alpha_back(kRows * kHeads * 4),
      beta_back(kRows * kHeads * 4);
  ASSERT_TRUE(backend->CopyD2H(**al_buf, alpha_back.data(), alpha_back.size())
                  .has_value());
  ASSERT_TRUE(backend->CopyD2H(**be_buf, beta_back.data(), beta_back.size())
                  .has_value());
  std::vector<float> alpha_ref(kRows * kHeads), beta_ref(kRows * kHeads);
  ASSERT_TRUE(core::SsmGateRef(std::span<const float>(a_log),
                               std::span<const float>(dt),
                               std::span<const float>(ar),
                               std::span<const float>(br),
                               std::span<float>(alpha_ref),
                               std::span<float>(beta_ref), kHeads, kRows)
                  .has_value());
  const auto* got_a = reinterpret_cast<const float*>(alpha_back.data());
  const auto* got_b = reinterpret_cast<const float*>(beta_back.data());
  for (std::size_t i = 0; i < kRows * kHeads; ++i) {
    EXPECT_NEAR(got_a[i], alpha_ref[i], 1e-5f);
    EXPECT_NEAR(got_b[i], beta_ref[i], 1e-5f);
  }
}

// Device: the head-batched delta step matches the host reference.
TEST(BackendTest, DeltaStepHeadsDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(84);
  constexpr std::size_t kHeads = 2;
  constexpr std::size_t kDk = 8;
  constexpr std::size_t kDv = 6;
  std::vector<float> s(kHeads * kDk * kDv), k(kHeads * kDk), v(kHeads * kDv);
  std::vector<float> q(kHeads * kDk), alpha(kHeads), beta(kHeads);
  for (auto& x : s) x = DrawValue(rng);
  for (auto& x : k) x = DrawValue(rng);
  for (auto& x : v) x = DrawValue(rng);
  for (auto& x : q) x = DrawValue(rng);
  for (auto& x : alpha) x = 0.9f;
  for (auto& x : beta) x = 0.5f;
  auto mk = [&](std::size_t n) {
    return backend->AllocateBuffer(n * 4, MemoryKind::Device);
  };
  auto s_buf = mk(s.size()), k_buf = mk(k.size()), v_buf = mk(v.size());
  auto q_buf = mk(q.size()), o_buf = mk(v.size());
  auto al_buf = mk(kHeads), be_buf = mk(kHeads);
  ASSERT_TRUE(s_buf && k_buf && v_buf && q_buf && o_buf && al_buf && be_buf);
  const auto upload = [&backend](auto& buf, const std::vector<float>& data) {
    return backend->CopyH2D(**buf, std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size() * 4));
  };
  ASSERT_TRUE(upload(s_buf, s).has_value());
  ASSERT_TRUE(upload(k_buf, k).has_value());
  ASSERT_TRUE(upload(v_buf, v).has_value());
  ASSERT_TRUE(upload(q_buf, q).has_value());
  ASSERT_TRUE(upload(al_buf, alpha).has_value());
  ASSERT_TRUE(upload(be_buf, beta).has_value());
  auto kernel = backend->LoadKernel("delta_step_heads", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = (kHeads * kDv + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*s_buf).get(), (*k_buf).get(), (*v_buf).get(),
                    (*q_buf).get(), (*o_buf).get(), (*al_buf).get(),
                    (*be_buf).get()};
  launch.scalars = {kHeads, kDk, kDv, 1, 0, 0, 0};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<float> s_ref = s;
  std::vector<float> o_ref(kHeads * kDv);
  ASSERT_TRUE(core::DeltaStepHeadsRef(
                  std::span<float>(s_ref), std::span<const float>(k),
                  std::span<const float>(v), std::span<const float>(q),
                  std::span<float>(o_ref), std::span<const float>(alpha),
                  std::span<const float>(beta), kHeads, kDk, kDv)
                  .has_value());
  std::vector<std::byte> o_back(o_ref.size() * 4), s_back(s.size() * 4);
  ASSERT_TRUE(backend->CopyD2H(**o_buf, o_back.data(), o_back.size())
                  .has_value());
  ASSERT_TRUE(backend->CopyD2H(**s_buf, s_back.data(), s_back.size())
                  .has_value());
  const auto* got_o = reinterpret_cast<const float*>(o_back.data());
  const auto* got_s = reinterpret_cast<const float*>(s_back.data());
  for (std::size_t i = 0; i < o_ref.size(); ++i) {
    EXPECT_NEAR(got_o[i], o_ref[i], 1e-4f);
  }
  for (std::size_t i = 0; i < s_ref.size(); ++i) {
    EXPECT_NEAR(got_s[i], s_ref[i], 1e-4f);
  }
}

// Device: with a non-zero state stride the delta step writes each row's
// post-state to its own history slot (row t reads slot t, writes slot t+1),
// which is what lets a batched verify restore any accepted prefix.

// Device: the current-step conv matches the host reference.
TEST(BackendTest, Conv1dStepDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(85);
  constexpr std::size_t kChannels = 8;
  constexpr std::size_t kWidth = 4;
  std::vector<float> x(kChannels * kWidth);
  std::vector<float> w(kChannels * kWidth);
  for (auto& v : x) v = DrawValue(rng);
  for (auto& v : w) v = DrawValue(rng);
  auto x_buf = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size() * 4, MemoryKind::Device);
  auto y_buf = backend->AllocateBuffer(kChannels * 4, MemoryKind::Device);
  ASSERT_TRUE(x_buf && w_buf && y_buf);
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(**buf, std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size() * 4));
  };
  ASSERT_TRUE(upload(x_buf, x).has_value());
  ASSERT_TRUE(upload(w_buf, w).has_value());
  auto kernel = backend->LoadKernel("conv1d_step", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = (kChannels + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*x_buf).get(), (*w_buf).get(), (*y_buf).get()};
  launch.scalars = {kChannels, kWidth};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(kChannels * 4);
  ASSERT_TRUE(backend->CopyD2H(**y_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(kChannels);
  ASSERT_TRUE(core::Conv1dStepRef(std::span<const float>(x),
                                  std::span<const float>(w),
                                  std::span<float>(ref), kChannels, kWidth)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  for (std::size_t i = 0; i < kChannels; ++i) {
    EXPECT_NEAR(got[i], ref[i], 1e-5f);
  }
}

// Device: the "conv1d_state" kernel (window from qkv + history, reversed
// taps, SiLU, q/k/v split, in-place history shift) matches a host
// reference.
TEST(BackendTest, Conv1dStateDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(96);
  constexpr std::size_t kConvDim = 8;
  constexpr std::size_t kKeyDim = 2;
  constexpr std::size_t kValueDim = kConvDim - 2 * kKeyDim;
  constexpr std::size_t kWidth = 3;
  constexpr std::size_t kHist = kConvDim * (kWidth - 1);
  std::vector<float> qkv(kConvDim), w(kConvDim * kWidth), hist(kHist);
  for (auto& v : qkv) v = DrawValue(rng);
  for (auto& v : w) v = DrawValue(rng);
  for (auto& v : hist) v = DrawValue(rng);
  auto qkv_buf = backend->AllocateBuffer(qkv.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size() * 4, MemoryKind::Device);
  auto hist_buf = backend->AllocateBuffer(hist.size() * 4, MemoryKind::Device);
  auto q_buf = backend->AllocateBuffer(kKeyDim * 4, MemoryKind::Device);
  auto k_buf = backend->AllocateBuffer(kKeyDim * 4, MemoryKind::Device);
  auto v_buf = backend->AllocateBuffer(kValueDim * 4, MemoryKind::Device);
  ASSERT_TRUE(qkv_buf && w_buf && hist_buf && q_buf && k_buf && v_buf);
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(**buf, std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size() * 4));
  };
  ASSERT_TRUE(upload(qkv_buf, qkv).has_value());
  ASSERT_TRUE(upload(w_buf, w).has_value());
  ASSERT_TRUE(upload(hist_buf, hist).has_value());
  auto kernel = backend->LoadKernel("conv1d_state", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = (kConvDim + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*qkv_buf).get(), (*w_buf).get(), (*hist_buf).get(),
                    (*q_buf).get(), (*k_buf).get(), (*v_buf).get(),
                    (*hist_buf).get()};
  launch.scalars = {kConvDim, kWidth, kKeyDim, 0, 1, 0};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  const auto download = [&backend](auto& buf, std::vector<float>& out) {
    std::vector<std::byte> raw(out.size() * 4);
    if (!backend->CopyD2H(**buf, raw.data(), raw.size()).has_value()) {
      return false;
    }
    std::memcpy(out.data(), raw.data(), raw.size());
    return true;
  };
  std::vector<float> got_q(kKeyDim), got_k(kKeyDim), got_v(kValueDim),
      got_hist(kHist);
  ASSERT_TRUE(download(q_buf, got_q));
  ASSERT_TRUE(download(k_buf, got_k));
  ASSERT_TRUE(download(v_buf, got_v));
  ASSERT_TRUE(download(hist_buf, got_hist));

  std::vector<float> ref_q(kKeyDim), ref_k(kKeyDim), ref_v(kValueDim);
  std::vector<float> ref_hist = hist;
  for (std::size_t c = 0; c < kConvDim; ++c) {
    const std::size_t base = c * (kWidth - 1);
    float acc = w[c * kWidth + (kWidth - 1)] * qkv[c];
    for (std::size_t i = 1; i < kWidth; ++i) {
      acc += w[c * kWidth + (kWidth - 1 - i)] * hist[base + (i - 1)];
    }
    const float result = acc / (1.0f + std::exp(-acc));
    if (c < kKeyDim) {
      ref_q[c] = result;
    } else if (c < 2 * kKeyDim) {
      ref_k[c - kKeyDim] = result;
    } else {
      ref_v[c - 2 * kKeyDim] = result;
    }
    for (std::size_t i = kWidth - 1; i > 1; --i) {
      ref_hist[base + (i - 1)] = hist[base + (i - 2)];
    }
    ref_hist[base] = qkv[c];
  }
  for (std::size_t i = 0; i < kKeyDim; ++i) {
    EXPECT_NEAR(got_q[i], ref_q[i], 1e-5f);
    EXPECT_NEAR(got_k[i], ref_k[i], 1e-5f);
  }
  for (std::size_t i = 0; i < kValueDim; ++i) {
    EXPECT_NEAR(got_v[i], ref_v[i], 1e-5f);
  }
  for (std::size_t i = 0; i < kHist; ++i) {
    EXPECT_NEAR(got_hist[i], ref_hist[i], 1e-5f);
  }
}

// Device: with a non-zero conv history stride the conv records each row's
// post-conv history into its own slot, which a batched verify restores from.

// Device: attention with several query rows (m > 1) matches the reference.
TEST(BackendTest, AttentionBatchedMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(86);
  constexpr std::size_t kM = 3;
  constexpr std::size_t kN = 6;
  constexpr std::size_t kHeads = 4;
  constexpr std::size_t kKvHeads = 2;
  constexpr std::size_t kDim = 16;
  constexpr std::uint64_t kQBase = 2;
  std::vector<float> q(kM * kHeads * kDim), k(kN * kKvHeads * kDim);
  std::vector<float> v(kN * kKvHeads * kDim);
  for (auto& x : q) x = DrawValue(rng);
  for (auto& x : k) x = DrawValue(rng);
  for (auto& x : v) x = DrawValue(rng);
  auto q_buf = backend->AllocateBuffer(q.size() * 4, MemoryKind::Device);
  auto k_buf = backend->AllocateBuffer(k.size() * 4, MemoryKind::Device);
  auto v_buf = backend->AllocateBuffer(v.size() * 4, MemoryKind::Device);
  auto o_buf =
      backend->AllocateBuffer(kM * kHeads * kDim * 4, MemoryKind::Device);
  ASSERT_TRUE(q_buf && k_buf && v_buf && o_buf);
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(**buf, std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size() * 4));
  };
  ASSERT_TRUE(upload(q_buf, q).has_value());
  ASSERT_TRUE(upload(k_buf, k).has_value());
  ASSERT_TRUE(upload(v_buf, v).has_value());
  auto kernel = backend->LoadKernel("attention", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = kM * kHeads;
  launch.block_x = 256;
  launch.buffers = {(*q_buf).get(), (*k_buf).get(), (*v_buf).get(),
                    (*o_buf).get()};
  launch.scalars = {kM, kN, kHeads, kKvHeads, kDim, kQBase, 0, 0, 1};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kHeads * kDim * 4);
  ASSERT_TRUE(backend->CopyD2H(**o_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(kM * kHeads * kDim);
  ASSERT_TRUE(core::AttentionRef(std::span<const float>(q),
                                 std::span<const float>(k),
                                 std::span<const float>(v),
                                 std::span<float>(ref), kM, kN, kHeads, kKvHeads,
                                 kDim, kQBase)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the DFlash2 grouped dynamic convolution matches the reference.
TEST(BackendTest, DflashConvDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(87);
  constexpr std::size_t kRows = 8;
  constexpr std::size_t kChannels = 8;
  constexpr std::size_t kTaps = 2;
  constexpr std::size_t kGroup = 4;
  constexpr std::size_t kBlock = 4;
  const std::size_t num_groups = kChannels / kGroup;
  std::vector<float> x(kRows * kChannels), base(kTaps * kChannels);
  std::vector<float> delta(kRows * kTaps * num_groups);
  for (auto& v : x) v = DrawValue(rng);
  for (auto& v : base) v = DrawValue(rng);
  for (auto& v : delta) v = DrawValue(rng);
  auto x_buf = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  auto d_buf = backend->AllocateBuffer(delta.size() * 4, MemoryKind::Device);
  auto b_buf = backend->AllocateBuffer(base.size() * 4, MemoryKind::Device);
  auto y_buf = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(x_buf && d_buf && b_buf && y_buf);
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(**buf, std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()), data.size() * 4));
  };
  ASSERT_TRUE(upload(x_buf, x).has_value());
  ASSERT_TRUE(upload(d_buf, delta).has_value());
  ASSERT_TRUE(upload(b_buf, base).has_value());
  auto kernel = backend->LoadKernel("dflash_conv", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = (kRows * kChannels + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*x_buf).get(), (*d_buf).get(), (*b_buf).get(),
                    (*y_buf).get()};
  launch.scalars = {kRows, kChannels, kTaps, kGroup, kBlock,
                    kTaps * num_groups, 0};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(x.size() * 4);
  ASSERT_TRUE(backend->CopyD2H(**y_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(x.size());
  ASSERT_TRUE(core::DflashConvRef(
                  std::span<const float>(x), std::span<const float>(delta),
                  std::span<const float>(base), std::span<float>(ref), kRows,
                  kChannels, kTaps, kGroup, kBlock, kTaps * num_groups)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the DFlash2 candidate-selector edge score matches the reference.
// The selector transition score adds the successor candidate's unary logit
// (vLLM `_score_edges` maps `unary_logits[:, :, None]` onto the successor
// axis), not the predecessor's. All-zero codebooks and hidden isolate the
// unary term, so out[pos, p, c] must equal unary[pos, c] for every p.

TEST(BackendTest, SelectorEdgeScoreDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(88);
  constexpr std::size_t kBatch = 1;
  constexpr std::size_t kSeq = 3;
  constexpr std::size_t kTopK = 3;
  constexpr std::size_t kRank = 4;
  constexpr std::size_t kVocab = 7;
  const std::size_t positions = kBatch * kSeq;
  std::vector<float> pred(kVocab * kRank), succ(kVocab * kRank);
  std::vector<float> hidden(positions * kRank), unary(positions * kTopK);
  for (auto& v : pred) v = DrawValue(rng);
  for (auto& v : succ) v = DrawValue(rng);
  for (auto& v : hidden) v = DrawValue(rng);
  for (auto& v : unary) v = DrawValue(rng);
  std::vector<std::int32_t> cand(positions * kTopK), anchor(positions);
  std::uniform_int_distribution<std::int32_t> pick(
      0, static_cast<std::int32_t>(kVocab - 1));
  for (auto& v : cand) v = pick(rng);
  for (auto& v : anchor) v = pick(rng);
  auto alloc = [&backend](std::size_t bytes) {
    return backend->AllocateBuffer(bytes, MemoryKind::Device);
  };
  auto pred_buf = alloc(pred.size() * 4);
  auto succ_buf = alloc(succ.size() * 4);
  auto hid_buf = alloc(hidden.size() * 4);
  auto cand_buf = alloc(cand.size() * 4);
  auto anchor_buf = alloc(anchor.size() * 4);
  auto unary_buf = alloc(unary.size() * 4);
  auto out_buf = alloc(positions * kTopK * kTopK * 4);
  ASSERT_TRUE(pred_buf && succ_buf && hid_buf && cand_buf && anchor_buf &&
              unary_buf && out_buf);
  const auto upload = [&backend](auto& buf, const void* data, std::size_t bytes) {
    return backend->CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data), bytes));
  };
  ASSERT_TRUE(upload(pred_buf, pred.data(), pred.size() * 4).has_value());
  ASSERT_TRUE(upload(succ_buf, succ.data(), succ.size() * 4).has_value());
  ASSERT_TRUE(upload(hid_buf, hidden.data(), hidden.size() * 4).has_value());
  ASSERT_TRUE(upload(cand_buf, cand.data(), cand.size() * 4).has_value());
  ASSERT_TRUE(upload(anchor_buf, anchor.data(), anchor.size() * 4).has_value());
  ASSERT_TRUE(upload(unary_buf, unary.data(), unary.size() * 4).has_value());
  auto kernel = backend->LoadKernel("selector_edge_score", {});
  ASSERT_TRUE(kernel.has_value());
  const std::size_t total = positions * kTopK * kTopK;
  tessera::KernelLaunch launch;
  launch.grid_x = (total + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*pred_buf).get(),   (*succ_buf).get(),
                    (*hid_buf).get(),    (*cand_buf).get(),
                    (*anchor_buf).get(), (*unary_buf).get(),
                    (*out_buf).get()};
  launch.scalars = {kBatch, kSeq, kTopK, kRank, kVocab};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(total * 4);
  ASSERT_TRUE(backend->CopyD2H(**out_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(total);
  ASSERT_TRUE(core::SelectorEdgeScoreRef(
                  std::span<const float>(pred), std::span<const float>(succ),
                  std::span<const float>(hidden),
                  std::span<const std::int32_t>(cand),
                  std::span<const std::int32_t>(anchor),
                  std::span<const float>(unary), std::span<float>(ref), kBatch,
                  kSeq, kTopK, kRank, kVocab)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: per-row top-k matches a host selection, including the row_base
// offset. Indices and values are exact (same source elements).
TEST(BackendTest, TopKRowsDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(99);
  constexpr std::size_t kRows = 3;
  constexpr std::size_t kVocab = 20;
  constexpr std::size_t kTopK = 4;
  constexpr std::size_t kRowBase = 1;
  std::vector<float> logits((kRows + kRowBase) * kVocab);
  for (auto& v : logits) {
    v = DrawValue(rng);
  }
  auto alloc = [&backend](std::size_t bytes) {
    return backend->AllocateBuffer(bytes, MemoryKind::Device);
  };
  auto in_buf = alloc(logits.size() * 4);
  auto ids_buf = alloc(kRows * kTopK * 4);
  auto vals_buf = alloc(kRows * kTopK * 4);
  ASSERT_TRUE(in_buf && ids_buf && vals_buf);
  ASSERT_TRUE(
      backend
          ->CopyH2D(**in_buf,
                    std::span<const std::byte>(
                        reinterpret_cast<const std::byte*>(logits.data()),
                        logits.size() * 4))
          .has_value());
  auto kernel = backend->LoadKernel("top_k_rows", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = kRows;
  launch.block_x = 256;
  launch.buffers = {(*in_buf).get(), (*ids_buf).get(), (*vals_buf).get()};
  launch.scalars = {kRows, kVocab, kTopK, kRowBase};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> ids_readback(kRows * kTopK * 4);
  std::vector<std::byte> vals_readback(kRows * kTopK * 4);
  ASSERT_TRUE(backend->CopyD2H(**ids_buf, ids_readback.data(),
                               ids_readback.size())
                  .has_value());
  ASSERT_TRUE(backend->CopyD2H(**vals_buf, vals_readback.data(),
                               vals_readback.size())
                  .has_value());
  const auto* got_ids =
      reinterpret_cast<const std::uint32_t*>(ids_readback.data());
  const auto* got_vals = reinterpret_cast<const float*>(vals_readback.data());
  for (std::size_t r = 0; r < kRows; ++r) {
    const float* row = logits.data() + (kRowBase + r) * kVocab;
    std::vector<std::uint32_t> order(kVocab);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + kTopK, order.end(),
                      [row](std::uint32_t a, std::uint32_t b) {
                        return row[a] > row[b];
                      });
    for (std::size_t k = 0; k < kTopK; ++k) {
      EXPECT_EQ(got_ids[r * kTopK + k], order[k]);
      EXPECT_FLOAT_EQ(got_vals[r * kTopK + k], row[order[k]]);
    }
  }
}

// Device: sliding-window attention matches the reference; a nonzero
// window trims the visible keys for the later query rows.
TEST(BackendTest, AttentionWindowDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(4321);
  constexpr std::size_t kM = 3;
  constexpr std::size_t kN = 5;
  constexpr std::size_t kHeads = 4;
  constexpr std::size_t kKvHeads = 2;
  constexpr std::size_t kDim = 16;
  constexpr std::uint64_t kQBase = 2;
  constexpr std::uint64_t kWindow = 2;
  std::vector<float> q(kM * kHeads * kDim), k(kN * kKvHeads * kDim);
  std::vector<float> v(kN * kKvHeads * kDim);
  for (auto& x : q) x = DrawValue(rng);
  for (auto& x : k) x = DrawValue(rng);
  for (auto& x : v) x = DrawValue(rng);
  auto q_buf = backend->AllocateBuffer(q.size() * 4, MemoryKind::Device);
  auto k_buf = backend->AllocateBuffer(k.size() * 4, MemoryKind::Device);
  auto v_buf = backend->AllocateBuffer(v.size() * 4, MemoryKind::Device);
  auto out_buf =
      backend->AllocateBuffer(kM * kHeads * kDim * 4, MemoryKind::Device);
  ASSERT_TRUE(q_buf && k_buf && v_buf && out_buf);
  const auto upload = [&backend](auto& buf, const auto& data) {
    return backend->CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data.data()),
                   data.size() * 4));
  };
  ASSERT_TRUE(upload(q_buf, q).has_value());
  ASSERT_TRUE(upload(k_buf, k).has_value());
  ASSERT_TRUE(upload(v_buf, v).has_value());
  auto kernel = backend->LoadKernel("attention", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = kM * kHeads;
  launch.block_x = 256;
  launch.buffers = {(*q_buf).get(), (*k_buf).get(), (*v_buf).get(),
                    (*out_buf).get()};
  launch.scalars = {kM, kN, kHeads, kKvHeads, kDim, kQBase, kWindow, 0, 1};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kHeads * kDim * 4);
  ASSERT_TRUE(backend->CopyD2H(**out_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(kM * kHeads * kDim);
  ASSERT_TRUE(core::AttentionRef(std::span<const float>(q),
                                 std::span<const float>(k),
                                 std::span<const float>(v),
                                 std::span<float>(ref), kM, kN, kHeads,
                                 kKvHeads, kDim, kQBase, kWindow)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: dflash_conv consumes one side of the kernel_projection output
// [rows, 2, taps, groups] directly via the delta row stride.

// Device: the shared KV append grows the cache geometrically and keeps
// every appended row. A struct with the same shape as the decode caches
// exercises the growth boundary (4 -> 8 -> 16).
TEST(BackendTest, AppendKvGrowsGeometrically) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  struct Kv {
    std::unique_ptr<tessera::Buffer> k;
    std::unique_ptr<tessera::Buffer> v;
    std::unique_ptr<tessera::Buffer> k_scale;
    std::unique_ptr<tessera::Buffer> v_scale;
    std::size_t rows = 0;
    std::size_t capacity = 0;
    tessera::KvCacheType type = tessera::KvCacheType::F32;
  };
  Kv kv;
  constexpr std::size_t kElems = 3;
  constexpr std::size_t kKvBytes = kElems * 4;
  std::vector<float> want_k, want_v;
  for (std::size_t r = 0; r < 10; ++r) {
    std::vector<float> k_row(kElems), v_row(kElems);
    for (std::size_t e = 0; e < kElems; ++e) {
      k_row[e] = static_cast<float>(r * 10 + e);
      v_row[e] = static_cast<float>(1000 + r * 10 + e);
    }
    auto k_buf = backend->AllocateBuffer(kKvBytes, MemoryKind::Device);
    auto v_buf = backend->AllocateBuffer(kKvBytes, MemoryKind::Device);
    ASSERT_TRUE(k_buf && v_buf);
    const auto upload = [&backend](auto& buf, const std::vector<float>& data) {
      return backend->CopyH2D(
          **buf, std::span<const std::byte>(
                     reinterpret_cast<const std::byte*>(data.data()),
                     data.size() * 4));
    };
    ASSERT_TRUE(upload(k_buf, k_row).has_value());
    ASSERT_TRUE(upload(v_buf, v_row).has_value());
    auto status =
        core::detail::AppendKv(*backend, nullptr, nullptr, nullptr, nullptr,
                             nullptr, kv, **k_buf, **v_buf, kElems);
    ASSERT_TRUE(status.has_value()) << tessera::ToString(status.error());
    want_k.insert(want_k.end(), k_row.begin(), k_row.end());
    want_v.insert(want_v.end(), v_row.begin(), v_row.end());
  }
  EXPECT_EQ(kv.rows, 10u);
  EXPECT_GE(kv.capacity, 10u);
  std::vector<float> got_k(want_k.size()), got_v(want_v.size());
  ASSERT_TRUE(backend->CopyD2H(*kv.k,
                                reinterpret_cast<std::byte*>(got_k.data()),
                                got_k.size() * 4)
                  .has_value());
  ASSERT_TRUE(backend->CopyD2H(*kv.v,
                                reinterpret_cast<std::byte*>(got_v.data()),
                                got_v.size() * 4)
                  .has_value());
  EXPECT_EQ(got_k, want_k);
  EXPECT_EQ(got_v, want_v);
}

// Device: the plain fp32 GEMM (unquantized weights) matches the reference.
TEST(BackendTest, GemmF32DeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(90);
  constexpr std::size_t kM = 3;
  constexpr std::size_t kN = 4;
  constexpr std::size_t kK = 5;
  std::vector<float> a(kM * kK), w(kN * kK);
  for (auto& v : a) v = DrawValue(rng);
  for (auto& v : w) v = DrawValue(rng);
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size() * 4, MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf && w_buf && c_buf);
  const auto upload = [&backend](auto& buf, const std::vector<float>& data) {
    return backend->CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data.data()),
                   data.size() * 4));
  };
  ASSERT_TRUE(upload(a_buf, a).has_value());
  ASSERT_TRUE(upload(w_buf, w).has_value());
  auto kernel = backend->LoadKernel("gemm_f32", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = tessera::core::detail::GemmGridFor(**kernel, kM, kN);
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kN * 4);
  ASSERT_TRUE(backend->CopyD2H(**c_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(kM * kN);
  ASSERT_TRUE(core::GemmF32Ref(std::span<const float>(a),
                               std::span<const float>(w),
                               std::span<float>(ref), kM, kN, kK)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the tiled batched F32/BF16 GEMMs (the DFlash2 draft and the
// target's batch-verify head) match the host reference for m > 1,
// including a k that is not a multiple of the four-wide inner step.
TEST(BackendTest, GemmBatchedF32Bf16DeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(97);
  constexpr std::size_t kM = 3;
  constexpr std::size_t kN = 4;
  constexpr std::size_t kK = 5;
  std::vector<float> a(kM * kK), w_f32(kN * kK);
  for (auto& v : a) v = DrawValue(rng);
  for (auto& v : w_f32) v = DrawValue(rng);
  std::vector<std::byte> w_bf16(kN * kK * 2);
  for (std::size_t i = 0; i < kN * kK; ++i) {
    const float value = DrawValue(rng);
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint16_t bf = static_cast<std::uint16_t>(bits >> 16);
    w_bf16[i * 2] = static_cast<std::byte>(bf & 0xFF);
    w_bf16[i * 2 + 1] = static_cast<std::byte>((bf >> 8) & 0xFF);
  }
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto wf_buf = backend->AllocateBuffer(w_f32.size() * 4, MemoryKind::Device);
  auto wb_buf = backend->AllocateBuffer(w_bf16.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf && wf_buf && wb_buf && c_buf);
  ASSERT_TRUE(backend->CopyH2D(
                  **a_buf,
                  std::span<const std::byte>(
                      reinterpret_cast<const std::byte*>(a.data()),
                      a.size() * 4))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(
                  **wf_buf,
                  std::span<const std::byte>(
                      reinterpret_cast<const std::byte*>(w_f32.data()),
                      w_f32.size() * 4))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(**wb_buf, std::span<const std::byte>(w_bf16))
                  .has_value());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  const std::pair<const char*, bool> cases[] = {
      {"gemm_f32_batched", false}, {"gemm_bf16_batched", true}};
  for (const auto& [name, bf16] : cases) {
    auto kernel = backend->LoadKernel(name, {});
    ASSERT_TRUE(kernel.has_value()) << name;
    tessera::KernelLaunch launch;
    launch.grid_x = tessera::core::detail::GemmGridFor(**kernel, kM, kN);
    launch.block_x = 256;
    launch.buffers = {(*a_buf).get(),
                      bf16 ? (*wb_buf).get() : (*wf_buf).get(),
                      (*c_buf).get()};
    launch.scalars = {kM, kN, kK};
    ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value()) << name;
    backend->Synchronize();
    std::vector<std::byte> readback(kM * kN * 4);
    ASSERT_TRUE(backend->CopyD2H(**c_buf, readback.data(), readback.size())
                    .has_value());
    std::vector<float> ref(kM * kN);
    if (bf16) {
      ASSERT_TRUE(core::GemmBf16Ref(
                      std::span<const float>(a),
                      std::span<const std::byte>(w_bf16),
                      std::span<float>(ref), kM, kN, kK)
                      .has_value());
    } else {
      ASSERT_TRUE(core::GemmF32Ref(std::span<const float>(a),
                                   std::span<const float>(w_f32),
                                   std::span<float>(ref), kM, kN, kK)
                      .has_value());
    }
    const auto* got = reinterpret_cast<const float*>(readback.data());
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
    }
    EXPECT_LE(max_abs, tol.abs)
        << name << " backend " << backend->Name() << " max_abs " << max_abs;
  }
}
// reference converts the same bf16 bits).
TEST(BackendTest, GemmBf16DeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(91);
  constexpr std::size_t kM = 3;
  constexpr std::size_t kN = 4;
  constexpr std::size_t kK = 5;
  std::vector<float> a(kM * kK, 0.0f);
  for (auto& v : a) v = DrawValue(rng);
  std::vector<std::byte> w_bf16(kN * kK * 2);
  for (std::size_t i = 0; i < kN * kK; ++i) {
    const float value = DrawValue(rng);
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint16_t bf = static_cast<std::uint16_t>(bits >> 16);
    w_bf16[i * 2] = static_cast<std::byte>(bf & 0xFF);
    w_bf16[i * 2 + 1] = static_cast<std::byte>((bf >> 8) & 0xFF);
  }
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w_bf16.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf && w_buf && c_buf);
  ASSERT_TRUE(backend->CopyH2D(
                  **a_buf, std::span<const std::byte>(
                               reinterpret_cast<const std::byte*>(a.data()),
                               a.size() * 4))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(**w_buf, std::span<const std::byte>(w_bf16))
                  .has_value());
  auto kernel = backend->LoadKernel("gemm_bf16", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = tessera::core::detail::GemmGridFor(**kernel, kM, kN);
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kN * 4);
  ASSERT_TRUE(backend->CopyD2H(**c_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(kM * kN);
  ASSERT_TRUE(core::GemmBf16Ref(std::span<const float>(a),
                                std::span<const std::byte>(w_bf16),
                                std::span<float>(ref), kM, kN, kK)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the token-embedding gather matches a host gather for f32, bf16
// and Q4_K tables; an out-of-range id writes a zero row.
TEST(BackendTest, EmbeddingGatherMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  constexpr std::size_t kRows = 4;
  constexpr std::size_t kCols = 512;
  constexpr std::size_t kVocab = 8;
  const std::vector<std::uint32_t> ids = {3, 0, 7, kVocab};
  std::mt19937 rng(4321);
  std::vector<float> values(kVocab * kCols);
  for (auto& v : values) {
    v = DrawValue(rng);
  }
  std::vector<std::byte> table_bf16(kVocab * kCols * 2);
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &values[i], sizeof(bits));
    const std::uint16_t half = static_cast<std::uint16_t>(bits >> 16);
    std::memcpy(table_bf16.data() + i * 2, &half, 2);
  }
  const std::vector<std::byte> table_q4k = QuantizeRows(values, kVocab, kCols);

  const DType dtypes[3] = {DType::F32, DType::BF16, DType::Q4K};
  for (const DType dtype : dtypes) {
    std::span<const std::byte> table;
    std::size_t row_bytes = 0;
    if (dtype == DType::F32) {
      row_bytes = kCols * 4;
      table = std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(values.data()),
          values.size() * 4);
    } else if (dtype == DType::BF16) {
      row_bytes = kCols * 2;
      table = table_bf16;
    } else {
      row_bytes = (kCols / 256) * core::kQ4KBlockBytes;
      table = table_q4k;
    }
    // Host reference: dequantize the requested row, zero for an OOB id.
    std::vector<float> ref(kRows * kCols, 0.0f);
    for (std::size_t r = 0; r < kRows; ++r) {
      const std::uint32_t id = ids[r];
      if (id >= kVocab) {
        continue;
      }
      float* dst = ref.data() + r * kCols;
      if (dtype == DType::F32) {
        std::memcpy(dst, values.data() + id * kCols, kCols * 4);
      } else if (dtype == DType::BF16) {
        for (std::size_t i = 0; i < kCols; ++i) {
          std::uint16_t half = 0;
          std::memcpy(&half, table_bf16.data() + (id * kCols + i) * 2, 2);
          dst[i] = core::Bf16ToFloat(half);
        }
      } else {
        auto status = core::DequantizeBlocks(
            dtype, table.subspan(id * row_bytes, row_bytes),
            std::span<float>(dst, kCols));
        ASSERT_TRUE(status.has_value()) << tessera::ToString(status.error());
      }
    }

    auto w_buf = backend->AllocateBuffer(table.size(), MemoryKind::Device);
    ASSERT_TRUE(w_buf.has_value());
    auto ids_buf = backend->AllocateBuffer(kRows * 4, MemoryKind::Device);
    ASSERT_TRUE(ids_buf.has_value());
    auto out_buf = backend->AllocateBuffer(kRows * kCols * 4, MemoryKind::Device);
    ASSERT_TRUE(out_buf.has_value());
    ASSERT_TRUE(backend->CopyH2D(**w_buf, table).has_value());
    ASSERT_TRUE(backend
                    ->CopyH2D(**ids_buf,
                              std::span<const std::byte>(
                                  reinterpret_cast<const std::byte*>(ids.data()),
                                  ids.size() * 4))
                    .has_value());
    const std::string_view name = core::detail::EmbeddingKernelName(dtype);
    ASSERT_FALSE(name.empty());
    auto kernel = backend->LoadKernel(name, {});
    ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
    auto launch = core::detail::GatherEmbeddingDevice(
        *backend, **kernel, **ids_buf, **w_buf, **out_buf, kRows, kCols, kVocab);
    ASSERT_TRUE(launch.has_value()) << tessera::ToString(launch.error());
    backend->Synchronize();

    std::vector<std::byte> readback(kRows * kCols * 4);
    ASSERT_TRUE(backend->CopyD2H(**out_buf, readback.data(), readback.size())
                    .has_value());
    const auto* got = reinterpret_cast<const float*>(readback.data());
    const GemmTolerance tol = ToleranceFor(backend->Name());
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
    }
    EXPECT_LE(max_abs, tol.abs)
        << "dtype " << static_cast<int>(dtype) << " backend "
        << backend->Name() << " max_abs " << max_abs;
  }
}

// Device: the block-scaled fp8 GEMM (one scale per 128x128 block) matches
// the reference. Bytes 0x7F/0xFF (E4M3 NaN) are avoided.
TEST(BackendTest, GemmFp8BlockDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(92);
  constexpr std::size_t kM = 2;
  constexpr std::size_t kN = 128;
  constexpr std::size_t kK = 128;
  std::vector<float> a(kM * kK, 0.0f);
  for (auto& v : a) v = DrawValue(rng);
  std::vector<std::byte> w(kN * kK);
  std::uniform_int_distribution<int> byte(0, 255);
  for (auto& v : w) {
    int b = byte(rng);
    if (b == 0x7F || b == 0xFF) b = 0x7E;
    v = static_cast<std::byte>(b);
  }
  std::vector<float> s(1, DrawValue(rng));
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  auto s_buf = backend->AllocateBuffer(s.size() * 4, MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf && w_buf && s_buf && c_buf);
  ASSERT_TRUE(backend->CopyH2D(
                  **a_buf, std::span<const std::byte>(
                               reinterpret_cast<const std::byte*>(a.data()),
                               a.size() * 4))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(**w_buf, std::span<const std::byte>(w))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(
                  **s_buf, std::span<const std::byte>(
                               reinterpret_cast<const std::byte*>(s.data()),
                               s.size() * 4))
                  .has_value());
  auto kernel = backend->LoadKernel("gemm_fp8_block", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = (kM * kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*a_buf).get(), (*w_buf).get(), (*s_buf).get(),
                    (*c_buf).get()};
  launch.scalars = {kM, kN, kK};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kN * 4);
  ASSERT_TRUE(backend->CopyD2H(**c_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(kM * kN);
  ASSERT_TRUE(core::GemmFp8BlockRef(std::span<const float>(a),
                                    std::span<const std::byte>(w),
                                    std::span<const float>(s),
                                    std::span<float>(ref), kM, kN, kK)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the DFlash2 grouped-conv layer stage (projection then one conv
// side) matches the host reference for both sides.

// Device: the DFlash2 MLP half of a draft layer (post-norm, mlp_conv
// prepare/finish, gated SiLU MLP) matches the host reference.

// Device: the DFlash2 attention half of a draft layer (input norm,
// attention_conv prepare/finish, QK-norm, RoPE, sliding attention, output
// projection) matches the host reference.

// Device: a full DFlash2 draft layer (residual + attention half + MLP
// half) matches the host reference, with and without a carried residual.
TEST(BackendTest, DraftLayerMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(96);
  constexpr std::size_t kRows = 2;
  constexpr std::size_t kHidden = 4;
  constexpr std::size_t kHeads = 2;
  constexpr std::size_t kKvHeads = 1;
  constexpr std::size_t kHeadDim = 2;
  constexpr std::size_t kFfn = 8;
  constexpr std::size_t kTaps = 2;
  constexpr std::size_t kGroup = 2;
  constexpr std::size_t kBlock = 2;
  constexpr std::size_t kWindow = 2;
  constexpr float kEps = 1e-6f;
  constexpr double kTheta = 10000.0;
  const std::size_t q_dim = kHeads * kHeadDim;
  const std::size_t kv_dim = kKvHeads * kHeadDim;
  const std::size_t num_groups = kHidden / kGroup;
  const std::size_t proj_n = 2 * kTaps * num_groups;
  auto rnd = [&rng](std::size_t n) {
    std::vector<float> out(n);
    for (auto& v : out) v = DrawValue(rng);
    return out;
  };
  std::vector<float> hidden = rnd(kRows * kHidden);
  std::vector<float> resid = rnd(kRows * kHidden);
  struct Span {
    std::vector<float> data;
  };
  const std::vector<float> inorm = rnd(kHidden);
  const std::vector<float> acp = rnd(proj_n * kHidden);
  const std::vector<float> acb = rnd(2 * kTaps * kHidden);
  const std::vector<float> qw = rnd(q_dim * kHidden);
  const std::vector<float> kw = rnd(kv_dim * kHidden);
  const std::vector<float> vw = rnd(kv_dim * kHidden);
  const std::vector<float> ow = rnd(kHidden * q_dim);
  const std::vector<float> qn = rnd(kHeadDim);
  const std::vector<float> kn = rnd(kHeadDim);
  const std::vector<float> pnorm = rnd(kHidden);
  const std::vector<float> mcp = rnd(proj_n * kHidden);
  const std::vector<float> mcb = rnd(2 * kTaps * kHidden);
  const std::vector<float> gw = rnd(kFfn * kHidden);
  const std::vector<float> uw = rnd(kFfn * kHidden);
  const std::vector<float> dw = rnd(kHidden * kFfn);
  const std::vector<float> hnorm = rnd(kHidden);
  constexpr std::size_t kCtx = 2;
  const std::vector<float> chidden = rnd(kCtx * kHidden);
  auto upload = [&backend](const std::vector<float>& data) {
    auto buf = backend->AllocateBuffer(data.size() * 4, MemoryKind::Device);
    EXPECT_TRUE(buf.has_value());
    backend->CopyH2D(**buf, std::span<const std::byte>(
                               reinterpret_cast<const std::byte*>(data.data()),
                               data.size() * 4));
    return std::move(*buf);
  };
  auto inorm_b = upload(inorm), acp_b = upload(acp), acb_b = upload(acb);
  auto qw_b = upload(qw), kw_b = upload(kw), vw_b = upload(vw);
  auto ow_b = upload(ow), qn_b = upload(qn), kn_b = upload(kn);
  auto pnorm_b = upload(pnorm), mcp_b = upload(mcp), mcb_b = upload(mcb);
  auto gw_b = upload(gw), uw_b = upload(uw), dw_b = upload(dw);
  auto hnorm_b = upload(hnorm);
  auto chidden_b = upload(chidden);
  auto h_b = upload(hidden), r_b = upload(resid);
  auto out_b = backend->AllocateBuffer(hidden.size() * 4, MemoryKind::Device);
  auto rout_b = backend->AllocateBuffer(hidden.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(out_b && rout_b);
  tessera::spec::DraftLayerBuffers wb;
  wb.input_norm = &*inorm_b;
  wb.attn_conv_proj = &*acp_b;
  wb.attn_conv_base = &*acb_b;
  wb.q_w = &*qw_b;
  wb.k_w = &*kw_b;
  wb.v_w = &*vw_b;
  wb.o_w = &*ow_b;
  wb.q_norm_w = &*qn_b;
  wb.k_norm_w = &*kn_b;
  wb.post_norm = &*pnorm_b;
  wb.mlp_conv_proj = &*mcp_b;
  wb.mlp_conv_base = &*mcb_b;
  wb.gate_w = &*gw_b;
  wb.up_w = &*uw_b;
  wb.down_w = &*dw_b;
  wb.hidden_norm = &*hnorm_b;
  auto rms = backend->LoadKernel("rmsnorm", {});
  auto gemm = backend->LoadKernel("gemm_f32", {});
  auto conv = backend->LoadKernel("dflash_conv", {});
  auto rope = backend->LoadKernel("rope", {});
  auto attn = backend->LoadKernel("attention", {});
  auto silu = backend->LoadKernel("silu_mul", {});
  auto add = backend->LoadKernel("add", {});
  ASSERT_TRUE(rms && gemm && conv && rope && attn && silu && add);
  tessera::spec::DraftLayerWeights wr;
  wr.input_norm = std::span<const float>(inorm);
  wr.attn_conv_proj = std::span<const float>(acp);
  wr.attn_conv_base = std::span<const float>(acb);
  wr.q_w = std::span<const float>(qw);
  wr.k_w = std::span<const float>(kw);
  wr.v_w = std::span<const float>(vw);
  wr.o_w = std::span<const float>(ow);
  wr.q_norm_w = std::span<const float>(qn);
  wr.k_norm_w = std::span<const float>(kn);
  wr.post_norm = std::span<const float>(pnorm);
  wr.mlp_conv_proj = std::span<const float>(mcp);
  wr.mlp_conv_base = std::span<const float>(mcb);
  wr.gate_w = std::span<const float>(gw);
  wr.up_w = std::span<const float>(uw);
  wr.down_w = std::span<const float>(dw);
  wr.hidden_norm = std::span<const float>(hnorm);
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  for (int use_residual = 0; use_residual < 2; ++use_residual) {
    for (const std::size_t ctx : {std::size_t{0}, kCtx}) {
      const tessera::Buffer* residual = use_residual ? &*r_b : nullptr;
      const tessera::Buffer* context = ctx > 0 ? &*chidden_b : nullptr;
      auto device = tessera::spec::DraftLayerDevice(
          *backend, **rms, **gemm, **conv, **rope, **attn, **silu, **add, *h_b,
          residual, wb, **out_b, **rout_b, kRows, kHidden, kHeads, kKvHeads,
          kHeadDim, kFfn, kTaps, kGroup, kBlock, kWindow, 0, kTheta, kEps,
          context, ctx);
      ASSERT_TRUE(device.has_value()) << tessera::ToString(device.error());
      backend->Synchronize();
      std::vector<float> out_got(hidden.size()), rout_got(hidden.size());
      backend->CopyD2H(**out_b, reinterpret_cast<std::byte*>(out_got.data()),
                       out_got.size() * 4);
      backend->CopyD2H(**rout_b, reinterpret_cast<std::byte*>(rout_got.data()),
                       rout_got.size() * 4);
      std::vector<float> ref(hidden.size()), rref(hidden.size());
      ASSERT_TRUE(tessera::spec::DraftLayerRef(
                      std::span<const float>(hidden),
                      use_residual ? std::span<const float>(resid)
                                   : std::span<const float>(),
                      wr, std::span<float>(ref), std::span<float>(rref), kRows,
                      kHidden, kHeads, kKvHeads, kHeadDim, kFfn, kTaps, kGroup,
                      kBlock, kWindow, 0, kTheta, kEps,
                      ctx > 0 ? std::span<const float>(chidden)
                              : std::span<const float>())
                      .has_value());
      float dout = 0.0f, dres = 0.0f;
      for (std::size_t i = 0; i < ref.size(); ++i) {
        dout = std::max(dout, std::abs(ref[i] - out_got[i]));
        dres = std::max(dres, std::abs(rref[i] - rout_got[i]));
      }
      EXPECT_LE(dout, tol.abs) << "residual=" << use_residual << " ctx=" << ctx;
      EXPECT_LE(dres, tol.abs) << "residual=" << use_residual << " ctx=" << ctx;
    }
  }
}

// Device: a two-layer DFlash2 draft stack (residual chaining + final
// norm) matches the host reference.
TEST(BackendTest, DraftStackMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(97);
  constexpr std::size_t kRows = 2, kHidden = 4, kHeads = 2, kKvHeads = 1,
                       kHeadDim = 2, kFfn = 8, kTaps = 2, kGroup = 2,
                       kBlock = 2, kWindow = 2;
  constexpr float kEps = 1e-6f;
  constexpr double kTheta = 10000.0;
  const std::size_t q_dim = kHeads * kHeadDim;
  const std::size_t kv_dim = kKvHeads * kHeadDim;
  const std::size_t proj_n = 2 * kTaps * (kHidden / kGroup);
  auto rnd = [&rng](std::size_t n) {
    std::vector<float> out(n);
    for (auto& v : out) v = DrawValue(rng);
    return out;
  };
  std::vector<std::unique_ptr<tessera::Buffer>> keep;
  auto mk = [&](const std::vector<float>& d) -> tessera::Buffer* {
    auto b = backend->AllocateBuffer(d.size() * 4, MemoryKind::Device);
    backend->CopyH2D(**b, std::span<const std::byte>(
                             reinterpret_cast<const std::byte*>(d.data()),
                             d.size() * 4));
    keep.push_back(std::move(*b));
    return keep.back().get();
  };
  struct Layer {
    std::vector<std::vector<float>> data;
    tessera::spec::DraftLayerWeights ref;
    tessera::spec::DraftLayerBuffers dev;
    Layer() { data.reserve(15); }
  };
  std::vector<Layer> layers(2);
  for (Layer& L : layers) {
    auto add = [&](std::size_t n) -> const std::vector<float>* {
      L.data.push_back(rnd(n));
      return &L.data.back();
    };
    const auto* inorm = add(kHidden);
    const auto* acp = add(proj_n * kHidden);
    const auto* acb = add(2 * kTaps * kHidden);
    const auto* qw = add(q_dim * kHidden);
    const auto* kw = add(kv_dim * kHidden);
    const auto* vw = add(kv_dim * kHidden);
    const auto* ow = add(kHidden * q_dim);
    const auto* qn = add(kHeadDim);
    const auto* kn = add(kHeadDim);
    const auto* pnorm = add(kHidden);
    const auto* mcp = add(proj_n * kHidden);
    const auto* mcb = add(2 * kTaps * kHidden);
    const auto* gw = add(kFfn * kHidden);
    const auto* uw = add(kFfn * kHidden);
    const auto* dw = add(kHidden * kFfn);
    L.ref.input_norm = *inorm;
    L.ref.attn_conv_proj = *acp;
    L.ref.attn_conv_base = *acb;
    L.ref.q_w = *qw;
    L.ref.k_w = *kw;
    L.ref.v_w = *vw;
    L.ref.o_w = *ow;
    L.ref.q_norm_w = *qn;
    L.ref.k_norm_w = *kn;
    L.ref.post_norm = *pnorm;
    L.ref.mlp_conv_proj = *mcp;
    L.ref.mlp_conv_base = *mcb;
    L.ref.gate_w = *gw;
    L.ref.up_w = *uw;
    L.ref.down_w = *dw;
    L.dev.input_norm = mk(*inorm);
    L.dev.attn_conv_proj = mk(*acp);
    L.dev.attn_conv_base = mk(*acb);
    L.dev.q_w = mk(*qw);
    L.dev.k_w = mk(*kw);
    L.dev.v_w = mk(*vw);
    L.dev.o_w = mk(*ow);
    L.dev.q_norm_w = mk(*qn);
    L.dev.k_norm_w = mk(*kn);
    L.dev.post_norm = mk(*pnorm);
    L.dev.mlp_conv_proj = mk(*mcp);
    L.dev.mlp_conv_base = mk(*mcb);
    L.dev.gate_w = mk(*gw);
    L.dev.up_w = mk(*uw);
    L.dev.down_w = mk(*dw);
  }
  const std::vector<float> embed = rnd(kRows * kHidden);
  const std::vector<float> fnormal = rnd(kHidden);
  auto embed_b = mk(embed);
  auto fnormal_b = mk(fnormal);
  auto out_b = backend->AllocateBuffer(embed.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(out_b.has_value());
  std::vector<tessera::spec::DraftLayerBuffers> dev_layers;
  for (Layer& L : layers) dev_layers.push_back(L.dev);
  std::vector<tessera::spec::DraftLayerWeights> ref_layers;
  for (Layer& L : layers) ref_layers.push_back(L.ref);
  auto rms = backend->LoadKernel("rmsnorm", {});
  auto gemm = backend->LoadKernel("gemm_f32", {});
  auto conv = backend->LoadKernel("dflash_conv", {});
  auto rope = backend->LoadKernel("rope", {});
  auto attn = backend->LoadKernel("attention", {});
  auto silu = backend->LoadKernel("silu_mul", {});
  auto add = backend->LoadKernel("add", {});
  ASSERT_TRUE(rms && gemm && conv && rope && attn && silu && add);
  auto device = tessera::spec::DraftStackDevice(
      *backend, **rms, **gemm, **conv, **rope, **attn, **silu, **add,
      *embed_b, dev_layers, *fnormal_b, **out_b, kRows, kHidden, kHeads,
      kKvHeads, kHeadDim, kFfn, kTaps, kGroup, kBlock, kWindow, 0, kTheta,
      kEps);
  ASSERT_TRUE(device.has_value()) << tessera::ToString(device.error());
  backend->Synchronize();
  std::vector<float> got(embed.size());
  backend->CopyD2H(**out_b, reinterpret_cast<std::byte*>(got.data()),
                   got.size() * 4);
  std::vector<float> ref(embed.size());
  ASSERT_TRUE(tessera::spec::DraftStackRef(
                  std::span<const float>(embed), ref_layers,
                  std::span<const float>(fnormal), std::span<float>(ref),
                  kRows, kHidden, kHeads, kKvHeads, kHeadDim, kFfn, kTaps,
                  kGroup, kBlock, kWindow, 0, kTheta, kEps)
                  .has_value());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the DFlash2 candidate-selector block (hidden projection +
// transition edge scores) matches the host reference.
TEST(BackendTest, DraftSelectorMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(98);
  constexpr std::size_t kRows = 2, kHidden = 4, kRank = 3, kVocab = 5,
                       kTopK = 2;
  std::vector<float> hidden(kRows * kHidden), pw(kRank * kHidden);
  std::vector<float> pred(kVocab * kRank), succ(kVocab * kRank);
  std::vector<float> unary(kRows * kTopK);
  for (auto& v : hidden) v = DrawValue(rng);
  for (auto& v : pw) v = DrawValue(rng);
  for (auto& v : pred) v = DrawValue(rng);
  for (auto& v : succ) v = DrawValue(rng);
  for (auto& v : unary) v = DrawValue(rng);
  std::vector<std::int32_t> cand(kRows * kTopK), anchor(kRows);
  std::uniform_int_distribution<std::int32_t> pick(
      0, static_cast<std::int32_t>(kVocab - 1));
  for (auto& v : cand) v = pick(rng);
  for (auto& v : anchor) v = pick(rng);
  auto alloc = [&backend](std::size_t bytes) {
    return backend->AllocateBuffer(bytes, MemoryKind::Device);
  };
  auto upf = [&backend](auto& buf, const std::vector<float>& data) {
    return backend->CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data.data()),
                   data.size() * 4));
  };
  auto h_b = alloc(hidden.size() * 4), pw_b = alloc(pw.size() * 4);
  auto pr_b = alloc(pred.size() * 4), su_b = alloc(succ.size() * 4);
  auto un_b = alloc(unary.size() * 4);
  auto ca_b = alloc(cand.size() * 4), an_b = alloc(anchor.size() * 4);
  auto proj_b = alloc(kRows * kRank * 4);
  auto out_b = alloc(kRows * kTopK * kTopK * 4);
  ASSERT_TRUE(h_b && pw_b && pr_b && su_b && un_b && ca_b && an_b && proj_b &&
              out_b);
  ASSERT_TRUE(upf(h_b, hidden).has_value());
  ASSERT_TRUE(upf(pw_b, pw).has_value());
  ASSERT_TRUE(upf(pr_b, pred).has_value());
  ASSERT_TRUE(upf(su_b, succ).has_value());
  ASSERT_TRUE(upf(un_b, unary).has_value());
  ASSERT_TRUE(backend->CopyH2D(
                  **ca_b, std::span<const std::byte>(
                               reinterpret_cast<const std::byte*>(cand.data()),
                               cand.size() * 4))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(
                  **an_b,
                  std::span<const std::byte>(
                      reinterpret_cast<const std::byte*>(anchor.data()),
                      anchor.size() * 4))
                  .has_value());
  auto gemm = backend->LoadKernel("gemm_f32", {});
  auto edge = backend->LoadKernel("selector_edge_score", {});
  ASSERT_TRUE(gemm.has_value() && edge.has_value());
  auto device = tessera::spec::DraftSelectorDevice(
      *backend, **gemm, **edge, **proj_b, **h_b, **pw_b, **pr_b, **su_b, **ca_b,
      **an_b, **un_b, **out_b, kRows, kHidden, kRank, kVocab, kTopK);
  ASSERT_TRUE(device.has_value()) << tessera::ToString(device.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kRows * kTopK * kTopK * 4);
  ASSERT_TRUE(backend->CopyD2H(**out_b, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(kRows * kTopK * kTopK);
  ASSERT_TRUE(tessera::spec::DraftSelectorRef(
                  std::span<const float>(hidden), std::span<const float>(pw),
                  std::span<const float>(pred), std::span<const float>(succ),
                  std::span<const std::int32_t>(cand),
                  std::span<const std::int32_t>(anchor),
                  std::span<const float>(unary), std::span<float>(ref), kRows,
                  kHidden, kRank, kVocab, kTopK)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the DFlash2 target-hidden fusion (concat n aux hidden tensors,
// then the fc projection) matches the host reference.
TEST(BackendTest, DraftFuseMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(99);
  constexpr std::size_t kN = 2, kRows = 2, kFeatures = 2, kHidden = 3;
  std::vector<float> aux(kN * kRows * kFeatures), fc(kHidden * kN * kFeatures);
  for (auto& v : aux) v = DrawValue(rng);
  for (auto& v : fc) v = DrawValue(rng);
  auto up = [&backend](const std::vector<float>& d) {
    auto b = backend->AllocateBuffer(d.size() * 4, MemoryKind::Device);
    backend->CopyH2D(**b, std::span<const std::byte>(
                              reinterpret_cast<const std::byte*>(d.data()),
                              d.size() * 4));
    return std::move(*b);
  };
  auto aux_b = up(aux), fc_b = up(fc);
  auto scratch_b = backend->AllocateBuffer(kRows * kN * kFeatures * 4,
                                            MemoryKind::Device);
  auto scale_b = backend->AllocateBuffer(kRows * 4, MemoryKind::Device);
  auto out_b = backend->AllocateBuffer(kRows * kHidden * 4, MemoryKind::Device);
  ASSERT_TRUE(scratch_b && scale_b && out_b);
  auto gemm = backend->LoadKernel("gemm_f32", {});
  auto concat = backend->LoadKernel("concat_features", {});
  auto quantize = backend->LoadKernel("quantize_fp8", {});
  ASSERT_TRUE(gemm.has_value() && concat.has_value() && quantize.has_value());
  auto device = tessera::spec::DraftFuseDevice(
      *backend, **gemm, **concat, **quantize, **scratch_b, **scale_b, *aux_b,
      *fc_b, **out_b, kN, kRows, kFeatures, kHidden);
  ASSERT_TRUE(device.has_value()) << tessera::ToString(device.error());
  backend->Synchronize();
  std::vector<float> got(kRows * kHidden);
  backend->CopyD2H(**out_b, reinterpret_cast<std::byte*>(got.data()),
                   got.size() * 4);
  std::vector<float> ref(kRows * kHidden);
  ASSERT_TRUE(tessera::spec::DraftFuseRef(
                  std::span<const float>(aux), std::span<const float>(fc),
                  std::span<float>(ref), kN, kRows, kFeatures, kHidden)
                  .has_value());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the DFlash2 context K/V projection matches the host reference.
TEST(BackendTest, DraftContextKvMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(100);
  constexpr std::size_t kCtx = 3, kHidden = 4, kKvHeads = 2, kHeadDim = 2;
  const std::size_t kv_dim = kKvHeads * kHeadDim;
  auto rnd = [&rng](std::size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = DrawValue(rng);
    return v;
  };
  std::vector<float> ctx = rnd(kCtx * kHidden), hnorm = rnd(kHidden);
  std::vector<float> kp = rnd(kv_dim * kHidden), vp = rnd(kv_dim * kHidden);
  std::vector<float> kn = rnd(kHeadDim);
  auto up = [&backend](const std::vector<float>& d) {
    auto b = backend->AllocateBuffer(d.size() * 4, MemoryKind::Device);
    backend->CopyH2D(**b, std::span<const std::byte>(
                              reinterpret_cast<const std::byte*>(d.data()),
                              d.size() * 4));
    return std::move(*b);
  };
  auto ctx_b = up(ctx), hn_b = up(hnorm), kp_b = up(kp), vp_b = up(vp),
       kn_b = up(kn);
  auto normed_b = backend->AllocateBuffer(kCtx * kHidden * 4, MemoryKind::Device);
  auto k_b = backend->AllocateBuffer(kCtx * kv_dim * 4, MemoryKind::Device);
  auto v_b = backend->AllocateBuffer(kCtx * kv_dim * 4, MemoryKind::Device);
  ASSERT_TRUE(normed_b && k_b && v_b);
  auto rms = backend->LoadKernel("rmsnorm", {});
  auto gemm = backend->LoadKernel("gemm_f32", {});
  auto rope = backend->LoadKernel("rope", {});
  ASSERT_TRUE(rms && gemm && rope);
  auto device = tessera::spec::DraftContextKvDevice(
      *backend, **rms, **gemm, **rope, **normed_b, *ctx_b, *hn_b, *kp_b, *vp_b,
      *kn_b, **k_b, **v_b, kCtx, kHidden, kKvHeads, kHeadDim, 0, 10000.0,
      1e-6f);
  ASSERT_TRUE(device.has_value()) << tessera::ToString(device.error());
  backend->Synchronize();
  std::vector<float> k_got(kCtx * kv_dim), v_got(kCtx * kv_dim);
  backend->CopyD2H(**k_b, reinterpret_cast<std::byte*>(k_got.data()),
                   k_got.size() * 4);
  backend->CopyD2H(**v_b, reinterpret_cast<std::byte*>(v_got.data()),
                   v_got.size() * 4);
  std::vector<float> k_ref(kCtx * kv_dim), v_ref(kCtx * kv_dim);
  ASSERT_TRUE(tessera::spec::DraftContextKvRef(
                  std::span<const float>(ctx), std::span<const float>(hnorm),
                  std::span<const float>(kp), std::span<const float>(vp),
                  std::span<const float>(kn), std::span<float>(k_ref),
                  std::span<float>(v_ref), kCtx, kHidden, kKvHeads, kHeadDim,
                  0, 10000.0, 1e-6f)
                  .has_value());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < k_ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(k_ref[i] - k_got[i]));
    max_abs = std::max(max_abs, std::abs(v_ref[i] - v_got[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: appending target hidden positions incrementally to the draft
// context cache matches one batch fuse plus context K/V reference.
TEST(BackendTest, DraftContextAppendMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(111);
  constexpr std::size_t kN = 2, kFirst = 2, kSecond = 1, kTotal = 3;
  constexpr std::size_t kHidden = 4, kKvHeads = 2, kHeadDim = 2;
  constexpr double kTheta = 10000.0;
  constexpr float kEps = 1e-6f;
  const std::size_t kv_dim = kKvHeads * kHeadDim;
  const std::size_t width = kN * kHidden;
  auto rnd = [&rng](std::size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = DrawValue(rng);
    return v;
  };
  std::vector<float> fc = rnd(kHidden * width);
  std::vector<std::vector<float>> hnorm(kN), kp(kN), vp(kN), kn(kN), cap(kN);
  for (std::size_t i = 0; i < kN; ++i) {
    hnorm[i] = rnd(kHidden);
    kp[i] = rnd(kv_dim * kHidden);
    vp[i] = rnd(kv_dim * kHidden);
    kn[i] = rnd(kHeadDim);
    cap[i] = rnd(kTotal * kHidden);
  }
  auto upload = [&backend](const std::vector<float>& d) {
    auto b = backend->AllocateBuffer(d.size() * 4, MemoryKind::Device);
    backend->CopyH2D(**b, std::span<const std::byte>(
                              reinterpret_cast<const std::byte*>(d.data()),
                              d.size() * 4));
    return std::move(*b);
  };
  auto fc_b = upload(fc);
  std::vector<std::unique_ptr<tessera::Buffer>> hnorm_b(kN), kp_b(kN),
      vp_b(kN), kn_b(kN);
  std::vector<const tessera::Buffer*> hnorm_p(kN), kp_p(kN), vp_p(kN), kn_p(kN);
  for (std::size_t i = 0; i < kN; ++i) {
    hnorm_b[i] = upload(hnorm[i]);
    kp_b[i] = upload(kp[i]);
    vp_b[i] = upload(vp[i]);
    kn_b[i] = upload(kn[i]);
    hnorm_p[i] = hnorm_b[i].get();
    kp_p[i] = kp_b[i].get();
    vp_p[i] = vp_b[i].get();
    kn_p[i] = kn_b[i].get();
  }
  auto rms = backend->LoadKernel("rmsnorm", {});
  auto gemm = backend->LoadKernel("gemm_f32", {});
  auto rope = backend->LoadKernel("rope", {});
  auto concat = backend->LoadKernel("concat_features", {});
  auto quantize = backend->LoadKernel("quantize_fp8", {});
  ASSERT_TRUE(rms && gemm && rope && concat && quantize);
  tessera::spec::DraftContextCache cache;
  const auto append_rows = [&](std::size_t begin, std::size_t rows) {
    std::vector<std::unique_ptr<tessera::Buffer>> parts(kN);
    std::vector<const tessera::Buffer*> part_p(kN);
    for (std::size_t i = 0; i < kN; ++i) {
      std::vector<float> part(cap[i].begin() + begin * kHidden,
                              cap[i].begin() + (begin + rows) * kHidden);
      parts[i] = upload(part);
      part_p[i] = parts[i].get();
    }
    return tessera::spec::DraftContextAppendDevice(
        *backend, **rms, **gemm, **rope, **concat, **quantize, cache, part_p,
        *fc_b, hnorm_p, kp_p, vp_p, kn_p, kN, kHidden, /*row_offset=*/0, rows,
        kHidden, kKvHeads, kHeadDim, kTheta, kEps);
  };
  ASSERT_TRUE(append_rows(0, kFirst).has_value());
  ASSERT_TRUE(append_rows(kFirst, kSecond).has_value());
  EXPECT_EQ(cache.rows, kTotal);
  // Reference: one batch fuse over the concatenated aux, then the context
  // K/V projection for layer 0.
  std::vector<float> aux_ref(kN * kTotal * kHidden);
  for (std::size_t i = 0; i < kN; ++i) {
    for (std::size_t t = 0; t < kTotal; ++t) {
      for (std::size_t f = 0; f < kHidden; ++f) {
        aux_ref[(i * kTotal + t) * kHidden + f] = cap[i][t * kHidden + f];
      }
    }
  }
  std::vector<float> fused(kTotal * kHidden);
  ASSERT_TRUE(tessera::spec::DraftFuseRef(
                  std::span<const float>(aux_ref), std::span<const float>(fc),
                  std::span<float>(fused), kN, kTotal, kHidden, kHidden)
                  .has_value());
  std::vector<float> k_ref(kTotal * kv_dim), v_ref(kTotal * kv_dim);
  ASSERT_TRUE(tessera::spec::DraftContextKvRef(
                  std::span<const float>(fused),
                  std::span<const float>(hnorm[0]),
                  std::span<const float>(kp[0]), std::span<const float>(vp[0]),
                  std::span<const float>(kn[0]), std::span<float>(k_ref),
                  std::span<float>(v_ref), kTotal, kHidden, kKvHeads, kHeadDim,
                  0, kTheta, kEps)
                  .has_value());
  backend->Synchronize();
  std::vector<float> k_got(kTotal * kv_dim), v_got(kTotal * kv_dim);
  backend->CopyD2H(*cache.layers[0].k, reinterpret_cast<std::byte*>(k_got.data()),
                   k_got.size() * 4);
  backend->CopyD2H(*cache.layers[0].v, reinterpret_cast<std::byte*>(v_got.data()),
                   v_got.size() * 4);
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < k_ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(k_ref[i] - k_got[i]));
    max_abs = std::max(max_abs, std::abs(v_ref[i] - v_got[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the DFlash2 attention half with a context K/V prefix (queries
// follow the context in position) matches the host reference.
TEST(BackendTest, DraftAttentionContextMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(101);
  constexpr std::size_t kRows = 2, kHidden = 4, kHeads = 2, kKvHeads = 1,
                       kHeadDim = 2, kTaps = 2, kGroup = 2, kBlock = 2,
                       kCtx = 2;
  constexpr float kEps = 1e-6f;
  constexpr double kTheta = 10000.0;
  const std::size_t q_dim = kHeads * kHeadDim;
  const std::size_t kv_dim = kKvHeads * kHeadDim;
  const std::size_t proj_n = 2 * kTaps * (kHidden / kGroup);
  auto rnd = [&rng](std::size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = DrawValue(rng);
    return v;
  };
  std::vector<float> x = rnd(kRows * kHidden), inorm = rnd(kHidden);
  std::vector<float> pw = rnd(proj_n * kHidden), base = rnd(2 * kTaps * kHidden);
  std::vector<float> qw = rnd(q_dim * kHidden), kw = rnd(kv_dim * kHidden);
  std::vector<float> vw = rnd(kv_dim * kHidden), ow = rnd(kHidden * q_dim);
  std::vector<float> qn = rnd(kHeadDim), kn = rnd(kHeadDim);
  std::vector<float> ck = rnd(kCtx * kv_dim), cv = rnd(kCtx * kv_dim);
  auto up = [&backend](const std::vector<float>& d) {
    auto b = backend->AllocateBuffer(d.size() * 4, MemoryKind::Device);
    backend->CopyH2D(**b, std::span<const std::byte>(
                              reinterpret_cast<const std::byte*>(d.data()),
                              d.size() * 4));
    return std::move(*b);
  };
  auto x_b = up(x), inorm_b = up(inorm), pw_b = up(pw), base_b = up(base);
  auto qw_b = up(qw), kw_b = up(kw), vw_b = up(vw), ow_b = up(ow);
  auto qn_b = up(qn), kn_b = up(kn), ck_b = up(ck), cv_b = up(cv);
  auto out_b = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  auto mk = [&backend](std::size_t n) {
    return backend->AllocateBuffer(n * 4, MemoryKind::Device);
  };
  auto xn_b = mk(kRows * kHidden), proj_b = mk(kRows * proj_n),
       h1_b = mk(kRows * kHidden), q_b = mk(kRows * q_dim),
       k_b = mk(kRows * kv_dim), v_b = mk(kRows * kv_dim),
       attn_b = mk(kRows * q_dim), oproj_b = mk(kRows * kHidden),
       side_b = mk(kTaps * kHidden);
  ASSERT_TRUE(out_b && xn_b && proj_b && h1_b && q_b && k_b && v_b && attn_b &&
              oproj_b && side_b);
  auto rms = backend->LoadKernel("rmsnorm", {});
  auto gemm = backend->LoadKernel("gemm_f32", {});
  auto conv = backend->LoadKernel("dflash_conv", {});
  auto rope = backend->LoadKernel("rope", {});
  auto attn = backend->LoadKernel("attention", {});
  ASSERT_TRUE(rms && gemm && conv && rope && attn);
  auto device = tessera::spec::DraftAttentionDevice(
      *backend, **rms, **gemm, **conv, **rope, **attn, **xn_b, **proj_b,
      **h1_b, **q_b, **k_b, **v_b, **attn_b, **oproj_b, **side_b, *x_b,
      *inorm_b, *pw_b, *base_b, *qw_b, *kw_b, *vw_b, *ow_b, *qn_b, *kn_b,
      **out_b, kRows, kHidden, kHeads, kKvHeads, kHeadDim, kTaps, kGroup,
      kBlock, 0, 0, kTheta, kEps, &*ck_b, &*cv_b, kCtx);
  ASSERT_TRUE(device.has_value()) << tessera::ToString(device.error());
  backend->Synchronize();
  std::vector<float> got(x.size());
  backend->CopyD2H(**out_b, reinterpret_cast<std::byte*>(got.data()),
                   got.size() * 4);
  std::vector<float> ref(x.size());
  ASSERT_TRUE(tessera::spec::DraftAttentionRef(
                  std::span<const float>(x), std::span<const float>(inorm),
                  std::span<const float>(pw), std::span<const float>(base),
                  std::span<const float>(qw), std::span<const float>(kw),
                  std::span<const float>(vw), std::span<const float>(ow),
                  std::span<const float>(qn), std::span<const float>(kn),
                  std::span<float>(ref), kRows, kHidden, kHeads, kKvHeads,
                  kHeadDim, kTaps, kGroup, kBlock, 0, 0, kTheta, kEps,
                  std::span<const float>(ck), std::span<const float>(cv), kCtx)
                  .has_value());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the full DFlash2 draft block (fc fusion of the target hidden,
// mask-query stack with context, final norm, logits head) matches the host
// reference.
TEST(BackendTest, DraftBlockMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(102);
  constexpr std::size_t kRows = 2, kCtx = 3, kHidden = 4, kN = 2,
                       kFeatures = 2, kVocab = 5, kHeads = 2, kKvHeads = 1,
                       kHeadDim = 2, kFfn = 8, kTaps = 2, kGroup = 2,
                       kBlock = 2, kWindow = 2;
  constexpr float kEps = 1e-6f;
  constexpr double kTheta = 10000.0;
  const std::size_t q_dim = kHeads * kHeadDim;
  const std::size_t kv_dim = kKvHeads * kHeadDim;
  const std::size_t proj_n = 2 * kTaps * (kHidden / kGroup);
  auto rnd = [&rng](std::size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = DrawValue(rng);
    return v;
  };
  std::vector<std::unique_ptr<tessera::Buffer>> keep;
  auto mk = [&](const std::vector<float>& d) -> tessera::Buffer* {
    auto b = backend->AllocateBuffer(d.size() * 4, MemoryKind::Device);
    backend->CopyH2D(**b, std::span<const std::byte>(
                             reinterpret_cast<const std::byte*>(d.data()),
                             d.size() * 4));
    keep.push_back(std::move(*b));
    return keep.back().get();
  };
  struct Layer {
    std::vector<std::vector<float>> data;
    tessera::spec::DraftLayerWeights ref;
    tessera::spec::DraftLayerBuffers dev;
    Layer() { data.reserve(16); }
  };
  std::vector<Layer> layers(2);
  for (Layer& L : layers) {
    auto add = [&](std::size_t n) -> const std::vector<float>* {
      L.data.push_back(rnd(n));
      return &L.data.back();
    };
    const auto* inorm = add(kHidden);
    const auto* acp = add(proj_n * kHidden);
    const auto* acb = add(2 * kTaps * kHidden);
    const auto* qw = add(q_dim * kHidden);
    const auto* kw = add(kv_dim * kHidden);
    const auto* vw = add(kv_dim * kHidden);
    const auto* ow = add(kHidden * q_dim);
    const auto* qn = add(kHeadDim);
    const auto* kn = add(kHeadDim);
    const auto* pnorm = add(kHidden);
    const auto* mcp = add(proj_n * kHidden);
    const auto* mcb = add(2 * kTaps * kHidden);
    const auto* gw = add(kFfn * kHidden);
    const auto* uw = add(kFfn * kHidden);
    const auto* dw = add(kHidden * kFfn);
    const auto* hnorm = add(kHidden);
    L.ref = {*inorm, *acp, *acb, *qw, *kw,      *vw,   *ow, *qn,
             *kn,    *pnorm, *mcp, *mcb, *gw,    *uw,   *dw, *hnorm};
    L.dev.input_norm = mk(*inorm);
    L.dev.attn_conv_proj = mk(*acp);
    L.dev.attn_conv_base = mk(*acb);
    L.dev.q_w = mk(*qw);
    L.dev.k_w = mk(*kw);
    L.dev.v_w = mk(*vw);
    L.dev.o_w = mk(*ow);
    L.dev.q_norm_w = mk(*qn);
    L.dev.k_norm_w = mk(*kn);
    L.dev.post_norm = mk(*pnorm);
    L.dev.mlp_conv_proj = mk(*mcp);
    L.dev.mlp_conv_base = mk(*mcb);
    L.dev.gate_w = mk(*gw);
    L.dev.up_w = mk(*uw);
    L.dev.down_w = mk(*dw);
    L.dev.hidden_norm = mk(*hnorm);
  }
  const std::vector<float> mask = rnd(kRows * kHidden);
  const std::vector<float> aux = rnd(kN * kCtx * kFeatures);
  const std::vector<float> fc = rnd(kHidden * kN * kFeatures);
  const std::vector<float> fnormal = rnd(kHidden);
  const std::vector<float> outw = rnd(kVocab * kHidden);
  auto mask_b = mk(mask), aux_b = mk(aux), fc_b = mk(fc), fnormal_b = mk(fnormal);
  auto outw_b = mk(outw);
  auto logits_b = backend->AllocateBuffer(kRows * kVocab * 4,
                                          MemoryKind::Device);
  ASSERT_TRUE(logits_b.has_value());
  std::vector<tessera::spec::DraftLayerBuffers> dev_layers;
  for (Layer& L : layers) dev_layers.push_back(L.dev);
  std::vector<tessera::spec::DraftLayerWeights> ref_layers;
  for (Layer& L : layers) ref_layers.push_back(L.ref);
  auto rms = backend->LoadKernel("rmsnorm", {});
  auto gemm = backend->LoadKernel("gemm_f32", {});
  auto conv = backend->LoadKernel("dflash_conv", {});
  auto rope = backend->LoadKernel("rope", {});
  auto attn = backend->LoadKernel("attention", {});
  auto silu = backend->LoadKernel("silu_mul", {});
  auto add = backend->LoadKernel("add", {});
  auto concat = backend->LoadKernel("concat_features", {});
  auto quantize = backend->LoadKernel("quantize_fp8", {});
  ASSERT_TRUE(rms && gemm && conv && rope && attn && silu && add && concat &&
              quantize);
  auto device = tessera::spec::DraftBlockDevice(
      *backend, **rms, **gemm, **gemm, **conv, **rope, **attn, **silu, **add,
      **concat, **quantize, *mask_b, aux_b, fc_b, dev_layers, *fnormal_b,
      *outw_b, **logits_b,
      kRows, kCtx, kHidden, kN, kFeatures, kVocab, kHeads, kKvHeads, kHeadDim,
      kFfn, kTaps, kGroup, kBlock, kWindow, 0, kTheta, kEps);
  ASSERT_TRUE(device.has_value()) << tessera::ToString(device.error());
  backend->Synchronize();
  std::vector<float> got(kRows * kVocab);
  backend->CopyD2H(**logits_b, reinterpret_cast<std::byte*>(got.data()),
                   got.size() * 4);
  std::vector<float> ref(kRows * kVocab);
  ASSERT_TRUE(tessera::spec::DraftBlockRef(
                  std::span<const float>(mask), std::span<const float>(aux),
                  std::span<const float>(fc), ref_layers,
                  std::span<const float>(fnormal), std::span<const float>(outw),
                  std::span<float>(ref), kRows, kCtx, kHidden, kN, kFeatures,
                  kVocab, kHeads, kKvHeads, kHeadDim, kFfn, kTaps, kGroup,
                  kBlock, kWindow, 0, kTheta, kEps)
                  .has_value());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Host: bf16 and block-scaled fp8 tensors decode to fp32.

// Host: the DFlash2 candidate extraction returns the top-K logits per row
// in descending order.

// Host: every row ranks the whole vocabulary. A candidate id >= top_k on a
// row after the first catches a shrinking reusable index vector (the old bug
// left later rows searching only ids 0..top_k-1).

// Device: fp16 keys/values attention (kv_f16) matches the fp16 host
// reference (both decode the same half bits).
TEST(BackendTest, AttentionF16DeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(103);
  constexpr std::size_t kM = 2, kN = 4, kHeads = 4, kKvHeads = 2, kDim = 16;
  constexpr std::uint64_t kQBase = 3;
  std::vector<float> q(kM * kHeads * kDim);
  std::vector<float> k(kN * kKvHeads * kDim), v(kN * kKvHeads * kDim);
  for (auto& x : q) x = DrawValue(rng);
  for (auto& x : k) x = DrawValue(rng);
  for (auto& x : v) x = DrawValue(rng);
  std::vector<std::byte> k16(k.size() * 2), v16(v.size() * 2);
  const auto encode = [](float value, std::byte* out) {
    const std::uint16_t half = core::Fp16FromFloat(value);
    out[0] = static_cast<std::byte>(half & 0xFF);
    out[1] = static_cast<std::byte>((half >> 8) & 0xFF);
  };
  for (std::size_t i = 0; i < k.size(); ++i) {
    encode(k[i], &k16[i * 2]);
    encode(v[i], &v16[i * 2]);
  }
  auto q_buf = backend->AllocateBuffer(q.size() * 4, MemoryKind::Device);
  auto k_buf = backend->AllocateBuffer(k16.size(), MemoryKind::Device);
  auto v_buf = backend->AllocateBuffer(v16.size(), MemoryKind::Device);
  auto out_buf = backend->AllocateBuffer(q.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(q_buf && k_buf && v_buf && out_buf);
  const auto upload = [&backend](auto& buf, const void* data, std::size_t bytes) {
    return backend->CopyH2D(
        **buf, std::span<const std::byte>(
                   reinterpret_cast<const std::byte*>(data), bytes));
  };
  ASSERT_TRUE(upload(q_buf, q.data(), q.size() * 4).has_value());
  ASSERT_TRUE(upload(k_buf, k16.data(), k16.size()).has_value());
  ASSERT_TRUE(upload(v_buf, v16.data(), v16.size()).has_value());
  auto kernel = backend->LoadKernel("attention", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = kM * kHeads;
  launch.block_x = 256;
  launch.buffers = {(*q_buf).get(), (*k_buf).get(), (*v_buf).get(),
                    (*out_buf).get()};
  launch.scalars = {kM, kN, kHeads, kKvHeads, kDim, kQBase, 0, 1, 1};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kHeads * kDim * 4);
  ASSERT_TRUE(backend->CopyD2H(**out_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(kM * kHeads * kDim);
  ASSERT_TRUE(core::AttentionRefF16(
                  std::span<const float>(q), std::span<const std::byte>(k16),
                  std::span<const std::byte>(v16), std::span<float>(ref), kM,
                  kN, kHeads, kKvHeads, kDim, kQBase, 0)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the fp32 -> fp16 cast matches the host reference (packed two per
// word).
TEST(BackendTest, CastF32F16MatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(104);
  constexpr std::size_t kN = 8;
  std::vector<float> in(kN);
  for (auto& v : in) v = DrawValue(rng);
  auto in_buf = backend->AllocateBuffer(in.size() * 4, MemoryKind::Device);
  auto out_buf = backend->AllocateBuffer(in.size() * 2, MemoryKind::Device);
  ASSERT_TRUE(in_buf && out_buf);
  ASSERT_TRUE(backend->CopyH2D(
                  **in_buf, std::span<const std::byte>(
                                reinterpret_cast<const std::byte*>(in.data()),
                                in.size() * 4))
                  .has_value());
  auto kernel = backend->LoadKernel("cast_f32_f16", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = (kN / 2 + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*in_buf).get(), (*out_buf).get()};
  launch.scalars = {kN};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> got(in.size() * 2);
  ASSERT_TRUE(backend->CopyD2H(**out_buf, got.data(), got.size()).has_value());
  std::vector<std::byte> ref(in.size() * 2);
  ASSERT_TRUE(core::CastF32F16Ref(std::span<const float>(in),
                                  std::span<std::byte>(ref))
                  .has_value());
  EXPECT_EQ(got, ref);
}

// Device: symmetric int8 quantization matches the host reference.
// Device: bf16 rounding (round to nearest even) matches the host reference
// and keeps exact bf16 values unchanged.
TEST(BackendTest, RoundBf16MatchesRef) {
  std::vector<float> known = {1.0f, -2.5f, 0.0f, 448.0f};
  std::vector<float> known_ref = known;
  ASSERT_TRUE(core::RoundBf16Ref(std::span<float>(known_ref)).has_value());
  for (std::size_t i = 0; i < known.size(); ++i) {
    EXPECT_FLOAT_EQ(known_ref[i], known[i]);
  }
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(112);
  constexpr std::size_t kN = 8;
  std::vector<float> in(kN);
  for (auto& v : in) v = DrawValue(rng) * 4.0f;
  std::vector<float> ref = in;
  ASSERT_TRUE(core::RoundBf16Ref(std::span<float>(ref)).has_value());
  auto buf = backend->AllocateBuffer(kN * 4, MemoryKind::Device);
  ASSERT_TRUE(buf.has_value());
  ASSERT_TRUE(backend->CopyH2D(
                  **buf, std::span<const std::byte>(
                             reinterpret_cast<const std::byte*>(in.data()),
                             in.size() * 4))
                  .has_value());
  auto kernel = backend->LoadKernel("round_bf16", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = (kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {buf->get()};
  launch.scalars = {kN};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<float> got(kN);
  backend->CopyD2H(**buf, reinterpret_cast<std::byte*>(got.data()),
                   got.size() * 4);
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(got[i], ref[i]) << "element " << i;
  }
}

TEST(BackendTest, QuantizeQ8MatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(105);
  constexpr std::size_t kRows = 2, kCols = 8;
  std::vector<float> in(kRows * kCols);
  for (auto& v : in) v = DrawValue(rng);
  auto in_buf = backend->AllocateBuffer(in.size() * 4, MemoryKind::Device);
  auto out_buf = backend->AllocateBuffer(in.size(), MemoryKind::Device);
  auto scale_buf = backend->AllocateBuffer(kRows * 4, MemoryKind::Device);
  ASSERT_TRUE(in_buf && out_buf && scale_buf);
  ASSERT_TRUE(backend->CopyH2D(
                  **in_buf, std::span<const std::byte>(
                                reinterpret_cast<const std::byte*>(in.data()),
                                in.size() * 4))
                  .has_value());
  auto kernel = backend->LoadKernel("quantize_q8", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = kRows;
  launch.block_x = 256;
  launch.buffers = {(*in_buf).get(), (*out_buf).get(), (*scale_buf).get()};
  launch.scalars = {kRows, kCols};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> got(in.size()), ref(in.size());
  std::vector<float> got_scale(kRows), ref_scale(kRows);
  backend->CopyD2H(**out_buf, got.data(), got.size());
  backend->CopyD2H(**scale_buf, reinterpret_cast<std::byte*>(got_scale.data()),
                   got_scale.size() * 4);
  ASSERT_TRUE(core::QuantizeQ8Ref(std::span<const float>(in),
                                  std::span<std::byte>(ref),
                                  std::span<float>(ref_scale), kRows, kCols)
                  .has_value());
  EXPECT_EQ(got, ref);
  for (std::size_t r = 0; r < kRows; ++r) {
    EXPECT_FLOAT_EQ(got_scale[r], ref_scale[r]);
  }
}

// Device: the row-parallel KV quantizer with a row wider than the workgroup
// (cols 1024 over 256 lanes) and multiple rows equals the host reference.
// This covers the strided within-row loop the small-shape tests never reach.

// Device: the per-token FP8 E4M3 quantize-dequantize (the W4A8 activation
// QDQ) matches the host reference in place.
TEST(BackendTest, QuantizeFp8MatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(107);
  constexpr std::size_t kRows = 2, kCols = 6;
  std::vector<float> in(kRows * kCols);
  for (auto& v : in) v = DrawValue(rng);
  auto data_buf = backend->AllocateBuffer(in.size() * 4, MemoryKind::Device);
  auto scale_buf = backend->AllocateBuffer(kRows * 4, MemoryKind::Device);
  ASSERT_TRUE(data_buf && scale_buf);
  ASSERT_TRUE(backend->CopyH2D(
                  **data_buf, std::span<const std::byte>(
                                  reinterpret_cast<const std::byte*>(in.data()),
                                  in.size() * 4))
                  .has_value());
  auto kernel = backend->LoadKernel("quantize_fp8", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = kRows;
  launch.block_x = 256;
  launch.buffers = {(*data_buf).get(), (*scale_buf).get()};
  launch.scalars = {kRows, kCols};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<float> got(kRows * kCols), got_scale(kRows);
  backend->CopyD2H(**data_buf, reinterpret_cast<std::byte*>(got.data()),
                   got.size() * 4);
  backend->CopyD2H(**scale_buf, reinterpret_cast<std::byte*>(got_scale.data()),
                   got_scale.size() * 4);
  std::vector<float> ref = in;
  std::vector<float> ref_scale(kRows);
  ASSERT_TRUE(core::QuantizeFp8Ref(std::span<float>(ref),
                                   std::span<float>(ref_scale), kRows, kCols)
                  .has_value());
  for (std::size_t i = 0; i < ref.size(); ++i) {
    EXPECT_FLOAT_EQ(got[i], ref[i]) << "element " << i;
  }
  for (std::size_t r = 0; r < kRows; ++r) {
    EXPECT_FLOAT_EQ(got_scale[r], ref_scale[r]);
  }
}

// Host: the FP8 per-token QDQ reproduces representable values exactly, and
// a zero row uses the smallest dynamic scale.

// Device: the W4A8 target projection quantizes an MXFP4 activation in
// place only when TESSERA_MXFP4_W4A8 is set; other dtypes are untouched.

// Device: int8 keys/values attention matches the host reference.
TEST(BackendTest, AttentionQ8DeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(106);
  constexpr std::size_t kM = 2, kN = 4, kHeads = 4, kKvHeads = 2, kDim = 16;
  constexpr std::uint64_t kQBase = 3;
  std::vector<float> q(kM * kHeads * kDim);
  std::vector<float> k(kN * kKvHeads * kDim), v(kN * kKvHeads * kDim);
  for (auto& x : q) x = DrawValue(rng);
  for (auto& x : k) x = DrawValue(rng);
  for (auto& x : v) x = DrawValue(rng);
  const std::size_t kv_dim = kKvHeads * kDim;
  std::vector<std::byte> k8(k.size()), v8(v.size());
  std::vector<float> ks(kN), vs(kN);
  ASSERT_TRUE(core::QuantizeQ8Ref(std::span<const float>(k),
                                  std::span<std::byte>(k8),
                                  std::span<float>(ks), kN, kv_dim)
                  .has_value());
  ASSERT_TRUE(core::QuantizeQ8Ref(std::span<const float>(v),
                                  std::span<std::byte>(v8),
                                  std::span<float>(vs), kN, kv_dim)
                  .has_value());
  auto up = [&backend](const void* data, std::size_t bytes) {
    auto b = backend->AllocateBuffer(bytes, MemoryKind::Device);
    backend->CopyH2D(**b, std::span<const std::byte>(
                              reinterpret_cast<const std::byte*>(data), bytes));
    return std::move(*b);
  };
  auto q_buf = up(q.data(), q.size() * 4);
  auto k_buf = up(k8.data(), k8.size());
  auto v_buf = up(v8.data(), v8.size());
  auto ks_buf = up(ks.data(), ks.size() * 4);
  auto vs_buf = up(vs.data(), vs.size() * 4);
  auto out_buf = backend->AllocateBuffer(q.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(out_buf.has_value());
  auto kernel = backend->LoadKernel("attention_q8", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = kM * kHeads;
  launch.block_x = 256;
  launch.buffers = {q_buf.get(), k_buf.get(), v_buf.get(), ks_buf.get(),
                    vs_buf.get(), out_buf->get()};
  launch.scalars = {kM, kN, kHeads, kKvHeads, kDim, kQBase, 0, 0};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(q.size() * 4);
  ASSERT_TRUE(backend->CopyD2H(**out_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(q.size());
  ASSERT_TRUE(core::AttentionQ8Ref(
                  std::span<const float>(q), std::span<const std::byte>(k8),
                  std::span<const std::byte>(v8), std::span<const float>(ks),
                  std::span<const float>(vs), std::span<float>(ref), kM, kN,
                  kHeads, kKvHeads, kDim, kQBase, 0)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the split-N (flash-decoding) quantized attention over a long
// single-token key range matches the host reference. This is the
// latency-bound path the single-token decode uses; n >= 1024 triggers it
// in the engine, so the test pins n = 1024 across 16 chunks.
TEST(BackendTest, AttentionQ8SplitMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(126);
  constexpr std::size_t kM = 1, kN = 1024, kHeads = 4, kKvHeads = 2;
  constexpr std::size_t kDim = 16, kSplit = 16;
  constexpr std::uint64_t kQBase = kN - 1;
  std::vector<float> q(kM * kHeads * kDim);
  std::vector<float> k(kN * kKvHeads * kDim), v(kN * kKvHeads * kDim);
  for (auto& x : q) x = DrawValue(rng);
  for (auto& x : k) x = DrawValue(rng);
  for (auto& x : v) x = DrawValue(rng);
  const std::size_t kv_dim = kKvHeads * kDim;
  std::vector<std::byte> k8(k.size()), v8(v.size());
  std::vector<float> ks(kN), vs(kN);
  ASSERT_TRUE(core::QuantizeQ8Ref(std::span<const float>(k),
                                  std::span<std::byte>(k8),
                                  std::span<float>(ks), kN, kv_dim)
                  .has_value());
  ASSERT_TRUE(core::QuantizeQ8Ref(std::span<const float>(v),
                                  std::span<std::byte>(v8),
                                  std::span<float>(vs), kN, kv_dim)
                  .has_value());
  auto up = [&backend](const void* data, std::size_t bytes) {
    auto b = backend->AllocateBuffer(bytes, MemoryKind::Device);
    backend->CopyH2D(**b, std::span<const std::byte>(
                              reinterpret_cast<const std::byte*>(data), bytes));
    return std::move(*b);
  };
  auto q_buf = up(q.data(), q.size() * 4);
  auto k_buf = up(k8.data(), k8.size());
  auto v_buf = up(v8.data(), v8.size());
  auto ks_buf = up(ks.data(), ks.size() * 4);
  auto vs_buf = up(vs.data(), vs.size() * 4);
  const std::size_t part = kM * kHeads * kSplit;
  auto pacc_buf = backend->AllocateBuffer(part * kDim * 4, MemoryKind::Device);
  auto pmax_buf = backend->AllocateBuffer(part * 4, MemoryKind::Device);
  auto psum_buf = backend->AllocateBuffer(part * 4, MemoryKind::Device);
  auto out_buf = backend->AllocateBuffer(q.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(pacc_buf && pmax_buf && psum_buf && out_buf);
  auto split_kernel = backend->LoadKernel("attention_q8_split", {});
  auto combine_kernel = backend->LoadKernel("attention_combine", {});
  ASSERT_TRUE(split_kernel.has_value());
  ASSERT_TRUE(combine_kernel.has_value());
  tessera::KernelLaunch split;
  split.grid_x = kM * kHeads * kSplit;
  split.block_x = 256;
  split.buffers = {q_buf.get(),  k_buf.get(),  v_buf.get(), ks_buf.get(),
                   vs_buf.get(), pacc_buf->get(), pmax_buf->get(),
                   psum_buf->get()};
  split.scalars = {kM, kN, kHeads, kKvHeads, kDim, kQBase, 0, 0, kSplit};
  ASSERT_TRUE(backend->LaunchKernel(**split_kernel, split).has_value());
  tessera::KernelLaunch combine;
  combine.grid_x = kM * kHeads;
  combine.block_x = 256;
  combine.buffers = {pacc_buf->get(), pmax_buf->get(), psum_buf->get(),
                     out_buf->get()};
  combine.scalars = {kM, kHeads, kDim, kSplit};
  ASSERT_TRUE(backend->LaunchKernel(**combine_kernel, combine).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(q.size() * 4);
  ASSERT_TRUE(backend->CopyD2H(**out_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(q.size());
  ASSERT_TRUE(core::AttentionQ8Ref(
                  std::span<const float>(q), std::span<const std::byte>(k8),
                  std::span<const std::byte>(v8), std::span<const float>(ks),
                  std::span<const float>(vs), std::span<float>(ref), kM, kN,
                  kHeads, kKvHeads, kDim, kQBase, 0)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: FP8 E4M3 packing (the fp8 KV quantize) matches the host
// reference byte for byte.
TEST(BackendTest, QuantizeFp8PackMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(109);
  constexpr std::size_t kRows = 2, kCols = 8;
  std::vector<float> in(kRows * kCols);
  for (auto& v : in) v = DrawValue(rng);
  auto in_buf = backend->AllocateBuffer(in.size() * 4, MemoryKind::Device);
  auto out_buf = backend->AllocateBuffer(in.size(), MemoryKind::Device);
  auto scale_buf = backend->AllocateBuffer(kRows * 4, MemoryKind::Device);
  ASSERT_TRUE(in_buf && out_buf && scale_buf);
  ASSERT_TRUE(backend->CopyH2D(
                  **in_buf, std::span<const std::byte>(
                                reinterpret_cast<const std::byte*>(in.data()),
                                in.size() * 4))
                  .has_value());
  auto kernel = backend->LoadKernel("quantize_fp8_pack", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = kRows;
  launch.block_x = 256;
  launch.buffers = {(*in_buf).get(), (*out_buf).get(), (*scale_buf).get()};
  launch.scalars = {kRows, kCols};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> got(in.size()), ref(in.size());
  std::vector<float> got_scale(kRows), ref_scale(kRows);
  backend->CopyD2H(**out_buf, got.data(), got.size());
  backend->CopyD2H(**scale_buf, reinterpret_cast<std::byte*>(got_scale.data()),
                   got_scale.size() * 4);
  ASSERT_TRUE(core::QuantizeFp8PackRef(std::span<const float>(in),
                                       std::span<std::byte>(ref),
                                       std::span<float>(ref_scale), kRows, kCols)
                  .has_value());
  EXPECT_EQ(got, ref);
  for (std::size_t r = 0; r < kRows; ++r) {
    EXPECT_FLOAT_EQ(got_scale[r], ref_scale[r]);
  }
}

// Device: FP8 E4M3 GQA attention (the fp8 KV read) matches the host
// reference within per-backend tolerance.
TEST(BackendTest, AttentionFp8DeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(110);
  constexpr std::size_t kM = 2, kN = 4, kHeads = 4, kKvHeads = 2, kDim = 16;
  constexpr std::uint64_t kQBase = 3;
  std::vector<float> q(kM * kHeads * kDim);
  std::vector<float> k(kN * kKvHeads * kDim), v(kN * kKvHeads * kDim);
  for (auto& x : q) x = DrawValue(rng);
  for (auto& x : k) x = DrawValue(rng);
  for (auto& x : v) x = DrawValue(rng);
  const std::size_t kv_dim = kKvHeads * kDim;
  std::vector<std::byte> k8(k.size()), v8(v.size());
  std::vector<float> ks(kN), vs(kN);
  ASSERT_TRUE(core::QuantizeFp8PackRef(std::span<const float>(k),
                                       std::span<std::byte>(k8),
                                       std::span<float>(ks), kN, kv_dim)
                  .has_value());
  ASSERT_TRUE(core::QuantizeFp8PackRef(std::span<const float>(v),
                                       std::span<std::byte>(v8),
                                       std::span<float>(vs), kN, kv_dim)
                  .has_value());
  auto up = [&backend](const void* data, std::size_t bytes) {
    auto b = backend->AllocateBuffer(bytes, MemoryKind::Device);
    backend->CopyH2D(**b, std::span<const std::byte>(
                              reinterpret_cast<const std::byte*>(data), bytes));
    return std::move(*b);
  };
  auto q_buf = up(q.data(), q.size() * 4);
  auto k_buf = up(k8.data(), k8.size());
  auto v_buf = up(v8.data(), v8.size());
  auto ks_buf = up(ks.data(), ks.size() * 4);
  auto vs_buf = up(vs.data(), vs.size() * 4);
  auto out_buf = backend->AllocateBuffer(q.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(out_buf.has_value());
  auto kernel = backend->LoadKernel("attention_fp8", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = kM * kHeads;
  launch.block_x = 256;
  launch.buffers = {q_buf.get(), k_buf.get(), v_buf.get(), ks_buf.get(),
                    vs_buf.get(), out_buf->get()};
  launch.scalars = {kM, kN, kHeads, kKvHeads, kDim, kQBase, 0, 2};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(q.size() * 4);
  ASSERT_TRUE(backend->CopyD2H(**out_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(q.size());
  ASSERT_TRUE(core::AttentionFp8Ref(
                  std::span<const float>(q), std::span<const std::byte>(k8),
                  std::span<const std::byte>(v8), std::span<const float>(ks),
                  std::span<const float>(vs), std::span<float>(ref), kM, kN,
                  kHeads, kKvHeads, kDim, kQBase, 0)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: fp32 attention over multiple 256-key tiles matches the host
// reference. The tiled kernel reduces each tile once (single thread),
// so a long context must agree across tile boundaries.
TEST(BackendTest, AttentionMultiTileMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(121);
  constexpr std::size_t kM = 2, kN = 600, kHeads = 4, kKvHeads = 2, kDim = 16;
  constexpr std::uint64_t kQBase = 598;
  std::vector<float> q(kM * kHeads * kDim), k(kN * kKvHeads * kDim);
  std::vector<float> v(kN * kKvHeads * kDim);
  for (auto& x : q) x = DrawValue(rng);
  for (auto& x : k) x = DrawValue(rng);
  for (auto& x : v) x = DrawValue(rng);
  auto up = [&backend](const void* data, std::size_t bytes) {
    auto b = backend->AllocateBuffer(bytes, MemoryKind::Device);
    backend->CopyH2D(**b, std::span<const std::byte>(
                              reinterpret_cast<const std::byte*>(data), bytes));
    return std::move(*b);
  };
  auto q_buf = up(q.data(), q.size() * 4);
  auto k_buf = up(k.data(), k.size() * 4);
  auto v_buf = up(v.data(), v.size() * 4);
  auto out_buf = backend->AllocateBuffer(q.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(out_buf.has_value());
  auto kernel = backend->LoadKernel("attention", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = kM * kHeads;
  launch.block_x = 256;
  launch.buffers = {q_buf.get(), k_buf.get(), v_buf.get(), out_buf->get()};
  launch.scalars = {kM, kN, kHeads, kKvHeads, kDim, kQBase, 0, 0, 1};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(q.size() * 4);
  ASSERT_TRUE(backend->CopyD2H(**out_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(q.size());
  ASSERT_TRUE(core::AttentionRef(std::span<const float>(q),
                                 std::span<const float>(k),
                                 std::span<const float>(v),
                                 std::span<float>(ref), kM, kN, kHeads, kKvHeads,
                                 kDim, kQBase)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: the split-N (flash-decoding) fp32 attention over a long
// single-token key range matches the host reference. Pins the fp32
// split/combine path used by the single-token decode.
TEST(BackendTest, AttentionSplitMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(127);
  constexpr std::size_t kM = 1, kN = 1024, kHeads = 4, kKvHeads = 2;
  constexpr std::size_t kDim = 16, kSplit = 16;
  constexpr std::uint64_t kQBase = kN - 1;
  std::vector<float> q(kM * kHeads * kDim), k(kN * kKvHeads * kDim);
  std::vector<float> v(kN * kKvHeads * kDim);
  for (auto& x : q) x = DrawValue(rng);
  for (auto& x : k) x = DrawValue(rng);
  for (auto& x : v) x = DrawValue(rng);
  auto up = [&backend](const void* data, std::size_t bytes) {
    auto b = backend->AllocateBuffer(bytes, MemoryKind::Device);
    backend->CopyH2D(**b, std::span<const std::byte>(
                              reinterpret_cast<const std::byte*>(data), bytes));
    return std::move(*b);
  };
  auto q_buf = up(q.data(), q.size() * 4);
  auto k_buf = up(k.data(), k.size() * 4);
  auto v_buf = up(v.data(), v.size() * 4);
  const std::size_t part = kM * kHeads * kSplit;
  auto pacc_buf = backend->AllocateBuffer(part * kDim * 4, MemoryKind::Device);
  auto pmax_buf = backend->AllocateBuffer(part * 4, MemoryKind::Device);
  auto psum_buf = backend->AllocateBuffer(part * 4, MemoryKind::Device);
  auto out_buf = backend->AllocateBuffer(q.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(pacc_buf && pmax_buf && psum_buf && out_buf);
  auto split_kernel = backend->LoadKernel("attention_split", {});
  auto combine_kernel = backend->LoadKernel("attention_combine", {});
  ASSERT_TRUE(split_kernel.has_value());
  ASSERT_TRUE(combine_kernel.has_value());
  tessera::KernelLaunch split;
  split.grid_x = kM * kHeads * kSplit;
  split.block_x = 256;
  split.buffers = {q_buf.get(),  k_buf.get(),   v_buf.get(),   pacc_buf->get(),
                   pmax_buf->get(), psum_buf->get()};
  split.scalars = {kM, kN, kHeads, kKvHeads, kDim,
                   kQBase, 0, 0, 1, kSplit};
  ASSERT_TRUE(backend->LaunchKernel(**split_kernel, split).has_value());
  tessera::KernelLaunch combine;
  combine.grid_x = kM * kHeads;
  combine.block_x = 256;
  combine.buffers = {pacc_buf->get(), pmax_buf->get(), psum_buf->get(),
                     out_buf->get()};
  combine.scalars = {kM, kHeads, kDim, kSplit};
  ASSERT_TRUE(backend->LaunchKernel(**combine_kernel, combine).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(q.size() * 4);
  ASSERT_TRUE(backend->CopyD2H(**out_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(q.size());
  ASSERT_TRUE(core::AttentionRef(std::span<const float>(q),
                                 std::span<const float>(k),
                                 std::span<const float>(v),
                                 std::span<float>(ref), kM, kN, kHeads, kKvHeads,
                                 kDim, kQBase)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: symmetric 4-bit quantization matches the host reference.
TEST(BackendTest, QuantizeQ4MatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(107);
  constexpr std::size_t kRows = 2, kCols = 16;
  std::vector<float> in(kRows * kCols);
  for (auto& v : in) v = DrawValue(rng);
  auto in_buf = backend->AllocateBuffer(in.size() * 4, MemoryKind::Device);
  auto out_buf = backend->AllocateBuffer(in.size() / 2, MemoryKind::Device);
  auto scale_buf = backend->AllocateBuffer(kRows * 4, MemoryKind::Device);
  ASSERT_TRUE(in_buf && out_buf && scale_buf);
  ASSERT_TRUE(backend->CopyH2D(
                  **in_buf, std::span<const std::byte>(
                                reinterpret_cast<const std::byte*>(in.data()),
                                in.size() * 4))
                  .has_value());
  auto kernel = backend->LoadKernel("quantize_q4", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = kRows;
  launch.block_x = 256;
  launch.buffers = {(*in_buf).get(), (*out_buf).get(), (*scale_buf).get()};
  launch.scalars = {kRows, kCols};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> got(in.size() / 2), ref(in.size() / 2);
  std::vector<float> got_scale(kRows), ref_scale(kRows);
  backend->CopyD2H(**out_buf, got.data(), got.size());
  backend->CopyD2H(**scale_buf, reinterpret_cast<std::byte*>(got_scale.data()),
                   got_scale.size() * 4);
  ASSERT_TRUE(core::QuantizeQ4Ref(std::span<const float>(in),
                                  std::span<std::byte>(ref),
                                  std::span<float>(ref_scale), kRows, kCols)
                  .has_value());
  EXPECT_EQ(got, ref);
  for (std::size_t r = 0; r < kRows; ++r) {
    EXPECT_FLOAT_EQ(got_scale[r], ref_scale[r]);
  }
}

// Device: 4-bit keys/values attention matches the host reference.
TEST(BackendTest, AttentionQ4DeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(108);
  constexpr std::size_t kM = 2, kN = 4, kHeads = 4, kKvHeads = 2, kDim = 16;
  constexpr std::uint64_t kQBase = 3;
  std::vector<float> q(kM * kHeads * kDim);
  std::vector<float> k(kN * kKvHeads * kDim), v(kN * kKvHeads * kDim);
  for (auto& x : q) x = DrawValue(rng);
  for (auto& x : k) x = DrawValue(rng);
  for (auto& x : v) x = DrawValue(rng);
  const std::size_t kv_dim = kKvHeads * kDim;
  std::vector<std::byte> k4(k.size() / 2), v4(v.size() / 2);
  std::vector<float> ks(kN), vs(kN);
  ASSERT_TRUE(core::QuantizeQ4Ref(std::span<const float>(k),
                                  std::span<std::byte>(k4),
                                  std::span<float>(ks), kN, kv_dim)
                  .has_value());
  ASSERT_TRUE(core::QuantizeQ4Ref(std::span<const float>(v),
                                  std::span<std::byte>(v4),
                                  std::span<float>(vs), kN, kv_dim)
                  .has_value());
  auto up = [&backend](const void* data, std::size_t bytes) {
    auto b = backend->AllocateBuffer(bytes, MemoryKind::Device);
    backend->CopyH2D(**b, std::span<const std::byte>(
                              reinterpret_cast<const std::byte*>(data), bytes));
    return std::move(*b);
  };
  auto q_buf = up(q.data(), q.size() * 4);
  auto k_buf = up(k4.data(), k4.size());
  auto v_buf = up(v4.data(), v4.size());
  auto ks_buf = up(ks.data(), ks.size() * 4);
  auto vs_buf = up(vs.data(), vs.size() * 4);
  auto out_buf = backend->AllocateBuffer(q.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(out_buf.has_value());
  auto kernel = backend->LoadKernel("attention_q4", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = kM * kHeads;
  launch.block_x = 256;
  launch.buffers = {q_buf.get(), k_buf.get(), v_buf.get(), ks_buf.get(),
                    vs_buf.get(), out_buf->get()};
  launch.scalars = {kM, kN, kHeads, kKvHeads, kDim, kQBase, 0, 1};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(q.size() * 4);
  ASSERT_TRUE(backend->CopyD2H(**out_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(q.size());
  ASSERT_TRUE(core::AttentionQ4Ref(
                  std::span<const float>(q), std::span<const std::byte>(k4),
                  std::span<const std::byte>(v4), std::span<const float>(ks),
                  std::span<const float>(vs), std::span<float>(ref), kM, kN,
                  kHeads, kKvHeads, kDim, kQBase, 0)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: LayerNorm matches the host reference.
TEST(BackendTest, LayerNormDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(109);
  constexpr std::size_t kRows = 3, kCols = 8;
  std::vector<float> x(kRows * kCols), w(kCols), b(kCols);
  for (auto& v : x) v = DrawValue(rng);
  for (auto& v : w) v = DrawValue(rng);
  for (auto& v : b) v = DrawValue(rng);
  auto up = [&backend](const std::vector<float>& d) {
    auto buf = backend->AllocateBuffer(d.size() * 4, MemoryKind::Device);
    backend->CopyH2D(**buf, std::span<const std::byte>(
                               reinterpret_cast<const std::byte*>(d.data()),
                               d.size() * 4));
    return std::move(*buf);
  };
  auto x_b = up(x), w_b = up(w), b_b = up(b);
  auto y_b = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(y_b.has_value());
  auto kernel = backend->LoadKernel("layernorm", {});
  ASSERT_TRUE(kernel.has_value());
  const float eps = 1e-5f;
  std::uint32_t bits = 0;
  std::memcpy(&bits, &eps, sizeof(bits));
  tessera::KernelLaunch launch;
  launch.grid_x = (kRows + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {x_b.get(), w_b.get(), b_b.get(), y_b->get()};
  launch.scalars = {kRows, kCols, bits};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(x.size() * 4);
  ASSERT_TRUE(backend->CopyD2H(**y_b, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(x.size());
  ASSERT_TRUE(core::LayerNormRef(std::span<const float>(x),
                                 std::span<const float>(w),
                                 std::span<const float>(b),
                                 std::span<float>(ref), kRows, kCols, eps)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: GELU matches the host reference.
TEST(BackendTest, GeluDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(110);
  constexpr std::size_t kN = 16;
  std::vector<float> x(kN);
  for (auto& v : x) v = DrawValue(rng);
  auto x_b = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  auto y_b = backend->AllocateBuffer(x.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(x_b && y_b);
  ASSERT_TRUE(backend->CopyH2D(
                  **x_b, std::span<const std::byte>(
                             reinterpret_cast<const std::byte*>(x.data()),
                             x.size() * 4))
                  .has_value());
  auto kernel = backend->LoadKernel("gelu", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = (kN + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*x_b).get(), (*y_b).get()};
  launch.scalars = {kN};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<float> got(kN), ref(kN);
  backend->CopyD2H(**y_b, reinterpret_cast<std::byte*>(got.data()),
                   got.size() * 4);
  ASSERT_TRUE(core::GeluRef(std::span<const float>(x), std::span<float>(ref),
                            kN)
                  .has_value());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: image normalize + patchify matches the host reference.

// Device: non-causal attention (causal flag 0) matches the reference.
TEST(BackendTest, AttentionNonCausalDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(112);
  constexpr std::size_t kM = 4, kN = 4, kHeads = 4, kKvHeads = 2, kDim = 16;
  std::vector<float> q(kM * kHeads * kDim), k(kN * kKvHeads * kDim);
  std::vector<float> v(kN * kKvHeads * kDim);
  for (auto& x : q) x = DrawValue(rng);
  for (auto& x : k) x = DrawValue(rng);
  for (auto& x : v) x = DrawValue(rng);
  auto up = [&backend](const std::vector<float>& d) {
    auto b = backend->AllocateBuffer(d.size() * 4, MemoryKind::Device);
    backend->CopyH2D(**b, std::span<const std::byte>(
                              reinterpret_cast<const std::byte*>(d.data()),
                              d.size() * 4));
    return std::move(*b);
  };
  auto q_buf = up(q), k_buf = up(k), v_buf = up(v);
  auto out_buf = backend->AllocateBuffer(q.size() * 4, MemoryKind::Device);
  ASSERT_TRUE(out_buf.has_value());
  auto kernel = backend->LoadKernel("attention", {});
  ASSERT_TRUE(kernel.has_value());
  tessera::KernelLaunch launch;
  launch.grid_x = kM * kHeads;
  launch.block_x = 256;
  launch.buffers = {q_buf.get(), k_buf.get(), v_buf.get(), out_buf->get()};
  launch.scalars = {kM, kN, kHeads, kKvHeads, kDim, 0, 0, 0, 0};
  ASSERT_TRUE(backend->LaunchKernel(**kernel, launch).has_value());
  backend->Synchronize();
  std::vector<std::byte> readback(q.size() * 4);
  ASSERT_TRUE(backend->CopyD2H(**out_buf, readback.data(), readback.size())
                  .has_value());
  std::vector<float> ref(q.size());
  ASSERT_TRUE(core::AttentionRef(std::span<const float>(q),
                                 std::span<const float>(k),
                                 std::span<const float>(v),
                                 std::span<float>(ref), kM, kN, kHeads, kKvHeads,
                                 kDim, 0, 0, false)
                  .has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  const AttentionTolerance tol = AttentionToleranceFor(backend->Name());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  }
  EXPECT_LE(max_abs, tol.abs)
      << "backend " << backend->Name() << " max_abs " << max_abs;
}

// Device: a CLIP vision transformer block matches the host reference.

// Device: the CLIP vision stack (patch embed + blocks + post-norm) matches
// the host reference.

// Device: the Qwen3VL vision merger (spatial merge + mm.0 + GELU + mm.2)
// matches the host reference.

// The warp-per-output GEMV family ("_vec"): one 32-lane group per output,
// eight per workgroup, vectorized weight reads and a shuffle reduction.
// ProjectDevice derives the launch from the kernel id, so one helper
// exercises the grid rule the engine uses for all four formats.
using GemmRefFn = std::expected<void, StatusCode> (*)(
    std::span<const float>, std::span<const std::byte>, std::span<float>,
    std::size_t, std::size_t, std::size_t);

static void ExpectGemvVecMatchesRef(std::unique_ptr<Backend>& backend,
                                    const char* kernel_name,
                                    const std::vector<std::byte>& w,
                                    std::size_t kM, std::size_t kN,
                                    std::size_t kK, GemmRefFn reference) {
  auto kernel = backend->LoadKernel(kernel_name, {});
  if (!kernel && kernel.error() == StatusCode::UnsupportedFeature) {
    GTEST_SKIP() << "no " << kernel_name << " kernel on " << backend->Name();
  }
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  std::mt19937 rng(97531 + static_cast<unsigned>(w.size()));
  // Small activations: the warp tree changes the summation order, so keep
  // the absolute rounding error far below the per-backend tolerance while
  // the relative error the tolerance is for stays the same.
  std::vector<float> a(kM * kK);
  for (auto& v : a) {
    v = DrawValue(rng) * 0.01f;
  }
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && w_buf.has_value() && c_buf.has_value());
  ASSERT_TRUE(backend->CopyH2D(**a_buf, std::span<const std::byte>(
                                           reinterpret_cast<const std::byte*>(a.data()),
                                           a.size() * 4))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(**w_buf, std::span<const std::byte>(w))
                  .has_value());
  // ProjectGemvDevice adds the split-K path (its reduce pass and partial
  // scratch) the engine uses for narrow projections.
  tessera::models::qwen3_5::Qwen35State state;
  auto launch = tessera::models::qwen3_5::ProjectGemvDevice(
      *backend, state, **kernel, **a_buf, **w_buf, **c_buf, kM, kN, kK);
  ASSERT_TRUE(launch.has_value()) << tessera::ToString(launch.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kN * 4);
  ASSERT_TRUE(
      backend->CopyD2H(**c_buf, readback.data(), readback.size()).has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  std::vector<float> ref(kM * kN);
  ASSERT_TRUE(reference(std::span<const float>(a),
                        std::span<const std::byte>(w), std::span<float>(ref),
                        kM, kN, kK)
                  .has_value());
  const GemmTolerance tol = ToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
    rel = std::max(rel,
                   std::abs(got[i] - ref[i]) / std::max(1.0f, std::abs(ref[i])));
  }
  EXPECT_LE(max_abs, tol.abs)
      << kernel_name << " on " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(rel, tol.rel)
      << kernel_name << " on " << backend->Name() << " max_rel " << rel;
}

// Random blocks with a finite block scale d (and, where the format has
// one, a small dmin so the min term stays in range).
static std::vector<std::byte> RandomVecBlocks(std::mt19937& rng,
                                              std::size_t rows,
                                              std::size_t k,
                                              std::size_t block_bytes,
                                              std::size_t d_off) {
  const std::size_t row_bytes = (k / 256) * block_bytes;
  std::vector<std::byte> w = RandomBlocks(rng, rows, row_bytes, d_off);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < k / 256; ++b) {
      std::byte* base = w.data() + r * row_bytes + b * block_bytes;
      base[d_off] = std::byte{0};
      base[d_off + 1] = std::byte{0x20};
    }
  }
  return w;
}





// A wide-enough output count keeps split-K off (one workgroup per output):
// the same kernel and the plain grid must match the reference.

// Split-K: a narrow projection (few output blocks) splits the k range
// across grid_y chunks and sums the partials in a second pass.

// One check for the multi-row GEMV family ("_rows"): one warp per column
// decodes each block once for all m rows. ProjectGemvRowsDevice owns the
// split-K pass the engine uses for narrow output counts.
static void ExpectGemvRowsMatchesRef(std::unique_ptr<Backend>& backend,
                                     const char* kernel_name,
                                     const std::vector<std::byte>& w,
                                     std::size_t kM, std::size_t kN,
                                     std::size_t kK, GemmRefFn reference) {
  auto kernel = backend->LoadKernel(kernel_name, {});
  if (!kernel && kernel.error() == StatusCode::UnsupportedFeature) {
    GTEST_SKIP() << "no " << kernel_name << " kernel on " << backend->Name();
  }
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  std::mt19937 rng(86420 + static_cast<unsigned>(w.size()) + kM);
  std::vector<float> a(kM * kK);
  for (auto& v : a) {
    v = DrawValue(rng) * 0.01f;
  }
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && w_buf.has_value() && c_buf.has_value());
  ASSERT_TRUE(backend->CopyH2D(**a_buf, std::span<const std::byte>(
                                           reinterpret_cast<const std::byte*>(a.data()),
                                           a.size() * 4))
                  .has_value());
  ASSERT_TRUE(backend->CopyH2D(**w_buf, std::span<const std::byte>(w))
                  .has_value());
  tessera::models::qwen3_5::Qwen35State state;
  auto launch = tessera::models::qwen3_5::ProjectGemvRowsDevice(
      *backend, state, **kernel, **a_buf, **w_buf, **c_buf, kM, kN, kK);
  ASSERT_TRUE(launch.has_value()) << tessera::ToString(launch.error());
  backend->Synchronize();
  std::vector<std::byte> readback(kM * kN * 4);
  ASSERT_TRUE(
      backend->CopyD2H(**c_buf, readback.data(), readback.size()).has_value());
  const auto* got = reinterpret_cast<const float*>(readback.data());
  std::vector<float> ref(kM * kN);
  ASSERT_TRUE(reference(std::span<const float>(a),
                        std::span<const std::byte>(w), std::span<float>(ref),
                        kM, kN, kK)
                  .has_value());
  const GemmTolerance tol = ToleranceFor(backend->Name());
  float max_abs = 0.0f;
  float rel = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
    rel = std::max(rel,
                   std::abs(got[i] - ref[i]) / std::max(1.0f, std::abs(ref[i])));
  }
  EXPECT_LE(max_abs, tol.abs)
      << kernel_name << " on " << backend->Name() << " max_abs " << max_abs;
  EXPECT_LE(rel, tol.rel)
      << kernel_name << " on " << backend->Name() << " max_rel " << rel;
}




