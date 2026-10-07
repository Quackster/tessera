#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <span>
#include <string_view>
#include <vector>

#include "core/numerics/attention.hpp"
#include "core/numerics/conv.hpp"
#include "core/numerics/gemm.hpp"
#include "core/numerics/norm.hpp"
#include "core/numerics/quant.hpp"
#include "test_helpers.hpp"
#include "tessera/backend.hpp"
#include "tessera/types.hpp"

using tessera::Backend;
using tessera::CreateBackend;
using tessera::MemoryKind;
using tessera::StatusCode;
using tessera::kQ4KBlockElements;
using tessera::testing::BlockU16;
using tessera::testing::DrawValue;
using tessera::testing::GemmTolerance;
using tessera::testing::QuantizeRows;
using tessera::testing::TestGetScaleMin;
using tessera::testing::ToleranceFor;
namespace core = tessera::core;

namespace {

// A device is required for these tests; skip cleanly without one.
void MakeBackendOrSkip(std::unique_ptr<Backend>& backend) {
  backend = CreateBackend();
  if (!backend) {
    GTEST_SKIP() << "no backend in this build";
  }
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

TEST(BackendTest, BufferRoundTrip) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  constexpr std::size_t kSize = 4096;
  auto device = backend->AllocateBuffer(kSize, MemoryKind::Device);
  auto host = backend->AllocateBuffer(kSize, MemoryKind::HostVisible);
  ASSERT_TRUE(device.has_value()) << tessera::ToString(device.error());
  ASSERT_TRUE(host.has_value()) << tessera::ToString(host.error());

  // Fixed pattern (no RNG, no wall clock): 0xAB, 0xCD, ...
  std::vector<std::byte> pattern(kSize);
  for (std::size_t i = 0; i < kSize; ++i) {
    pattern[i] = static_cast<std::byte>((i * 31 + 0xAB) & 0xFF);
  }
  auto upload = backend->CopyH2D(**device, std::span<const std::byte>(pattern));
  ASSERT_TRUE(upload.has_value()) << tessera::ToString(upload.error());
  std::vector<std::byte> readback(kSize);
  auto download = backend->CopyD2H(
      **device, readback.data(), readback.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  backend->Synchronize();
  EXPECT_EQ(pattern, readback);

  // The host-visible buffer is mapped for direct host access.
  ASSERT_NE((**host).HostMap(), nullptr);
  auto upload_host =
      backend->CopyH2D(**host, std::span<const std::byte>(pattern));
  ASSERT_TRUE(upload_host.has_value()) << tessera::ToString(upload_host.error());
  auto read_mapped =
      std::equal(static_cast<const std::byte*>((**host).HostMap()),
                 static_cast<const std::byte*>((**host).HostMap()) + kSize,
                 pattern.begin());
  EXPECT_TRUE(read_mapped);
}

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
  EXPECT_FLOAT_EQ(core::Fp8E4M3ToFloat(0x38), 0.5f);
  EXPECT_FLOAT_EQ(core::Fp8E4M3ToFloat(0x40), 1.0f);
  EXPECT_FLOAT_EQ(core::Fp8E4M3ToFloat(0xBC), -0.75f);
  EXPECT_FLOAT_EQ(core::Fp8E4M3ToFloat(0xC4), -1.5f);
  EXPECT_FLOAT_EQ(core::Fp8E4M3ToFloat(0x76), 112.0f);
  EXPECT_FLOAT_EQ(core::Fp8E4M3ToFloat(0x7E), 448.0f);
  EXPECT_TRUE(std::isnan(core::Fp8E4M3ToFloat(0x7F)));
  EXPECT_TRUE(std::isnan(core::Fp8E4M3ToFloat(0xFF)));
  EXPECT_EQ(core::Fp32ToFp8E4M3Bits(1.0f), 0x40);
  EXPECT_EQ(core::Fp32ToFp8E4M3Bits(-0.0f), 0x80);
  EXPECT_EQ(core::Fp32ToFp8E4M3Bits(448.0f), 0x7E);
  EXPECT_EQ(core::Fp32ToFp8E4M3Bits(1e30f), 0x7F);
  const float grid[7] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};
  for (std::uint32_t i = 0; i < 7; ++i) {
    EXPECT_FLOAT_EQ(core::F4E2M1ToFloat(static_cast<std::uint8_t>(i)),
                                       grid[i])
        << "nibble " << i;
    EXPECT_FLOAT_EQ(core::F4E2M1ToFloat(static_cast<std::uint8_t>(8 | i)),
                                       -grid[i])
        << "nibble " << (8 | i);
    EXPECT_EQ(core::Fp32ToF4E2M1Nibble(grid[i]),
              static_cast<std::uint8_t>(i));
  }
  EXPECT_TRUE(std::isnan(core::F4E2M1ToFloat(7)));
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
  constexpr float kGrid[7] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};
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
            kGrid[(j + t) % 7] * (t % 3 == 0 ? -1.0f : 1.0f));
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
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  auto s_buf = backend->AllocateBuffer(s.size(), MemoryKind::Device);
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
  ASSERT_TRUE(upload(s_buf, s, 1).has_value());

  auto kernel = backend->LoadKernel("gemm_mxfp4", {});
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

// Host: the fp GEMM references reject malformed shapes.
TEST(BackendTest, FpGemmRefsRejectBadArgs) {
  std::vector<float> a(2 * 64, 0.5f);
  std::vector<std::byte> w(4 * 64, std::byte{0x40});
  std::vector<float> s(4, 1.0f);
  std::vector<float> c(2 * 4);
  auto bad = core::GemmFp8Ref(
      std::span<const float>(a), std::span<const std::byte>(w),
      std::span<const float>(s), std::span<float>(c), 2, 4, 0);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  std::vector<float> short_s(3, 1.0f);
  bad = core::GemmFp8Ref(
      std::span<const float>(a), std::span<const std::byte>(w),
      std::span<const float>(short_s), std::span<float>(c), 2, 4, 64);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  std::vector<std::byte> wm(4 * 32, std::byte{0});
  std::vector<std::byte> sm(4 * 2, std::byte{127});
  auto bad_mx = core::GemmMxFp4Ref(
      std::span<const float>(a), std::span<const std::byte>(wm),
      std::span<const std::byte>(sm), std::span<float>(c), 2, 4, 100);
  ASSERT_FALSE(bad_mx.has_value());
  EXPECT_EQ(bad_mx.error(), StatusCode::InvalidArgument);
  auto ok_mx = core::GemmMxFp4Ref(
      std::span<const float>(a), std::span<const std::byte>(wm),
      std::span<const std::byte>(sm), std::span<float>(c), 2, 4, 64);
  ASSERT_TRUE(ok_mx.has_value()) << tessera::ToString(ok_mx.error());
}

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

// Device: IQ3_S GEMM matches the host reference (2 x 8 outputs).
TEST(BackendTest, GemmIq3SDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(60);
  constexpr std::size_t kM = 2;
  constexpr std::size_t kN = 8;
  constexpr std::size_t kK = 256;
  std::vector<float> a(kM * kK);
  for (auto& v : a) {
    v = DrawValue(rng);
  }
  std::vector<std::byte> w = RandomBlocks(rng, kN, 110, 0);
  auto a_buf = backend->AllocateBuffer(a.size() * 4, MemoryKind::Device);
  auto w_buf = backend->AllocateBuffer(w.size(), MemoryKind::Device);
  auto c_buf = backend->AllocateBuffer(kM * kN * 4, MemoryKind::Device);
  ASSERT_TRUE(a_buf.has_value() && w_buf.has_value() && c_buf.has_value());
  auto up_a = backend->CopyH2D(**a_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(a.data()), a.size() * 4));
  auto up_w = backend->CopyH2D(**w_buf, std::span<const std::byte>(w));
  ASSERT_TRUE(up_a.has_value() && up_w.has_value());
  auto kernel = backend->LoadKernel("gemm_iq3s", {});
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
  auto ref_status = core::GemmIq3SRef(
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
TEST(BackendTest, QuantGemmRejectsBadContract) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto q5k = backend->LoadKernel("gemm_q5k", {});
  ASSERT_TRUE(q5k.has_value()) << tessera::ToString(q5k.error());
  auto q6k = backend->LoadKernel("gemm_q6k", {});
  ASSERT_TRUE(q6k.has_value()) << tessera::ToString(q6k.error());
  auto iq4nl = backend->LoadKernel("gemm_iq4nl", {});
  ASSERT_TRUE(iq4nl.has_value()) << tessera::ToString(iq4nl.error());
  const std::vector<const tessera::Buffer*> two_buffers{nullptr, nullptr};
  const std::vector<const tessera::Buffer*> three_buffers{
      nullptr, nullptr, nullptr};
  tessera::KernelLaunch launch;
  launch.buffers = two_buffers;
  launch.scalars = {2, 8, 256};
  auto short_buffers = backend->LaunchKernel(**q5k, launch);
  ASSERT_FALSE(short_buffers.has_value());
  EXPECT_EQ(short_buffers.error(), StatusCode::InvalidArgument);
  launch.buffers = three_buffers;
  launch.scalars = {2, 8, 100};
  auto bad_k = backend->LaunchKernel(**q6k, launch);
  ASSERT_FALSE(bad_k.has_value());
  EXPECT_EQ(bad_k.error(), StatusCode::InvalidArgument);
  launch.scalars = {2, 8, 100};
  auto bad_iq = backend->LaunchKernel(**iq4nl, launch);
  ASSERT_FALSE(bad_iq.has_value());
  EXPECT_EQ(bad_iq.error(), StatusCode::InvalidArgument);
}

// Device: the fp GEMM contracts reject bad shapes.
TEST(BackendTest, FpGemmRejectsBadContract) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto fp8 = backend->LoadKernel("gemm_fp8", {});
  ASSERT_TRUE(fp8.has_value()) << tessera::ToString(fp8.error());
  auto mxfp4 = backend->LoadKernel("gemm_mxfp4", {});
  ASSERT_TRUE(mxfp4.has_value()) << tessera::ToString(mxfp4.error());
  const std::vector<const tessera::Buffer*> three_buffers{
      nullptr, nullptr, nullptr};
  const std::vector<const tessera::Buffer*> four_buffers{
      nullptr, nullptr, nullptr, nullptr};
  tessera::KernelLaunch launch;
  launch.buffers = three_buffers;
  launch.scalars = {1, 4, 64};
  auto short_buffers = backend->LaunchKernel(**fp8, launch);
  ASSERT_FALSE(short_buffers.has_value());
  EXPECT_EQ(short_buffers.error(), StatusCode::InvalidArgument);
  launch.buffers = four_buffers;
  launch.scalars = {1, 4, 100};
  auto bad_k = backend->LaunchKernel(**mxfp4, launch);
  ASSERT_FALSE(bad_k.has_value());
  EXPECT_EQ(bad_k.error(), StatusCode::InvalidArgument);
}

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
}

// Host: the K-quant and IQ GEMM references reject malformed shapes.
TEST(BackendTest, QuantGemmRefsRejectBadArgs) {
  std::vector<float> a(2 * 256, 0.5f);
  std::vector<float> c(2 * 4);
  std::vector<std::byte> w5(4 * 176, std::byte{0});
  auto bad = core::GemmQ5KRef(std::span<const float>(a),
                              std::span<const std::byte>(w5),
                              std::span<float>(c), 2, 4, 100);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  std::vector<std::byte> w6(4 * 210, std::byte{0});
  bad = core::GemmQ6KRef(std::span<const float>(a),
                         std::span<const std::byte>(w6),
                         std::span<float>(c), 0, 4, 256);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  std::vector<std::byte> w3(3 * 110, std::byte{0});
  bad = core::GemmQ3KRef(std::span<const float>(a),
                         std::span<const std::byte>(w3),
                         std::span<float>(c), 2, 4, 256);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  std::vector<float> a32(2 * 32, 0.5f);
  std::vector<std::byte> wnl(4 * 18, std::byte{0});
  bad = core::GemmIq4NlRef(std::span<const float>(a32),
                           std::span<const std::byte>(wnl),
                           std::span<float>(c), 2, 4, 100);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  std::vector<std::byte> wxs(4 * 136 - 1, std::byte{0});
  bad = core::GemmIq4XsRef(std::span<const float>(a),
                           std::span<const std::byte>(wxs),
                           std::span<float>(c), 2, 4, 256);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  std::vector<std::byte> ws3(4 * 110, std::byte{0});
  std::vector<float> short_c(7);
  bad = core::GemmIq3SRef(std::span<const float>(a),
                          std::span<const std::byte>(ws3),
                          std::span<float>(short_c), 2, 4, 256);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
}

TEST(BackendTest, CopyD2HAtReadsSlice) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  constexpr std::size_t kSize = 64;
  auto device = backend->AllocateBuffer(kSize, MemoryKind::Device);
  ASSERT_TRUE(device.has_value()) << tessera::ToString(device.error());
  std::vector<std::byte> pattern(kSize);
  for (std::size_t i = 0; i < kSize; ++i) {
    pattern[i] = static_cast<std::byte>(i);
  }
  auto upload = backend->CopyH2D(**device, std::span<const std::byte>(pattern));
  ASSERT_TRUE(upload.has_value()) << tessera::ToString(upload.error());
  std::vector<std::byte> slice(16);
  auto download =
      backend->CopyD2HAt(**device, 16, slice.data(), slice.size());
  ASSERT_TRUE(download.has_value()) << tessera::ToString(download.error());
  for (std::size_t i = 0; i < slice.size(); ++i) {
    EXPECT_EQ(slice[i], static_cast<std::byte>(16 + i)) << "byte " << i;
  }
  // Past-the-end reads are rejected at the boundary.
  auto over = backend->CopyD2HAt(**device, 64, slice.data(), 1);
  ASSERT_FALSE(over.has_value());
  EXPECT_EQ(over.error(), StatusCode::InvalidArgument);
  auto spanning = backend->CopyD2HAt(**device, 56, slice.data(), 16);
  ASSERT_FALSE(spanning.has_value());
  EXPECT_EQ(spanning.error(), StatusCode::InvalidArgument);
}

TEST(BackendTest, AllocateZeroBytes) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto result = backend->AllocateBuffer(0, MemoryKind::Device);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), StatusCode::InvalidArgument);
}

TEST(BackendTest, OversizedH2D) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto device = backend->AllocateBuffer(4096, MemoryKind::Device);
  ASSERT_TRUE(device.has_value()) << tessera::ToString(device.error());
  std::vector<std::byte> big(8192, std::byte{0});
  auto result =
      backend->CopyH2D(**device, std::span<const std::byte>(big));
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), StatusCode::InvalidArgument);
}

TEST(BackendTest, OversizedD2H) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto device = backend->AllocateBuffer(4096, MemoryKind::Device);
  ASSERT_TRUE(device.has_value()) << tessera::ToString(device.error());
  std::vector<std::byte> big(8192, std::byte{0});
  auto result =
      backend->CopyD2H(**device, big.data(), big.size());
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), StatusCode::InvalidArgument);
}

TEST(BackendTest, LoadKernelUnknownBuiltIn) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto result = backend->LoadKernel("no_such_kernel", {});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), StatusCode::UnsupportedFeature);
}

TEST(BackendTest, LoadKernelMalformedCode) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::vector<std::byte> garbage = {std::byte{0x00}, std::byte{0x01},
                                   std::byte{0x02}, std::byte{0x03}};
  auto result =
      backend->LoadKernel("fill", std::span<const std::byte>(garbage));
  ASSERT_FALSE(result.has_value());
  // The vulkan backend parses the module; rocm has no module loading.
  if (backend->Name() == "vulkan") {
    EXPECT_EQ(result.error(), StatusCode::MalformedFile);
  } else {
    EXPECT_EQ(result.error(), StatusCode::UnsupportedFeature);
  }
}

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

TEST(BackendTest, LaunchKernelZeroBlock) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto kernel = backend->LoadKernel("fill", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.block_x = 0;
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), StatusCode::InvalidArgument);
}

TEST(BackendTest, LaunchFillMissingArgs) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto kernel = backend->LoadKernel("fill", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  // Valid shape, but the fill contract wants 1 buffer and 2 scalars.
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), StatusCode::InvalidArgument);
}

// The "gemm_q4k" contract (backend.hpp) checked on the device: k must
// be a positive multiple of 256; n and m must be positive.
TEST(BackendTest, GemmQ4KRejectsBadContract) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto kernel = backend->LoadKernel("gemm_q4k", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  const std::vector<const tessera::Buffer*> two_buffers{nullptr, nullptr};
  const std::vector<const tessera::Buffer*> three_buffers{
      nullptr, nullptr, nullptr};

  tessera::KernelLaunch launch;
  // k is not a multiple of 256.
  launch.buffers = three_buffers;
  launch.scalars = {100, 1, 1};
  auto bad_k = backend->LaunchKernel(**kernel, launch);
  ASSERT_FALSE(bad_k.has_value());
  EXPECT_EQ(bad_k.error(), StatusCode::InvalidArgument);
  // k is zero.
  launch.scalars = {0, 1, 1};
  auto zero_k = backend->LaunchKernel(**kernel, launch);
  ASSERT_FALSE(zero_k.has_value());
  EXPECT_EQ(zero_k.error(), StatusCode::InvalidArgument);
  // n is zero.
  launch.scalars = {256, 0, 1};
  auto zero_n = backend->LaunchKernel(**kernel, launch);
  ASSERT_FALSE(zero_n.has_value());
  EXPECT_EQ(zero_n.error(), StatusCode::InvalidArgument);
  // Two buffers instead of three.
  launch.buffers = two_buffers;
  launch.scalars = {256, 1, 1};
  auto short_buffers = backend->LaunchKernel(**kernel, launch);
  ASSERT_FALSE(short_buffers.has_value());
  EXPECT_EQ(short_buffers.error(), StatusCode::InvalidArgument);
  // Two scalars instead of three.
  launch.buffers = three_buffers;
  launch.scalars = {256, 1};
  auto short_scalars = backend->LaunchKernel(**kernel, launch);
  ASSERT_FALSE(short_scalars.has_value());
  EXPECT_EQ(short_scalars.error(), StatusCode::InvalidArgument);
}

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
TEST(BackendTest, GemmQ4KRefInvalidArgs) {
  std::vector<float> a(2 * 256);
  std::vector<std::byte> w(2 * 1 * core::kQ4KBlockBytes);
  std::vector<float> c(2 * 2);
  auto bad = core::GemmQ4KRef(std::span<const float>(a),
                             std::span<const std::byte>(w),
                             std::span<float>(c), 2, 2, 255);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  // k not a multiple of 256.
  bad = core::GemmQ4KRef(std::span<const float>(a),
                        std::span<const std::byte>(w),
                        std::span<float>(c), 2, 2, 100);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  // k is zero.
  bad = core::GemmQ4KRef(std::span<const float>(a),
                        std::span<const std::byte>(w),
                        std::span<float>(c), 2, 2, 0);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  // Wrong activation count.
  std::vector<float> short_a(256);
  bad = core::GemmQ4KRef(std::span<const float>(short_a),
                        std::span<const std::byte>(w),
                        std::span<float>(c), 2, 2, 256);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  // Wrong weight size (one block short).
  std::vector<std::byte> short_w(core::kQ4KBlockBytes);
  bad = core::GemmQ4KRef(std::span<const float>(a),
                        std::span<const std::byte>(short_w),
                        std::span<float>(c), 2, 2, 256);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
}

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
  launch.scalars = {kK, kN, kM};
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
  launch.grid_x = (kM * kHeads * kDim + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*q_buf).get(), (*k_buf).get(), (*v_buf).get(),
                    (*out_buf).get()};
  launch.scalars = {kM, kN, kHeads, kKvHeads, kDim, kQBase};
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
TEST(BackendTest, AttentionRefsRejectBadArgs) {
  std::vector<float> io(2 * 2 * 8, 1.0f);
  auto bad = core::RopeRef(std::span<float>(io), 0, 2, 8, 8, 0, 1e4);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  bad = core::RopeRef(std::span<float>(io), 2, 2, 8, 16, 0, 1e4);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  std::vector<float> short_io(8, 1.0f);
  bad = core::RopeRef(std::span<float>(short_io), 2, 2, 8, 8, 0, 1e4);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);

  std::vector<float> q(2 * 4 * 8, 0.5f);
  std::vector<float> k(4 * 2 * 8, 0.5f);
  std::vector<float> v(4 * 2 * 8, 0.5f);
  std::vector<float> out(2 * 4 * 8);
  auto bad_attn = core::AttentionRef(
      std::span<const float>(q), std::span<const float>(k),
      std::span<const float>(v), std::span<float>(out), 2, 4, 4, 3, 8, 0);
  ASSERT_FALSE(bad_attn.has_value());
  EXPECT_EQ(bad_attn.error(), StatusCode::InvalidArgument);
  std::vector<float> short_q(8, 0.5f);
  bad_attn = core::AttentionRef(
      std::span<const float>(short_q), std::span<const float>(k),
      std::span<const float>(v), std::span<float>(out), 2, 4, 4, 2, 8, 0);
  ASSERT_FALSE(bad_attn.has_value());
  EXPECT_EQ(bad_attn.error(), StatusCode::InvalidArgument);
}

// Device: the rope and attention contracts (backend.hpp) reject bad
// shapes before anything reaches the device.
TEST(BackendTest, AttentionRejectsBadContract) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto rope = backend->LoadKernel("rope", {});
  ASSERT_TRUE(rope.has_value()) << tessera::ToString(rope.error());
  auto attention = backend->LoadKernel("attention", {});
  ASSERT_TRUE(attention.has_value())
      << tessera::ToString(attention.error());

  tessera::KernelLaunch launch;
  // Rope wants 1 buffer and 6 scalars.
  launch.scalars = {8, 4, 32, 16, 0, 0};
  auto no_buffers = backend->LaunchKernel(**rope, launch);
  ASSERT_FALSE(no_buffers.has_value());
  EXPECT_EQ(no_buffers.error(), StatusCode::InvalidArgument);
  // Attention wants 4 buffers and 6 scalars; 3 heads do not split
  // into 2 kv groups.
  const std::vector<const tessera::Buffer*> three_buffers{
      nullptr, nullptr, nullptr};
  const std::vector<const tessera::Buffer*> four_buffers{
      nullptr, nullptr, nullptr, nullptr};
  launch.buffers = three_buffers;
  launch.scalars = {1, 4, 4, 2, 16, 0};
  auto short_buffers = backend->LaunchKernel(**attention, launch);
  ASSERT_FALSE(short_buffers.has_value());
  EXPECT_EQ(short_buffers.error(), StatusCode::InvalidArgument);
  launch.buffers = four_buffers;
  launch.scalars = {1, 4, 3, 2, 16, 0};
  auto bad_groups = backend->LaunchKernel(**attention, launch);
  ASSERT_FALSE(bad_groups.has_value());
  EXPECT_EQ(bad_groups.error(), StatusCode::InvalidArgument);
}

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
  launch.grid_x = (kRows + 255) / 256;
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
TEST(BackendTest, NormRefsRejectBadArgs) {
  std::vector<float> x(4 * 8, 0.5f);
  std::vector<float> w(8, 1.0f);
  std::vector<float> y(4 * 8);
  auto bad = core::RmsNormRef(std::span<const float>(x),
                              std::span<const float>(w), std::span<float>(y),
                              0, 8, 1e-6f);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  bad = core::RmsNormRef(std::span<const float>(x), std::span<const float>(w),
                         std::span<float>(y), 4, 8, -1.0f);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  std::vector<float> short_w(4, 1.0f);
  bad = core::RmsNormRef(std::span<const float>(x),
                         std::span<const float>(short_w), std::span<float>(y),
                         4, 8, 1e-6f);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);

  std::vector<float> a(8, 0.5f);
  std::vector<float> g(8, 0.5f);
  std::vector<float> o(8);
  auto bad_gate = core::SigmoidGateRef(std::span<const float>(a),
                                       std::span<const float>(g),
                                       std::span<float>(o), 0);
  ASSERT_FALSE(bad_gate.has_value());
  EXPECT_EQ(bad_gate.error(), StatusCode::InvalidArgument);
  std::vector<float> short_a(4, 0.5f);
  bad_gate = core::SigmoidGateRef(std::span<const float>(short_a),
                                  std::span<const float>(g),
                                  std::span<float>(o), 8);
  ASSERT_FALSE(bad_gate.has_value());
  EXPECT_EQ(bad_gate.error(), StatusCode::InvalidArgument);
}

// Device: the rmsnorm and sigmoid_gate contracts (backend.hpp) reject
// bad shapes before anything reaches the device.
TEST(BackendTest, NormRejectsBadContract) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto rmsnorm = backend->LoadKernel("rmsnorm", {});
  ASSERT_TRUE(rmsnorm.has_value()) << tessera::ToString(rmsnorm.error());
  auto gate = backend->LoadKernel("sigmoid_gate", {});
  ASSERT_TRUE(gate.has_value()) << tessera::ToString(gate.error());

  tessera::KernelLaunch launch;
  // Rmsnorm wants 3 buffers and 3 scalars; zero rows are rejected.
  const std::vector<const tessera::Buffer*> three_buffers{
      nullptr, nullptr, nullptr};
  launch.buffers = three_buffers;
  launch.scalars = {0, 64, 0};
  auto no_buffers = backend->LaunchKernel(**rmsnorm, launch);
  ASSERT_FALSE(no_buffers.has_value());
  EXPECT_EQ(no_buffers.error(), StatusCode::InvalidArgument);
  // Sigmoid gate wants 3 buffers and 1 scalar; zero length rejected.
  launch.scalars = {0};
  auto zero_len = backend->LaunchKernel(**gate, launch);
  ASSERT_FALSE(zero_len.has_value());
  EXPECT_EQ(zero_len.error(), StatusCode::InvalidArgument);
}

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
TEST(BackendTest, ConvRejectsBadArgs) {
  std::vector<float> x(8 * 16, 0.5f);
  std::vector<float> w(8 * 4, 0.5f);
  std::vector<float> y(8 * 16);
  auto bad = core::ConvRef(std::span<const float>(x), std::span<const float>(w),
                           std::span<float>(y), 8, 16, 0);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  std::vector<float> short_w(8 * 2, 0.5f);
  bad = core::ConvRef(std::span<const float>(x), std::span<const float>(short_w),
                      std::span<float>(y), 8, 16, 4);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);

  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto conv = backend->LoadKernel("conv1d", {});
  ASSERT_TRUE(conv.has_value()) << tessera::ToString(conv.error());
  tessera::KernelLaunch launch;
  // Conv1d wants 3 buffers and 3 scalars; zero width is rejected.
  const std::vector<const tessera::Buffer*> three_buffers{
      nullptr, nullptr, nullptr};
  launch.buffers = three_buffers;
  launch.scalars = {8, 16, 0};
  auto zero_width = backend->LaunchKernel(**conv, launch);
  ASSERT_FALSE(zero_width.has_value());
  EXPECT_EQ(zero_width.error(), StatusCode::InvalidArgument);
}

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
TEST(BackendTest, DeltaRejectsBadArgs) {
  std::vector<float> s(8 * 12, 0.5f);
  std::vector<float> k(8, 0.5f);
  std::vector<float> v(12, 0.5f);
  std::vector<float> q(8, 0.5f);
  std::vector<float> o(12);
  auto bad = core::DeltaStepRef(std::span<float>(s), std::span<const float>(k),
                                std::span<const float>(v),
                                std::span<const float>(q), std::span<float>(o),
                                0, 12, 0.9f, 0.5f);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  std::vector<float> short_k(4, 0.5f);
  bad = core::DeltaStepRef(std::span<float>(s), std::span<const float>(short_k),
                           std::span<const float>(v), std::span<const float>(q),
                           std::span<float>(o), 8, 12, 0.9f, 0.5f);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);

  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto delta = backend->LoadKernel("delta_step", {});
  ASSERT_TRUE(delta.has_value()) << tessera::ToString(delta.error());
  tessera::KernelLaunch launch;
  // Delta step wants 5 buffers and 4 scalars; zero dk is rejected.
  const std::vector<const tessera::Buffer*> five_buffers{
      nullptr, nullptr, nullptr, nullptr, nullptr};
  launch.buffers = five_buffers;
  launch.scalars = {0, 12, 0, 0};
  auto zero_dk = backend->LaunchKernel(**delta, launch);
  ASSERT_FALSE(zero_dk.has_value());
  EXPECT_EQ(zero_dk.error(), StatusCode::InvalidArgument);
}

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
TEST(BackendTest, MropeRejectsBadArgs) {
  std::vector<float> io(4 * 2 * 32, 0.5f);
  std::vector<std::uint64_t> pos(4 * 3, 1);
  std::vector<float> ref = io;
  auto bad = core::MropeRef(std::span<float>(ref),
                            std::span<const std::uint64_t>(pos), 4, 2, 32, 16,
                            5, 5, 5, 1e4);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  std::vector<std::uint64_t> short_pos(4, 1);
  bad = core::MropeRef(std::span<float>(ref),
                       std::span<const std::uint64_t>(short_pos), 4, 2, 32, 16,
                       3, 3, 2, 1e4);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);

  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto mrope = backend->LoadKernel("mrope", {});
  ASSERT_TRUE(mrope.has_value()) << tessera::ToString(mrope.error());
  tessera::KernelLaunch launch;
  // Mrope wants 2 buffers and 8 scalars; overflowing sections rejected.
  const std::vector<const tessera::Buffer*> two_buffers{nullptr, nullptr};
  launch.buffers = two_buffers;
  launch.scalars = {4, 2, 32, 16, 0, 5, 5, 5};
  auto bad_sections = backend->LaunchKernel(**mrope, launch);
  ASSERT_FALSE(bad_sections.has_value());
  EXPECT_EQ(bad_sections.error(), StatusCode::InvalidArgument);
}

// Device: the fused gated-attention split matches the host reference
// (3 heads of head dim 8, distinct values per segment).
TEST(BackendTest, QGateSplitDeviceMatchesRef) {
  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  std::mt19937 rng(47);
  constexpr std::size_t kHeads = 3;
  constexpr std::size_t kHeadDim = 8;
  std::vector<float> fused(kHeads * 2 * kHeadDim);
  for (auto& v : fused) {
    v = DrawValue(rng);
  }
  auto fused_buf =
      backend->AllocateBuffer(fused.size() * 4, MemoryKind::Device);
  auto q_buf =
      backend->AllocateBuffer(kHeads * kHeadDim * 4, MemoryKind::Device);
  auto gate_buf =
      backend->AllocateBuffer(kHeads * kHeadDim * 4, MemoryKind::Device);
  ASSERT_TRUE(fused_buf.has_value() && q_buf.has_value() &&
              gate_buf.has_value());
  auto upload = backend->CopyH2D(**fused_buf, std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(fused.data()), fused.size() * 4));
  ASSERT_TRUE(upload.has_value()) << tessera::ToString(upload.error());

  auto kernel = backend->LoadKernel("qgate_split", {});
  ASSERT_TRUE(kernel.has_value()) << tessera::ToString(kernel.error());
  tessera::KernelLaunch launch;
  launch.grid_x = (kHeads * kHeadDim + 255) / 256;
  launch.block_x = 256;
  launch.buffers = {(*fused_buf).get(), (*q_buf).get(), (*gate_buf).get()};
  launch.scalars = {kHeads, kHeadDim};
  auto result = backend->LaunchKernel(**kernel, launch);
  ASSERT_TRUE(result.has_value()) << tessera::ToString(result.error());
  backend->Synchronize();

  std::vector<float> ref_q(kHeads * kHeadDim);
  std::vector<float> ref_gate(kHeads * kHeadDim);
  auto ref_status = core::QGateSplitRef(
      std::span<const float>(fused), std::span<float>(ref_q),
      std::span<float>(ref_gate), kHeads, kHeadDim);
  ASSERT_TRUE(ref_status.has_value())
      << tessera::ToString(ref_status.error());
  std::vector<std::byte> readback_q(kHeads * kHeadDim * 4);
  std::vector<std::byte> readback_gate(kHeads * kHeadDim * 4);
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
TEST(BackendTest, QGateSplitRejectsBadArgs) {
  std::vector<float> fused(2 * 2 * 4, 0.5f);
  std::vector<float> q(2 * 4);
  std::vector<float> gate(2 * 4);
  auto bad = core::QGateSplitRef(std::span<const float>(fused),
                                 std::span<float>(q), std::span<float>(gate),
                                 0, 4);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
  std::vector<float> short_q(2, 0.5f);
  bad = core::QGateSplitRef(std::span<const float>(fused),
                            std::span<float>(short_q), std::span<float>(gate),
                            2, 4);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);

  std::unique_ptr<Backend> backend;
  MakeBackendOrSkip(backend);
  auto split = backend->LoadKernel("qgate_split", {});
  ASSERT_TRUE(split.has_value()) << tessera::ToString(split.error());
  tessera::KernelLaunch launch;
  // Qgate split wants 3 buffers and 2 scalars; zero head dim rejected.
  const std::vector<const tessera::Buffer*> three_buffers{
      nullptr, nullptr, nullptr};
  launch.buffers = three_buffers;
  launch.scalars = {2, 0};
  auto bad_contract = backend->LaunchKernel(**split, launch);
  ASSERT_FALSE(bad_contract.has_value());
  EXPECT_EQ(bad_contract.error(), StatusCode::InvalidArgument);
}
