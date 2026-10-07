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
#include "core/numerics/gemm.hpp"
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
