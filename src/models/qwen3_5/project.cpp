#include "models/qwen3_5/internal.hpp"

#include <cstdlib>

#include "core/decode_internal.hpp"
#include "models/qwen3_5/state.hpp"

// Qwen3.5 projection helper. With TESSERA_MXFP4_W4A8=1 an MXFP4 (F4E2M1)
// weight runs as W4A8: its activation is quantize-dequantized per token to
// FP8 E4M3 in place before the GEMM, matching the quantize step the served
// Qwen3.8 target applies to every MXFP4 linear input. It is off by default:
// on the short DFlash2 fixture it did not raise acceptance and it costs
// about 2.5x decode time, so it is opt-in pending a long-generation
// measurement. Non-MXFP4 weights are always unchanged.

namespace tessera::models::qwen3_5 {

namespace detail = ::tessera::core::detail;

std::expected<void, StatusCode> QuantizeMxFp4Input(
    Backend& backend, Qwen35State& h, DType dtype, Buffer& data,
    std::size_t rows, std::size_t cols) {
  if (dtype != DType::F4E2M1) {
    return {};
  }
  const char* enabled = std::getenv("TESSERA_MXFP4_W4A8");
  if (enabled == nullptr || std::atoi(enabled) == 0) {
    return {};
  }
  if (h.fp8_quant_kernel == nullptr) {
    auto kernel = backend.LoadKernel("quantize_fp8", {});
    if (!kernel) {
      return std::unexpected(kernel.error());
    }
    h.fp8_quant_kernel = std::move(*kernel);
  }
  if (h.fp8_scale == nullptr || h.fp8_scale->Size() < rows * 4) {
    auto scale = backend.AllocateBuffer(rows * 4, MemoryKind::Device);
    if (!scale) {
      return std::unexpected(scale.error());
    }
    h.fp8_scale = std::move(*scale);
  }
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>(rows);
  launch.block_x = 256;
  launch.buffers = {&data, h.fp8_scale.get()};
  launch.scalars = {rows, cols};
  return backend.LaunchKernel(*h.fp8_quant_kernel, launch);
}

// Opt-in fp8 tensor-core MXFP4 GEMM. Returns true when it projected, false
// when the backend/shape has no path (caller falls through to the scalar
// kernels). Gated by TESSERA_MXFP4_WMMA. A is packed to fp8 E4M3 per token,
// the weight stays the packed MXFP4 (the kernel folds it in), and the per-row
// Wref is cached per weight buffer.
std::expected<bool, StatusCode> ProjectWmma(Backend& backend, Qwen35State& h,
                                            Buffer& a, const Buffer& w,
                                            Buffer& out, std::size_t m,
                                            std::size_t n, std::size_t k) {
  if (m > 16 || k % 128 != 0 || k % 4 != 0) {
    return false;
  }
  const auto load = [&](std::unique_ptr<Kernel>& slot,
                        const char* name) -> std::expected<bool, StatusCode> {
    if (slot != nullptr) {
      return true;
    }
    auto kernel = backend.LoadKernel(name, {});
    if (!kernel) {
      if (kernel.error() == StatusCode::UnsupportedFeature) {
        return false;
      }
      return std::unexpected(kernel.error());
    }
    slot = std::move(*kernel);
    return true;
  };
  auto pack = load(h.fp8_pack_kernel, "quantize_fp8_pack_rows");
  if (!pack) return pack;
  if (!*pack) return false;
  auto refk = load(h.mxfp4_rowref_kernel, "mxfp4_row_ref");
  if (!refk) return refk;
  if (!*refk) return false;
  auto wk = load(h.mxfp4_wmma_kernel, "gemm_mxfp4_wmma");
  if (!wk) return wk;
  if (!*wk) return false;
  if (h.wmma_a == nullptr || h.wmma_a->Size() < m * k) {
    auto buf = backend.AllocateBuffer(m * k, MemoryKind::Device);
    if (!buf) return std::unexpected(buf.error());
    h.wmma_a = std::move(*buf);
  }
  if (h.wmma_as == nullptr || h.wmma_as->Size() < m * 4) {
    auto buf = backend.AllocateBuffer(m * 4, MemoryKind::Device);
    if (!buf) return std::unexpected(buf.error());
    h.wmma_as = std::move(*buf);
  }
  Buffer* wref = nullptr;
  auto it = h.mxfp4_wref.find(&w);
  if (it == h.mxfp4_wref.end()) {
    auto buf = backend.AllocateBuffer(n, MemoryKind::Device);
    if (!buf) return std::unexpected(buf.error());
    KernelLaunch rl;
    rl.grid_x = static_cast<std::uint32_t>((n + 255) / 256);
    rl.block_x = 256;
    rl.buffers = {&w, buf->get()};
    rl.scalars = {n, k};
    auto st = backend.LaunchKernel(*h.mxfp4_rowref_kernel, rl);
    if (!st) return std::unexpected(st.error());
    wref = buf->get();
    h.mxfp4_wref.emplace(&w, std::move(*buf));
  } else {
    wref = it->second.get();
  }
  KernelLaunch ql;
  ql.grid_x = static_cast<std::uint32_t>(m);
  ql.block_x = 1024;
  ql.buffers = {&a, h.wmma_a.get(), h.wmma_as.get()};
  ql.scalars = {m, k};
  if (auto st = backend.LaunchKernel(*h.fp8_pack_kernel, ql); !st) {
    return std::unexpected(st.error());
  }
  // Split-K: aim for a target workgroup count so the fp8 GEMM saturates the
  // device on the narrow projections (n ~ 5120) where one block per 64
  // columns leaves it latency-bound. Each split writes its K partial to its
  // own region of a scratch buffer; a reduce kernel sums the regions into the
  // output (no atomics, no pre-zeroing). Both bounds are overridable.
  std::uint64_t target = 320, cap = 4;
  if (const char* e = std::getenv("TESSERA_MXFP4_SPLIT")) {
    target = std::strtoull(e, nullptr, 10);
  }
  if (const char* e = std::getenv("TESSERA_MXFP4_SPLITCAP")) {
    cap = std::strtoull(e, nullptr, 10);
  }
  std::uint64_t split = (target * 64ull + n - 1) / n;
  const std::uint64_t chunks = k / 128;
  if (split < 1) split = 1;
  if (split > chunks) split = chunks;
  if (split > cap) split = cap;
  Buffer* dst = &out;
  if (split > 1) {
    const std::size_t part_elems = static_cast<std::size_t>(split) * m * n;
    if (h.wmma_part == nullptr || h.wmma_part->Size() < part_elems * 4) {
      auto buf = backend.AllocateBuffer(part_elems * 4, MemoryKind::Device);
      if (!buf) return std::unexpected(buf.error());
      h.wmma_part = std::move(*buf);
    }
    dst = h.wmma_part.get();
  }
  KernelLaunch wl;
  wl.grid_x = static_cast<std::uint32_t>((n + 63) / 64);
  wl.grid_y = static_cast<std::uint32_t>(split);
  wl.block_x = 128;
  wl.buffers = {h.wmma_a.get(), &w, h.wmma_as.get(), wref, dst};
  wl.scalars = {m, n, k, split};
  if (auto st = backend.LaunchKernel(*h.mxfp4_wmma_kernel, wl); !st) {
    return std::unexpected(st.error());
  }
  if (split > 1) {
    auto rk = load(h.mxfp4_reduce_kernel, "gemm_mxfp4_wmma_reduce");
    if (!rk) return rk;
    if (!*rk) return false;
    const std::uint64_t total = m * n;
    KernelLaunch rl;
    rl.grid_x = static_cast<std::uint32_t>((total + 255) / 256);
    rl.block_x = 256;
    rl.buffers = {h.wmma_part.get(), &out};
    rl.scalars = {total, split};
    if (auto st = backend.LaunchKernel(*h.mxfp4_reduce_kernel, rl); !st) {
      return std::unexpected(st.error());
    }
  }
  return true;
}

std::expected<void, StatusCode> ProjectBatch(
    Backend& backend, Qwen35State& h, DType dtype, Buffer& a,
    const Buffer& w, Buffer& out, std::size_t m, std::size_t n,
    std::size_t k) {
  // fp8 tensor-core MXFP4 GEMM for the small-m verify (m 2..16). On by
  // default where the backend provides it (ROCm gfx12); the m=1 greedy path
  // and Vulkan (no fp8 tensor cores) keep the scalar kernels, so the
  // non-speculative baseline is unchanged. TESSERA_MXFP4_WMMA=0 disables it.
  const char* wmma_env = std::getenv("TESSERA_MXFP4_WMMA");
  const bool wmma_on = wmma_env == nullptr || std::atoi(wmma_env) != 0;
  if (dtype == DType::F4E2M1 && wmma_on && m >= 2 && m <= 16) {
    auto projected = ProjectWmma(backend, h, a, w, out, m, n, k);
    if (!projected) {
      return std::unexpected(projected.error());
    }
    if (*projected) {
      return {};
    }
  }
  if (auto quantized = QuantizeMxFp4Input(backend, h, dtype, a, m, k);
      !quantized) {
    return quantized;
  }
  std::expected<void, StatusCode> projected = {};
  bool projected_done = false;
  // The speculative verifier runs the target trunk at a small m (about 2 to
  // 7). The warp-per-column multi-row GEMV reads each weight column once for
  // all m rows, unlike the single-row GEMV, which re-reads it per row.
  // The speculative verifier runs the target trunk at a small m (about 2 to
  // 7). The warp-per-column multi-row GEMV reads each weight column once for
  // all m rows, unlike the single-row GEMV, which re-reads it per row.
  if (dtype == DType::F4E2M1 && m >= 2 && m <= 16) {
    auto rows = detail::CachedKernel(
        backend, h.gemms, static_cast<int>(DType::F4E2M1) + 0x2000,
        "gemm_mxfp4_rows");
    if (rows) {
      projected = detail::ProjectDevice(backend, **rows, a, w, out, m, n, k);
      projected_done = true;
    } else if (rows.error() != StatusCode::UnsupportedFeature) {
      return std::unexpected(rows.error());
    }
  }
  if (!projected_done &&
      (m >= detail::kGemmTiledMinRows ||
       (m >= 2 && n >= detail::kGemmTiledMinCols)) &&
      !detail::GemmTiledKernelName(dtype).empty()) {
    auto tiled = detail::GemmTiledFor(backend, h.gemm_tiled, dtype);
    if (tiled) {
      projected = detail::ProjectTiledDevice(backend, **tiled, a, w, out, m, n,
                                             k);
      projected_done = true;
    } else if (tiled.error() != StatusCode::UnsupportedFeature) {
      return std::unexpected(tiled.error());
    }
    // The backend has no tiled kernel for this dtype; fall through to GEMV.
  }
  if (!projected_done) {
    auto gemm = detail::GemmFor(backend, h.gemms, dtype);
    if (!gemm) {
      return std::unexpected(gemm.error());
    }
    projected = detail::ProjectDevice(backend, **gemm, a, w, out, m, n, k);
  }
  if (!projected) {
    return projected;
  }
  // The served target rounds every linear output to bf16 (the GEMM C is
  // bf16 and the fused epilogue stores a bf16 residual). Opt in for the
  // DFlash2 aux-value match; the default fp32 path is unchanged.
  const char* bf16 = std::getenv("TESSERA_TARGET_BF16");
  if (bf16 != nullptr && std::atoi(bf16) != 0) {
    if (h.bf16_kernel == nullptr) {
      auto kernel = backend.LoadKernel("round_bf16", {});
      if (!kernel) {
        return std::unexpected(kernel.error());
      }
      h.bf16_kernel = std::move(*kernel);
    }
    return detail::RoundBf16Device(backend, *h.bf16_kernel, out, m * n);
  }
  return {};
}

}  // namespace tessera::models::qwen3_5
