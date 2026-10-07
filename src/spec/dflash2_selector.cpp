#include "spec/dflash2_selector.hpp"

#include <vector>

#include "core/decode_internal.hpp"
#include "core/numerics/gemm.hpp"
#include "core/numerics/selector.hpp"

namespace tessera::spec {

std::expected<void, StatusCode> DraftSelectorRef(
    std::span<const float> hidden, std::span<const float> projection_w,
    std::span<const float> predecessor, std::span<const float> successor,
    std::span<const std::int32_t> candidate_ids,
    std::span<const std::int32_t> anchor_ids, std::span<const float> unary,
    std::span<float> out, std::size_t rows, std::size_t hidden_dim,
    std::size_t rank, std::size_t vocab, std::size_t top_k) {
  if (rows == 0 || hidden_dim == 0 || rank == 0 || vocab == 0 || top_k == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (hidden.size() != rows * hidden_dim ||
      projection_w.size() != rank * hidden_dim ||
      predecessor.size() != vocab * rank || successor.size() != vocab * rank ||
      candidate_ids.size() != rows * top_k || anchor_ids.size() != rows ||
      unary.size() != rows * top_k ||
      out.size() != rows * top_k * top_k) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> proj(rows * rank);
  if (!core::GemmF32Ref(hidden, projection_w, std::span<float>(proj), rows,
                        rank, hidden_dim)) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return core::SelectorEdgeScoreRef(
      predecessor, successor, std::span<const float>(proj), candidate_ids,
      anchor_ids, unary, out, 1, rows, top_k, rank, vocab);
}

std::expected<void, StatusCode> DraftSelectorDevice(
    Backend& backend, const Kernel& gemm, const Kernel& edge, Buffer& proj,
    const Buffer& hidden, const Buffer& projection_w,
    const Buffer& predecessor, const Buffer& successor,
    const Buffer& candidate_ids, const Buffer& anchor_ids,
    const Buffer& unary, Buffer& out, std::size_t rows, std::size_t hidden_dim,
    std::size_t rank, std::size_t vocab, std::size_t top_k) {
  if (rows == 0 || hidden_dim == 0 || rank == 0 || vocab == 0 || top_k == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto projected = core::detail::ProjectDevice(backend, gemm, hidden,
                                               projection_w, proj, rows,
                                               rank, hidden_dim);
  if (!projected) {
    return std::unexpected(projected.error());
  }
  KernelLaunch launch;
  launch.grid_x = static_cast<std::uint32_t>(
      (rows * top_k * top_k + 255) / 256);
  launch.block_x = 256;
  launch.buffers = {&predecessor, &successor, &proj,        &candidate_ids,
                    &anchor_ids,  &unary,     &out};
  launch.scalars = {1, rows, top_k, rank, vocab};
  return backend.LaunchKernel(edge, launch);
}

}  // namespace tessera::spec
