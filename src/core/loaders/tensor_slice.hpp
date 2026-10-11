#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <vector>

#include "tessera/types.hpp"

namespace tessera::core {

// A row-major 2D tensor sliced for one tensor-parallel rank.
struct SlicedTensor {
  std::vector<std::byte> bytes;
  std::size_t rows = 0;  // output rows after the slice
  std::size_t cols = 0;  // elements per row after the slice
};

// Slice `src` (a `rows` x `cols` row-major tensor of `dtype`) for `rank` of
// `world` ranks. Column keeps the output rows [rank*rows/world, ...); Row
// keeps the input columns [rank*cols/world, ...) at block boundaries.
// InvalidArgument when the split is not block aligned or the source is too
// small. Replicated (or world <= 1) copies the whole tensor. F4E2M1 handles
// the packed blob||scales layout.
//
// Usage:
//   auto sliced = SliceTensor2D(DType::Q4K, 10240, 5120,
//                               ParallelKind::Row, 0, 2, blob);
[[nodiscard]] std::expected<SlicedTensor, StatusCode> SliceTensor2D(
    DType dtype, std::size_t rows, std::size_t cols, ParallelKind kind,
    int rank, int world, std::span<const std::byte> src);

// A rank-3 packed expert tensor sliced for one rank: `experts` is the
// number of experts kept and `bytes` their contiguous blocks.
struct SlicedExperts {
  std::vector<std::byte> bytes;
  std::size_t experts = 0;
};

// Slice the outer (expert) dim of a packed rank-3 expert tensor: keep
// experts [rank*num_experts/world, (rank+1)*num_experts/world). Experts are
// contiguous blocks, so this is a contiguous byte range. InvalidArgument
// when num_experts is not divisible by world.
[[nodiscard]] std::expected<SlicedExperts, StatusCode> SliceExperts(
    DType dtype, std::size_t num_experts, std::size_t per_expert_elements,
    int rank, int world, std::span<const std::byte> src);

}  // namespace tessera::core
