#include "core/loaders/tensor_slice.hpp"

#include <cstring>
#include <utility>

namespace tessera::core {

namespace {

// Elements and bytes of one quantization block. Plain types use a one
// element block.
struct Block {
  std::size_t elements;
  std::size_t bytes;
};

Block BlockOf(DType dtype) {
  switch (dtype) {
    case DType::Q40:
      return {32, 17};
    case DType::Q80:
      return {32, 34};
    case DType::Q3K:
      return {256, 110};
    case DType::Q4K:
      return {256, 144};
    case DType::Q5K:
      return {256, 176};
    case DType::Q6K:
      return {256, 210};
    case DType::Q8K:
      return {256, 258};
    case DType::IQ4_NL:
      return {32, 18};
    case DType::IQ4_XS:
      return {256, 136};
    case DType::IQ3_S:
      return {256, 110};
    default:
      return {1, static_cast<std::size_t>(ElementBytes(dtype))};
  }
}

std::expected<SlicedTensor, StatusCode> SliceMxFp4(
    std::size_t rows, std::size_t cols, ParallelKind kind, int rank,
    int world, std::span<const std::byte> src) {
  // Packed layout: blob (rows*cols/2 bytes) then scales (rows*cols/32).
  const std::size_t blob_bytes = rows * cols / 2;
  const std::size_t scale_bytes = rows * cols / 32;
  if (blob_bytes + scale_bytes > src.size() ||
      (cols % 32) != 0 || (rows % static_cast<std::size_t>(world)) != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::span<const std::byte> blob = src.subspan(0, blob_bytes);
  const std::span<const std::byte> scales = src.subspan(blob_bytes, scale_bytes);
  SlicedTensor out;
  if (kind == ParallelKind::Column) {
    // Split the output rows.
    const std::size_t r0 = rows * static_cast<std::size_t>(rank) /
                           static_cast<std::size_t>(world);
    const std::size_t r1 = rows * static_cast<std::size_t>(rank + 1) /
                           static_cast<std::size_t>(world);
    const std::size_t brow = cols / 2;
    const std::size_t srow = cols / 32;
    out.rows = r1 - r0;
    out.cols = cols;
    out.bytes.resize(out.rows * (brow + srow));
    std::memcpy(out.bytes.data(), blob.data() + r0 * brow, out.rows * brow);
    std::memcpy(out.bytes.data() + out.rows * brow, scales.data() + r0 * srow,
                out.rows * srow);
    return out;
  }
  // Row: split the input columns at 32-element block boundaries.
  const std::size_t c0 = cols * static_cast<std::size_t>(rank) /
                         static_cast<std::size_t>(world);
  const std::size_t c1 = cols * static_cast<std::size_t>(rank + 1) /
                         static_cast<std::size_t>(world);
  if ((c0 % 32) != 0 || (c1 % 32) != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t brow = cols / 2;
  const std::size_t srow = cols / 32;
  const std::size_t nb = (c1 - c0) / 2;   // blob bytes per row
  const std::size_t ns = (c1 - c0) / 32;  // scale bytes per row
  out.rows = rows;
  out.cols = c1 - c0;
  out.bytes.resize(rows * (nb + ns));
  for (std::size_t r = 0; r < rows; ++r) {
    std::memcpy(out.bytes.data() + r * (nb + ns), blob.data() + r * brow + c0 / 2,
                nb);
    std::memcpy(out.bytes.data() + r * (nb + ns) + nb,
                scales.data() + r * srow + c0 / 32, ns);
  }
  return out;
}

}  // namespace

std::expected<SlicedTensor, StatusCode> SliceTensor2D(
    DType dtype, std::size_t rows, std::size_t cols, ParallelKind kind,
    int rank, int world, std::span<const std::byte> src) {
  if (kind == ParallelKind::Replicated || world <= 1) {
    SlicedTensor whole;
    whole.rows = rows;
    whole.cols = cols;
    whole.bytes.assign(src.begin(), src.end());
    return whole;
  }
  if (rank < 0 || rank >= world || rows == 0 || cols == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (dtype == DType::F4E2M1) {
    return SliceMxFp4(rows, cols, kind, rank, world, src);
  }
  const Block block = BlockOf(dtype);
  if ((cols % block.elements) != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t row_bytes = (cols / block.elements) * block.bytes;
  if (rows * row_bytes > src.size()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  SlicedTensor out;
  if (kind == ParallelKind::Column) {
    const std::size_t r0 = rows * static_cast<std::size_t>(rank) /
                           static_cast<std::size_t>(world);
    const std::size_t r1 = rows * static_cast<std::size_t>(rank + 1) /
                           static_cast<std::size_t>(world);
    out.rows = r1 - r0;
    out.cols = cols;
    out.bytes.assign(src.begin() + static_cast<std::ptrdiff_t>(r0 * row_bytes),
                     src.begin() + static_cast<std::ptrdiff_t>(r1 * row_bytes));
    return out;
  }
  const std::size_t c0 = cols * static_cast<std::size_t>(rank) /
                         static_cast<std::size_t>(world);
  const std::size_t c1 = cols * static_cast<std::size_t>(rank + 1) /
                         static_cast<std::size_t>(world);
  if ((c0 % block.elements) != 0 || (c1 % block.elements) != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t blk0 = c0 / block.elements;
  const std::size_t blk1 = c1 / block.elements;
  const std::size_t kept = (blk1 - blk0) * block.bytes;
  out.rows = rows;
  out.cols = (blk1 - blk0) * block.elements;
  out.bytes.resize(rows * kept);
  for (std::size_t r = 0; r < rows; ++r) {
    std::memcpy(out.bytes.data() + r * kept,
                src.data() + r * row_bytes + blk0 * block.bytes, kept);
  }
  return out;
}

std::expected<SlicedExperts, StatusCode> SliceExperts(
    DType dtype, std::size_t num_experts, std::size_t per_expert_elements,
    int rank, int world, std::span<const std::byte> src) {
  if (world <= 1) {
    SlicedExperts whole;
    whole.experts = num_experts;
    whole.bytes.assign(src.begin(), src.end());
    return whole;
  }
  if (num_experts == 0 || rank < 0 || rank >= world ||
      (num_experts % static_cast<std::size_t>(world)) != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto per = TensorBytes(dtype, per_expert_elements);
  if (!per) {
    return std::unexpected(per.error());
  }
  const std::size_t begin =
      num_experts * static_cast<std::size_t>(rank) /
      static_cast<std::size_t>(world);
  const std::size_t end =
      num_experts * static_cast<std::size_t>(rank + 1) /
      static_cast<std::size_t>(world);
  if (end * *per > src.size()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  SlicedExperts out;
  out.experts = end - begin;
  out.bytes.assign(src.begin() + static_cast<std::ptrdiff_t>(begin * *per),
                   src.begin() + static_cast<std::ptrdiff_t>(end * *per));
  return out;
}

}  // namespace tessera::core
