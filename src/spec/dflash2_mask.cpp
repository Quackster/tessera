#include "spec/dflash2_mask.hpp"

#include <algorithm>
#include <vector>

#include "core/decode_internal.hpp"

namespace tessera::spec {

namespace {

// Gather embedding row `token` and write it to `rows` consecutive rows of
// `out` starting at row `offset`.
std::expected<void, StatusCode> TileToken(Backend& backend,
                                          const DeviceTensor& embed,
                                          std::uint32_t token,
                                          std::vector<float>& out,
                                          std::size_t offset, std::size_t rows,
                                          std::size_t hidden) {
  std::vector<float> row(hidden);
  auto gathered =
      core::detail::GatherEmbedding(backend, embed, token, hidden, row);
  if (!gathered) {
    return std::unexpected(gathered.error());
  }
  for (std::size_t r = 0; r < rows; ++r) {
    std::copy(row.begin(), row.end(), out.begin() + (offset + r) * hidden);
  }
  return {};
}

std::expected<std::unique_ptr<Buffer>, StatusCode> Upload(
    Backend& backend, const std::vector<float>& data) {
  auto buffer = backend.AllocateBuffer(data.size() * 4, MemoryKind::Device);
  if (!buffer) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  if (!backend.CopyH2D(**buffer,
                       std::span<const std::byte>(
                           reinterpret_cast<const std::byte*>(data.data()),
                           data.size() * 4))) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return std::move(*buffer);
}

}  // namespace

std::expected<std::unique_ptr<Buffer>, StatusCode> QueryEmbeddings(
    Backend& backend, const DeviceTensor& embed, std::uint32_t anchor_token,
    std::uint32_t mask_token, std::size_t mask_rows, std::size_t hidden) {
  const std::size_t vocab = embed.manifest.shape.dims[embed.manifest.shape.rank - 1];
  if (mask_rows == 0 || hidden == 0 || anchor_token >= vocab ||
      mask_token >= vocab) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> queries((1 + mask_rows) * hidden);
  auto anchor = TileToken(backend, embed, anchor_token, queries, 0, 1, hidden);
  if (!anchor) {
    return std::unexpected(anchor.error());
  }
  auto masks =
      TileToken(backend, embed, mask_token, queries, 1, mask_rows, hidden);
  if (!masks) {
    return std::unexpected(masks.error());
  }
  return Upload(backend, queries);
}

}  // namespace tessera::spec
