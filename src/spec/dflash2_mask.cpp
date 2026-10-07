#include "spec/dflash2_mask.hpp"

#include <vector>

#include "core/decode_internal.hpp"

namespace tessera::spec {

std::expected<std::unique_ptr<Buffer>, StatusCode> MaskEmbeddings(
    Backend& backend, const DeviceTensor& embed, std::uint32_t mask_token,
    std::size_t rows, std::size_t hidden) {
  if (rows == 0 || hidden == 0 ||
      mask_token >= embed.manifest.shape.dims[embed.manifest.shape.rank - 1]) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<float> row(hidden);
  auto gathered =
      core::detail::GatherEmbedding(backend, embed, mask_token, hidden, row);
  if (!gathered) {
    return std::unexpected(gathered.error());
  }
  std::vector<float> tiled(rows * hidden);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < hidden; ++c) {
      tiled[r * hidden + c] = row[c];
    }
  }
  auto buffer = backend.AllocateBuffer(tiled.size() * 4, MemoryKind::Device);
  if (!buffer) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  if (!backend.CopyH2D(**buffer,
                       std::span<const std::byte>(
                           reinterpret_cast<const std::byte*>(tiled.data()),
                           tiled.size() * 4))) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return std::move(*buffer);
}

}  // namespace tessera::spec
