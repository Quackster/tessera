#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>

#include "tessera/backend.hpp"
#include "tessera/model.hpp"
#include "tessera/types.hpp"

namespace tessera::spec {

// The DFlash2 mask-token query embeddings: gather the target embedding row
// at `mask_token` (any quantized format, via the shared gather) and tile it
// to `rows` identical rows of `hidden` floats. Returns a new device buffer
// of rows x hidden, or MalformedFile/DeviceError on failure.
[[nodiscard]] std::expected<std::unique_ptr<Buffer>, StatusCode>
MaskEmbeddings(Backend& backend, const DeviceTensor& embed,
               std::uint32_t mask_token, std::size_t rows, std::size_t hidden);

}  // namespace tessera::spec
