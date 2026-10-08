#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>

#include "tessera/backend.hpp"
#include "tessera/model.hpp"
#include "tessera/types.hpp"

namespace tessera::spec {

// The DFlash2 draft queries: the anchor (bonus) token embedding at row 0,
// followed by `mask_rows` identical mask-token embeddings. The draft block
// runs 1 + mask_rows queries and predicts from the mask rows only. Returns a
// new device buffer of (1 + mask_rows) x hidden, or MalformedFile/DeviceError.
[[nodiscard]] std::expected<std::unique_ptr<Buffer>, StatusCode>
QueryEmbeddings(Backend& backend, const DeviceTensor& embed,
                std::uint32_t anchor_token, std::uint32_t mask_token,
                std::size_t mask_rows, std::size_t hidden);

}  // namespace tessera::spec
