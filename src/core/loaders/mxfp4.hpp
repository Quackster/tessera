#pragma once

#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/loaders/safetensors.hpp"
#include "tessera/model.hpp"
#include "tessera/types.hpp"

namespace tessera {
class Architecture;
class Backend;
namespace core {

// The architecture name from an MXFP4 config.json: the first entry of
// `architectures` when present, else `model_type`. Empty when absent or
// unparseable, so the caller falls back to the raw directory load.
[[nodiscard]] std::string MxFp4ArchitectureName(std::string_view json);

// Build internal device tensors from a parsed MXFP4 safetensors map, using
// the architecture module: map each tensor name to the internal name,
// apply the module's value conversion (value-head reorder, A_log -> F32
// -exp), reverse the shape to the internal convention, and pack each
// F4E2M1 blob with its paired E8M0 scale (blob then scale). Plain
// (BF16/F32) tensors become F32 except the embedding, which stays BF16.
// Tensors the module ignores and the MTP FP8 tensors are skipped. The
// returned weights are in internal-name order.
[[nodiscard]] std::expected<std::vector<DeviceTensor>, StatusCode>
BuildMxFp4Weights(Backend& backend, std::span<const std::byte> file,
                  std::span<const SafetensorsTensor> tensors,
                  const Architecture& module, const TransformerConfig& config);

}  // namespace core
}  // namespace tessera
