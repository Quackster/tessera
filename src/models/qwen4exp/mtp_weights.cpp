#include "models/qwen4exp/mtp_weights.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/files.hpp"
#include "core/json.hpp"
#include "core/numerics/quant.hpp"

namespace tessera::models::qwen4exp {

namespace {

using core::detail::UploadF32;

constexpr std::string_view kManifestName = "mtp-manifest.json";
constexpr std::string_view kExpertGateUp =
    "mtp.layers.0.mlp.experts.gate_up_proj";
constexpr std::string_view kExpertDown = "mtp.layers.0.mlp.experts.down_proj";
constexpr std::string_view kSharedGate =
    "mtp.layers.0.mlp.shared_expert_gate.weight";

// The MTP RMSNorm gammas. The checkpoint stores them raw; the engine's norm
// kernels expect the Gemma convention (1 + w), so the loader adds 1. The
// shared-expert gate is a linear gate, not a norm, and is not in this set.
bool IsNorm(std::string_view name) {
  static const std::set<std::string_view> kNames = {
      "mtp.pre_fc_norm_embedding.weight",
      "mtp.pre_fc_norm_hidden.weight",
      "mtp.layers.0.self_attn.q_norm.weight",
      "mtp.layers.0.self_attn.k_norm.weight",
      "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
      "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
      "mtp.hyper_connection_mixer.hc_norm.weight",
      "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
      "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
  };
  return kNames.count(name) != 0;
}

std::string DtypeName(const core::Json* value) {
  return value != nullptr && value->isString() ? value->AsString() : std::string{};
}

// Decode the checkpoint bytes to f32: BF16 or F32 only. F16 is not used by
// this head.
std::expected<std::vector<float>, StatusCode> ToF32(
    std::string_view dtype, std::span<const std::byte> bytes) {
  if (dtype == "F32") {
    if (bytes.size() % 4 != 0) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    std::vector<float> out(bytes.size() / 4);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
  }
  if (dtype == "BF16") {
    if (bytes.size() % 2 != 0) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    std::vector<float> out(bytes.size() / 2);
    if (!core::DequantizeBf16(bytes, std::span<float>(out))) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    return out;
  }
  return std::unexpected(StatusCode::UnsupportedFeature);
}

std::expected<std::unique_ptr<Buffer>, StatusCode> UploadRaw(
    Backend& backend, std::span<const std::byte> bytes) {
  auto buffer = backend.AllocateBuffer(bytes.size(), MemoryKind::Device);
  if (!buffer) {
    return std::unexpected(buffer.error());
  }
  if (!backend.CopyH2D(**buffer, bytes)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return std::move(*buffer);
}

}  // namespace

const Buffer* MtpWeights::Get(std::string_view name) const {
  const auto it = tensors.find(std::string(name));
  return it == tensors.end() ? nullptr : it->second.get();
}

std::expected<MtpWeights, StatusCode> LoadMtpWeights(
    Backend& backend, const std::filesystem::path& dir) {
  auto manifest_bytes = core::ReadFile(dir / std::string(kManifestName));
  if (!manifest_bytes) {
    return std::unexpected(manifest_bytes.error());
  }
  const std::string text(
      reinterpret_cast<const char*>(manifest_bytes->data()),
      manifest_bytes->size());
  auto parsed = core::Json::Parse(text);
  if (parsed == nullptr || !parsed->isArray()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  MtpWeights weights;
  const std::filesystem::path tensors_dir = dir / "tensors";
  for (const core::Json& item : parsed->AsArray()) {
    const core::Json* name_json = item.Find("name");
    const core::Json* bytes_json = item.Find("bytes");
    const core::Json* shape_json = item.Find("shape");
    if (name_json == nullptr || !name_json->isString() || bytes_json == nullptr ||
        shape_json == nullptr || !shape_json->isArray()) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    const std::string name = name_json->AsString();
    const std::string dtype = DtypeName(item.Find("dtype"));
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(bytes_json->AsNumber());
    auto raw = core::ReadFile(tensors_dir / (name + ".bin"));
    if (!raw) {
      return std::unexpected(raw.error());
    }
    if (raw->size() != bytes) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    if (name == kExpertGateUp) {
      if (shape_json->AsArray().size() != 3) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      weights.experts = static_cast<std::size_t>(shape_json->AsArray()[0].AsNumber());
      weights.inter = static_cast<std::size_t>(shape_json->AsArray()[1].AsNumber()) / 2;
      weights.hidden = static_cast<std::size_t>(shape_json->AsArray()[2].AsNumber());
    } else if (name == kExpertDown) {
      if (shape_json->AsArray().size() != 3) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      weights.hidden = static_cast<std::size_t>(shape_json->AsArray()[1].AsNumber());
      weights.inter = static_cast<std::size_t>(shape_json->AsArray()[2].AsNumber());
    }
    if (IsNorm(name) || name == kSharedGate) {
      // Norms become f32 with the Gemma +1; the shared gate is lifted to f32
      // as it is.
      auto f32 = ToF32(dtype, std::span<const std::byte>(raw->data(), raw->size()));
      if (!f32) {
        return std::unexpected(f32.error());
      }
      if (IsNorm(name)) {
        for (float& value : *f32) {
          value += 1.0f;
        }
      }
      auto buffer = backend.AllocateBuffer(f32->size() * 4, MemoryKind::Device);
      if (!buffer) {
        return std::unexpected(buffer.error());
      }
      if (!UploadF32(backend, **buffer, *f32)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      weights.tensors.emplace(name, std::move(*buffer));
      continue;
    }
    if (dtype != "BF16") {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    auto buffer = UploadRaw(backend, std::span<const std::byte>(raw->data(), raw->size()));
    if (!buffer) {
      return std::unexpected(buffer.error());
    }
    weights.tensors.emplace(name, std::move(*buffer));
  }
  if (weights.experts == 0 || weights.inter == 0 || weights.hidden == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return weights;
}

}  // namespace tessera::models::qwen4exp
