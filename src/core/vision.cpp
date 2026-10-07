#include "tessera/vision.hpp"

#include <string>
#include <vector>

#include "core/loaders/gguf.hpp"

namespace tessera {

namespace {

using core::GgufFile;

std::expected<std::uint64_t, StatusCode> U64(const GgufFile& gguf,
                                             const std::string& key) {
  const core::GgufValue* value = gguf.Find(key);
  if (value == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (const auto* v = std::get_if<std::uint32_t>(value)) {
    return *v;
  }
  if (const auto* v = std::get_if<std::int32_t>(value)) {
    return static_cast<std::uint64_t>(*v);
  }
  if (const auto* v = std::get_if<std::uint64_t>(value)) {
    return *v;
  }
  if (const auto* v = std::get_if<std::int64_t>(value)) {
    return static_cast<std::uint64_t>(*v);
  }
  return std::unexpected(StatusCode::MalformedFile);
}

std::expected<float, StatusCode> F32(const GgufFile& gguf,
                                     const std::string& key) {
  const core::GgufValue* value = gguf.Find(key);
  if (value == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (const auto* v = std::get_if<float>(value)) {
    return *v;
  }
  if (const auto* v = std::get_if<double>(value)) {
    return static_cast<float>(*v);
  }
  return std::unexpected(StatusCode::MalformedFile);
}

bool ReadFloats(const GgufFile& gguf, const std::string& key,
                std::span<float> out) {
  auto it = gguf.small_arrays.find(key);
  if (it == gguf.small_arrays.end() || it->second.size() != out.size()) {
    return false;
  }
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (const auto* v = std::get_if<float>(&it->second[i])) {
      out[i] = *v;
    } else if (const auto* d = std::get_if<double>(&it->second[i])) {
      out[i] = static_cast<float>(*d);
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace

std::expected<VisionConfig, StatusCode> LoadVisionConfig(
    const std::filesystem::path& path) {
  auto gguf = core::ParseGgufFile(path);
  if (!gguf) {
    return std::unexpected(gguf.error());
  }
  const core::GgufValue* arch = gguf->Find("general.architecture");
  if (arch == nullptr || std::get_if<std::string>(arch) == nullptr ||
      std::get<std::string>(*arch) != "clip") {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const core::GgufValue* has_vision = gguf->Find("clip.has_vision_encoder");
  if (has_vision == nullptr || std::get_if<bool>(has_vision) == nullptr ||
      !std::get<bool>(*has_vision)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  VisionConfig config;
  auto image_size = U64(*gguf, "clip.vision.image_size");
  auto patch = U64(*gguf, "clip.vision.patch_size");
  auto embed = U64(*gguf, "clip.vision.embedding_length");
  auto ffn = U64(*gguf, "clip.vision.feed_forward_length");
  auto blocks = U64(*gguf, "clip.vision.block_count");
  auto heads = U64(*gguf, "clip.vision.attention.head_count");
  auto proj = U64(*gguf, "clip.vision.projection_dim");
  auto merge = U64(*gguf, "clip.vision.spatial_merge_size");
  auto eps = F32(*gguf, "clip.vision.attention.layer_norm_epsilon");
  if (!image_size || !patch || !embed || !ffn || !blocks || !heads || !proj ||
      !merge || !eps) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  config.image_size = *image_size;
  config.patch_size = *patch;
  config.embedding_length = *embed;
  config.feed_forward_length = *ffn;
  config.block_count = *blocks;
  config.head_count = *heads;
  config.projection_dim = *proj;
  config.spatial_merge_size = *merge;
  config.layer_norm_eps = *eps;
  if (config.image_size == 0 || config.patch_size == 0 ||
      config.embedding_length == 0 || config.block_count == 0 ||
      config.head_count == 0 || config.projection_dim == 0 ||
      config.spatial_merge_size == 0 ||
      config.image_size % config.patch_size != 0 ||
      config.embedding_length % config.head_count != 0 ||
      !ReadFloats(*gguf, "clip.vision.image_mean", config.image_mean) ||
      !ReadFloats(*gguf, "clip.vision.image_std", config.image_std)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (const core::GgufValue* type = gguf->Find("clip.projector_type");
      type != nullptr) {
    if (const auto* s = std::get_if<std::string>(type)) {
      config.projector_type = *s;
    }
  }
  if (const core::GgufValue* gelu = gguf->Find("clip.use_gelu");
      gelu != nullptr) {
    if (const auto* b = std::get_if<bool>(gelu)) {
      config.use_gelu = *b;
    }
  }
  return config;
}

}  // namespace tessera
