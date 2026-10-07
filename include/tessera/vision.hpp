#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <string>

#include "tessera/types.hpp"

namespace tessera {

// The vision projector (mmproj) definition: a CLIP vision encoder plus a
// merger that projects the image tokens into the language model's hidden
// space.
struct VisionConfig {
  std::size_t image_size = 0;
  std::size_t patch_size = 0;
  std::size_t embedding_length = 0;
  std::size_t feed_forward_length = 0;
  std::size_t block_count = 0;
  std::size_t head_count = 0;
  std::size_t projection_dim = 0;
  std::size_t spatial_merge_size = 0;
  float layer_norm_eps = 0.0f;
  float image_mean[3] = {0.0f, 0.0f, 0.0f};
  float image_std[3] = {0.0f, 0.0f, 0.0f};
  std::string projector_type;
  bool use_gelu = false;
};

// Parse a vision projector (mmproj) GGUF file. The file must be a CLIP
// vision encoder (`general.architecture` "clip",
// `clip.has_vision_encoder` true). FileNotFound when missing,
// MalformedFile for a wrong architecture or a missing/invalid field.
//
// Usage:
//   auto config = LoadVisionConfig("~/models/.../mmproj-BF16.gguf");
[[nodiscard]] std::expected<VisionConfig, StatusCode> LoadVisionConfig(
    const std::filesystem::path& path);

}  // namespace tessera
