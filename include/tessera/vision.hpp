#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "tessera/backend.hpp"
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
[[nodiscard]] std::expected<VisionConfig, StatusCode> LoadVisionConfig(
    const std::filesystem::path& path);

// The loaded vision projector weights (fp32 on the device) plus the
// kernels the encoder needs. `Encode` preprocesses an [h, w, 3] fp32 image
// (values in [0, 1]), runs the CLIP encoder and the merger, and returns the
// image embeddings (out_tokens x projection_dim) as a host vector.
class VisionModel {
 public:
  VisionModel();
  ~VisionModel();
  VisionModel(VisionModel&&) noexcept;
  VisionModel& operator=(VisionModel&&) noexcept;
  VisionModel(const VisionModel&) = delete;
  VisionModel& operator=(const VisionModel&) = delete;

  [[nodiscard]] static std::expected<VisionModel, StatusCode> Load(
      Backend& backend, const std::filesystem::path& path);

  [[nodiscard]] const VisionConfig& Config() const;

  [[nodiscard]] std::expected<std::vector<float>, StatusCode> Encode(
      Backend& backend, std::span<const float> image, std::size_t h,
      std::size_t w);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tessera
