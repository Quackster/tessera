#pragma once

#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tessera/backend.hpp"
#include "tessera/types.hpp"

namespace tessera {

// On-disk form of a model.
enum class ModelFormat : int {
  Gguf = 0,   // a single .gguf file
  MxFp4 = 1,  // a directory with config.json + one *.safetensors
};

// Options for loading a model.
struct ModelOptions {
  // Path to a .gguf file, or to a model directory (MXFP4 layout).
  std::string path;
  // Maximum context length to prepare for; a tunable.
  std::size_t max_context_length = 4096;
};

// A loaded model: format + options + parsed tensor manifest.
// A tensor on the device: its manifest entry plus the buffer that
// holds its bytes (allocated and uploaded by Model::Load).
struct DeviceTensor {
  TensorEntry manifest;
  std::unique_ptr<Buffer> device;
};

// A loaded model: format + options + parsed tensor manifest + the
// uploaded weight buffers (one per manifest tensor, GGUF only).
//
// Usage:
//   auto model = engine.LoadModel(ModelOptions{path});
//   if (model) for (auto& t : model->Weights()) ...
class Model {
 public:
  // Load and validate from options.path on `backend`. The format is
  // detected automatically: a regular file is GGUF, a directory MXFP4.
  static std::expected<std::unique_ptr<Model>, StatusCode> Load(
      Backend& backend, const ModelOptions& options);

  [[nodiscard]] ModelFormat Format() const;
  [[nodiscard]] std::span<const TensorEntry> Tensors() const;
  // Uploaded weights, parallel to Tensors() (empty for MXFP4, whose
  // manifest is parsed in a later milestone).
  [[nodiscard]] std::span<const DeviceTensor> Weights() const;
  // Device buffer for the tensor with this name; nullptr when absent.
  [[nodiscard]] const Buffer* FindWeight(std::string_view name) const;
  [[nodiscard]] const std::string& Path() const;
  [[nodiscard]] std::size_t MaxContextLength() const;
  // Model name from the file metadata when present ("" otherwise).
  [[nodiscard]] std::string_view Name() const;
  // Attention parameters from the model definition, for sizing kernel
  // launches (heads, kv groups, head dim, RoPE range and base). GGUF
  // reads <arch>.attention.head_count, head_count_kv, embedding_length
  // and <arch>.rope.dimension_count, rope.freq_base. MalformedFile
  // when the definition lacks them; UnsupportedFeature for MXFP4
  // (config.json parsing is a later milestone).
  //
  // Usage:
  //   auto params = model.Attention();
  //   if (params) launch.scalars = {m, n, params->heads, ...};
  [[nodiscard]] std::expected<AttentionParams, StatusCode> Attention() const;

 private:
  Model(Backend& backend, ModelOptions options, ModelFormat format,
        std::vector<TensorEntry> tensors, std::string name,
        std::string architecture, std::optional<AttentionParams> attention,
        std::vector<DeviceTensor> weights);
  Backend& backend_;
  ModelOptions options_;
  ModelFormat format_;
  std::vector<TensorEntry> tensors_;
  std::string name_;
  std::string architecture_;
  std::optional<AttentionParams> attention_;
  std::vector<DeviceTensor> weights_;
};

}  // namespace tessera
