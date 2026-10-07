#pragma once

#include <expected>
#include <memory>
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
// Weight upload to the device lands in milestone 4 (docs/PROGRESS.md);
// Load today validates the file and parses the manifest only.
//
// Usage:
//   auto model = engine.LoadModel(ModelOptions{path});
//   if (model) for (auto& t : model->Tensors()) ...
class Model {
 public:
  // Load and validate from options.path on `backend`. The format is
  // detected automatically: a regular file is GGUF, a directory MXFP4.
  static std::expected<std::unique_ptr<Model>, StatusCode> Load(
      Backend& backend, const ModelOptions& options);

  [[nodiscard]] ModelFormat Format() const;
  [[nodiscard]] std::span<const TensorEntry> Tensors() const;
  [[nodiscard]] const std::string& Path() const;
  [[nodiscard]] std::size_t MaxContextLength() const;
  // Model name from the file metadata when present ("" otherwise).
  [[nodiscard]] std::string_view Name() const;

 private:
  Model(Backend& backend, ModelOptions options, ModelFormat format,
        std::vector<TensorEntry> tensors, std::string name);
  Backend& backend_;
  ModelOptions options_;
  ModelFormat format_;
  std::vector<TensorEntry> tensors_;
  std::string name_;
};

}  // namespace tessera
