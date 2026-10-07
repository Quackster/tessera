#include "tessera/model.hpp"

#include <filesystem>
#include <system_error>

#include "core/loaders/gguf.hpp"
#include "core/loaders/safetensors.hpp"

namespace tessera {

Model::Model(Backend& backend, ModelOptions options, ModelFormat format,
             std::vector<TensorEntry> tensors, std::string name)
    : backend_(backend), options_(std::move(options)), format_(format),
      tensors_(std::move(tensors)), name_(std::move(name)) {}

std::expected<std::unique_ptr<Model>, StatusCode> Model::Load(
    Backend& backend, const ModelOptions& options) {
  if (options.path.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::error_code ec;
  const std::filesystem::path path(options.path);
  if (!std::filesystem::exists(path, ec) || ec) {
    return std::unexpected(StatusCode::FileNotFound);
  }
  if (std::filesystem::is_regular_file(path, ec) && !ec) {
    auto gguf = core::ParseGgufFile(path);
    if (!gguf) {
      return std::unexpected(gguf.error());
    }
    std::string name;
    if (const auto* value = gguf->Find("general.name"); value != nullptr) {
      if (const auto* name_str = std::get_if<std::string>(value)) {
        name = *name_str;
      }
    }
    return std::unique_ptr<Model>(new Model(
        backend, options, ModelFormat::Gguf, std::move(gguf->tensors),
        std::move(name)));
  }
  if (std::filesystem::is_directory(path, ec) && !ec) {
    // MXFP4 layout check; the tensor map is parsed in milestone 6, so the
    // manifest is empty until then.
    auto layout = core::InspectMxFp4Directory(path);
    if (!layout) {
      return std::unexpected(layout.error());
    }
    return std::unique_ptr<Model>(new Model(
        backend, options, ModelFormat::MxFp4, {}, std::string{}));
  }
  return std::unexpected(StatusCode::InvalidArgument);
}

ModelFormat Model::Format() const {
  return format_;
}

std::span<const TensorEntry> Model::Tensors() const {
  return std::span<const TensorEntry>(tensors_);
}

const std::string& Model::Path() const {
  return options_.path;
}

std::size_t Model::MaxContextLength() const {
  return options_.max_context_length;
}

std::string_view Model::Name() const {
  return name_;
}

}  // namespace tessera
