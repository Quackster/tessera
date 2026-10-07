#include "tessera/model.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

#include "core/files.hpp"
#include "core/loaders/gguf.hpp"
#include "core/loaders/safetensors.hpp"

namespace tessera {

namespace {

// Integral GGUF metadata as u64; nullopt for bool, float and text.
std::optional<std::uint64_t> AsU64(const core::GgufValue& value) {
  if (const auto* v = std::get_if<std::uint32_t>(&value)) {
    return *v;
  }
  if (const auto* v = std::get_if<std::int32_t>(&value)) {
    return *v < 0 ? std::nullopt : std::optional<std::uint64_t>(*v);
  }
  if (const auto* v = std::get_if<std::uint64_t>(&value)) {
    return *v;
  }
  if (const auto* v = std::get_if<std::int64_t>(&value)) {
    return *v < 0 ? std::nullopt : std::optional<std::uint64_t>(*v);
  }
  return std::nullopt;
}

// Floating GGUF metadata as double; integrals convert exactly.
std::optional<double> AsDouble(const core::GgufValue& value) {
  if (const auto* v = std::get_if<float>(&value)) {
    return *v;
  }
  if (const auto* v = std::get_if<double>(&value)) {
    return *v;
  }
  if (const auto count = AsU64(value)) {
    return static_cast<double>(*count);
  }
  return std::nullopt;
}

// Attention parameters from GGUF metadata; nullopt when the file
// carries no attention keys, MalformedFile when they are incomplete.
std::expected<std::optional<AttentionParams>, StatusCode>
ParseAttention(const core::GgufFile& gguf, std::string_view architecture) {
  if (architecture.empty()) {
    return std::optional<AttentionParams>{};
  }
  const std::string prefix = std::string(architecture);
  const auto* heads = gguf.Find(prefix + ".attention.head_count");
  const auto* kv = gguf.Find(prefix + ".attention.head_count_kv");
  const auto* embed = gguf.Find(prefix + ".embedding_length");
  const auto* rope_dim = gguf.Find(prefix + ".rope.dimension_count");
  const auto* theta = gguf.Find(prefix + ".rope.freq_base");
  if (heads == nullptr && kv == nullptr && embed == nullptr &&
      rope_dim == nullptr && theta == nullptr) {
    return std::optional<AttentionParams>{};
  }
  if (heads == nullptr || kv == nullptr || embed == nullptr ||
      rope_dim == nullptr || theta == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const auto h = AsU64(*heads);
  const auto k = AsU64(*kv);
  const auto e = AsU64(*embed);
  const auto r = AsU64(*rope_dim);
  const auto t = AsDouble(*theta);
  if (!h || !k || !e || !r || !t || *h == 0 || *k == 0 || *e == 0 ||
      *r == 0 || !(*t > 0.0) || (*h % *k) != 0 || (*e % *h) != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const std::uint64_t head_dim = *e / *h;
  if (*r > head_dim || (*r % 2) != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  AttentionParams params;
  params.heads = static_cast<std::size_t>(*h);
  params.kv_heads = static_cast<std::size_t>(*k);
  params.head_dim = static_cast<std::size_t>(head_dim);
  params.rope_dim = static_cast<std::size_t>(*r);
  params.rope_theta = *t;
  return std::optional<AttentionParams>{params};
}

}  // namespace

Model::Model(Backend& backend, ModelOptions options, ModelFormat format,
             std::vector<TensorEntry> tensors, std::string name,
             std::string architecture, std::optional<AttentionParams> attention,
             std::vector<DeviceTensor> weights)
    : backend_(backend), options_(std::move(options)), format_(format),
      tensors_(std::move(tensors)), name_(std::move(name)),
      architecture_(std::move(architecture)),
      attention_(std::move(attention)), weights_(std::move(weights)) {}

// Upload every manifest tensor to a device buffer. `bytes` is the
// whole file; offsets come from the parsed manifest.
std::expected<std::vector<DeviceTensor>, StatusCode> UploadWeights(
    Backend& backend, const std::vector<TensorEntry>& tensors,
    const std::vector<std::uint64_t>& offsets, std::uint64_t data_start,
    std::span<const std::byte> bytes) {
  std::vector<DeviceTensor> weights;
  weights.reserve(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& entry = tensors[i];
    auto sized = TensorBytes(entry.dtype, entry.shape.Numel());
    if (!sized) {
      return std::unexpected(sized.error());
    }
    const std::size_t count = *sized;
    const std::uint64_t begin = data_start + offsets[i];
    if (begin > bytes.size() ||
        static_cast<std::uint64_t>(count) > bytes.size() - begin) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    auto buffer = backend.AllocateBuffer(count, MemoryKind::Device);
    if (!buffer) {
      return std::unexpected(buffer.error());
    }
    auto uploaded = backend.CopyH2D(
        **buffer, bytes.subspan(static_cast<std::size_t>(begin), count));
    if (!uploaded) {
      return std::unexpected(uploaded.error());
    }
    weights.push_back(DeviceTensor{entry, std::move(*buffer)});
  }
  return weights;
}

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
    auto bytes = core::ReadFile(path);
    if (!bytes) {
      return std::unexpected(bytes.error());
    }
    auto gguf = core::ParseGguf(std::span<const std::byte>(*bytes));
    if (!gguf) {
      return std::unexpected(gguf.error());
    }
    std::string name;
    if (const auto* value = gguf->Find("general.name"); value != nullptr) {
      if (const auto* name_str = std::get_if<std::string>(value)) {
        name = *name_str;
      }
    }
    std::string architecture;
    if (const auto* value = gguf->Find("general.architecture");
        value != nullptr) {
      if (const auto* arch_str = std::get_if<std::string>(value)) {
        architecture = *arch_str;
      }
    }
    auto attention = ParseAttention(*gguf, architecture);
    if (!attention) {
      return std::unexpected(attention.error());
    }
    auto weights =
        UploadWeights(backend, gguf->tensors, gguf->tensor_offsets,
                      gguf->tensor_data_start, std::span<const std::byte>(*bytes));
    if (!weights) {
      return std::unexpected(weights.error());
    }
    return std::unique_ptr<Model>(new Model(
        backend, options, ModelFormat::Gguf, std::move(gguf->tensors),
        std::move(name), std::move(architecture), std::move(*attention),
        std::move(*weights)));
  }
  if (std::filesystem::is_directory(path, ec) && !ec) {
    // MXFP4 layout check; the tensor map is parsed in milestone 6, so the
    // manifest is empty until then.
    auto layout = core::InspectMxFp4Directory(path);
    if (!layout) {
      return std::unexpected(layout.error());
    }
    return std::unique_ptr<Model>(
        new Model(backend, options, ModelFormat::MxFp4, {}, std::string{},
                  std::string{}, std::nullopt, {}));
  }
  return std::unexpected(StatusCode::InvalidArgument);
}

ModelFormat Model::Format() const {
  return format_;
}

std::span<const TensorEntry> Model::Tensors() const {
  return std::span<const TensorEntry>(tensors_);
}

std::span<const DeviceTensor> Model::Weights() const {
  return std::span<const DeviceTensor>(weights_);
}

const Buffer* Model::FindWeight(std::string_view name) const {
  for (const auto& weight : weights_) {
    if (weight.manifest.name == name) {
      return weight.device.get();
    }
  }
  return nullptr;
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

std::expected<AttentionParams, StatusCode> Model::Attention() const {
  if (format_ != ModelFormat::Gguf) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  if (!attention_) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return *attention_;
}

}  // namespace tessera
