#include "tessera/model.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <variant>
#include <vector>

#include "core/files.hpp"
#include "core/loaders/gguf.hpp"
#include "core/loaders/hf_tokenizer.hpp"
#include "core/loaders/mxfp4.hpp"
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

// Attention parameters from GGUF metadata; nullopt without keys.
// Explicit key/value lengths set head dim (hybrid); else embed/heads.
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
      *r == 0 || !(*t > 0.0) || (*h % *k) != 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::uint64_t head_dim = 0;
  const auto* key_len = gguf.Find(prefix + ".attention.key_length");
  const auto* value_len = gguf.Find(prefix + ".attention.value_length");
  if (key_len != nullptr || value_len != nullptr) {
    if (key_len == nullptr || value_len == nullptr) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    const auto kl = AsU64(*key_len);
    const auto vl = AsU64(*value_len);
    if (!kl || !vl || *kl == 0 || *vl == 0) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    if (*kl != *vl) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    head_dim = *kl;
  } else {
    if ((*e % *h) != 0) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    head_dim = *e / *h;
  }
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

// Hybrid recurrent definition (ssm.* plus full_attention_interval).
// Nullopt without any key, MalformedFile when partial or zero.
struct HybridDef {
  SsmParams ssm;
  std::uint64_t interval = 0;
};

std::expected<std::optional<HybridDef>, StatusCode> ParseHybrid(
    const core::GgufFile& gguf, std::string_view architecture) {
  const std::string prefix = std::string(architecture);
  const auto* conv = gguf.Find(prefix + ".ssm.conv_kernel");
  const auto* state = gguf.Find(prefix + ".ssm.state_size");
  const auto* groups = gguf.Find(prefix + ".ssm.group_count");
  const auto* rank = gguf.Find(prefix + ".ssm.time_step_rank");
  const auto* inner = gguf.Find(prefix + ".ssm.inner_size");
  const auto* interval = gguf.Find(prefix + ".full_attention_interval");
  if (conv == nullptr && state == nullptr && groups == nullptr &&
      rank == nullptr && inner == nullptr && interval == nullptr) {
    return std::optional<HybridDef>{};
  }
  if (conv == nullptr || state == nullptr || groups == nullptr ||
      rank == nullptr || inner == nullptr || interval == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const auto c = AsU64(*conv);
  const auto s = AsU64(*state);
  const auto g = AsU64(*groups);
  const auto r = AsU64(*rank);
  const auto n = AsU64(*inner);
  const auto v = AsU64(*interval);
  if (!c || !s || !g || !r || !n || !v || *c == 0 || *s == 0 || *g == 0 ||
      *r == 0 || *n == 0 || *v == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  HybridDef def;
  def.ssm.conv_kernel = static_cast<std::size_t>(*c);
  def.ssm.state_size = static_cast<std::size_t>(*s);
  def.ssm.group_count = static_cast<std::size_t>(*g);
  def.ssm.time_step_rank = static_cast<std::size_t>(*r);
  def.ssm.inner_size = static_cast<std::size_t>(*n);
  def.interval = *v;
  return std::optional<HybridDef>{def};
}

// mRoPE section pair counts from the section array; nullopt when the
// file carries none. Three integer entries, or four with a zero pad
// (the file convention); the leading three cover at most rope_dim/2
// pairs. MalformedFile on any other shape.
std::expected<std::optional<std::vector<std::uint64_t>>, StatusCode>
ParseRopeSections(const core::GgufFile& gguf, std::string_view architecture,
                  std::size_t rope_dim) {
  const auto found = gguf.small_arrays.find(
      std::string(architecture) + ".rope.dimension_sections");
  if (found == gguf.small_arrays.end()) {
    return std::optional<std::vector<std::uint64_t>>{};
  }
  if (found->second.size() != 3 && found->second.size() != 4) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::vector<std::uint64_t> sections;
  sections.reserve(found->second.size());
  for (const auto& entry : found->second) {
    const auto value = AsU64(entry);
    if (!value) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    sections.push_back(*value);
  }
  if (sections.size() == 4) {
    if (sections[3] != 0) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    sections.pop_back();
  }
  if (sections[0] + sections[1] + sections[2] > rope_dim / 2) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return std::optional<std::vector<std::uint64_t>>{sections};
}

// Build the byte-level BPE tokenizer from the retained tokenizer arrays;
// nullopt when the file declares no supported tokenizer (not gpt2, or no
// token array). MalformedFile when the arrays are inconsistent.
std::expected<std::optional<Tokenizer>, StatusCode> ParseTokenizer(
    const core::GgufFile& gguf) {
  const auto* model = gguf.Find("tokenizer.ggml.model");
  const auto* model_str = model == nullptr ? nullptr : std::get_if<std::string>(model);
  if (model_str == nullptr || *model_str != "gpt2") {
    return std::optional<Tokenizer>{};
  }
  const auto tokens_it = gguf.small_arrays.find("tokenizer.ggml.tokens");
  if (tokens_it == gguf.small_arrays.end()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::vector<std::string> vocab;
  vocab.reserve(tokens_it->second.size());
  for (const auto& value : tokens_it->second) {
    const auto* text = std::get_if<std::string>(&value);
    if (text == nullptr) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    vocab.push_back(*text);
  }
  std::vector<std::int32_t> types(vocab.size(), 1);
  const auto types_it = gguf.small_arrays.find("tokenizer.ggml.token_type");
  if (types_it != gguf.small_arrays.end()) {
    if (types_it->second.size() != vocab.size()) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    for (std::size_t i = 0; i < vocab.size(); ++i) {
      const auto value = AsU64(types_it->second[i]);
      if (!value) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      types[i] = static_cast<std::int32_t>(*value);
    }
  }
  std::vector<std::string> merges;
  const auto merges_it = gguf.small_arrays.find("tokenizer.ggml.merges");
  if (merges_it != gguf.small_arrays.end()) {
    merges.reserve(merges_it->second.size());
    for (const auto& value : merges_it->second) {
      const auto* text = std::get_if<std::string>(&value);
      if (text == nullptr) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      merges.push_back(*text);
    }
  }
  return std::optional<Tokenizer>{
      std::in_place, std::move(vocab), std::move(types), std::move(merges)};
}

// Transformer config from GGUF metadata; nullopt without layer keys.
// Hybrid layers counts trunk blocks (block_count minus nextn blocks).
std::expected<std::optional<TransformerConfig>, StatusCode> ParseConfig(
    const core::GgufFile& gguf, std::string_view architecture) {
  if (architecture.empty()) {
    return std::optional<TransformerConfig>{};
  }
  const std::string prefix = std::string(architecture);
  const auto* layers_key = gguf.Find(prefix + ".block_count");
  const auto* ffn_key = gguf.Find(prefix + ".feed_forward_length");
  const auto* eps_key =
      gguf.Find(prefix + ".attention.layer_norm_rms_epsilon");
  if (layers_key == nullptr && ffn_key == nullptr && eps_key == nullptr) {
    return std::optional<TransformerConfig>{};
  }
  auto attention = ParseAttention(gguf, architecture);
  if (!attention) {
    return std::unexpected(attention.error());
  }
  if (!*attention) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const auto* embed_key = gguf.Find(prefix + ".embedding_length");
  if (layers_key == nullptr || ffn_key == nullptr || eps_key == nullptr ||
      embed_key == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const auto layers = AsU64(*layers_key);
  const auto ffn = AsU64(*ffn_key);
  const auto eps = AsDouble(*eps_key);
  const auto hidden = AsU64(*embed_key);
  if (!layers || !ffn || !eps || !hidden || *layers == 0 || *ffn == 0 ||
      *hidden == 0 || !(*eps > 0.0)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto hybrid = ParseHybrid(gguf, architecture);
  if (!hybrid) {
    return std::unexpected(hybrid.error());
  }
  std::uint64_t trunk = *layers;
  if (*hybrid) {
    std::uint64_t nextn = 0;
    if (const auto* nextn_key = gguf.Find(prefix + ".nextn_predict_layers");
        nextn_key != nullptr) {
      const auto parsed = AsU64(*nextn_key);
      if (!parsed) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      nextn = *parsed;
    }
    if (nextn > *layers) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    trunk = *layers - nextn;
    if (trunk == 0) {
      return std::unexpected(StatusCode::MalformedFile);
    }
  }
  std::size_t vocab = 0;
  for (const auto& tensor : gguf.tensors) {
    if (tensor.name == "output.weight") {
      const std::size_t numel = tensor.shape.Numel();
      if (numel % *hidden != 0) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      vocab = numel / static_cast<std::size_t>(*hidden);
    }
  }
  if (vocab == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  TransformerConfig config;
  config.attention = **attention;
  config.layers = static_cast<std::size_t>(trunk);
  config.hidden_dim = static_cast<std::size_t>(*hidden);
  config.ffn_dim = static_cast<std::size_t>(*ffn);
  config.vocab_size = vocab;
  config.norm_eps = *eps;
  if (*hybrid) {
    config.hybrid = true;
    config.ssm = (*hybrid)->ssm;
    config.full_attention_interval =
        static_cast<std::size_t>((*hybrid)->interval);
  }
  auto sections = ParseRopeSections(gguf, architecture,
                                     config.attention.rope_dim);
  if (!sections) {
    return std::unexpected(sections.error());
  }
  if (*hybrid && !sections->has_value()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (sections->has_value()) {
    config.rope_sections = **sections;
  }
  return std::optional<TransformerConfig>{config};
}

}  // namespace

Model::Model(Backend& backend, ModelOptions options, ModelFormat format,
             std::vector<TensorEntry> tensors, std::string name,
             std::string architecture, std::optional<AttentionParams> attention,
             std::optional<TransformerConfig> config,
             std::vector<DeviceTensor> weights,
             std::optional<Tokenizer> tokenizer, std::string chat_template,
             std::vector<std::uint32_t> stop_tokens)
    : backend_(backend), options_(std::move(options)), format_(format),
      tensors_(std::move(tensors)), name_(std::move(name)),
      architecture_(std::move(architecture)),
      attention_(std::move(attention)), config_(std::move(config)),
      weights_(std::move(weights)), tokenizer_(std::move(tokenizer)),
      chat_template_(std::move(chat_template)),
      stop_tokens_(std::move(stop_tokens)) {
  module_ = CreateArchitecture(architecture_);
}

const Architecture* Model::Arch() const { return module_.get(); }

// Upload every manifest tensor to a device buffer. `bytes` is the
// whole file; offsets come from the parsed manifest.
std::expected<std::vector<DeviceTensor>, StatusCode> UploadWeights(
    Backend& backend, const std::vector<TensorEntry>& tensors,
    const std::vector<std::uint64_t>& offsets, std::uint64_t data_start,
    std::span<const std::byte> bytes) {
  std::vector<DeviceTensor> weights;
  weights.reserve(tensors.size());
  std::vector<Backend::HostCopy> copies;
  copies.reserve(tensors.size());
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
    weights.push_back(DeviceTensor{entry, std::move(*buffer)});
    copies.push_back(Backend::HostCopy{
        weights.back().device.get(),
        bytes.subspan(static_cast<std::size_t>(begin), count)});
  }
  // One batched upload: backends pipeline the copies and synchronize once
  // per chunk instead of a round trip per tensor.
  if (auto uploaded = backend.CopyH2DBatch(copies); !uploaded) {
    return std::unexpected(uploaded.error());
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
    // Map the file instead of reading it: a multi-gigabyte GGUF would
    // otherwise be copied into a zero-filled host vector (a full memset
    // of the file) before parsing and uploading.
    auto mapped = core::MappedFile::Open(path);
    if (!mapped) {
      return std::unexpected(mapped.error());
    }
    const std::span<const std::byte> bytes = mapped->bytes();
    auto gguf = core::ParseGguf(bytes);
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
    auto config = ParseConfig(*gguf, architecture);
    if (!config) {
      return std::unexpected(config.error());
    }
    auto weights =
        UploadWeights(backend, gguf->tensors, gguf->tensor_offsets,
                      gguf->tensor_data_start, bytes);
    if (!weights) {
      return std::unexpected(weights.error());
    }
    auto tokenizer = ParseTokenizer(*gguf);
    if (!tokenizer) {
      return std::unexpected(tokenizer.error());
    }
    std::string chat_template;
    if (const auto* value = gguf->Find("tokenizer.chat_template");
        value != nullptr) {
      if (const auto* text = std::get_if<std::string>(value)) {
        chat_template = *text;
      }
    }
    std::vector<std::uint32_t> stop_tokens = core::GgufStopTokens(*gguf);
    return std::unique_ptr<Model>(new Model(
        backend, options, ModelFormat::Gguf, std::move(gguf->tensors),
        std::move(name), std::move(architecture), std::move(*attention),
        std::move(*config), std::move(*weights), std::move(*tokenizer),
        std::move(chat_template), std::move(stop_tokens)));
  }
  if (std::filesystem::is_directory(path, ec) && !ec) {
    auto layout = core::InspectMxFp4Directory(path);
    if (!layout) {
      return std::unexpected(layout.error());
    }
    auto mapped = core::MappedFile::Open(layout->weights_path);
    if (!mapped) {
      return std::unexpected(mapped.error());
    }
    const std::span<const std::byte> file_bytes = mapped->bytes();
    auto parsed = core::ParseSafetensorsMap(file_bytes);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    // When config.json names a registered architecture, build the
    // internal weights through its module: renamed, value-converted and
    // MXFP4-packed. Other directories (test fixtures, unknown models)
    // fall back to the raw safetensors map.
    auto config_bytes = core::ReadFile(path / "config.json");
    if (config_bytes) {
      const std::string_view text(
          reinterpret_cast<const char*>(config_bytes->data()),
          config_bytes->size());
      std::string architecture = core::MxFp4ArchitectureName(text);
      auto module = CreateArchitecture(architecture);
      if (module != nullptr) {
        auto config = module->ParseConfigJson(text);
        if (!config) {
          return std::unexpected(config.error());
        }
        auto weights = core::BuildMxFp4Weights(
            backend, file_bytes, *parsed, *module, *config);
        if (!weights) {
          return std::unexpected(weights.error());
        }
        std::vector<TensorEntry> tensors;
        tensors.reserve(weights->size());
        for (const auto& weight : *weights) {
          tensors.push_back(weight.manifest);
        }
        auto tokenizer = core::LoadHfTokenizer(path);
        if (!tokenizer) {
          return std::unexpected(tokenizer.error());
        }
        std::string chat_template = core::LoadHfChatTemplate(path);
        std::vector<std::uint32_t> stop_tokens = core::LoadHfStopTokens(path);
        std::optional<AttentionParams> attention = config->attention;
        std::optional<TransformerConfig> parsed_config = std::move(*config);
        return std::unique_ptr<Model>(new Model(
            backend, options, ModelFormat::MxFp4, std::move(tensors),
            std::string{}, std::move(architecture), std::move(attention),
            std::move(parsed_config), std::move(*weights),
            std::move(*tokenizer), std::move(chat_template),
            std::move(stop_tokens)));
      }
    }
    std::vector<TensorEntry> tensors;
    std::vector<std::uint64_t> offsets;
    tensors.reserve(parsed->size());
    offsets.reserve(parsed->size());
    for (const auto& tensor : *parsed) {
      tensors.push_back(tensor.entry);
      offsets.push_back(tensor.begin);
    }
    // Map offsets count from the file start, so the base is zero.
    auto weights = UploadWeights(backend, tensors, offsets, 0, file_bytes);
    if (!weights) {
      return std::unexpected(weights.error());
    }
    return std::unique_ptr<Model>(new Model(
        backend, options, ModelFormat::MxFp4, std::move(tensors),
        std::string{}, std::string{}, std::nullopt, std::nullopt,
        std::move(*weights), std::nullopt, std::string{},
        core::LoadHfStopTokens(path)));
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

std::size_t Model::EffectiveMaxTokens(std::size_t prompt_size,
                                      std::size_t requested) const {
  if (requested > 0) {
    return requested;
  }
  return options_.max_context_length > prompt_size
             ? options_.max_context_length - prompt_size
             : 0;
}

std::string_view Model::Name() const {
  return name_;
}

std::expected<AttentionParams, StatusCode> Model::Attention() const {
  if (!attention_) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return *attention_;
}

std::expected<TransformerConfig, StatusCode> Model::Config() const {
  if (!config_) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return *config_;
}

const Tokenizer* Model::GetTokenizer() const {
  return tokenizer_ ? &*tokenizer_ : nullptr;
}

std::string_view Model::ChatTemplate() const { return chat_template_; }

std::span<const std::uint32_t> Model::StopTokens() const {
  return std::span<const std::uint32_t>(stop_tokens_);
}

}  // namespace tessera
