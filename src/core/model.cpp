#include "tessera/model.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "core/files.hpp"
#include "core/loaders/gguf.hpp"
#include "core/loaders/hf_tokenizer.hpp"
#include "core/loaders/mxfp4.hpp"
#include "core/loaders/safetensors.hpp"
#include "core/loaders/shard.hpp"
#include "core/loaders/tensor_slice.hpp"
#include "core/timing.hpp"

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

// The `general.name` metadata as text, or empty when absent or not text.
// One canonical extraction for Model::Load and Model::ProbeName.
std::string GgufName(const core::GgufFile& gguf) {
  const auto* value = gguf.Find("general.name");
  if (value == nullptr) {
    return {};
  }
  const auto* name = std::get_if<std::string>(value);
  return name != nullptr ? *name : std::string{};
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

// Sparse mixture-of-experts definition (expert_count and friends).
// Nullopt without any expert key, MalformedFile when partial or invalid.
struct MoeDef {
  std::uint64_t num_experts = 0;
  std::uint64_t experts_per_tok = 0;
  std::uint64_t intermediate = 0;
  std::uint64_t shared_intermediate = 0;
};

std::expected<std::optional<MoeDef>, StatusCode> ParseMoe(
    const core::GgufFile& gguf, std::string_view architecture) {
  const std::string prefix = std::string(architecture);
  const auto* count = gguf.Find(prefix + ".expert_count");
  const auto* used = gguf.Find(prefix + ".expert_used_count");
  const auto* inter = gguf.Find(prefix + ".expert_feed_forward_length");
  const auto* shared =
      gguf.Find(prefix + ".expert_shared_feed_forward_length");
  if (count == nullptr && used == nullptr && inter == nullptr &&
      shared == nullptr) {
    return std::optional<MoeDef>{};
  }
  if (count == nullptr || used == nullptr || inter == nullptr ||
      shared == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const auto c = AsU64(*count);
  const auto u = AsU64(*used);
  const auto i = AsU64(*inter);
  const auto s = AsU64(*shared);
  if (!c || !u || !i || !s || *c == 0 || *u == 0 || *i == 0 || *s == 0 ||
      *u > *c) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  MoeDef def;
  def.num_experts = *c;
  def.experts_per_tok = *u;
  def.intermediate = *i;
  def.shared_intermediate = *s;
  return std::optional<MoeDef>{def};
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

// Read a retained integer array metadata value; nullopt when the key is
// absent. MalformedFile when any entry is not integral.
std::expected<std::optional<std::vector<std::uint64_t>>, StatusCode>
ReadU64Array(const core::GgufFile& gguf, const std::string& key) {
  const auto it = gguf.small_arrays.find(key);
  if (it == gguf.small_arrays.end()) {
    return std::optional<std::vector<std::uint64_t>>{};
  }
  std::vector<std::uint64_t> out;
  out.reserve(it->second.size());
  for (const auto& entry : it->second) {
    const auto value = AsU64(entry);
    if (!value) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    out.push_back(*value);
  }
  return std::optional<std::vector<std::uint64_t>>{std::move(out)};
}

// Gated residual stack dimensions (hyper_connection.count / low_rank).
// Nullopt when the definition carries none; MalformedFile when partial.
std::expected<std::optional<HyperConnectionParams>, StatusCode>
ParseHyperConnection(const core::GgufFile& gguf,
                     std::string_view architecture) {
  const std::string prefix = std::string(architecture);
  const auto* count = gguf.Find(prefix + ".hyper_connection.count");
  const auto* rank = gguf.Find(prefix + ".hyper_connection.low_rank");
  if (count == nullptr && rank == nullptr) {
    return std::optional<HyperConnectionParams>{};
  }
  if (count == nullptr || rank == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const auto c = AsU64(*count);
  const auto r = AsU64(*rank);
  if (!c || !r || *c == 0 || *r == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  HyperConnectionParams params;
  params.count = static_cast<std::size_t>(*c);
  params.low_rank = static_cast<std::size_t>(*r);
  return std::optional<HyperConnectionParams>{params};
}

// Sparse-attention indexer dimensions (attention.indexer.*). Nullopt when
// the definition carries none; MalformedFile when partial.
std::expected<std::optional<IndexerParams>, StatusCode> ParseIndexer(
    const core::GgufFile& gguf, std::string_view architecture) {
  const std::string prefix = std::string(architecture);
  const auto* heads = gguf.Find(prefix + ".attention.indexer.head_count");
  const auto* key = gguf.Find(prefix + ".attention.indexer.key_length");
  const auto* topk = gguf.Find(prefix + ".attention.indexer.top_k");
  if (heads == nullptr && key == nullptr && topk == nullptr) {
    return std::optional<IndexerParams>{};
  }
  if (heads == nullptr || key == nullptr || topk == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const auto h = AsU64(*heads);
  const auto k = AsU64(*key);
  const auto t = AsU64(*topk);
  if (!h || !k || !t || *h == 0 || *k == 0 || *t == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  IndexerParams params;
  params.heads = static_cast<std::size_t>(*h);
  params.key_length = static_cast<std::size_t>(*k);
  params.top_k = static_cast<std::size_t>(*t);
  return std::optional<IndexerParams>{params};
}

// Per-layer n-gram embedding dimensions (ple.*). Nullopt when the
// definition carries no ple.layers array; MalformedFile when any other
// required key is missing or inconsistent with `hidden_dim`.
std::expected<std::optional<PleParams>, StatusCode> ParsePle(
    const core::GgufFile& gguf, std::string_view architecture,
    std::size_t hidden_dim) {
  const std::string prefix = std::string(architecture);
  auto layers = ReadU64Array(gguf, prefix + ".ple.layers");
  if (!layers) {
    return std::unexpected(layers.error());
  }
  if (!*layers) {
    return std::optional<PleParams>{};
  }
  const auto* ngram = gguf.Find(prefix + ".ple.ngram_size");
  const auto* hpng = gguf.Find(prefix + ".ple.heads_per_ngram");
  const auto* conv = gguf.Find(prefix + ".ple.conv_kernel");
  const auto* embd = gguf.Find(prefix + ".embedding_length_per_layer_input");
  auto offsets = ReadU64Array(gguf, prefix + ".ple.head_offsets");
  auto vocabs = ReadU64Array(gguf, prefix + ".ple.head_vocab_sizes");
  auto mult = ReadU64Array(gguf, prefix + ".ple.layer_multipliers");
  if (ngram == nullptr || hpng == nullptr || conv == nullptr ||
      embd == nullptr || !offsets || !vocabs || !mult || !*offsets ||
      !*vocabs || !*mult) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const auto n = AsU64(*ngram);
  const auto p = AsU64(*hpng);
  const auto c = AsU64(*conv);
  const auto e = AsU64(*embd);
  if (!n || !p || !c || !e || *n == 0 || *p == 0 || *c == 0 || *e == 0 ||
      *e > hidden_dim || (*n - 1) * *p == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (mult->value().size() != *n) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  PleParams params;
  params.layers = std::move(**layers);
  params.ngram_size = static_cast<std::size_t>(*n);
  params.heads_per_ngram = static_cast<std::size_t>(*p);
  params.conv_kernel = static_cast<std::size_t>(*c);
  params.embd_per_layer = static_cast<std::size_t>(*e);
  params.layer_multipliers = std::move(**mult);
  params.head_offsets = std::move(**offsets);
  params.head_vocab_sizes = std::move(**vocabs);
  if (params.head_offsets.size() != params.head_vocab_sizes.size()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (const auto* image = gguf.Find(prefix + ".ple.image_token_id")) {
    if (const auto value = AsU64(*image)) {
      params.image_token_id = static_cast<std::size_t>(*value);
    }
  }
  if (const auto* eos = gguf.Find(prefix + ".ple.eos_token_id")) {
    if (const auto value = AsU64(*eos)) {
      params.eos_token_id = static_cast<std::size_t>(*value);
    }
  }
  return std::optional<PleParams>{std::move(params)};
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
  auto moe = ParseMoe(gguf, architecture);
  if (!moe) {
    return std::unexpected(moe.error());
  }
  const auto* embed_key = gguf.Find(prefix + ".embedding_length");
  // A dense definition carries feed_forward_length; an MoE definition
  // carries the expert widths instead and may omit it.
  if (layers_key == nullptr || eps_key == nullptr || embed_key == nullptr ||
      (ffn_key == nullptr && !*moe)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const auto layers = AsU64(*layers_key);
  const auto eps = AsDouble(*eps_key);
  const auto hidden = AsU64(*embed_key);
  if (!layers || !eps || !hidden || *layers == 0 || *hidden == 0 ||
      !(*eps > 0.0)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const auto ffn = ffn_key == nullptr ? std::optional<std::uint64_t>{}
                                      : AsU64(*ffn_key);
  if (ffn_key != nullptr && (!ffn || *ffn == 0)) {
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
  config.ffn_dim = ffn ? static_cast<std::size_t>(*ffn) : 0;
  config.vocab_size = vocab;
  config.norm_eps = *eps;
  if (*moe) {
    config.num_experts = static_cast<std::size_t>((*moe)->num_experts);
    config.experts_per_tok = static_cast<std::size_t>((*moe)->experts_per_tok);
    config.moe_intermediate = static_cast<std::size_t>((*moe)->intermediate);
    config.shared_expert_intermediate =
        static_cast<std::size_t>((*moe)->shared_intermediate);
  }
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
  auto hc = ParseHyperConnection(gguf, architecture);
  if (!hc) {
    return std::unexpected(hc.error());
  }
  if (*hc) {
    config.hyper_connection = **hc;
  }
  auto indexer = ParseIndexer(gguf, architecture);
  if (!indexer) {
    return std::unexpected(indexer.error());
  }
  if (*indexer) {
    config.indexer = **indexer;
  }
  auto ple = ParsePle(gguf, architecture, config.hidden_dim);
  if (!ple) {
    return std::unexpected(ple.error());
  }
  if (*ple) {
    config.ple = **ple;
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
             std::vector<std::uint32_t> stop_tokens,
             std::vector<PagedTensor> paged,
             std::vector<std::pair<std::string, std::unique_ptr<Buffer>>> staged,
             std::shared_ptr<const void> file_backing,
             std::shared_ptr<const void> retained_backing)
    : backend_(backend), options_(std::move(options)), format_(format),
      tensors_(std::move(tensors)), name_(std::move(name)),
      architecture_(std::move(architecture)),
      attention_(std::move(attention)), config_(std::move(config)),
      weights_(std::move(weights)), paged_(std::move(paged)),
      file_backing_(std::move(file_backing)),
      retained_backing_(std::move(retained_backing)),
      tokenizer_(std::move(tokenizer)),
      chat_template_(std::move(chat_template)),
      stop_tokens_(std::move(stop_tokens)) {
  // Index the weights by name once, so the decode loop's per-op lookups do
  // not scan every tensor (a 27B model has about 866). The keys are views
  // into weights_, which is immutable after load.
  weight_index_.reserve(weights_.size());
  for (std::size_t i = 0; i < weights_.size(); ++i) {
    weight_index_.emplace(weights_[i].manifest.name, i);
  }
  paged_index_.reserve(paged_.size());
  for (std::size_t i = 0; i < paged_.size(); ++i) {
    paged_index_.emplace(paged_[i].manifest.name, i);
  }
  // Group the paged indices by trunk layer once, so the per-token
  // ensures walk only the layer they stage (about ten tensors, not
  // every paged tensor of the model).
  for (std::size_t i = 0; i < paged_.size(); ++i) {
    auto& table = core::IsRoutedExpertKind(paged_[i].kind)
                      ? paged_expert_by_layer_
                      : paged_dense_by_layer_;
    if (table.size() <= paged_[i].layer) {
      table.resize(paged_[i].layer + 1);
    }
    table[paged_[i].layer].push_back(i);
  }
  // One staging view per paged kind; the manifest is filled when a layer
  // is ensured and the device buffer is the kind's staging slot.
  stage_views_.reserve(staged.size());
  for (auto& slot : staged) {
    stage_index_.emplace(slot.first, stage_views_.size());
    stage_views_.push_back(DeviceTensor{TensorEntry{}, std::move(slot.second)});
  }
  module_ = CreateArchitecture(architecture_);
}

const Architecture* Model::Arch() const { return module_.get(); }

const ModelShard& Model::Shard() const { return options_.shard; }

const TensorShard& Model::Tensor() const { return options_.tensor; }

// Upload every manifest tensor to a device buffer. `reader` is a pread
// reader over the file; offsets come from the parsed manifest and each
// tensor is read into a bounded buffer instead of a whole-file mapping, so
// the load does not fill the page cache with file pages that memory
// pressure evicts and re-faults. `log` receives one phase line per upload
// stage (allocation, transfer) so a slow load is traceable to a stage; it
// may be null. Under `--gpu-layers` paging only the resident tensors are
// uploaded; the rest return as RAM-backed paged entries, and stage_sizes
// carries one staging size per paged tensor kind.
struct SplitUpload {
  std::vector<DeviceTensor> resident;
  std::vector<PagedTensor> paged;
  std::vector<std::vector<std::byte>> sliced;
  std::vector<std::pair<std::string, std::size_t>> stage_sizes;
};

std::expected<SplitUpload, StatusCode> UploadWeights(
    Backend& backend, const std::vector<TensorEntry>& tensors,
    const std::vector<std::uint64_t>& offsets, std::uint64_t data_start,
    const core::FileReader& reader, const log::Diagnostics* log,
    const ModelShard& shard, std::size_t trunk_layers,
    const TensorShard& tensor, const Architecture* module,
    std::size_t gpu_layers, bool has_moe) {
  SplitUpload out;
  out.resident.reserve(tensors.size());
  std::vector<Backend::HostCopy> copies;
  copies.reserve(tensors.size());
  // Sliced tensor bytes and paged RAM copies must outlive the upload, so
  // keep them in one stable vector (moving the inner vectors preserves
  // their data).
  std::vector<std::vector<std::byte>> sliced_storage;
  // Read buffers for the current upload chunk (the HostCopy spans point
  // into them); they are freed after the chunk is uploaded, so the load
  // does not hold every resident tensor's bytes at once and its peak stays
  // under the host's available memory.
  std::vector<std::vector<std::byte>> chunk_buffers;
  chunk_buffers.reserve(tensors.size());
  std::size_t chunk_bytes = 0;
  std::unordered_map<std::string, std::size_t> stage_max;
  std::uint64_t total_bytes = 0;
  // Upload the resident tensors in bounded chunks so the read buffers are
  // freed as each chunk uploads; 2 GiB keeps the peak far under a 60 GB
  // host that already holds the 31 GiB paged copy.
  const std::size_t kUploadChunkBytes = 2u * 1024u * 1024u * 1024u;
  auto flush_chunk = [&backend, &copies, &chunk_buffers, &chunk_bytes]()
      -> std::expected<void, StatusCode> {
    if (chunk_bytes == 0) {
      return {};
    }
    if (auto uploaded = backend.CopyH2DBatch(copies); !uploaded) {
      return std::unexpected(uploaded.error());
    }
    copies.clear();
    chunk_buffers.clear();
    chunk_bytes = 0;
    return {};
  };
  {
    core::PhaseTimer timer(log, "model", "allocate device buffers");
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      TensorEntry entry = tensors[i];
      if (!core::ShardKeepsTensor(entry.name, shard, trunk_layers)) {
        continue;
      }
      auto sized = TensorBytes(entry.dtype, entry.shape.Numel());
      if (!sized) {
        return std::unexpected(sized.error());
      }
      const std::size_t count = *sized;
      const std::uint64_t begin = data_start + offsets[i];
      if (begin > reader.size() ||
          static_cast<std::uint64_t>(count) > reader.size() - begin) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      auto raw = reader.ReadRange(begin, count);
      if (!raw) {
        return std::unexpected(raw.error());
      }
      std::span<const std::byte> payload(raw->data(), raw->size());
      // Set when the slicing below consumes `raw` into sliced_storage, so
      // the paged/resident handling does not move it a second time.
      bool raw_consumed = false;
      if (tensor.enabled && module != nullptr &&
          (entry.shape.rank == 2 || entry.shape.rank == 3)) {
        const ParallelKind kind = module->TensorParallelKind(entry.name);
        if (kind == ParallelKind::Expert && entry.shape.rank == 3) {
          // The outer dim is the expert count; slice it for this rank.
          const std::size_t num_experts = entry.shape.dims[2];
          if (num_experts == 0) {
            return std::unexpected(StatusCode::MalformedFile);
          }
          const std::size_t per = entry.shape.Numel() / num_experts;
          auto sliced = core::SliceExperts(entry.dtype, num_experts, per,
                                           tensor.rank, tensor.world, payload);
          if (!sliced) {
            return std::unexpected(sliced.error());
          }
          entry.shape.dims[2] = sliced->experts;
          sliced_storage.push_back(std::move(sliced->bytes));
          payload = sliced_storage.back();
          raw_consumed = true;
        } else if (kind != ParallelKind::Replicated &&
                   entry.shape.rank == 2) {
          // GGUF dims are [row_length, rows] (ne[0] fastest); the slicer
          // takes (rows, row_length).
          auto sliced = core::SliceTensor2D(
              entry.dtype, entry.shape.dims[1], entry.shape.dims[0], kind,
              tensor.rank, tensor.world, payload);
          if (!sliced) {
            return std::unexpected(sliced.error());
          }
          entry.shape.dims[1] = sliced->rows;
          entry.shape.dims[0] = sliced->cols;
          sliced_storage.push_back(std::move(sliced->bytes));
          payload = sliced_storage.back();
          raw_consumed = true;
        }
      }
      // Under paging the early trunk blocks stay RAM-backed: no device
      // buffer is allocated now; the decode loop stages them per layer.
      // Anything outside `blk.<layer>.` (or past the trunk) stays
      // resident, as does everything when paging is off. In a model with
      // routed experts the dense block tensors stay resident too and only
      // the routed experts page (see PagedMoeDenseResident).
      bool resident =
          core::GpuLayerResident(entry.name, gpu_layers, trunk_layers) ||
          core::PagedMoeDenseResident(entry.name, gpu_layers, trunk_layers,
                                      has_moe);
      std::size_t layer = 0;
      std::string block_kind;
      if (!resident) {
        std::string_view kind_view;
        if (!core::SplitBlockName(entry.name, &layer, &kind_view) ||
            layer >= trunk_layers) {
          resident = true;
        } else {
          block_kind = std::string(kind_view);
        }
      }
      if (!resident) {
        auto staged = stage_max.find(block_kind);
        if (staged == stage_max.end()) {
          stage_max.emplace(block_kind, payload.size());
        } else if (staged->second < payload.size()) {
          staged->second = payload.size();
        }
        // Hold the paged bytes in system RAM. The decode loop re-reads
        // them every token; a private copy keeps those reads stable when
        // the file's page cache is reclaimed, or returns stale bytes under
        // memory pressure. `sliced_storage` keeps the copies alive and its
        // inner vectors do not move their data on reallocation.
        if (!raw_consumed) {
          // Move the read buffer into the paged RAM copy (no extra copy).
          sliced_storage.push_back(std::move(*raw));
          payload = std::span(sliced_storage.back());
        }
        out.paged.push_back(PagedTensor{std::move(entry),
                                         sliced_storage.back(), layer,
                                         std::move(block_kind)});
        continue;
      }
      if (!raw_consumed) {
        // Hold the read buffer in the current upload chunk (freed after
        // the chunk uploads); paged read buffers are in sliced_storage.
        chunk_buffers.push_back(std::move(*raw));
        payload = std::span(chunk_buffers.back());
      }
      auto buffer = backend.AllocateBuffer(payload.size(), MemoryKind::Device);
      if (!buffer) {
        return std::unexpected(buffer.error());
      }
      total_bytes += payload.size();
      out.resident.push_back(
          DeviceTensor{std::move(entry), std::move(*buffer)});
      copies.push_back(Backend::HostCopy{out.resident.back().device.get(),
                                         payload});
      chunk_bytes += payload.size();
      if (chunk_bytes >= kUploadChunkBytes) {
        if (auto flushed = flush_chunk(); !flushed) {
          return std::unexpected(flushed.error());
        }
      }
    }
    timer.Stop(std::to_string(tensors.size()) + " tensors");
  }
  // Flush the final partial chunk; the earlier chunks were uploaded (and
  // their read buffers freed) inside the loop as each reached the bound, so
  // the load's peak stays under the host's available memory instead of
  // holding every resident tensor at once.
  {
    core::PhaseTimer timer(log, "model", "upload weights to device");
    if (auto flushed = flush_chunk(); !flushed) {
      return std::unexpected(flushed.error());
    }
    timer.Stop(std::to_string(total_bytes) + " bytes in " +
               std::to_string(out.resident.size()) + " copies");
  }
  for (const auto& staged : stage_max) {
    out.stage_sizes.emplace_back(staged.first, staged.second);
  }
  out.sliced = std::move(sliced_storage);
  return out;
}

// One staging slot per paged tensor kind; the decode loop reuses them for
// every paged layer.
std::expected<std::vector<std::pair<std::string, std::unique_ptr<Buffer>>>,
              StatusCode>
AllocateStaging(
    Backend& backend,
    const std::vector<std::pair<std::string, std::size_t>>& stage_sizes) {
  std::vector<std::pair<std::string, std::unique_ptr<Buffer>>> staged;
  staged.reserve(stage_sizes.size());
  for (const auto& sized : stage_sizes) {
    auto slot = backend.AllocateBuffer(sized.second, MemoryKind::Device);
    if (!slot) {
      return std::unexpected(slot.error());
    }
    staged.emplace_back(sized.first, std::move(*slot));
  }
  return staged;
}

// RAM backing produced by downloading: resident device tensors plus the
// downloaded overflow under `--gpu-layers` paging.
struct EvictedWeights {
  std::vector<DeviceTensor> resident;
  std::vector<PagedTensor> paged;
  std::vector<std::vector<std::byte>> converted;
  std::vector<std::pair<std::string, std::size_t>> stage_sizes;
};

// Download every non-resident weight to RAM and release its device buffer,
// so a loader that must upload everything (MXFP4 conversion) still pages.
// The resident path is untouched; only paged tensors pay the extra
// download. Transiently holds the full model on the device, so a model
// that never fits cannot load paged either.
std::expected<EvictedWeights, StatusCode> EvictPagedLayers(
    Backend& backend, std::vector<DeviceTensor> uploaded,
    std::size_t trunk_layers, std::size_t gpu_layers, bool has_moe) {
  EvictedWeights out;
  out.resident.reserve(uploaded.size());
  std::unordered_map<std::string, std::size_t> stage_max;
  for (auto& weight : uploaded) {
    bool resident =
        weight.device == nullptr ||
        core::GpuLayerResident(weight.manifest.name, gpu_layers,
                               trunk_layers) ||
        core::PagedMoeDenseResident(weight.manifest.name, gpu_layers,
                                    trunk_layers, has_moe);
    std::size_t layer = 0;
    std::string block_kind;
    if (!resident) {
      std::string_view kind_view;
      if (!core::SplitBlockName(weight.manifest.name, &layer, &kind_view) ||
          layer >= trunk_layers) {
        resident = true;
      } else {
        block_kind = std::string(kind_view);
      }
    }
    if (resident) {
      out.resident.push_back(std::move(weight));
      continue;
    }
    std::vector<std::byte> host(weight.device->Size());
    if (auto down = backend.CopyD2H(*weight.device, host.data(), host.size());
        !down) {
      return std::unexpected(down.error());
    }
    auto staged = stage_max.find(block_kind);
    if (staged == stage_max.end()) {
      stage_max.emplace(block_kind, host.size());
    } else if (staged->second < host.size()) {
      staged->second = host.size();
    }
    out.converted.push_back(std::move(host));
    TensorEntry manifest = weight.manifest;
    weight.device.reset();
    out.paged.push_back(PagedTensor{
        std::move(manifest),
        std::span<const std::byte>(out.converted.back()), layer,
        std::move(block_kind)});
  }
  for (const auto& staged : stage_max) {
    out.stage_sizes.emplace_back(staged.first, staged.second);
  }
  return out;
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
  const log::Diagnostics* log = backend.Diagnostics();
  if (std::filesystem::is_regular_file(path, ec) && !ec) {
    // Map the file instead of reading it: a multi-gigabyte GGUF would
    // otherwise be copied into a zero-filled host vector (a full memset
    // of the file) before parsing and uploading.
    auto mapped = core::MappedFile::Open(path);
    if (!mapped) {
      return std::unexpected(mapped.error());
    }
    const std::span<const std::byte> bytes = mapped->bytes();
    core::PhaseTimer gguf_timer(log, "model", "parse gguf");
    auto gguf = core::ParseGguf(bytes);
    if (!gguf) {
      return std::unexpected(gguf.error());
    }
    gguf_timer.Stop(std::to_string(gguf->tensors.size()) + " tensors");
    std::string name = GgufName(*gguf);
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
    std::expected<std::optional<TransformerConfig>, StatusCode> config;
    {
      core::PhaseTimer config_timer(log, "model", "parse config");
      config = ParseConfig(*gguf, architecture);
    }
    if (!config) {
      return std::unexpected(config.error());
    }
    // The module answers the tensor-parallel role of each internal tensor
    // so the loader can slice it; a null module loads every tensor whole.
    std::unique_ptr<Architecture> module = CreateArchitecture(architecture);
    const bool has_moe = config->has_value() && (*config)->IsMoe();
    // Read the tensor data with a bounded pread read instead of the whole-
    // file mapping above: a multi-gigabyte load would otherwise fill the
    // page cache with file pages that memory pressure evicts and re-faults,
    // returning stale bytes to the paged copy and the device upload.
    auto reader = core::FileReader::Open(path);
    if (!reader) {
      return std::unexpected(reader.error());
    }
    auto uploaded =
        UploadWeights(backend, gguf->tensors, gguf->tensor_offsets,
                      gguf->tensor_data_start, *reader, log, options.shard,
                      config->has_value() ? (*config)->layers : 0,
                      options.tensor, module.get(), options.gpu_layers,
                      has_moe);
    if (!uploaded) {
      return std::unexpected(uploaded.error());
    }
    // One staging slot per paged tensor kind; the decode loop reuses them
    // for every paged layer.
    auto staged = AllocateStaging(backend, uploaded->stage_sizes);
    if (!staged) {
      return std::unexpected(staged.error());
    }
    // The upload copies every paged tensor into RAM (`retained_backing`),
    // so no file mapping is retained and the file's page cache is released
    // once the load returns.
    std::shared_ptr<const void> file_backing{};
    // `sliced` owns the paged tensors' RAM copies and any tensor-parallel
    // slices those spans point into. It is retained whenever paging is on;
    // without paging the resident bytes are already on the device, so the
    // default path keeps its memory behavior.
    std::shared_ptr<const void> retained_backing =
        uploaded->paged.empty() || uploaded->sliced.empty()
            ? nullptr
            : std::make_shared<std::vector<std::vector<std::byte>>>(
                  std::move(uploaded->sliced));
    std::expected<std::optional<Tokenizer>, StatusCode> tokenizer;
    {
      core::PhaseTimer tokenizer_timer(log, "model", "build tokenizer");
      tokenizer = ParseTokenizer(*gguf);
    }
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
        std::move(*config), std::move(uploaded->resident),
        std::move(*tokenizer), std::move(chat_template),
        std::move(stop_tokens), std::move(uploaded->paged), std::move(*staged),
        std::move(file_backing), std::move(retained_backing)));
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
    std::expected<std::vector<core::SafetensorsTensor>, StatusCode> parsed;
    {
      core::PhaseTimer map_timer(log, "model", "parse safetensors map");
      parsed = core::ParseSafetensorsMap(file_bytes);
    }
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
        std::expected<std::vector<DeviceTensor>, StatusCode> weights;
        {
          core::PhaseTimer build_timer(log, "model", "build mxfp4 weights");
          weights = core::BuildMxFp4Weights(
              backend, file_bytes, *parsed, *module, *config,
              WeightLoadOptions{options.load_mtp_head}, options.shard,
              options.tensor);
        }
        if (!weights) {
          return std::unexpected(weights.error());
        }
        // Converted weights upload fully; paging downloads the early
        // trunk blocks back to RAM afterwards. A zero layer count keeps
        // the full upload with no extra traffic.
        std::vector<DeviceTensor> resident = std::move(*weights);
        std::vector<PagedTensor> paged;
        std::vector<std::vector<std::byte>> converted;
        std::vector<std::pair<std::string, std::size_t>> stage_sizes;
        if (options.gpu_layers != 0) {
          auto evicted = EvictPagedLayers(backend, std::move(resident),
                                          config->layers, options.gpu_layers,
                                          config->IsMoe());
          if (!evicted) {
            return std::unexpected(evicted.error());
          }
          resident = std::move(evicted->resident);
          paged = std::move(evicted->paged);
          converted = std::move(evicted->converted);
          stage_sizes = std::move(evicted->stage_sizes);
        }
        auto staged = AllocateStaging(backend, stage_sizes);
        if (!staged) {
          return std::unexpected(staged.error());
        }
        std::vector<TensorEntry> tensors;
        tensors.reserve(resident.size() + paged.size());
        for (const auto& weight : resident) {
          tensors.push_back(weight.manifest);
        }
        for (const auto& entry : paged) {
          tensors.push_back(entry.manifest);
        }
        auto tokenizer = core::LoadHfTokenizer(path);
        if (!tokenizer) {
          return std::unexpected(tokenizer.error());
        }
        std::string chat_template = core::LoadHfChatTemplate(path);
        std::vector<std::uint32_t> stop_tokens = core::LoadHfStopTokens(path);
        std::optional<AttentionParams> attention = config->attention;
        std::optional<TransformerConfig> parsed_config = std::move(*config);
        // The downloaded blobs stay alive in the model while paged spans
        // point at them; null without paging, so the default path keeps
        // its memory behavior.
        std::shared_ptr<const void> retained_backing =
            converted.empty()
                ? nullptr
                : std::make_shared<std::vector<std::vector<std::byte>>>(
                      std::move(converted));
        return std::unique_ptr<Model>(new Model(
            backend, options, ModelFormat::MxFp4, std::move(tensors),
            std::string{}, std::move(architecture), std::move(attention),
            std::move(parsed_config), std::move(resident),
            std::move(*tokenizer), std::move(chat_template),
            std::move(stop_tokens), std::move(paged), std::move(*staged),
            nullptr, std::move(retained_backing)));
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
    // A raw directory has no trunk count, so every tensor stays resident;
    // the backing is still retained for uniformity. Read the tensor data
    // with a bounded pread read (see the GGUF path above) so the load does
    // not fill the page cache with stale-able file pages.
    auto reader = core::FileReader::Open(layout->weights_path);
    if (!reader) {
      return std::unexpected(reader.error());
    }
    auto uploaded = UploadWeights(backend, tensors, offsets, 0, *reader,
                                  log, options.shard, 0, options.tensor,
                                  nullptr, options.gpu_layers,
                                  /*has_moe=*/false);
    if (!uploaded) {
      return std::unexpected(uploaded.error());
    }
    auto staged = AllocateStaging(backend, uploaded->stage_sizes);
    if (!staged) {
      return std::unexpected(staged.error());
    }
    auto file_backing = uploaded->paged.empty()
                              ? std::shared_ptr<const void>{}
                              : std::make_shared<core::MappedFile>(
                                    std::move(*mapped));
    std::shared_ptr<const void> retained_backing =
        uploaded->paged.empty() || uploaded->sliced.empty()
            ? nullptr
            : std::make_shared<std::vector<std::vector<std::byte>>>(
                  std::move(uploaded->sliced));
    return std::unique_ptr<Model>(new Model(
        backend, options, ModelFormat::MxFp4, std::move(tensors),
        std::string{}, std::string{}, std::nullopt, std::nullopt,
        std::move(uploaded->resident), std::nullopt, std::string{},
        core::LoadHfStopTokens(path), std::move(uploaded->paged),
        std::move(*staged), std::move(file_backing),
        std::move(retained_backing)));
  }
  return std::unexpected(StatusCode::InvalidArgument);
}

std::expected<TransformerConfig, StatusCode> Model::ProbeConfig(
    const std::string& path) {
  if (path.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::error_code ec;
  const std::filesystem::path file_path(path);
  if (!std::filesystem::exists(file_path, ec) || ec) {
    return std::unexpected(StatusCode::FileNotFound);
  }
  if (std::filesystem::is_regular_file(file_path, ec) && !ec) {
    auto mapped = core::MappedFile::Open(file_path);
    if (!mapped) {
      return std::unexpected(mapped.error());
    }
    auto gguf = core::ParseGguf(mapped->bytes());
    if (!gguf) {
      return std::unexpected(gguf.error());
    }
    std::string architecture;
    if (const auto* value = gguf->Find("general.architecture");
        value != nullptr) {
      if (const auto* arch_str = std::get_if<std::string>(value)) {
        architecture = *arch_str;
      }
    }
    auto config = ParseConfig(*gguf, architecture);
    if (!config) {
      return std::unexpected(config.error());
    }
    if (!*config) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    return **config;
  }
  if (std::filesystem::is_directory(file_path, ec) && !ec) {
    auto config_bytes = core::ReadFile(file_path / "config.json");
    if (!config_bytes) {
      return std::unexpected(config_bytes.error());
    }
    const std::string_view text(
        reinterpret_cast<const char*>(config_bytes->data()),
        config_bytes->size());
    const std::string architecture = core::MxFp4ArchitectureName(text);
    auto module = CreateArchitecture(architecture);
    if (module == nullptr) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    return module->ParseConfigJson(text);
  }
  return std::unexpected(StatusCode::InvalidArgument);
}

std::expected<std::string, StatusCode> Model::ProbeName(
    const std::string& path) {
  if (path.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::error_code ec;
  const std::filesystem::path file_path(path);
  if (!std::filesystem::exists(file_path, ec) || ec) {
    return std::unexpected(StatusCode::FileNotFound);
  }
  // A directory (MXFP4) carries no GGUF name; Model::Load leaves the name
  // empty there, so the caller keys on the path basename.
  if (!std::filesystem::is_regular_file(file_path, ec) || ec) {
    return std::string{};
  }
  auto mapped = core::MappedFile::Open(file_path);
  if (!mapped) {
    return std::unexpected(mapped.error());
  }
  auto gguf = core::ParseGguf(mapped->bytes());
  if (!gguf) {
    return std::unexpected(gguf.error());
  }
  return GgufName(*gguf);
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

const DeviceTensor* Model::FindDeviceTensor(std::string_view name) const {
  const auto it = weight_index_.find(name);
  if (it != weight_index_.end()) {
    return &weights_[it->second];
  }
  // A paged name resolves only while its layer is staged; otherwise the
  // caller missed its EnsureLayerResident and gets a miss, which the
  // NeedWeight helpers report as MalformedFile. Routed experts resolve
  // through the expert staging instead (see EnsureLayerExperts): the
  // whole-layer ensure skips them.
  const auto paged = paged_index_.find(name);
  if (paged == paged_index_.end()) {
    return nullptr;
  }
  const PagedTensor& entry = paged_[paged->second];
  if (core::IsRoutedExpertKind(entry.kind)) {
    if (!experts_layer_.has_value() || *experts_layer_ != entry.layer) {
      return nullptr;
    }
  } else if (!ensured_layer_.has_value() || *ensured_layer_ != entry.layer) {
    return nullptr;
  }
  const auto staged = stage_index_.find(entry.kind);
  return staged == stage_index_.end() ? nullptr
                                      : &stage_views_[staged->second];
}

std::optional<DType> Model::TensorDType(std::string_view name) const {
  for (const TensorEntry& entry : tensors_) {
    if (entry.name == name) {
      return entry.dtype;
    }
  }
  return std::nullopt;
}

const Buffer* Model::FindWeight(std::string_view name) const {
  const DeviceTensor* found = FindDeviceTensor(name);
  return found == nullptr ? nullptr : found->device.get();
}

std::span<const PagedTensor> Model::Paged() const {
  return std::span<const PagedTensor>(paged_);
}

std::expected<void, StatusCode> Model::EnsureLayerResident(
    std::size_t layer) const {
  if (paged_.empty()) {
    return {};
  }
  if (ensured_layer_.has_value() && *ensured_layer_ == layer) {
    return {};
  }
  // One batched upload for the layer's paged tensors, then point the
  // kind views at this layer's manifests. Resident layers stage nothing
  // but still mark the layer ensured. Routed experts are skipped: the
  // MoE path stages only the selected experts (see EnsureLayerExperts),
  // which moves top_k slices instead of every expert. The per-layer
  // index keeps this to the layer's own entries; an empty layer skips
  // the upload round trip entirely.
  const std::vector<std::size_t>* rows =
      layer < paged_dense_by_layer_.size() ? &paged_dense_by_layer_[layer]
                                           : nullptr;
  if (rows != nullptr && !rows->empty()) {
    std::vector<Backend::HostCopy> copies;
    std::vector<std::size_t> slots;
    copies.reserve(rows->size());
    slots.reserve(rows->size());
    for (const std::size_t i : *rows) {
      const PagedTensor& entry = paged_[i];
      const auto staged = stage_index_.find(entry.kind);
      if (staged == stage_index_.end()) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      copies.push_back(Backend::HostCopy{stage_views_[staged->second].device.get(),
                                         entry.host});
      slots.push_back(staged->second);
    }
    if (auto uploaded = backend_.CopyH2DBatch(copies); !uploaded) {
      return std::unexpected(uploaded.error());
    }
    for (std::size_t k = 0; k < rows->size(); ++k) {
      stage_views_[slots[k]].manifest = paged_[(*rows)[k]].manifest;
    }
  }
  ensured_layer_ = layer;
  return {};
}

bool Model::ExpertsPaged(std::size_t layer) const {
  return layer < paged_expert_by_layer_.size() &&
         !paged_expert_by_layer_[layer].empty();
}

// One expert's byte size and count from a rank-3 packed expert manifest.
// MalformedFile when the shape cannot hold experts.
std::expected<std::pair<std::size_t, std::size_t>, StatusCode> ExpertSpan(
    const TensorEntry& manifest) {
  if (manifest.shape.rank != 3 || manifest.shape.dims[2] == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const std::size_t count = manifest.shape.dims[2];
  auto per = TensorBytes(manifest.dtype, manifest.shape.Numel() / count);
  if (!per || *per == 0) {
    return std::unexpected(per ? StatusCode::MalformedFile : per.error());
  }
  return std::make_pair(*per, count);
}

// A repeated expert costs one device-side gather instead of a host upload,
// so a larger store means fewer per-token uploads. The slot count trades
// the free device memory against the reserve (see ExpertCacheSlots).
std::size_t Model::ExpertCacheEntries() const {
  if (expert_cache_entries_ != 0) {
    return expert_cache_entries_;
  }
  // Every paged (layer, kind) pair gets the same slot count, so the cache
  // spends one slot-set per pair: entries = budget / total_per, where
  // total_per sums the per-expert slice bytes over every pair.
  std::size_t total_per = 0;
  for (const PagedTensor& entry : paged_) {
    if (!core::IsRoutedExpertKind(entry.kind)) {
      continue;
    }
    auto span = ExpertSpan(entry.manifest);
    if (span) {
      total_per += span->first;
    }
  }
  expert_cache_entries_ =
      core::ExpertCacheSlots(backend_.MemoryInfo().free_bytes, total_per);
  return expert_cache_entries_;
}

std::expected<void, StatusCode> Model::EnsureLayerExperts(
    std::size_t layer) const {
  if (paged_.empty()) {
    return {};
  }
  if (experts_layer_.has_value() && *experts_layer_ == layer) {
    return {};
  }
  // Whole upload for the batched case: every expert tensor lands fully.
  const std::vector<std::size_t>* rows =
      layer < paged_expert_by_layer_.size() ? &paged_expert_by_layer_[layer]
                                            : nullptr;
  if (rows != nullptr && !rows->empty()) {
    std::vector<Backend::HostCopy> copies;
    std::vector<std::size_t> slots;
    copies.reserve(rows->size());
    slots.reserve(rows->size());
    for (const std::size_t i : *rows) {
      const PagedTensor& entry = paged_[i];
      const auto staged = stage_index_.find(entry.kind);
      if (staged == stage_index_.end()) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      copies.push_back(Backend::HostCopy{stage_views_[staged->second].device.get(),
                                         entry.host});
      slots.push_back(staged->second);
    }
    if (auto uploaded = backend_.CopyH2DBatch(copies); !uploaded) {
      return std::unexpected(uploaded.error());
    }
    for (std::size_t k = 0; k < rows->size(); ++k) {
      stage_views_[slots[k]].manifest = paged_[(*rows)[k]].manifest;
    }
  }
  // A whole upload covers every expert id, so later selective calls for
  // this layer stage nothing.
  experts_staged_.clear();
  experts_staged_all_ = true;
  experts_layer_ = layer;
  return {};
}

std::expected<void, StatusCode> Model::EnsureLayerExperts(
    std::size_t layer, std::span<const std::uint32_t> experts) const {
  if (paged_.empty()) {
    return {};
  }
  // Selective upload for the single-row case: only the routed experts
  // land, at their expert offsets, so kernels keep the unchanged ids.
  // One batched upload covers every slice, so a token pays one staging
  // round trip per layer instead of one per expert. The per-layer index
  // keeps this to the layer's own expert entries, and the staged set
  // below stages only ids missing from an earlier call for the same
  // layer: a prefill batch whose rows share experts uploads each one
  // once instead of once per row. The staged bytes are identical either
  // way, so the kernels read the same weights.
  //
  // An expert already in the kind's device cache gathers from the store
  // into the staging view with a device-side copy (ordered with the
  // kernels on both backends, no host round trip); only cache misses
  // upload from the host, into a store slot, then gather the same way.
  const std::vector<std::size_t>* rows =
      layer < paged_expert_by_layer_.size() ? &paged_expert_by_layer_[layer]
                                           : nullptr;
  const bool same_layer =
      experts_layer_.has_value() && *experts_layer_ == layer;
  if (rows == nullptr || rows->empty()) {
    experts_layer_ = layer;
    experts_staged_.clear();
    experts_staged_all_ = false;
    return {};
  }
  if (same_layer && experts_staged_all_) {
    return {};
  }
  // Sorted unique ids for the set difference below. An id outside the
  // tensor's expert count is MalformedFile, reported before any upload.
  std::vector<std::uint32_t> wanted(experts.begin(), experts.end());
  std::sort(wanted.begin(), wanted.end());
  wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
  std::vector<std::uint32_t> missing;
  if (same_layer && !experts_staged_all_) {
    missing.reserve(wanted.size());
    for (const std::uint32_t id : wanted) {
      if (!std::binary_search(experts_staged_.begin(), experts_staged_.end(),
                              id)) {
        missing.push_back(id);
      }
    }
    if (missing.empty()) {
      return {};
    }
  } else {
    missing = wanted;
  }
  // One cache per (layer, kind) row, parallel to the layer's expert
  // rows and resolved before any upload so a malformed manifest still
  // reports before bytes move. A null cache bypasses caching and uploads
  // straight into the staging view.
  struct KindWork {
    std::size_t row = 0;
    std::size_t stage_slot = 0;
    std::size_t per = 0;
    ExpertCache* cache = nullptr;
  };
  if (expert_cache_by_layer_.size() <= layer) {
    expert_cache_by_layer_.resize(layer + 1);
  }
  std::vector<ExpertCache>& layer_cache = expert_cache_by_layer_[layer];
  if (layer_cache.size() < rows->size()) {
    layer_cache.resize(rows->size());
  }
  const std::size_t cache_entries = ExpertCacheEntries();
  std::vector<KindWork> kinds;
  kinds.reserve(rows->size());
  for (std::size_t k = 0; k < rows->size(); ++k) {
    const std::size_t i = (*rows)[k];
    const PagedTensor& entry = paged_[i];
    auto span = ExpertSpan(entry.manifest);
    if (!span) {
      return std::unexpected(span.error());
    }
    const auto [per, count] = *span;
    const auto staged = stage_index_.find(entry.kind);
    if (staged == stage_index_.end()) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    Buffer* slot = stage_views_[staged->second].device.get();
    if (slot == nullptr) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    for (const std::uint32_t expert : missing) {
      const std::size_t id = expert;
      if (id >= count || per > entry.host.size() ||
          id > (entry.host.size() - per) / per) {
        return std::unexpected(StatusCode::MalformedFile);
      }
    }
    ExpertCache* cache = nullptr;
    ExpertCache& held = layer_cache[k];
    if (held.store == nullptr) {
      auto store =
          backend_.AllocateBuffer(cache_entries * per, MemoryKind::Device);
      if (store) {
        held.stage_slot = staged->second;
        held.per_bytes = per;
        held.store = std::move(*store);
        held.entries.resize(cache_entries);
      }
    }
    if (held.store != nullptr && held.per_bytes == per &&
        held.entries.size() == cache_entries) {
      cache = &held;
    }
    kinds.push_back(KindWork{i, staged->second, per, cache});
  }
  // Cache fills upload into store slots; every wanted slice (hit or fill)
  // then gathers store to staging at the expert offset.
  struct Gather {
    Buffer* store = nullptr;
    std::size_t src_offset = 0;
    Buffer* stage = nullptr;
    std::size_t dst_offset = 0;
    std::size_t bytes = 0;
  };
  std::vector<Backend::HostCopy> uploads;
  std::vector<Gather> gathers;
  uploads.reserve(rows->size() * missing.size());
  gathers.reserve(rows->size() * missing.size());
  std::vector<std::size_t> slots;
  slots.reserve(rows->size());
  for (const KindWork& kind : kinds) {
    const PagedTensor& entry = paged_[kind.row];
    Buffer* stage = stage_views_[kind.stage_slot].device.get();
    slots.push_back(kind.stage_slot);
    for (const std::uint32_t expert : missing) {
      const std::size_t begin = static_cast<std::size_t>(expert) * kind.per;
      if (kind.cache == nullptr) {
        uploads.push_back(Backend::HostCopy{
            stage, entry.host.subspan(begin, kind.per), begin});
        continue;
      }
      ExpertCache& cache = *kind.cache;
      std::size_t slot = cache_entries;
      for (std::size_t s = 0; s < cache.entries.size(); ++s) {
        const ExpertCacheEntry& held = cache.entries[s];
        if (held.occupied && held.expert == expert) {
          slot = s;
          break;
        }
      }
      if (slot < cache_entries) {
        cache.entries[slot].tick = ++expert_cache_tick_;
      } else {
        slot = 0;
        for (std::size_t s = 1; s < cache.entries.size(); ++s) {
          const ExpertCacheEntry& held = cache.entries[s];
          const ExpertCacheEntry& victim = cache.entries[slot];
          if (!held.occupied ||
              (victim.occupied && held.tick < victim.tick)) {
            slot = s;
          }
        }
        cache.entries[slot] =
            ExpertCacheEntry{expert, ++expert_cache_tick_, true};
        uploads.push_back(Backend::HostCopy{
            cache.store.get(), entry.host.subspan(begin, kind.per),
            slot * kind.per});
      }
      gathers.push_back(Gather{cache.store.get(), slot * kind.per, stage,
                               begin, kind.per});
    }
  }
  if (!uploads.empty()) {
    if (auto uploaded = backend_.CopyH2DBatch(uploads); !uploaded) {
      return std::unexpected(uploaded.error());
    }
  }
  // Ordered after the uploads on both backends (Vulkan records into the
  // pending compute buffer, ROCm onto the same stream), so the kernels
  // that run next read the gathered bytes. Runs of gathers that share a
  // (store, stage) pair go through one batched copy, so a layer's dozens of
  // per-expert gathers cost a few launches.
  std::size_t gather_begin = 0;
  while (gather_begin < gathers.size()) {
    std::size_t gather_end = gather_begin + 1;
    while (gather_end < gathers.size() &&
           gathers[gather_end].store == gathers[gather_begin].store &&
           gathers[gather_end].stage == gathers[gather_begin].stage) {
      ++gather_end;
    }
    std::vector<Backend::DeviceCopySpan> spans;
    spans.reserve(gather_end - gather_begin);
    for (std::size_t k = gather_begin; k < gather_end; ++k) {
      spans.push_back(Backend::DeviceCopySpan{
          gathers[k].src_offset, gathers[k].dst_offset, gathers[k].bytes});
    }
    if (auto copied = backend_.CopyD2DBatch(*gathers[gather_begin].store,
                                            *gathers[gather_begin].stage,
                                            spans);
        !copied) {
      return std::unexpected(copied.error());
    }
    gather_begin = gather_end;
  }
  for (std::size_t k = 0; k < kinds.size(); ++k) {
    stage_views_[slots[k]].manifest = paged_[kinds[k].row].manifest;
  }
  if (!same_layer) {
    experts_staged_ = wanted;
  } else {
    experts_staged_.insert(experts_staged_.end(), missing.begin(),
                           missing.end());
    std::sort(experts_staged_.begin(), experts_staged_.end());
  }
  experts_layer_ = layer;
  return {};
}

bool Model::ExpertCached(std::size_t layer, std::uint32_t expert) const {
  if (layer >= paged_expert_by_layer_.size()) {
    return true;  // resident layer (or no paging)
  }
  const std::vector<std::size_t>& rows = paged_expert_by_layer_[layer];
  if (rows.empty()) {
    return true;
  }
  if (layer >= expert_cache_by_layer_.size()) {
    return false;
  }
  const std::vector<ExpertCache>& layer_cache = expert_cache_by_layer_[layer];
  if (layer_cache.size() < rows.size()) {
    return false;
  }
  // An expert counts as cached only when every kind holds it: the GPU needs
  // gate, up and down together to compute it without a transfer.
  for (std::size_t k = 0; k < rows.size(); ++k) {
    const ExpertCache& cache = layer_cache[k];
    if (cache.store == nullptr) {
      return false;
    }
    bool found = false;
    for (const ExpertCacheEntry& held : cache.entries) {
      if (held.occupied && held.expert == expert) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

std::span<const std::byte> Model::PagedExpertHost(
    std::string_view tensor_name, std::uint32_t expert) const {
  const auto it = paged_index_.find(tensor_name);
  if (it == paged_index_.end()) {
    return {};
  }
  const PagedTensor& entry = paged_[it->second];
  auto span = ExpertSpan(entry.manifest);
  if (!span) {
    return {};
  }
  const std::size_t per = span->first;
  if (expert >= span->second) {
    return {};
  }
  const std::size_t begin = static_cast<std::size_t>(expert) * per;
  if (begin > entry.host.size() || per > entry.host.size() - begin) {
    return {};
  }
  return entry.host.subspan(begin, per);
}

std::span<const std::byte> Model::PagedTensorHost(
    std::string_view tensor_name) const {
  const auto it = paged_index_.find(tensor_name);
  if (it == paged_index_.end()) {
    return {};
  }
  return paged_[it->second].host;
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
