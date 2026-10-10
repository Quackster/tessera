#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tessera/architecture.hpp"
#include "tessera/backend.hpp"
#include "tessera/tokenizer.hpp"
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
  // The uploaded weight with this name, or nullptr when absent. Backed by
  // a name index, so the decode loop's per-op lookups are O(1) instead of
  // a scan over every tensor.
  [[nodiscard]] const DeviceTensor* FindDeviceTensor(
      std::string_view name) const;
  // Device buffer for the tensor with this name; nullptr when absent.
  [[nodiscard]] const Buffer* FindWeight(std::string_view name) const;
  [[nodiscard]] const std::string& Path() const;
  [[nodiscard]] std::size_t MaxContextLength() const;
  // New tokens to generate for `requested` after a prompt of
  // `prompt_size` tokens. An explicit count passes through unchanged.
  // Zero means no explicit count: fill the remaining context
  // (MaxContextLength minus `prompt_size`, saturating at zero).
  //
  // Usage:
  //   const std::size_t max_completion_tokens = model.EffectiveMaxTokens(
  //       prompt.size(), options.max_completion_tokens);
  [[nodiscard]] std::size_t EffectiveMaxTokens(std::size_t prompt_size,
                                              std::size_t requested) const;
  // Model name from the file metadata when present ("" otherwise).
  [[nodiscard]] std::string_view Name() const;
  // Attention parameters from the model definition, for sizing kernel
  // launches (heads, kv groups, head dim, RoPE range and base). GGUF
  // reads <arch>.attention.head_count, head_count_kv, embedding_length
  // and <arch>.rope.dimension_count, rope.freq_base. Hybrid definitions
  // carry <arch>.attention.key_length/value_length instead; the head
  // dim comes from those and embedding divisibility is not required.
  // An MXFP4 model reads them from config.json through its module.
  // MalformedFile when the definition lacks them.
  //
  // Usage:
  //   auto params = model.Attention();
  //   if (params) launch.scalars = {m, n, params->heads, ...};
  [[nodiscard]] std::expected<AttentionParams, StatusCode> Attention() const;
  // Full transformer config for the decode loop. GGUF reads block_count,
  // embedding_length, feed_forward_length and the attention keys above
  // plus <arch>.attention.layer_norm_rms_epsilon; the vocabulary comes
  // from the output weight shape. Hybrid definitions also provide the
  // ssm.* keys and full_attention_interval, plus the rope section
  // array; layers counts trunk blocks (block_count minus
  // nextn_predict_layers). An MXFP4 model reads them from config.json
  // through its module. MalformedFile when the definition lacks them.
  //
  // Usage:
  //   auto config = model.Config();
  //   if (config) for (std::size_t l = 0; l < config->layers; ++l) ...
  [[nodiscard]] std::expected<TransformerConfig, StatusCode> Config() const;
  // The model's tokenizer (GGUF only); nullptr when the file carries no
  // recognized tokenizer definition.
  //
  // Usage:
  //   if (auto* tok = model.Tokenizer()) tok->Encode("hi");
  [[nodiscard]] const Tokenizer* GetTokenizer() const;
  // The model's chat template (GGUF tokenizer.chat_template); empty when
  // the file carries none.
  [[nodiscard]] std::string_view ChatTemplate() const;
  // Stop token ids the definition declares: GGUF
  // `tokenizer.ggml.eos_token_id`, or the HuggingFace `eos_token_id` from
  // `generation_config.json` then `config.json`. A generation loop ends
  // and does not emit a token in this set. Empty when the file declares
  // none.
  //
  // Usage:
  //   for (std::uint32_t id : model.StopTokens()) ...
  [[nodiscard]] std::span<const std::uint32_t> StopTokens() const;
  // Render a single user message through the model's chat template with
  // the generation prompt appended, so a tokenizer can produce the ids the
  // model was trained on. UnsupportedFeature when the model has no chat
  // template.
  [[nodiscard]] std::expected<std::string, StatusCode> ChatPrompt(
      std::string_view user_text, bool enable_thinking = false) const;
  // The architecture module for this model (from general.architecture), or
  // nullptr when no module is registered. Generic code dispatches through
  // it for architecture specific behavior.
  [[nodiscard]] const Architecture* Arch() const;

 private:
  Model(Backend& backend, ModelOptions options, ModelFormat format,
        std::vector<TensorEntry> tensors, std::string name,
        std::string architecture, std::optional<AttentionParams> attention,
        std::optional<TransformerConfig> config,
        std::vector<DeviceTensor> weights,
        std::optional<Tokenizer> tokenizer, std::string chat_template,
        std::vector<std::uint32_t> stop_tokens);
  Backend& backend_;
  ModelOptions options_;
  ModelFormat format_;
  std::vector<TensorEntry> tensors_;
  std::string name_;
  std::string architecture_;
  std::optional<AttentionParams> attention_;
  std::optional<TransformerConfig> config_;
  std::vector<DeviceTensor> weights_;
  // Weight-name index: name -> position in weights_. Keys are string_views
  // into the (immutable) DeviceTensor names, so no string is copied.
  std::unordered_map<std::string_view, std::size_t> weight_index_;
  std::optional<Tokenizer> tokenizer_;
  std::string chat_template_;
  std::vector<std::uint32_t> stop_tokens_;
  std::unique_ptr<Architecture> module_;
};

}  // namespace tessera
