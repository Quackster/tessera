#pragma once

#include <cstddef>
#include <cstdint>
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

// Recurrent linear-attention dimensions of a hybrid model definition.
// They size the fused qkv/gate projections, the causal conv1d and the
// recurrent state; generic kernels take them as launch data, never as
// per-model branches.
struct SsmParams {
  std::size_t conv_kernel = 0;    // causal conv1d width over the mix
  std::size_t state_size = 0;     // recurrent state dim per head
  std::size_t group_count = 0;    // key heads sharing a value group
  std::size_t time_step_rank = 0;  // value heads (gate rank)
  std::size_t inner_size = 0;     // fused qkv/gate projection width
};

// Transformer hyper-parameters from the model definition. The decode
// loop sizes every launch from these; no code branches on the
// architecture.
struct TransformerConfig {
  AttentionParams attention;
  std::size_t layers = 0;  // trunk blocks (excludes MTP draft blocks)
  std::size_t hidden_dim = 0;
  std::size_t ffn_dim = 0;
  std::size_t vocab_size = 0;
  double norm_eps = 1e-5;
  // True for hybrid attention/SSM definitions (recurrent linear layers
  // interleaved with full attention). The vanilla decode path rejects
  // these as UnsupportedFeature until the recurrent kernels land.
  bool hybrid = false;
  // SSM dimensions, valid only when hybrid is true.
  SsmParams ssm;
  // Trunk layer l is full attention iff (l + 1) % interval == 0.
  // Zero means every layer is full attention (vanilla).
  std::size_t full_attention_interval = 0;
  // True when trunk layer l runs full attention (never recurrent for
  // vanilla configs, where the interval is zero).
  //
  // Usage:
  //   if (config.IsFullAttentionLayer(l)) { /* GQA path */ }
  [[nodiscard]] bool IsFullAttentionLayer(std::size_t layer) const {
    if (!hybrid || full_attention_interval == 0) {
      return true;
    }
    return (layer + 1) % full_attention_interval == 0;
  }
  // mRoPE section pair counts (temporal, height, width[, pad]) from the
  // model definition. Empty when the file carries no section array;
  // hybrid definitions require it (the mrope built-in takes the counts
  // as launch scalars).
  std::vector<std::uint64_t> rope_sections;
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
  // and <arch>.rope.dimension_count, rope.freq_base. Hybrid definitions
  // carry <arch>.attention.key_length/value_length instead; the head
  // dim comes from those and embedding divisibility is not required.
  // MalformedFile when the definition lacks them; UnsupportedFeature
  // for MXFP4 (config.json parsing is a later milestone).
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
  // nextn_predict_layers). MalformedFile when the definition lacks
  // them; UnsupportedFeature for MXFP4.
  //
  // Usage:
  //   auto config = model.Config();
  //   if (config) for (std::size_t l = 0; l < config->layers; ++l) ...
  [[nodiscard]] std::expected<TransformerConfig, StatusCode> Config() const;

 private:
  Model(Backend& backend, ModelOptions options, ModelFormat format,
        std::vector<TensorEntry> tensors, std::string name,
        std::string architecture, std::optional<AttentionParams> attention,
        std::optional<TransformerConfig> config,
        std::vector<DeviceTensor> weights);
  Backend& backend_;
  ModelOptions options_;
  ModelFormat format_;
  std::vector<TensorEntry> tensors_;
  std::string name_;
  std::string architecture_;
  std::optional<AttentionParams> attention_;
  std::optional<TransformerConfig> config_;
  std::vector<DeviceTensor> weights_;
};

}  // namespace tessera
