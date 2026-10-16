#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tessera/defaults.hpp"

namespace tessera {

// Storage type of the full-attention KV cache.
enum class KvCacheType : int {
  F32 = 0,
  F16 = 1,
  Q8 = 2,
  Q4 = 3,
  FP8 = 4,
};

// How a projection is split across tensor-parallel ranks. The architecture
// module answers this per internal tensor; the loader slices a Column or Row
// tensor at block boundaries and the forward all-reduces a Row projection.
enum class ParallelKind : int {
  Replicated = 0,  // every rank holds the full tensor
  Column = 1,      // split the output rows across ranks
  Row = 2,         // split the input columns (the row-parallel reduce)
  Expert = 3,      // split the outer (expert) dim of a packed expert tensor
};

// Status codes for result-style returns (runtime I/O, parsing, device errors).
// See AGENTS.md rule 6: exceptions are for programmer errors only.
enum class StatusCode : int {
  Ok = 0,
  InvalidArgument = 1,
  FileNotFound = 2,
  MalformedFile = 3,
  UnsupportedFeature = 4,
  OutOfMemory = 5,
  DeviceError = 6,
  NonFinite = 7,
  NetworkError = 8,
};

// Human-readable name of a status code, for diagnostics and messages.
//
// Usage:
//   auto result = model.Load(backend, options);
//   if (!result) log.Error("model", ToString(result.error()));
[[nodiscard]] inline std::string_view ToString(StatusCode code) {
  switch (code) {
    case StatusCode::Ok: return "ok";
    case StatusCode::InvalidArgument: return "invalid_argument";
    case StatusCode::FileNotFound: return "file_not_found";
    case StatusCode::MalformedFile: return "malformed_file";
    case StatusCode::UnsupportedFeature: return "unsupported_feature";
    case StatusCode::OutOfMemory: return "out_of_memory";
    case StatusCode::DeviceError: return "device_error";
    case StatusCode::NonFinite: return "non_finite_logits";
    case StatusCode::NetworkError: return "network_error";
  }
  return "unknown";
}

// Human-readable name of the full-attention KV cache storage type.
//
// Usage:
//   log.Info("cli", std::string("kv cache: ") +
//                       std::string(ToString(options.kv_type)));
[[nodiscard]] inline std::string_view ToString(KvCacheType type) {
  switch (type) {
    case KvCacheType::F32: return "fp32";
    case KvCacheType::F16: return "fp16";
    case KvCacheType::Q8: return "int8";
    case KvCacheType::Q4: return "4-bit";
    case KvCacheType::FP8: return "fp8";
  }
  return "fp32";
}

// Element types the engine can hold. Block-quantized types (Q*) and
// microscaled F4E2M1 have no fixed per-element byte size; use the block
// layout helpers of the weight-loader layer for those.
enum class DType : int {
  F32 = 0,
  F16 = 1,
  BF16 = 2,
  F8E4M3 = 3,
  F8E5M2 = 4,
  F8E8M0 = 5,  // MXFP4 per-block scale (E8M0)
  F4E2M1 = 6,  // MXFP4 element (2 elements per byte + block scale)
  Q2K = 7,
  Q3K = 8,
  Q40 = 9,
  Q4K = 10,
  Q5K = 11,
  Q6K = 12,
  Q80 = 13,
  Q8K = 14,
  I32 = 15,
  I64 = 16,
  IQ4_NL = 17,  // 4-bit codebook blocks (ggml 2026 numbering)
  IQ4_XS = 18,  // 4-bit fine-scale blocks
  IQ3_S = 19,   // 3-bit grid blocks
  IQ3_XXS = 20, // 3-bit grid blocks, extra-small scales
  IQ2_S = 21,   // 2-bit grid blocks, small
  Q2_0 = 22,    // 2-bit sign/magnitude blocks (64 elements, 18 bytes)
};

// Bytes per element of a plain (non-block) dtype.
// Throws std::invalid_argument for block-quantized and microscaled types
// (programmer error: they need the block layout helpers instead).
[[nodiscard]] inline int ElementBytes(DType dtype) {
  switch (dtype) {
    case DType::F32:
    case DType::I32: return 4;
    case DType::F16:
    case DType::BF16: return 2;
    case DType::F8E4M3:
    case DType::F8E5M2:
    case DType::F8E8M0: return 1;
    case DType::I64: return 8;
    default:
      throw std::invalid_argument(
          "ElementBytes: block-quantized dtype has no fixed element size");
  }
}

// Q4_K block layout constant (ggml 2026 numbering): the elements one
// 144-byte Q4_K block holds. Shared by the "gemm_q4k" built-in
// contract and the weight loaders.
constexpr std::size_t kQ4KBlockElements = 256;

// Byte size of a dense tensor with `numel` elements. Block types use
// their block layout (ggml 2026 numbering, F4E2M1 packs two elements
// per byte); plain types use ElementBytes. UnsupportedFeature for
// layouts without a known size (Q2_K, Q3_K); InvalidArgument when
// numel is not a multiple of the block elements or the product
// overflows. This is the single canonical sizer (weight upload,
// file validation).
//
// Usage:
//   auto bytes = TensorBytes(DType::Q4K, 256);  // 144 when present
[[nodiscard]] inline std::expected<std::size_t, StatusCode> TensorBytes(
    DType dtype, std::size_t numel) {
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  switch (dtype) {
    case DType::Q40:
    case DType::Q80:
    case DType::Q3K:
    case DType::Q4K:
    case DType::Q5K:
    case DType::Q6K:
    case DType::Q8K:
    case DType::IQ4_NL:
    case DType::IQ4_XS:
    case DType::IQ3_S:
    case DType::IQ3_XXS:
    case DType::IQ2_S:
    case DType::Q2_0: {
      std::size_t elements = 0;
      std::size_t bytes = 0;
      switch (dtype) {
        case DType::Q40: elements = 32; bytes = 17; break;
        case DType::Q80: elements = 32; bytes = 34; break;
        case DType::Q3K: elements = 256; bytes = 110; break;
        case DType::Q4K: elements = 256; bytes = 144; break;
        case DType::Q5K: elements = 256; bytes = 176; break;
        case DType::Q6K: elements = 256; bytes = 210; break;
        case DType::IQ4_NL: elements = 32; bytes = 18; break;
        case DType::IQ4_XS: elements = 256; bytes = 136; break;
        case DType::IQ3_S: elements = 256; bytes = 110; break;
        case DType::IQ3_XXS: elements = 256; bytes = 98; break;
        case DType::IQ2_S: elements = 256; bytes = 82; break;
        case DType::Q2_0: elements = 64; bytes = 18; break;
        default: elements = 256; bytes = 258; break;  // Q8K
      }
      if (numel % elements != 0 || numel / elements > kMax / bytes) {
        return std::unexpected(StatusCode::InvalidArgument);
      }
      return numel / elements * bytes;
    }
    case DType::F4E2M1:
      if ((numel % 2) != 0 || numel / 2 > kMax) {
        return std::unexpected(StatusCode::InvalidArgument);
      }
      return numel / 2;
    default: break;
  }
  try {
    const int per_element = ElementBytes(dtype);
    if (per_element <= 0 ||
        numel > kMax / static_cast<std::size_t>(per_element)) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    return numel * static_cast<std::size_t>(per_element);
  } catch (const std::invalid_argument&) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
}

// Attention architecture parameters, parsed from the model definition
// (Model::Attention). Query heads share key/value heads in groups of
// heads / kv_heads (grouped-query attention). RoPE rotates rope_dim
// elements of every head with base rope_theta; the tail of a wider
// head is left alone. All fields come from model data; no kernel
// branches on the architecture.
//
// Usage:
//   auto params = model.Attention();
//   if (params) launch.buffers = {...};  // heads=params->heads, ...
struct AttentionParams {
  std::size_t heads = 0;
  std::size_t kv_heads = 0;
  std::size_t head_dim = 0;
  std::size_t rope_dim = 0;
  double rope_theta = 10000.0;
};

// Multi-head latent attention (MLA) dimensions of a definition whose
// full-attention layers compress keys and values into a low-rank latent.
// kv_lora_rank == 0 means the definition uses plain GQA instead (then
// every field is zero).
struct MlaParams {
  std::size_t q_lora_rank = 0;       // query compression width
  std::size_t kv_lora_rank = 0;      // key/value latent width (the cache row)
  std::size_t qk_nope_head_dim = 0;  // per-head q/k width without rotary dims
  std::size_t qk_rope_head_dim = 0;  // per-head rotary width (0 for NoPE)
  std::size_t v_head_dim = 0;        // per-head value width
  [[nodiscard]] bool IsPresent() const { return kv_lora_rank > 0; }
};

// Recurrent linear-attention dimensions of a hybrid model definition.
// They size the fused qkv/gate projections, the causal conv1d and the
// recurrent state; generic kernels take them as launch data, never as
// per-model branches.
struct SsmParams {
  std::size_t conv_kernel = 0;     // causal conv1d width over the mix
  std::size_t state_size = 0;      // recurrent state dim per head
  std::size_t group_count = 0;     // key heads sharing a value group
  std::size_t time_step_rank = 0;  // value heads (gate rank)
  std::size_t inner_size = 0;      // fused qkv/gate projection width
};

// KDA (gated delta-rule linear attention) dimensions of a hybrid
// definition whose recurrent layers keep per-head delta-rule states.
// head_dim == 0 means the definition has no KDA layers (then every field
// is zero). The projection width is heads * head_dim; q, k and v each
// pass their own causal conv of width conv_kernel. The log decay gate is
// g = gate_lower_bound * sigmoid(...) so the per-step decay is in
// [exp(gate_lower_bound), 1].
struct KdaParams {
  std::size_t heads = 0;
  std::size_t head_dim = 0;
  std::size_t conv_kernel = 0;
  double gate_lower_bound = -5.0;
  [[nodiscard]] bool IsPresent() const { return head_dim > 0; }
};

// Hyper-connection dimensions of a definition that carries a gated
// residual stack instead of one residual stream. `count` is the number
// of streams. `low_rank` is the mixer bottleneck width for the gated
// residual variant; 0 selects the manifold-constrained variant (mHC),
// which mixes with pre/post/combination weights projected from the
// normalized input and projects the combination matrix onto the doubly
// stochastic manifold with `sinkhorn_iterations` Sinkhorn steps and
// `epsilon`. count == 0 means the definition has no hyper-connections.
struct HyperConnectionParams {
  std::size_t count = 0;
  std::size_t low_rank = 0;
  std::size_t sinkhorn_iterations = 0;
  double epsilon = 0.0;
  [[nodiscard]] bool IsManifold() const { return count > 0 && low_rank == 0; }
};

// Per-layer-embedding (PLE) dimensions of a definition whose token
// embedding is augmented by an n-gram table. `layers` names the trunk
// layers that carry the PLE weights; the table itself is not a weight
// tensor that uploads whole. All fields come from model data.
struct PleParams {
  std::vector<std::uint64_t> layers;
  std::size_t ngram_size = 0;
  std::size_t heads_per_ngram = 0;
  std::size_t conv_kernel = 0;
  std::size_t embd_per_layer = 0;  // width of one gathered row
  std::size_t image_token_id = 0;
  std::size_t eos_token_id = 0;
  std::vector<std::uint64_t> layer_multipliers;
  std::vector<std::uint64_t> head_offsets;
  std::vector<std::uint64_t> head_vocab_sizes;
};

// Sparse-attention indexer dimensions of a definition that selects a
// subset of the key/value cells before attention. heads == 0 means the
// definition has no indexer. `kpool` is the pool size of the compressed
// key index (1 disables pooling); `select_tail` always appends the last
// visible tokens so the query attends to its own neighborhood. Each
// value-selection round keeps top_k / kpool pools. norm_eps is the
// epsilon of the indexer's layer norm (0 when the definition carries
// none).
struct IndexerParams {
  std::size_t heads = 0;
  std::size_t key_length = 0;
  std::size_t top_k = 0;
  std::size_t kpool = 1;
  bool select_tail = false;
  double norm_eps = 0.0;
};

// Routed-expert gating function as encoded in the model definition (the
// GGUF expert_gating_func values). kSoftmax scores all experts with a
// softmax over the full expert axis and keeps the top-k; kSigmoid scores
// each expert independently and keeps the top-k scores.
enum class MoeGating : std::uint8_t {
  kSoftmax = 1,
  kSigmoid = 2,
};

// Transformer hyper-parameters from the model definition. The decode
// loop sizes every launch from these; no code branches on the
// architecture.
struct TransformerConfig {
  AttentionParams attention;
  // MLA dimensions of the full-attention layers; all-zero when the
  // definition uses plain GQA attention.
  MlaParams mla;
  std::size_t layers = 0;  // trunk blocks (excludes MTP draft blocks)
  std::size_t hidden_dim = 0;
  std::size_t ffn_dim = 0;
  std::size_t vocab_size = 0;
  double norm_eps = 1e-5;
  // Sparse mixture-of-experts feed-forward. num_experts == 0 means a dense
  // FFN sized by ffn_dim. When num_experts > 0 the block FFN is the sum of
  // `experts_per_tok` routed experts with width moe_intermediate plus one
  // always-on shared expert with width shared_expert_intermediate; ffn_dim
  // is then unused and may be 0.
  std::size_t num_experts = 0;
  std::size_t experts_per_tok = 0;
  std::size_t moe_intermediate = 0;
  std::size_t shared_expert_intermediate = 0;
  // Routed-expert scoring and combination. moe_gating is the score
  // function; moe_weights_norm renormalizes the kept weights to sum to
  // one before scaling; moe_weights_scale multiplies the final mixture
  // (0 = no scaling).
  MoeGating moe_gating = MoeGating::kSoftmax;
  bool moe_weights_norm = false;
  double moe_weights_scale = 0.0;
  // Trunk layers [0, leading_dense_layers) run a dense FFN sized by
  // ffn_dim even when the definition is an MoE; later layers run the
  // mixture.
  //
  // Usage:
  //   const bool dense = l < config.leading_dense_layers;
  std::size_t leading_dense_layers = 0;
  // True for hybrid attention/SSM definitions (recurrent linear layers
  // interleaved with full attention). The vanilla decode path rejects
  // these as UnsupportedFeature until the recurrent kernels land.
  bool hybrid = false;
  // SSM dimensions, valid only when hybrid is true.
  SsmParams ssm;
  // KDA dimensions of the recurrent layers, valid when KdaParams::
  // IsPresent() (a hybrid variant); head_dim stays 0 for SSM hybrids.
  KdaParams kda;
  // True when the feed-forward is a sparse mixture of experts.
  //
  // Usage:
  //   if (config.IsMoe()) { /* routed experts plus shared expert */ }
  [[nodiscard]] bool IsMoe() const { return num_experts > 0; }
  // Explicit per-layer full-attention flags parsed from a definition
  // that carries a per-layer head-count array (0 = recurrent layer).
  // Empty when the definition carries none; the interval below then
  // selects the full-attention layers. `layer_full_attention[l]` is
  // valid for l < layers.
  std::vector<std::uint8_t> layer_full_attention;
  // Trunk layer l is full attention iff (l + 1) % interval == 0.
  // Zero means every layer is full attention (vanilla).
  std::size_t full_attention_interval = 0;
  // True when trunk layer l runs full attention (never recurrent for
  // vanilla configs, where the interval is zero).
  //
  // Usage:
  //   if (config.IsFullAttentionLayer(l)) { /* full attention path */ }
  [[nodiscard]] bool IsFullAttentionLayer(std::size_t layer) const {
    if (!layer_full_attention.empty()) {
      return layer < layer_full_attention.size() &&
             layer_full_attention[layer] != 0;
    }
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
  // Gated residual stack (hyper-connections). count == 0 when absent.
  HyperConnectionParams hyper_connection;
  // Width of the residual stream that crosses a pipeline stage boundary:
  // `hidden_dim`, times the hyper-connection stream count when the model
  // carries a gated residual stack. A single-stream model returns
  // `hidden_dim`.
  [[nodiscard]] std::size_t ResidualWidth() const {
    return hidden_dim *
           (hyper_connection.count == 0 ? 1 : hyper_connection.count);
  }
  // Per-layer n-gram embedding. layers empty when absent.
  PleParams ple;
  // Sparse-attention indexer. heads == 0 when absent.
  IndexerParams indexer;
  // Per-trunk-layer SwiGLU clamp on the gate and up activations (0 = no
  // clamp). Empty when the definition carries none. The shared expert
  // uses `swiglu_clamp_shared`, which falls back to `swiglu_clamp` when
  // empty.
  std::vector<float> swiglu_clamp;
  std::vector<float> swiglu_clamp_shared;
};

// Dense tensor shape, at most 4 dimensions, row-major.
//
// Usage:
//   TensorShape shape;
//   shape.rank = 2;
//   shape.dims[0] = 4096;
//   shape.dims[1] = 8;
struct TensorShape {
  // Multimodal checkpoints carry rank-5 vision tensors (for example a
  // patch-embedding conv weight [out, in, t, h, w]), so the cap is 6.
  static constexpr std::size_t kMaxRank = 6;

  std::size_t rank = 0;
  std::array<std::size_t, kMaxRank> dims{};

  // Product of dims[0..rank-1]; 1 for rank 0 (scalar).
  [[nodiscard]] std::size_t Numel() const {
    std::size_t product = 1;
    for (std::size_t i = 0; i < rank; ++i) product *= dims[i];
    return product;
  }

  [[nodiscard]] bool operator==(const TensorShape& other) const {
    return rank == other.rank && dims == other.dims;
  }
};

// Tensor metadata without its data: name, shape, dtype.
struct TensorEntry {
  std::string name;
  TensorShape shape;
  DType dtype = DType::F32;
};

// Parsed, validated list of the tensors a model file contains.
struct TensorManifest {
  std::vector<TensorEntry> tensors;

  // Total element count over all tensors (informational; quantized types
  // count elements, not bytes).
  [[nodiscard]] std::size_t TotalNumel() const {
    std::size_t total = 0;
    for (const auto& tensor : tensors) total += tensor.shape.Numel();
    return total;
  }
};

}  // namespace tessera
