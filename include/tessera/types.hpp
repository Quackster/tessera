#pragma once

#include <array>
#include <cstddef>
#include <expected>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tessera {

// Storage type of the full-attention KV cache.
enum class KvCacheType : int {
  F32 = 0,
  F16 = 1,
  Q8 = 2,
  Q4 = 3,
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
  }
  return "unknown";
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
    case DType::IQ3_S: {
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
