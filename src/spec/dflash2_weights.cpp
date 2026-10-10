#include "spec/dflash2_weights.hpp"

#include <cstring>
#include <string>
#include <unordered_map>

#include "core/files.hpp"
#include "core/loaders/safetensors.hpp"
#include "core/numerics/quant.hpp"

namespace tessera::spec {

namespace {

using core::SafetensorsTensor;

std::expected<std::vector<float>, StatusCode> ToF32(
    const std::unordered_map<std::string, SafetensorsTensor>& index,
    std::span<const std::byte> file, const std::string& name) {
  auto it = index.find(name);
  if (it == index.end()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const SafetensorsTensor& tensor = it->second;
  const std::size_t numel = tensor.entry.shape.Numel();
  std::vector<float> out(numel);
  const std::span<const std::byte> bytes =
      file.subspan(tensor.begin, tensor.end - tensor.begin);
  switch (tensor.entry.dtype) {
    case DType::F32: {
      if (bytes.size() != numel * 4) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      std::memcpy(out.data(), bytes.data(), bytes.size());
      return out;
    }
    case DType::BF16: {
      if (!core::DequantizeBf16(bytes, std::span<float>(out))) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      return out;
    }
    case DType::F8E4M3: {
      auto scale_it = index.find(name + "_scale_inv");
      if (tensor.entry.shape.rank != 2 || scale_it == index.end()) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      const SafetensorsTensor& scale = scale_it->second;
      std::vector<float> scales(scale.entry.shape.Numel());
      const std::span<const std::byte> scale_bytes =
          file.subspan(scale.begin, scale.end - scale.begin);
      if (scale.entry.dtype == DType::F32) {
        if (scale_bytes.size() != scales.size() * 4) {
          return std::unexpected(StatusCode::MalformedFile);
        }
        std::memcpy(scales.data(), scale_bytes.data(), scale_bytes.size());
      } else if (scale.entry.dtype == DType::BF16) {
        if (!core::DequantizeBf16(scale_bytes, std::span<float>(scales))) {
          return std::unexpected(StatusCode::MalformedFile);
        }
      } else {
        return std::unexpected(StatusCode::MalformedFile);
      }
      if (!core::DequantizeFp8Block(bytes, std::span<const float>(scales),
                                    std::span<float>(out),
                                    tensor.entry.shape.dims[0],
                                    tensor.entry.shape.dims[1])) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      return out;
    }
    default:
      return std::unexpected(StatusCode::UnsupportedFeature);
  }
}

}  // namespace

std::expected<DraftWeightStore, StatusCode> DraftWeightStore::Load(
    Backend& backend, const std::filesystem::path& dir,
    const DFlash2Config& config) {
  auto layout = core::InspectMxFp4Directory(dir);
  if (!layout) {
    return std::unexpected(layout.error());
  }
  auto file = core::ReadFile(layout->weights_path);
  if (!file) {
    return std::unexpected(file.error());
  }
  std::span<const std::byte> bytes(*file);
  auto parsed = core::ParseSafetensorsMap(bytes);
  if (!parsed) {
    return std::unexpected(parsed.error());
  }
  std::unordered_map<std::string, SafetensorsTensor> index;
  for (const SafetensorsTensor& tensor : *parsed) {
    index.emplace(tensor.name, tensor);
  }
  DraftWeightStore store;
  auto upload = [&](const std::vector<float>& data) -> const Buffer* {
    auto buffer =
        backend.AllocateBuffer(data.size() * 4, MemoryKind::Device);
    if (!buffer) {
      return nullptr;
    }
    const std::span<const std::byte> src(
        reinterpret_cast<const std::byte*>(data.data()), data.size() * 4);
    if (!backend.CopyH2D(**buffer, src)) {
      return nullptr;
    }
    store.owned_.push_back(std::move(*buffer));
    return store.owned_.back().get();
  };
  auto bind = [&](const std::string& name) -> const Buffer* {
    auto converted = ToF32(index, bytes, name);
    if (!converted) {
      return nullptr;
    }
    return upload(*converted);
  };
  // GEMM weights are stored bf16: the draft projections dominate the draft
  // forward, and bf16 is more precise than the fp8 source while reading half
  // the bytes of the fp32 the loader would otherwise upload. Norm, conv and
  // codebook tensors keep fp32 (their kernels take fp32 weights).
  auto bind_bf16 = [&](const std::string& name) -> const Buffer* {
    auto converted = ToF32(index, bytes, name);
    if (!converted) {
      return nullptr;
    }
    std::vector<std::byte> bf16(converted->size() * 2);
    for (std::size_t i = 0; i < converted->size(); ++i) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &(*converted)[i], sizeof(bits));
      const std::uint16_t b =
          static_cast<std::uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
      bf16[i * 2] = static_cast<std::byte>(b & 0xFF);
      bf16[i * 2 + 1] = static_cast<std::byte>((b >> 8) & 0xFF);
    }
    auto buffer = backend.AllocateBuffer(bf16.size(), MemoryKind::Device);
    if (!buffer) {
      return nullptr;
    }
    if (!backend.CopyH2D(**buffer, std::span<const std::byte>(bf16))) {
      return nullptr;
    }
    store.owned_.push_back(std::move(*buffer));
    return store.owned_.back().get();
  };
  store.weights_.layers.resize(config.num_layers);
  for (std::size_t layer = 0; layer < config.num_layers; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    DraftLayerBuffers& w = store.weights_.layers[layer];
    w.input_norm = bind(prefix + "input_layernorm.weight");
    w.attn_conv_proj =
        bind_bf16(prefix + "attention_conv.kernel_projection.weight");
    w.attn_conv_base = bind(prefix + "attention_conv.base_kernel");
    w.q_w = bind_bf16(prefix + "self_attn.q_proj.weight");
    w.k_w = bind_bf16(prefix + "self_attn.k_proj.weight");
    w.v_w = bind_bf16(prefix + "self_attn.v_proj.weight");
    w.o_w = bind_bf16(prefix + "self_attn.o_proj.weight");
    w.q_norm_w = bind(prefix + "self_attn.q_norm.weight");
    w.k_norm_w = bind(prefix + "self_attn.k_norm.weight");
    w.post_norm = bind(prefix + "post_attention_layernorm.weight");
    w.mlp_conv_proj = bind_bf16(prefix + "mlp_conv.kernel_projection.weight");
    w.mlp_conv_base = bind(prefix + "mlp_conv.base_kernel");
    w.gate_w = bind_bf16(prefix + "mlp.gate_proj.weight");
    w.up_w = bind_bf16(prefix + "mlp.up_proj.weight");
    w.down_w = bind_bf16(prefix + "mlp.down_proj.weight");
    w.hidden_norm = nullptr;
    if (w.input_norm == nullptr || w.attn_conv_proj == nullptr ||
        w.attn_conv_base == nullptr || w.q_w == nullptr ||
        w.k_w == nullptr || w.v_w == nullptr || w.o_w == nullptr ||
        w.q_norm_w == nullptr || w.k_norm_w == nullptr ||
        w.post_norm == nullptr || w.mlp_conv_proj == nullptr ||
        w.mlp_conv_base == nullptr || w.gate_w == nullptr || w.up_w == nullptr ||
        w.down_w == nullptr) {
      return std::unexpected(StatusCode::MalformedFile);
    }
  }
  store.weights_.hidden_norm = bind("hidden_norm.weight");
  store.weights_.fc = bind_bf16("fc.weight");
  store.weights_.final_norm = bind("norm.weight");
  for (DraftLayerBuffers& w : store.weights_.layers) {
    w.hidden_norm = store.weights_.hidden_norm;
  }
  if (store.weights_.hidden_norm == nullptr || store.weights_.fc == nullptr ||
      store.weights_.final_norm == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  // The candidate selector is optional; the drafter falls back to the
  // unary top-1 when it is absent.
  store.selector_.projection =
      bind_bf16("candidate_selector.hidden_projection.weight");
  store.selector_.predecessor = bind("candidate_selector.predecessor_codebook");
  store.selector_.successor = bind("candidate_selector.successor_codebook");
  return store;
}

}  // namespace tessera::spec
