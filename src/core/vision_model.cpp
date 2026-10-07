#include "tessera/vision.hpp"

#include <cstring>
#include <string>
#include <unordered_map>

#include "core/decode_internal.hpp"
#include "core/files.hpp"
#include "core/loaders/gguf.hpp"
#include "core/numerics/quant.hpp"
#include "core/vision_block.hpp"
#include "core/vision_merger.hpp"
#include "core/vision_stack.hpp"

namespace tessera {

namespace {

using core::GgufFile;

// Decode a whole tensor to fp32 (F32 copy, BF16 dequantize over the raw
// bytes at the tensor's file offset).
std::expected<std::vector<float>, StatusCode> TensorToF32(
    const GgufFile& gguf, std::span<const std::byte> file,
    const std::string& name, std::size_t expected_numel) {
  for (std::size_t i = 0; i < gguf.tensors.size(); ++i) {
    const TensorEntry& entry = gguf.tensors[i];
    if (entry.name != name) {
      continue;
    }
    const std::size_t numel = entry.shape.Numel();
    if (expected_numel != 0 && numel != expected_numel) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    auto size = TensorBytes(entry.dtype, numel);
    if (!size) {
      return std::unexpected(size.error());
    }
    const std::size_t offset =
        gguf.tensor_data_start + gguf.tensor_offsets[i];
    const std::span<const std::byte> bytes = file.subspan(offset, *size);
    std::vector<float> out(numel);
    if (entry.dtype == DType::F32) {
      if (bytes.size() != numel * 4) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      std::memcpy(out.data(), bytes.data(), bytes.size());
      return out;
    }
    if (entry.dtype == DType::BF16) {
      if (!core::DequantizeBf16(bytes, std::span<float>(out))) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      return out;
    }
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  return std::unexpected(StatusCode::MalformedFile);
}

}  // namespace

struct VisionModel::Impl {
  VisionConfig config;
  std::vector<std::unique_ptr<Buffer>> owned;
  std::vector<core::VisionBlockBuffers> blocks;
  const Buffer* patch_w = nullptr;
  const Buffer* patch_bias = nullptr;
  const Buffer* position = nullptr;
  const Buffer* post_ln_weight = nullptr;
  const Buffer* post_ln_bias = nullptr;
  const Buffer* mm0_weight = nullptr;
  const Buffer* mm0_bias = nullptr;
  const Buffer* mm2_weight = nullptr;
  const Buffer* mm2_bias = nullptr;
  std::unique_ptr<Kernel> layernorm;
  std::unique_ptr<Kernel> gemm;
  std::unique_ptr<Kernel> attention;
  std::unique_ptr<Kernel> gelu;
  std::unique_ptr<Kernel> bias_add;
  std::unique_ptr<Kernel> add;
  std::unique_ptr<Kernel> spatial_merge;
  std::unique_ptr<Kernel> patchify;
};

VisionModel::VisionModel() : impl_(std::make_unique<Impl>()) {}
VisionModel::~VisionModel() = default;
VisionModel::VisionModel(VisionModel&&) noexcept = default;
VisionModel& VisionModel::operator=(VisionModel&&) noexcept = default;

const VisionConfig& VisionModel::Config() const { return impl_->config; }

std::expected<VisionModel, StatusCode> VisionModel::Load(
    Backend& backend, const std::filesystem::path& path) {
  auto config = LoadVisionConfig(path);
  if (!config) {
    return std::unexpected(config.error());
  }
  auto file = core::ReadFile(path);
  if (!file) {
    return std::unexpected(file.error());
  }
  auto gguf = core::ParseGguf(std::span<const std::byte>(*file));
  if (!gguf) {
    return std::unexpected(gguf.error());
  }
  VisionModel model;
  model.impl_->config = *config;
  Impl& impl = *model.impl_;
  const std::span<const std::byte> bytes(*file);
  auto upload = [&](const std::vector<float>& data) -> const Buffer* {
    auto buffer = backend.AllocateBuffer(data.size() * 4, MemoryKind::Device);
    if (!buffer) {
      return nullptr;
    }
    backend.CopyH2D(**buffer, std::span<const std::byte>(
                                 reinterpret_cast<const std::byte*>(data.data()),
                                 data.size() * 4));
    impl.owned.push_back(std::move(*buffer));
    return impl.owned.back().get();
  };
  auto bind = [&](const std::string& name,
                  std::size_t numel) -> const Buffer* {
    auto converted = TensorToF32(*gguf, bytes, name, numel);
    if (!converted) {
      return nullptr;
    }
    return upload(*converted);
  };
  const std::size_t embed = config->embedding_length;
  const std::size_t ffn = config->feed_forward_length;
  const std::size_t patch_dim = 3 * config->patch_size * config->patch_size;
  const std::size_t tokens =
      (config->image_size / config->patch_size) *
      (config->image_size / config->patch_size);
  impl.patch_w = bind("v.patch_embd.weight", embed * patch_dim);
  impl.patch_bias = bind("v.patch_embd.bias", embed);
  impl.position = bind("v.position_embd.weight", tokens * embed);
  impl.post_ln_weight = bind("v.post_ln.weight", embed);
  impl.post_ln_bias = bind("v.post_ln.bias", embed);
  impl.blocks.resize(config->block_count);
  for (std::size_t l = 0; l < config->block_count; ++l) {
    const std::string p = "v.blk." + std::to_string(l) + ".";
    core::VisionBlockBuffers& b = impl.blocks[l];
    b.ln1_weight = bind(p + "ln1.weight", embed);
    b.ln1_bias = bind(p + "ln1.bias", embed);
    b.qkv_weight = bind(p + "attn_qkv.weight", 3 * embed * embed);
    b.qkv_bias = bind(p + "attn_qkv.bias", 3 * embed);
    b.out_weight = bind(p + "attn_out.weight", embed * embed);
    b.out_bias = bind(p + "attn_out.bias", embed);
    b.ln2_weight = bind(p + "ln2.weight", embed);
    b.ln2_bias = bind(p + "ln2.bias", embed);
    b.up_weight = bind(p + "ffn_up.weight", ffn * embed);
    b.up_bias = bind(p + "ffn_up.bias", ffn);
    b.down_weight = bind(p + "ffn_down.weight", embed * ffn);
    b.down_bias = bind(p + "ffn_down.bias", embed);
    if (b.ln1_weight == nullptr || b.qkv_weight == nullptr ||
        b.out_weight == nullptr || b.up_weight == nullptr ||
        b.down_weight == nullptr) {
      return std::unexpected(StatusCode::MalformedFile);
    }
  }
  const std::size_t merged =
      config->spatial_merge_size * config->spatial_merge_size * embed;
  impl.mm0_weight = bind("mm.0.weight", merged * merged);
  impl.mm0_bias = bind("mm.0.bias", merged);
  impl.mm2_weight = bind("mm.2.weight", config->projection_dim * merged);
  impl.mm2_bias = bind("mm.2.bias", config->projection_dim);
  if (impl.patch_w == nullptr || impl.patch_bias == nullptr ||
      impl.position == nullptr || impl.post_ln_weight == nullptr ||
      impl.post_ln_bias == nullptr || impl.mm0_weight == nullptr ||
      impl.mm0_bias == nullptr || impl.mm2_weight == nullptr ||
      impl.mm2_bias == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto load_kernel = [&backend](std::unique_ptr<Kernel>& slot,
                                const char* name) -> StatusCode {
    auto kernel = backend.LoadKernel(name, {});
    if (!kernel) {
      return kernel.error();
    }
    slot = std::move(*kernel);
    return StatusCode::Ok;
  };
  const std::pair<std::unique_ptr<Kernel>*, const char*> kernels[] = {
      {&impl.layernorm, "layernorm"}, {&impl.gemm, "gemm_f32"},
      {&impl.attention, "attention"}, {&impl.gelu, "gelu"},
      {&impl.bias_add, "bias_add"},   {&impl.add, "add"},
      {&impl.spatial_merge, "spatial_merge"},
      {&impl.patchify, "image_patchify"}};
  for (const auto& [slot, name] : kernels) {
    const StatusCode status = load_kernel(*slot, name);
    if (status != StatusCode::Ok) {
      return std::unexpected(status);
    }
  }
  return model;
}

std::expected<std::vector<float>, StatusCode> VisionModel::Encode(
    Backend& backend, std::span<const float> image, std::size_t h, std::size_t w) {
  const VisionConfig& cfg = impl_->config;
  if (h != cfg.image_size || w != cfg.image_size ||
      image.size() != h * w * 3) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t patch = cfg.patch_size;
  const std::size_t grid_w = w / patch;
  const std::size_t grid_h = h / patch;
  const std::size_t tokens = grid_w * grid_h;
  const std::size_t patch_dim = 3 * patch * patch;
  const std::size_t embed = cfg.embedding_length;
  auto image_buf = backend.AllocateBuffer(image.size() * 4, MemoryKind::Device);
  auto mean_buf = backend.AllocateBuffer(3 * 4, MemoryKind::Device);
  auto std_buf = backend.AllocateBuffer(3 * 4, MemoryKind::Device);
  auto patches = backend.AllocateBuffer(tokens * patch_dim * 4,
                                        MemoryKind::Device);
  auto hidden = backend.AllocateBuffer(tokens * embed * 4, MemoryKind::Device);
  const std::size_t out_tokens = (grid_h / cfg.spatial_merge_size) *
                                 (grid_w / cfg.spatial_merge_size);
  auto embeddings = backend.AllocateBuffer(
      out_tokens * cfg.projection_dim * 4, MemoryKind::Device);
  if (!image_buf || !mean_buf || !std_buf || !patches || !hidden ||
      !embeddings) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  backend.CopyH2D(**image_buf, std::span<const std::byte>(
                                   reinterpret_cast<const std::byte*>(
                                       image.data()),
                                   image.size() * 4));
  backend.CopyH2D(**mean_buf, std::span<const std::byte>(
                                  reinterpret_cast<const std::byte*>(
                                      cfg.image_mean),
                                  3 * 4));
  backend.CopyH2D(**std_buf, std::span<const std::byte>(
                                 reinterpret_cast<const std::byte*>(
                                     cfg.image_std),
                                 3 * 4));
  KernelLaunch patchify_launch;
  patchify_launch.grid_x =
      static_cast<std::uint32_t>((tokens * patch_dim + 255) / 256);
  patchify_launch.block_x = 256;
  patchify_launch.buffers = {image_buf->get(), mean_buf->get(), std_buf->get(),
                             patches->get()};
  patchify_launch.scalars = {h, w, patch};
  if (!backend.LaunchKernel(*impl_->patchify, patchify_launch)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  auto stack = core::VisionStackDevice(
      backend, *impl_->layernorm, *impl_->gemm, *impl_->attention, *impl_->gelu,
      *impl_->bias_add, *impl_->add, **patches, *impl_->patch_w,
      *impl_->patch_bias, *impl_->position, impl_->blocks,
      *impl_->post_ln_weight, *impl_->post_ln_bias, **hidden, tokens, embed,
      patch_dim, cfg.head_count, cfg.embedding_length / cfg.head_count,
      cfg.feed_forward_length, cfg.layer_norm_eps);
  if (!stack) {
    return std::unexpected(stack.error());
  }
  auto merger = core::VisionMergerDevice(
      backend, *impl_->gemm, *impl_->gelu, *impl_->bias_add,
      *impl_->spatial_merge, **hidden, *impl_->mm0_weight, *impl_->mm0_bias,
      *impl_->mm2_weight, *impl_->mm2_bias, **embeddings, grid_h, grid_w, embed,
      cfg.spatial_merge_size, cfg.projection_dim);
  if (!merger) {
    return std::unexpected(merger.error());
  }
  backend.Synchronize();
  std::vector<float> out(out_tokens * cfg.projection_dim);
  if (!backend.CopyD2H(**embeddings, reinterpret_cast<std::byte*>(out.data()),
                       out.size() * 4)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return out;
}

}  // namespace tessera
