#include "tessera/budget.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/files.hpp"
#include "core/loaders/gguf.hpp"
#include "core/loaders/mxfp4.hpp"
#include "core/loaders/safetensors.hpp"
#include "core/loaders/shard.hpp"
#include "tessera/architecture.hpp"
#include "tessera/model.hpp"

namespace tessera::core {

namespace {

// Driver, scratch and fragmentation reserve kept free on top of the
// weights and the cache, so an applied recommendation does not land on
// an exactly full device. llama.cpp fits to 1024 MiB free the same way.
constexpr std::uint64_t kRecommendHeadroomBytes = 1024ull << 20;

// True for a regular file (GGUF); a directory is the MXFP4 layout.
bool IsGgufPath(const std::string& path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(std::filesystem::path(path), ec) &&
         !ec;
}

// Converted device bytes of one MXFP4 tensor. Mirrors BuildMxFp4Weights:
// F4E2M1 packs blob plus scale, F8E4M3 dequantizes to F32, the embedding
// and the head keep their layout, every other plain tensor widens to
// F32 (permutations preserve the size). Same errors on the same shapes.
std::expected<std::uint64_t, StatusCode> MxFp4ConvertedBytes(
    DType dtype, std::size_t rows, std::size_t cols,
    std::uint64_t payload_bytes, std::uint64_t scale_bytes,
    bool keep_layout) {
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  if (dtype == DType::F4E2M1) {
    if (payload_bytes > kMax - scale_bytes) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    return payload_bytes + scale_bytes;
  }
  if (rows == 0 || cols == 0 ||
      static_cast<std::uint64_t>(rows) > kMax / 4 / static_cast<std::uint64_t>(cols)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const std::uint64_t elements =
      static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols);
  if (dtype == DType::F8E4M3) {
    return elements * 4;
  }
  if (keep_layout) {
    return payload_bytes;
  }
  return elements * 4;
}

}  // namespace

HostMemoryInfo HostMemory() {
  HostMemoryInfo info;
#if defined(__linux__)
  std::ifstream stream("/proc/meminfo");
  if (!stream) {
    return info;
  }
  std::string line;
  while (std::getline(stream, line)) {
    // Each line reads "<name>: <value> kB"; parse from after the colon so
    // the alignment spaces do not matter.
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const char* value = line.c_str() + colon + 1;
    if (line.rfind("MemTotal", 0) == 0) {
      info.total_bytes = std::strtoull(value, nullptr, 10) * 1024;
    } else if (line.rfind("MemAvailable", 0) == 0) {
      info.available_bytes = std::strtoull(value, nullptr, 10) * 1024;
    }
  }
#endif
  return info;
}

std::expected<WeightBudget, StatusCode> MeasureWeights(
    const std::string& path, bool include_mtp_head) {
  if (path.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto config = Model::ProbeConfig(path);
  if (!config) {
    return std::unexpected(config.error());
  }
  WeightBudget budget;
  budget.trunk_layers = config->layers;
  budget.has_moe = config->IsMoe();
  if (IsGgufPath(path)) {
    auto mapped = MappedFile::Open(std::filesystem::path(path));
    if (!mapped) {
      return std::unexpected(mapped.error());
    }
    auto gguf = ParseGguf(mapped->bytes());
    if (!gguf) {
      return std::unexpected(gguf.error());
    }
    for (const TensorEntry& tensor : gguf->tensors) {
      auto sized = TensorBytes(tensor.dtype, tensor.shape.Numel());
      if (!sized) {
        return std::unexpected(sized.error());
      }
      budget.tensors.emplace_back(tensor.name, *sized);
    }
    return budget;
  }
  auto layout = InspectMxFp4Directory(std::filesystem::path(path));
  if (!layout) {
    return std::unexpected(layout.error());
  }
  auto mapped = MappedFile::Open(layout->weights_path);
  if (!mapped) {
    return std::unexpected(mapped.error());
  }
  const std::span<const std::byte> file_bytes = mapped->bytes();
  auto parsed = ParseSafetensorsMap(file_bytes);
  if (!parsed) {
    return std::unexpected(parsed.error());
  }
  auto config_bytes = ReadFile(std::filesystem::path(path) / "config.json");
  if (!config_bytes) {
    return std::unexpected(config_bytes.error());
  }
  const std::string_view text(
      reinterpret_cast<const char*>(config_bytes->data()),
      config_bytes->size());
  auto module = CreateArchitecture(MxFp4ArchitectureName(text));
  if (module == nullptr) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  std::unordered_map<std::string, const SafetensorsTensor*> by_name;
  for (const auto& tensor : *parsed) {
    by_name.emplace(tensor.name, &tensor);
  }
  const WeightLoadOptions options{include_mtp_head};
  for (const auto& tensor : *parsed) {
    if (tensor.entry.dtype == DType::F8E8M0) {
      continue;  // E8M0 scales pair with their F4E2M1 blob
    }
    auto internal = module->MapWeightName(tensor.name, *config, options);
    if (!internal) {
      continue;
    }
    const TensorShape& shape = tensor.entry.shape;
    if (shape.rank == 0 || shape.rank > 3) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    const std::size_t rows = shape.dims[0];
    std::size_t cols = 1;
    for (std::size_t d = 1; d < shape.rank; ++d) {
      cols *= shape.dims[d];
    }
    const std::uint64_t payload_bytes = tensor.end - tensor.begin;
    std::uint64_t scale_bytes = 0;
    if (tensor.entry.dtype == DType::F4E2M1 ||
        tensor.entry.dtype == DType::F8E4M3) {
      const std::string scale_name = tensor.name + "_scale";
      const auto scale = by_name.find(scale_name);
      if (scale == by_name.end()) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      scale_bytes = scale->second->end - scale->second->begin;
    }
    const bool keep_layout = *internal == "token_embd.weight" ||
                             *internal == "output.weight";
    auto sized = MxFp4ConvertedBytes(tensor.entry.dtype, rows, cols,
                                     payload_bytes, scale_bytes, keep_layout);
    if (!sized) {
      return std::unexpected(sized.error());
    }
    budget.tensors.emplace_back(std::move(*internal), *sized);
  }
  return budget;
}

std::uint64_t FullKvLayerBytes(const TransformerConfig& config,
                               KvCacheType type, std::size_t context) {
  const std::size_t kv_dim =
      config.attention.kv_heads * config.attention.head_dim;
  if (kv_dim == 0 || context == 0) {
    return 0;
  }
  // Row bytes mirror AppendKv; quantized types add one fp32 scale row
  // each for the keys and the values.
  const std::size_t row =
      type == KvCacheType::F32   ? kv_dim * 4
      : type == KvCacheType::F16 ? kv_dim * 2
      : type == KvCacheType::Q4  ? kv_dim / 2
                                 : kv_dim;
  const bool quantized = type == KvCacheType::Q8 || type == KvCacheType::Q4 ||
                         type == KvCacheType::FP8;
  std::size_t capacity = detail::kInitialKvRows;
  while (capacity < context &&
         capacity <= std::numeric_limits<std::size_t>::max() / 2) {
    capacity *= 2;
  }
  return 2 * static_cast<std::uint64_t>(capacity) *
             static_cast<std::uint64_t>(row) +
         (quantized ? 2 * static_cast<std::uint64_t>(capacity) * 4 : 0);
}

std::uint64_t LinearStateLayerBytes(const TransformerConfig& config) {
  if (!config.hybrid) {
    return 0;
  }
  // Products mirror DeriveGeometry and the InitScratch histories: the
  // causal conv history plus the recurrent state, both fp32.
  const std::size_t head_k = config.ssm.state_size;
  const std::size_t num_k = config.ssm.group_count;
  const std::size_t value_dim = config.ssm.inner_size;
  const std::size_t num_v = config.ssm.time_step_rank;
  const std::size_t width = config.ssm.conv_kernel;
  if (head_k == 0 || num_k == 0 || value_dim == 0 || num_v == 0 ||
      width == 0 || value_dim % num_v != 0 || num_v % num_k != 0) {
    return 0;
  }
  const std::size_t key_dim = head_k * num_k;
  const std::size_t conv_dim = key_dim * 2 + value_dim;
  const std::size_t head_v = value_dim / num_v;
  return static_cast<std::uint64_t>(conv_dim) *
             static_cast<std::uint64_t>(width - 1) * 4 +
         static_cast<std::uint64_t>(num_v) *
             static_cast<std::uint64_t>(head_k) *
             static_cast<std::uint64_t>(head_v) * 4;
}

std::uint64_t ResidentWeightBytes(const WeightBudget& budget,
                                  std::size_t gpu_layers) {
  std::uint64_t total = 0;
  for (const auto& [name, bytes] : budget.tensors) {
    if (GpuLayerResident(name, gpu_layers, budget.trunk_layers) ||
        PagedMoeDenseResident(name, gpu_layers, budget.trunk_layers,
                              budget.has_moe)) {
      total += bytes;
    }
  }
  return total;
}

std::uint64_t PagedWeightBytes(const WeightBudget& budget,
                               std::size_t gpu_layers) {
  std::uint64_t total = 0;
  for (const auto& [name, bytes] : budget.tensors) {
    total += bytes;
  }
  const std::uint64_t resident = ResidentWeightBytes(budget, gpu_layers);
  return total > resident ? total - resident : 0;
}

std::uint64_t StagingWeightBytes(const WeightBudget& budget,
                                 std::size_t gpu_layers) {
  // A full upload stages nothing. Under paging the loader keeps one device
  // buffer per paged tensor kind, sized to that kind's largest paged layer
  // (see UploadWeights' stage_max), and reuses it for every paged layer.
  if (gpu_layers == 0) {
    return 0;
  }
  std::map<std::string, std::uint64_t> max_by_kind;
  for (const auto& [name, bytes] : budget.tensors) {
    if (GpuLayerResident(name, gpu_layers, budget.trunk_layers) ||
        PagedMoeDenseResident(name, gpu_layers, budget.trunk_layers,
                              budget.has_moe)) {
      continue;
    }
    std::size_t layer = 0;
    std::string_view kind;
    if (!SplitBlockName(name, &layer, &kind) || layer >= budget.trunk_layers) {
      continue;
    }
    auto [it, inserted] = max_by_kind.try_emplace(std::string(kind), bytes);
    if (!inserted && it->second < bytes) {
      it->second = bytes;
    }
  }
  std::uint64_t total = 0;
  for (const auto& entry : max_by_kind) {
    total += entry.second;
  }
  return total;
}

// Cache bytes at context, split by where they live: the full-attention KV
// cache (offloadable to system RAM) and the context-independent linear
// state (always on the device). One geometrically grown full-attention
// layer each, plus one MTP KV layer for the "mtp" strategy. Shared with
// RecommendGpuLayers so the help text and the fit check judge the same
// cost.
struct CacheCost {
  std::uint64_t full_kv_bytes = 0;
  std::uint64_t linear_bytes = 0;
};

CacheCost CacheBytes(const TransformerConfig& config, std::size_t trunk_layers,
                     std::size_t context, KvCacheType kv_type,
                     const std::string& strategy) {
  std::size_t full_layers = 0;
  for (std::size_t l = 0; l < trunk_layers; ++l) {
    if (config.IsFullAttentionLayer(l)) {
      ++full_layers;
    }
  }
  const std::size_t linear_layers =
      config.hybrid ? trunk_layers - full_layers : 0;
  CacheCost cost;
  cost.full_kv_bytes = static_cast<std::uint64_t>(full_layers) *
                       FullKvLayerBytes(config, kv_type, context);
  cost.linear_bytes = static_cast<std::uint64_t>(linear_layers) *
                      LinearStateLayerBytes(config);
  if (strategy == "mtp") {
    cost.full_kv_bytes += FullKvLayerBytes(config, kv_type, context);
  }
  return cost;
}

// Device bytes of the cache: the linear state always, the full-attention
// KV only when it is not offloaded to system RAM.
std::uint64_t DeviceCacheBytes(const CacheCost& cost, bool kv_offload) {
  return (kv_offload ? 0 : cost.full_kv_bytes) + cost.linear_bytes;
}

// Overflow-safe fit: resident weights plus the paging staging buffers plus
// the cache plus the headroom reserve stays within the budget. The staging
// term runs the resident set close to the device, so omitting it lets a
// recommendation land on an out-of-memory load.
bool FitsBudget(std::uint64_t resident, std::uint64_t staging,
                std::uint64_t kv_bytes, std::uint64_t budget) {
  if (resident > budget) {
    return false;
  }
  const std::uint64_t after_resident = budget - resident;
  if (staging > after_resident) {
    return false;
  }
  const std::uint64_t rest = after_resident - staging;
  return kv_bytes <= rest && kRecommendHeadroomBytes <= rest - kv_bytes;
}

std::expected<GpuLayersAdvice, StatusCode> RecommendGpuLayers(
    const std::string& path, std::size_t context, KvCacheType kv_type,
    std::uint64_t free_bytes, const std::string& strategy, bool kv_offload) {
  auto config = Model::ProbeConfig(path);
  if (!config) {
    return std::unexpected(config.error());
  }
  const bool include_mtp_head = strategy != "dflash2";
  auto budget = MeasureWeights(path, include_mtp_head);
  if (!budget) {
    return std::unexpected(budget.error());
  }
  std::size_t full_layers = 0;
  for (std::size_t l = 0; l < budget->trunk_layers; ++l) {
    if (config->IsFullAttentionLayer(l)) {
      ++full_layers;
    }
  }
  const std::size_t linear_layers = config->hybrid
                                        ? budget->trunk_layers - full_layers
                                        : 0;
  const CacheCost cost =
      CacheBytes(*config, budget->trunk_layers, context, kv_type, strategy);
  const std::uint64_t device_cache = DeviceCacheBytes(cost, kv_offload);
  GpuLayersAdvice advice;
  advice.trunk_layers = budget->trunk_layers;
  advice.total_layers = budget->trunk_layers + 1;
  advice.full_layers = full_layers;
  advice.linear_layers = linear_layers;
  advice.full_bytes = ResidentWeightBytes(*budget, 0);
  advice.kv_bytes = cost.full_kv_bytes + cost.linear_bytes;
  advice.kv_offload = kv_offload;
  advice.free_bytes = free_bytes;
  if (free_bytes == 0) {
    return advice;
  }
  if (FitsBudget(advice.full_bytes, /*staging=*/0, device_cache, free_bytes)) {
    advice.has_best = true;
    advice.best = 0;
    advice.best_bytes = advice.full_bytes;
    return advice;
  }
  for (std::size_t layers = budget->trunk_layers; layers > 0; --layers) {
    const std::uint64_t resident = ResidentWeightBytes(*budget, layers);
    const std::uint64_t staging = StagingWeightBytes(*budget, layers);
    if (FitsBudget(resident, staging, device_cache, free_bytes)) {
      advice.has_best = true;
      advice.best = layers;
      advice.best_bytes = resident;
      advice.staging_bytes = staging;
      return advice;
    }
  }
  return advice;
}

std::expected<GpuLayersFit, StatusCode> FitGpuLayers(
    const std::string& path, std::size_t context, KvCacheType kv_type,
    std::uint64_t budget_bytes, const std::string& strategy,
    std::size_t gpu_layers, bool kv_offload) {
  auto config = Model::ProbeConfig(path);
  if (!config) {
    return std::unexpected(config.error());
  }
  const bool include_mtp_head = strategy != "dflash2";
  auto budget = MeasureWeights(path, include_mtp_head);
  if (!budget) {
    return std::unexpected(budget.error());
  }
  const CacheCost cost =
      CacheBytes(*config, budget->trunk_layers, context, kv_type, strategy);
  GpuLayersFit fit;
  fit.total_layers = budget->trunk_layers + 1;
  fit.resident_bytes = ResidentWeightBytes(*budget, gpu_layers);
  fit.staging_bytes = StagingWeightBytes(*budget, gpu_layers);
  fit.kv_bytes = cost.full_kv_bytes + cost.linear_bytes;
  fit.kv_offload = kv_offload;
  fit.budget_bytes = budget_bytes;
  fit.fits = FitsBudget(fit.resident_bytes, fit.staging_bytes,
                        DeviceCacheBytes(cost, kv_offload), budget_bytes);
  return fit;
}

}  // namespace tessera::core
