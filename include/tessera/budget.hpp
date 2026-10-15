#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "tessera/types.hpp"

// VRAM budget advisor for `--gpu-layers`: how many layers a model has,
// what they weigh, what the KV cache costs, and which layer count fits
// the free device memory. Reads definitions only, never uploads.

namespace tessera::core {

// On-disk weight bytes per tensor in manifest order.
struct WeightBudget {
  // Trunk block count from the model definition.
  std::size_t trunk_layers = 0;
  // True when the definition carries routed experts: under paging the
  // dense block tensors stay resident and only experts page (see
  // PagedMoeDenseResident), so the resident bytes include them.
  bool has_moe = false;
  // (internal tensor name, device bytes) in manifest order.
  std::vector<std::pair<std::string, std::uint64_t>> tensors;
};

// Measure the weight bytes of the model at `path` (a .gguf file or an
// MXFP4 directory, the same dispatch as Model::Load) without uploading
// anything. `include_mtp_head` selects the MTP head, mirroring the
// loader. FileNotFound on a bad path; MalformedFile or
// UnsupportedFeature on an unreadable definition.
//
// Usage:
//   auto budget = MeasureWeights(path, true);
//   if (budget) for (const auto& [name, bytes] : budget->tensors) ...
[[nodiscard]] std::expected<WeightBudget, StatusCode> MeasureWeights(
    const std::string& path, bool include_mtp_head);

// Device bytes of one full-attention layer's KV cache at `context` rows
// for `type`, mirroring AppendKv: geometrically grown K and V plus one
// fp32 scale row each for the quantized types.
[[nodiscard]] std::uint64_t FullKvLayerBytes(const TransformerConfig& config,
                                             KvCacheType type,
                                             std::size_t context);

// Device bytes of one linear-attention layer's recurrent state, context
// independent, mirroring the InitScratch products. Zero without a hybrid
// definition.
[[nodiscard]] std::uint64_t LinearStateLayerBytes(
    const TransformerConfig& config);

// Resident weight bytes for `gpu_layers` (llama.cpp -ngl order: the head
// plus the last gpu_layers - 1 trunk blocks; 0 keeps everything).
[[nodiscard]] std::uint64_t ResidentWeightBytes(const WeightBudget& budget,
                                                std::size_t gpu_layers);

// Device bytes of the paging staging buffers for `gpu_layers`: one buffer
// per paged tensor kind, sized to that kind's largest layer, as the loader
// allocates them (see UploadWeights). Zero for a full upload (0) and for a
// model with no paged tensors. A fit must count it: the staging runs the
// resident set close to the device.
[[nodiscard]] std::uint64_t StagingWeightBytes(const WeightBudget& budget,
                                               std::size_t gpu_layers);

// --gpu-layers sizing advice for one load.
struct GpuLayersAdvice {
  std::size_t trunk_layers = 0;
  // Trunk blocks plus the output head: the largest useful layer count.
  std::size_t total_layers = 0;
  std::size_t full_layers = 0;
  std::size_t linear_layers = 0;
  // Resident bytes at 0 (the full upload).
  std::uint64_t full_bytes = 0;
  // Staging bytes at `best` (0 for the full upload).
  std::uint64_t staging_bytes = 0;
  // Cache bytes at context, with one MTP layer when strategy is "mtp".
  std::uint64_t kv_bytes = 0;
  // Device budget in bytes: free memory when the backend reports it, else
  // the device total (an upper bound; other processes may hold memory).
  std::uint64_t free_bytes = 0;
  // True when free_bytes is the device total, not current free memory.
  bool budget_is_total = false;
  bool has_best = false;
  // Largest fitting layer count (0 means the full upload fits, so no
  // paging is needed).
  std::size_t best = 0;
  // Its resident bytes.
  std::uint64_t best_bytes = 0;
};

// Recommend a `--gpu-layers` value for `path` at `context` with `kv_type`
// against `free_bytes` of device memory. `strategy` is "mtp", "dflash2"
// or "none" (the calibration label): it decides the MTP head and its KV
// row. An estimate: weights are exact, the cache follows the loader's
// growth, and a headroom reserve covers the driver, the scratch and the
// draft. FileNotFound/MalformedFile/UnsupportedFeature as MeasureWeights.
[[nodiscard]] std::expected<GpuLayersAdvice, StatusCode> RecommendGpuLayers(
    const std::string& path, std::size_t context, KvCacheType kv_type,
    std::uint64_t free_bytes, const std::string& strategy);

// Fit of one requested `--gpu-layers` value against device memory.
struct GpuLayersFit {
  // Trunk blocks plus the output head: the largest useful layer count.
  std::size_t total_layers = 0;
  // Resident weight bytes at the requested value.
  std::uint64_t resident_bytes = 0;
  // Staging buffer bytes at the requested value (0 for the full upload).
  std::uint64_t staging_bytes = 0;
  // Cache bytes at context, with one MTP layer when strategy is "mtp".
  std::uint64_t kv_bytes = 0;
  // Device budget in bytes (free memory or the device total).
  std::uint64_t budget_bytes = 0;
  // True when budget_bytes is the device total, not current free memory.
  bool budget_is_total = false;
  // Resident plus cache plus the headroom reserve fits the budget.
  bool fits = false;
};

// Check a requested `--gpu-layers` value for `path` at `context` with
// `kv_type` against `budget_bytes` of device memory. `budget_bytes`
// must be nonzero: pass free memory when the backend reports it, else
// the device total (then budget_is_total is set). `strategy` is "mtp",
// "dflash2" or "none". Same errors as MeasureWeights.
//
// Usage:
//   auto fit = FitGpuLayers(path, 4096, kv, free, "none", 20);
//   if (fit && !fit->fits) refuse the load before it starts;
[[nodiscard]] std::expected<GpuLayersFit, StatusCode> FitGpuLayers(
    const std::string& path, std::size_t context, KvCacheType kv_type,
    std::uint64_t budget_bytes, const std::string& strategy,
    std::size_t gpu_layers);

}  // namespace tessera::core
