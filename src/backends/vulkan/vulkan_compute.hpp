#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include <vulkan/vulkan.h>

#include "tessera/backend.hpp"
#include "tessera/log.hpp"
#include "tessera/types.hpp"

namespace tessera::backends::vulkan {

// Fence wait bound: 30 s of device silence is a hang, not a slow kernel.
constexpr std::uint64_t kFenceTimeoutNs = 30ull * 1000 * 1000 * 1000;

// Map a Vulkan result onto a StatusCode for error returns.
[[nodiscard]] inline StatusCode FromVkResult(VkResult result) {
  switch (result) {
    case VK_SUCCESS: return StatusCode::Ok;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return StatusCode::OutOfMemory;
    case VK_ERROR_INITIALIZATION_FAILED:
    case VK_ERROR_DEVICE_LOST:
    default: return StatusCode::DeviceError;
  }
}

// One resolved buffer binding: the vendor buffer object + byte size.
struct VkBinding {
  VkBuffer buffer = VK_NULL_HANDLE;
  std::size_t size = 0;
};

// A loaded Vulkan kernel: one compute pipeline on the shared layout.
// read_mask/write_mask have bit i set when binding i is read/written, decoded
// from the SPIR-V NonWritable/NonReadable decorations. They drive the
// inter-dispatch memory barriers: a barrier is needed only when a dispatch
// reads a buffer written since the last barrier, or writes a buffer touched
// since the last barrier. All-access (the default) reproduces the
// conservative one-barrier-per-dispatch behavior.
struct VkKernel {
  VkPipeline pipeline = VK_NULL_HANDLE;
  std::string name;
  std::uint32_t read_mask = 0xffffffffu;
  std::uint32_t write_mask = 0xffffffffu;
};

// Key for the descriptor-set cache: the pipeline plus the bound buffers
// and their sizes. Only the first `count` entries matter.
struct DescKey {
  VkPipeline pipeline = VK_NULL_HANDLE;
  std::uint32_t count = 0;
  std::array<VkBuffer, kMaxBoundBuffers> buffers{};
  std::array<std::size_t, kMaxBoundBuffers> sizes{};
  bool operator==(const DescKey& other) const = default;
};

// Mix prime for the descriptor-set cache hash.
constexpr std::size_t kDescKeyHashMix = 1315423911u;

struct DescKeyHash {
  std::size_t operator()(const DescKey& key) const noexcept {
    std::size_t hash =
        std::hash<void*>{}(static_cast<void*>(key.pipeline));
    hash = hash * kDescKeyHashMix + key.count;
    for (std::uint32_t i = 0; i < key.count; ++i) {
      hash = hash * kDescKeyHashMix +
             std::hash<void*>{}(static_cast<void*>(key.buffers[i]));
      hash = hash * kDescKeyHashMix + key.sizes[i];
    }
    return hash;
  }
};

// Kernel launch plumbing for the vulkan backend: SPIR-V pipelines on one
// shared descriptor layout (kMaxBoundBuffers buffer bindings, scalars as
// push constants). Created after device creation, torn down before the
// device is destroyed.
class VulkanCompute {
 public:
  [[nodiscard]] bool Ready() const {
    return ready_;
  }
  // Set up the command pool, descriptor layout, and descriptor pool.
  std::expected<void, StatusCode> Init(VkDevice device, VkQueue queue,
                                      log::Diagnostics* diagnostics);
  void Shutdown();
  // See Backend::LoadKernel for the name/code contract.
  std::expected<std::unique_ptr<VkKernel>, StatusCode> LoadKernel(
      std::string_view name, std::span<const std::byte> code);
  // Release a VkKernel; destroys its pipeline.
  void FreeKernel(VkKernel* kernel);
  // Attribute descriptor-set lookups to the core's prefill/decode phase
  // for hit-rate accounting; the backend forwards its phase here.
  void SetCachePhase(CachePhase phase) { cache_phase_ = phase; }
  // Record a compute dispatch into the pending command buffer. The
  // submission is deferred: the dispatch is flushed by Synchronize() (or
  // when the descriptor pool fills), so a decode step that issues
  // thousands of small launches pays one queue submit, not thousands.
  std::expected<void, StatusCode> LaunchKernel(
      const VkKernel& kernel, const KernelLaunch& launch,
      std::span<const VkBinding> bindings);
  // Record a device-to-device copy into the pending command buffer, so it
  // stays ordered with the dispatches around it without a queue wait.
  std::expected<void, StatusCode> RecordCopy(VkBuffer src, VkBuffer dst,
                                             std::size_t src_offset,
                                             std::size_t dst_offset,
                                             std::size_t bytes);
  // Record a small host-to-device update into the pending command buffer
  // (vkCmdUpdateBuffer), so a decode step's token id, position and
  // activation uploads do not flush and wait between dispatches. The data
  // must be at most 64 KiB and a multiple of 4 bytes.
  std::expected<void, StatusCode> RecordUpdate(
      VkBuffer dst, std::size_t dst_offset, std::span<const std::byte> data);
  // Flush the pending command buffer (submit) and wait for it. Idempotent.
  [[nodiscard]] std::expected<void, StatusCode> Synchronize();
  // Submit pending work without waiting, so a transfer can queue behind
  // it on the same queue and one fence wait covers both. Idempotent.
  // Call Wait() once the queued work must be complete.
  [[nodiscard]] std::expected<void, StatusCode> Flush();
  // Wait for a prior Flush and release its descriptor sets. Idempotent.
  [[nodiscard]] std::expected<void, StatusCode> Wait();
  // Drop the cached sets bound to `buffer`. The backend calls this when
  // it destroys a buffer: handle values recycle, so a set cached for a
  // dead buffer would otherwise hit for a later buffer with the same
  // value and read the wrong memory.
  void InvalidateBuffer(VkBuffer buffer);

 private:
  void DestroyResources();
  void LogError(std::string_view message) const;
  // Count one descriptor-set lookup in the current phase.
  void CountDescLookup(bool hit);
  // Log the per-phase descriptor-set hit rates (phases with traffic
  // only); silent without diagnostics or without lookups.
  void ReportDescHitRate() const;
  // End and submit the pending command buffer, wait on its fence, and
  // reset the descriptor pool. A no-op when nothing is recording.
  [[nodiscard]] std::expected<void, StatusCode> FlushAndWait();
  // Start the pending command buffer if it is not recording.
  [[nodiscard]] std::expected<void, StatusCode> EnsureRecording();
  // Decide whether a dispatch over `bindings` (with the given per-binding
  // access masks) needs a barrier, record one when it does, and fold the
  // bindings into the pending hazard sets.
  void OrderAccess(std::span<const VkBinding> bindings, std::uint32_t read_mask,
                   std::uint32_t write_mask);
  // Drop every cached descriptor set and reset the pool (after a flush).
  // The pool outgrowing path and teardown call this.
  void InvalidateDescriptors();

  // Buffers touched since the last barrier, split by access. A dispatch needs
  // a barrier only against these sets, so independent dispatches overlap.
  std::unordered_set<VkBuffer> read_pending_;
  std::unordered_set<VkBuffer> write_pending_;

  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  // The pending command buffer and the fence for its submission. Dispatches
  // and device copies record into it; it is submitted once per flush.
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  bool recording_ = false;
  // A Flush() submission is still unwaited: the next Wait() drains it
  // and releases its descriptor sets.
  bool submitted_ = false;
  // Descriptor sets are allocated on demand from one pool and released in
  // bulk by vkResetDescriptorPool after each flush.
  static constexpr std::uint32_t kDescriptorSetBudget = 8192;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  log::Diagnostics* diagnostics_ = nullptr;
  bool ready_ = false;
  // Descriptor-set lookup counters per generation phase, for the
  // end-of-run hit-rate report (see Shutdown). A hit reuses a set with
  // no alloc or update; a miss allocates and writes one.
  std::uint64_t desc_prefill_hits_ = 0;
  std::uint64_t desc_prefill_misses_ = 0;
  std::uint64_t desc_decode_hits_ = 0;
  std::uint64_t desc_decode_misses_ = 0;
  // Phase the backend forwarded for hit-rate accounting (default decode).
  CachePhase cache_phase_ = CachePhase::kDecode;
  // Cached descriptor sets by pipeline and bindings: a decode step binds
  // the same weights every token, so repeats skip alloc and update.
  std::unordered_map<DescKey, VkDescriptorSet, DescKeyHash> desc_cache_;
  // Per-launch scratch for the descriptor writes on a cache miss. Sized
  // for the binding limit and kept on the backend, so a launch pays no
  // per-launch heap cost.
  std::array<VkDescriptorBufferInfo, kMaxBoundBuffers> info_scratch_{};
  std::array<VkWriteDescriptorSet, kMaxBoundBuffers> write_scratch_{};
  // Tensor-core (cooperative matrix) support, probed at Init. False keeps
  // the tensor-core kernels on their scalar fallbacks.
  bool coopmat_ = false;
};

}  // namespace tessera::backends::vulkan
