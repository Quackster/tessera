#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
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

 private:
  void DestroyResources();
  void LogError(std::string_view message) const;
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
  // Descriptor sets are allocated on demand from one pool and released in
  // bulk by vkResetDescriptorPool after each flush.
  static constexpr std::uint32_t kDescriptorSetBudget = 8192;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  log::Diagnostics* diagnostics_ = nullptr;
  bool ready_ = false;
};

}  // namespace tessera::backends::vulkan
