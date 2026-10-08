#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>

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
struct VkKernel {
  VkPipeline pipeline = VK_NULL_HANDLE;
  std::string name;
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
  // Record, submit, and wait a launch (synchronous, like the copy path).
  std::expected<void, StatusCode> LaunchKernel(
      const VkKernel& kernel, const KernelLaunch& launch,
      std::span<const VkBinding> bindings);

 private:
  void DestroyResources();
  void LogError(std::string_view message) const;

  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  // Reused across launches: the API is synchronous, so at most one launch
  // is in flight and the command buffer and fence can be reset.
  VkCommandBuffer command_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  log::Diagnostics* diagnostics_ = nullptr;
  bool ready_ = false;
};

}  // namespace tessera::backends::vulkan
