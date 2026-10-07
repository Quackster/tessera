#include "backends/vulkan/vulkan_compute.hpp"

#include "tessera/backend.hpp"

// The vendor C API is used (not the generated C++ bindings): it is stable
// across Khronos releases, while the C++ bindings changed their calling
// convention (explicit Dispatch) in 2025-2026.
#include <vulkan/vulkan.h>

#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace tessera::backends::vulkan {

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

// A single allocated memory block. The core sees only an opaque handle;
// the vendor objects live here.
struct MemoryRecord {
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  void* mapped = nullptr;
};

// The Vulkan objects owned by the backend.
struct VulkanState {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
};

void DestroyState(VulkanState& state) {
  if (state.device != VK_NULL_HANDLE) {
    if (state.queue != VK_NULL_HANDLE) {
      vkQueueWaitIdle(state.queue);
    }
    if (state.pool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(state.device, state.pool, nullptr);
    }
    vkDestroyDevice(state.device, nullptr);
  }
  if (state.instance != VK_NULL_HANDLE) {
    vkDestroyInstance(state.instance, nullptr);
  }
  state = VulkanState{};
}

}  // namespace

class VulkanBackend final : public Backend {
 public:
  ~VulkanBackend() override {
    compute_.Shutdown();
    FreeAllMemory();
    DestroyState(state_);
  }

  std::string_view Name() const override {
    return "vulkan";
  }

  std::string_view DeviceName() const override {
    return device_name_;
  }

  std::expected<void, StatusCode> Init() override {
    if (initialized_) {
      return {};
    }
    // Instance, with the validation layer when the ICD provides it.
    std::vector<const char*> layers;
    std::uint32_t layer_count = 0;
    if (vkEnumerateInstanceLayerProperties(&layer_count, nullptr) ==
        VK_SUCCESS) {
      std::vector<VkLayerProperties> properties(layer_count);
      if (vkEnumerateInstanceLayerProperties(&layer_count, properties.data()) ==
          VK_SUCCESS) {
        for (const auto& property : properties) {
          if (std::string_view(property.layerName) == kValidationLayer) {
            layers.push_back(kValidationLayer);
            break;
          }
        }
      }
    }
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "tessera";
    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    instance_info.enabledLayerCount =
        static_cast<std::uint32_t>(layers.size());
    instance_info.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
    VkResult result =
        vkCreateInstance(&instance_info, nullptr, &state_.instance);
    if (result != VK_SUCCESS) {
      LogError(std::string("vkCreateInstance failed (") +
               std::to_string(static_cast<int>(result)) +
               "); no Vulkan device available, check the ICD list");
      return std::unexpected(FromVkResult(result));
    }
    // First physical device with a compute queue.
    std::uint32_t device_count = 0;
    result = vkEnumeratePhysicalDevices(state_.instance, &device_count, nullptr);
    if (result != VK_SUCCESS || device_count == 0) {
      LogError("vkEnumeratePhysicalDevices found no devices; run vulkaninfo "
               "to list devices and check the ICDs");
      DestroyState(state_);
      return std::unexpected(FromVkResult(result));
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    result =
        vkEnumeratePhysicalDevices(state_.instance, &device_count, devices.data());
    if (result != VK_SUCCESS) {
      LogError(std::string("vkEnumeratePhysicalDevices failed (") +
               std::to_string(static_cast<int>(result)) + ")");
      DestroyState(state_);
      return std::unexpected(FromVkResult(result));
    }
    // Vulkan 1.4 removed the shaderCompute feature; a compute-capable
    // device is one with a compute queue family.
    for (auto candidate : devices) {
      std::uint32_t family_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count,
                                              nullptr);
      std::vector<VkQueueFamilyProperties> probe(family_count);
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count,
                                              probe.data());
      bool has_compute = false;
      for (const auto& family : probe) {
        if ((family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
          has_compute = true;
          break;
        }
      }
      if (has_compute) {
        state_.physical = candidate;
        break;
      }
    }
    if (state_.physical == VK_NULL_HANDLE) {
      LogError("no physical device with a compute queue family; the device "
               "cannot run kernels");
      DestroyState(state_);
      return std::unexpected(StatusCode::DeviceError);
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(state_.physical, &props);
    device_name_ = props.deviceName;

    std::uint32_t queue_family = 0;
    bool found_family = false;
    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(state_.physical, &family_count,
                                            nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(state_.physical, &family_count,
                                            families.data());
    for (const auto& family : families) {
      if ((family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
        found_family = true;
        break;
      }
      ++queue_family;
    }
    if (!found_family) {
      LogError("compute queue family not found on " + device_name_ +
               "; the device cannot run kernels");
      DestroyState(state_);
      return std::unexpected(StatusCode::DeviceError);
    }
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    result =
        vkCreateDevice(state_.physical, &device_info, nullptr, &state_.device);
    if (result != VK_SUCCESS) {
      LogError(std::string("vkCreateDevice failed (") +
               std::to_string(static_cast<int>(result)) + "); check the "
               "queue family and device extensions");
      DestroyState(state_);
      return std::unexpected(FromVkResult(result));
    }
    vkGetDeviceQueue(state_.device, queue_family, 0, &state_.queue);
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family;
    result = vkCreateCommandPool(state_.device, &pool_info, nullptr,
                                &state_.pool);
    if (result != VK_SUCCESS) {
      LogError(std::string("vkCreateCommandPool failed (") +
               std::to_string(static_cast<int>(result)) + ")");
      DestroyState(state_);
      return std::unexpected(FromVkResult(result));
    }
    auto compute = compute_.Init(state_.device, state_.queue, diagnostics_);
    if (!compute) {
      DestroyState(state_);
      return compute;
    }
    initialized_ = true;
    return {};
  }

  std::expected<std::unique_ptr<Buffer>, StatusCode> AllocateBuffer(
      std::size_t bytes, MemoryKind kind) override {
    if (bytes == 0) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    if (!initialized_) {
      return std::unexpected(StatusCode::DeviceError);
    }
    auto type_index = PickMemoryIndex(kind);
    if (!type_index) {
      LogError(std::string("no ") +
               (kind == MemoryKind::Device ? "device-local" : "host-visible") +
               " memory type on " + device_name_);
      return std::unexpected(StatusCode::OutOfMemory);
    }
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = bytes;
    alloc_info.memoryTypeIndex = *type_index;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    auto result =
        vkAllocateMemory(state_.device, &alloc_info, nullptr, &memory);
    if (result != VK_SUCCESS) {
      LogError(std::string("vkAllocateMemory failed for ") +
               std::to_string(bytes) + " bytes (" +
               std::to_string(static_cast<int>(result)) + "); the device is "
               "out of that memory pool");
      return std::unexpected(FromVkResult(result));
    }
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = bytes;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    // Default sharing mode is exclusive; no queue family list needed.
    VkBuffer buffer = VK_NULL_HANDLE;
    result =
        vkCreateBuffer(state_.device, &buffer_info, nullptr, &buffer);
    if (result != VK_SUCCESS) {
      vkFreeMemory(state_.device, memory, nullptr);
      LogError(std::string("vkCreateBuffer failed (") +
               std::to_string(static_cast<int>(result)) + ")");
      return std::unexpected(FromVkResult(result));
    }
    result = vkBindBufferMemory(state_.device, buffer, memory, 0);
    if (result != VK_SUCCESS) {
      vkDestroyBuffer(state_.device, buffer, nullptr);
      vkFreeMemory(state_.device, memory, nullptr);
      LogError(std::string("vkBindBufferMemory failed (") +
               std::to_string(static_cast<int>(result)) + "); the buffer has "
               "no backing memory");
      return std::unexpected(FromVkResult(result));
    }
    void* mapped = nullptr;
    if (kind == MemoryKind::HostVisible) {
      result = vkMapMemory(state_.device, memory, 0, VK_WHOLE_SIZE, 0, &mapped);
      if (result != VK_SUCCESS) {
        vkDestroyBuffer(state_.device, buffer, nullptr);
        vkFreeMemory(state_.device, memory, nullptr);
        LogError(std::string("vkMapMemory failed (") +
                 std::to_string(static_cast<int>(result)) + ")");
        return std::unexpected(FromVkResult(result));
      }
    }
    const std::size_t id = next_id_++;
    memories_.emplace(id, MemoryRecord{memory, buffer, mapped});
    return AdoptBuffer(ToHandle(id), bytes, kind, mapped);
  }

  void FreeBuffer(MemoryKind kind, void* handle) override {
    (void)kind;
    const auto it = memories_.find(FromHandle(handle));
    if (it == memories_.end()) {
      LogError("FreeBuffer: unknown handle; the buffer was already freed");
      return;
    }
    ReleaseRecord(it->second);
    memories_.erase(it);
  }

  std::expected<void, StatusCode> CopyH2D(Buffer& dst,
                                         std::span<const std::byte> src) override {
    if (src.size() > dst.Size()) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    auto staging = AllocateStaging(src.size());
    if (!staging) {
      return std::unexpected(staging.error());
    }
    std::memcpy(staging->mapped, src.data(), src.size());
    auto record = LookupRecord(dst.Handle());
    if (!record) {
      return std::unexpected(StatusCode::DeviceError);
    }
    auto copy = SubmitCopy(*staging, *record, src.size());
    ReleaseStaging(*staging);
    return copy;
  }

  std::expected<void, StatusCode> CopyD2H(const Buffer& src, std::byte* dst,
                                         std::size_t bytes) override {
    if (bytes > src.Size()) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    auto staging = AllocateStaging(bytes);
    if (!staging) {
      return std::unexpected(staging.error());
    }
    auto record = LookupRecord(src.Handle());
    if (!record) {
      return std::unexpected(StatusCode::DeviceError);
    }
    auto copy = SubmitCopy(*record, *staging, bytes);
    if (copy) {
      std::memcpy(dst, staging->mapped, bytes);
    }
    ReleaseStaging(*staging);
    return copy;
  }

  std::expected<std::unique_ptr<Kernel>, StatusCode> LoadKernel(
      std::string_view name, std::span<const std::byte> code) override {
    auto kernel = compute_.LoadKernel(name, code);
    if (!kernel) {
      return std::unexpected(kernel.error());
    }
    // Detach: the Kernel handle owns the VkKernel until FreeKernel.
    return AdoptKernel((*kernel).release(), name);
  }

  void FreeKernel(void* handle) override {
    compute_.FreeKernel(static_cast<VkKernel*>(handle));
  }

  std::expected<void, StatusCode> LaunchKernel(const Kernel& kernel,
                                              const KernelLaunch& launch) override {
    auto invalid = ValidateLaunch(launch);
    if (invalid != StatusCode::Ok) {
      return std::unexpected(invalid);
    }
    auto missing = CheckBuiltInArgs(kernel, launch);
    if (missing != StatusCode::Ok) {
      return std::unexpected(missing);
    }
    std::vector<VkBinding> bindings;
    for (const auto* buffer : launch.buffers) {
      if (buffer == nullptr) {
        return std::unexpected(StatusCode::InvalidArgument);
      }
      const auto* record = LookupRecord(buffer->Handle());
      if (record == nullptr) {
        LogError("kernel launch: buffer handle no longer exists; the buffer "
                 "was freed before the launch");
        return std::unexpected(StatusCode::DeviceError);
      }
      bindings.push_back(VkBinding{record->buffer, buffer->Size()});
    }
    return compute_.LaunchKernel(
        *static_cast<const VkKernel*>(kernel.Handle()), launch,
        std::span<const VkBinding>(bindings));
  }

  void Synchronize() override {
    auto result = vkQueueWaitIdle(state_.queue);
    if (result != VK_SUCCESS) {
      LogError(std::string("vkQueueWaitIdle failed (") +
               std::to_string(static_cast<int>(result)) + "); the device did "
               "not reach a quiescent state");
    }
  }

 private:
  // Opaque handle: the record id + 1, so null never collides.
  static void* ToHandle(std::size_t id) {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(id) + 1);
  }
  static std::size_t FromHandle(void* handle) {
    return static_cast<std::size_t>(
               reinterpret_cast<std::uintptr_t>(handle)) -
           1;
  }

  std::optional<std::uint32_t> PickMemoryIndex(MemoryKind kind) const {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(state_.physical, &properties);
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      auto flags = properties.memoryTypes[i].propertyFlags;
      if (kind == MemoryKind::Device) {
        if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
          return i;
        }
      } else if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 &&
                 (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
        return i;
      }
    }
    return std::nullopt;
  }

  const MemoryRecord* LookupRecord(void* handle) const {
    const auto it = memories_.find(FromHandle(handle));
    return it == memories_.end() ? nullptr : &it->second;
  }

  void ReleaseRecord(MemoryRecord& record) {
    if (record.mapped != nullptr) {
      vkUnmapMemory(state_.device, record.memory);
    }
    if (record.buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(state_.device, record.buffer, nullptr);
    }
    if (record.memory != VK_NULL_HANDLE) {
      vkFreeMemory(state_.device, record.memory, nullptr);
    }
  }

  void FreeAllMemory() {
    for (auto& [id, record] : memories_) {
      (void)id;
      ReleaseRecord(record);
    }
    memories_.clear();
  }

  struct Staging {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    void* mapped = nullptr;
  };

  std::expected<Staging, StatusCode> AllocateStaging(std::size_t bytes) {
    Staging staging;
    auto type_index = PickMemoryIndex(MemoryKind::HostVisible);
    if (!type_index) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = bytes;
    alloc_info.memoryTypeIndex = *type_index;
    auto result =
        vkAllocateMemory(state_.device, &alloc_info, nullptr, &staging.memory);
    if (result != VK_SUCCESS) {
      return std::unexpected(FromVkResult(result));
    }
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = bytes;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    // Default sharing mode is exclusive; no queue family list needed.
    result = vkCreateBuffer(state_.device, &buffer_info, nullptr,
                           &staging.buffer);
    if (result != VK_SUCCESS) {
      vkFreeMemory(state_.device, staging.memory, nullptr);
      return std::unexpected(FromVkResult(result));
    }
    result = vkBindBufferMemory(state_.device, staging.buffer, staging.memory,
                               0);
    if (result != VK_SUCCESS) {
      vkDestroyBuffer(state_.device, staging.buffer, nullptr);
      vkFreeMemory(state_.device, staging.memory, nullptr);
      return std::unexpected(FromVkResult(result));
    }
    result = vkMapMemory(state_.device, staging.memory, 0, VK_WHOLE_SIZE, 0,
                        &staging.mapped);
    if (result != VK_SUCCESS) {
      vkDestroyBuffer(state_.device, staging.buffer, nullptr);
      vkFreeMemory(state_.device, staging.memory, nullptr);
      return std::unexpected(FromVkResult(result));
    }
    return staging;
  }

  void ReleaseStaging(Staging& staging) {
    if (staging.mapped != nullptr) {
      vkUnmapMemory(state_.device, staging.memory);
    }
    if (staging.buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(state_.device, staging.buffer, nullptr);
    }
    if (staging.memory != VK_NULL_HANDLE) {
      vkFreeMemory(state_.device, staging.memory, nullptr);
    }
  }

  // One transfer: command buffer + fence, then wait for completion.
  std::expected<void, StatusCode> SubmitCopy(
      const VkBuffer source, const VkBuffer destination, std::size_t bytes) {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = state_.pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    auto result =
        vkAllocateCommandBuffers(state_.device, &alloc_info, &command);
    if (result != VK_SUCCESS) {
      return std::unexpected(FromVkResult(result));
    }
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    result = vkBeginCommandBuffer(command, &begin_info);
    if (result != VK_SUCCESS) {
      vkFreeCommandBuffers(state_.device, state_.pool, 1, &command);
      return std::unexpected(FromVkResult(result));
    }
    // The 1.4+ C API takes a region table, not offsets.
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = bytes;
    vkCmdCopyBuffer(command, source, destination, 1, &region);
    result = vkEndCommandBuffer(command);
    if (result != VK_SUCCESS) {
      vkFreeCommandBuffers(state_.device, state_.pool, 1, &command);
      return std::unexpected(FromVkResult(result));
    }
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    result = vkCreateFence(state_.device, &fence_info, nullptr, &fence);
    if (result != VK_SUCCESS) {
      vkFreeCommandBuffers(state_.device, state_.pool, 1, &command);
      return std::unexpected(FromVkResult(result));
    }
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command;
    result = vkQueueSubmit(state_.queue, 1, &submit_info, fence);
    if (result == VK_SUCCESS) {
      result = vkWaitForFences(state_.device, 1, &fence, VK_TRUE,
                              kFenceTimeoutNs);
    }
    if (fence != VK_NULL_HANDLE) {
      vkDestroyFence(state_.device, fence, nullptr);
    }
    vkFreeCommandBuffers(state_.device, state_.pool, 1, &command);
    if (result == VK_TIMEOUT) {
      LogError("fence wait timed out after 30 s; the device is hung or the "
               "ICD is unresponsive");
      return std::unexpected(StatusCode::DeviceError);
    }
    if (result != VK_SUCCESS) {
      LogError(std::string("device transfer failed (") +
               std::to_string(static_cast<int>(result)) + ")");
      return std::unexpected(FromVkResult(result));
    }
    return {};
  }

  // Overloads so the two record types can be passed uniformly.
  std::expected<void, StatusCode> SubmitCopy(const Staging& source,
                                            const MemoryRecord& destination,
                                            std::size_t bytes) {
    return SubmitCopy(source.buffer, destination.buffer, bytes);
  }
  std::expected<void, StatusCode> SubmitCopy(const MemoryRecord& source,
                                            const Staging& destination,
                                            std::size_t bytes) {
    return SubmitCopy(source.buffer, destination.buffer, bytes);
  }

  VulkanState state_;
  std::unordered_map<std::size_t, MemoryRecord> memories_;
  VulkanCompute compute_;
  std::size_t next_id_ = 0;
  std::string device_name_;
  bool initialized_ = false;
};

}  // namespace tessera::backends::vulkan

namespace tessera {

std::unique_ptr<Backend> CreateBackend() {
  return std::make_unique<backends::vulkan::VulkanBackend>();
}

}  // namespace tessera
