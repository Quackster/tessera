#include "backends/vulkan/vulkan_compute.hpp"

// Generated at configure time from src/backends/vulkan/kernels/*.comp.
#include "kernels_spirv.hpp"

#include <array>
#include <cstring>
#include <vector>

namespace tessera::backends::vulkan {

namespace {

// Push constants: up to 16 x u64 scalars, zero padded (128 bytes, below
// the 256 byte spec minimum for push constant ranges).
constexpr std::size_t kPushConstantBytes = 128;
// SPIR-V module magic.
constexpr std::uint32_t kSpirvMagic = 0x07230203;

// The built-in kernels: name + compiled SPIR-V (compiled at configure
// time; the standard "main" entry, this radv build crashes compiling a
// non-main entry name).
struct BuiltInKernel {
  std::string_view name;
  const std::byte* code;
  std::size_t size;
};
const BuiltInKernel kBuiltInKernels[] = {
    {"fill", kFillSpirV, sizeof(kFillSpirV)},
    {"gemm_q4k", kGemmQ4kSpirV, sizeof(kGemmQ4kSpirV)},
    {"attention", kAttentionSpirV, sizeof(kAttentionSpirV)},
    {"rope", kRopeSpirV, sizeof(kRopeSpirV)},
    {"gemm_fp8", kGemmFp8SpirV, sizeof(kGemmFp8SpirV)},
    {"gemm_fp8_block", kGemmFp8BlockSpirV, sizeof(kGemmFp8BlockSpirV)},
    {"gemm_f32", kGemmF32SpirV, sizeof(kGemmF32SpirV)},
    {"gemm_f32_batched", kGemmF32BatchedSpirV, sizeof(kGemmF32BatchedSpirV)},
    {"gemm_bf16", kGemmBf16SpirV, sizeof(kGemmBf16SpirV)},
    {"gemm_bf16_batched", kGemmBf16BatchedSpirV,
     sizeof(kGemmBf16BatchedSpirV)},
    {"gemm_mxfp4", kGemmMxfp4SpirV, sizeof(kGemmMxfp4SpirV)},
    {"gemm_q5k", kGemmQ5KSpirV, sizeof(kGemmQ5KSpirV)},
    {"gemm_q6k", kGemmQ6KSpirV, sizeof(kGemmQ6KSpirV)},
    {"gemm_q3k", kGemmQ3KSpirV, sizeof(kGemmQ3KSpirV)},
    {"gemm_iq4nl", kGemmIq4NlSpirV, sizeof(kGemmIq4NlSpirV)},
    {"gemm_iq4xs", kGemmIq4XsSpirV, sizeof(kGemmIq4XsSpirV)},
    {"gemm_iq3s", kGemmIq3SSpirV, sizeof(kGemmIq3SSpirV)},
    {"gemm_q80", kGemmQ80SpirV, sizeof(kGemmQ80SpirV)},
    {"rmsnorm", kRmsnormSpirV, sizeof(kRmsnormSpirV)},
    {"sigmoid_gate", kSigmoidGateSpirV, sizeof(kSigmoidGateSpirV)},
    {"l2norm", kL2NormSpirV, sizeof(kL2NormSpirV)},
    {"rmsnorm_gated", kRmsnormGatedSpirV, sizeof(kRmsnormGatedSpirV)},
    {"conv1d", kConv1dSpirV, sizeof(kConv1dSpirV)},
    {"conv1d_step", kConv1dStepSpirV, sizeof(kConv1dStepSpirV)},
    {"conv1d_state", kConv1dStateSpirV, sizeof(kConv1dStateSpirV)},
    {"dflash_conv", kDflashConvSpirV, sizeof(kDflashConvSpirV)},
    {"selector_edge_score", kSelectorEdgeScoreSpirV, sizeof(kSelectorEdgeScoreSpirV)},
    {"top_k_rows", kTopKRowsSpirV, sizeof(kTopKRowsSpirV)},
    {"concat_features", kConcatFeaturesSpirV, sizeof(kConcatFeaturesSpirV)},
    {"cast_f32_f16", kCastF32F16SpirV, sizeof(kCastF32F16SpirV)},
    {"round_bf16", kRoundBf16SpirV, sizeof(kRoundBf16SpirV)},
    {"quantize_q8", kQuantizeQ8SpirV, sizeof(kQuantizeQ8SpirV)},
    {"quantize_fp8", kQuantizeFp8SpirV, sizeof(kQuantizeFp8SpirV)},
    {"quantize_fp8_pack", kQuantizeFp8PackSpirV, sizeof(kQuantizeFp8PackSpirV)},
    {"attention_fp8", kAttentionQuantSpirV, sizeof(kAttentionQuantSpirV)},
    {"attention_q8", kAttentionQuantSpirV, sizeof(kAttentionQuantSpirV)},
    {"quantize_q4", kQuantizeQ4SpirV, sizeof(kQuantizeQ4SpirV)},
    {"attention_q4", kAttentionQuantSpirV, sizeof(kAttentionQuantSpirV)},
    {"attention_split", kAttentionSplitSpirV, sizeof(kAttentionSplitSpirV)},
    {"attention_q8_split",
     kAttentionQuantSplitSpirV, sizeof(kAttentionQuantSplitSpirV)},
    {"attention_fp8_split",
     kAttentionQuantSplitSpirV, sizeof(kAttentionQuantSplitSpirV)},
    {"attention_q4_split",
     kAttentionQuantSplitSpirV, sizeof(kAttentionQuantSplitSpirV)},
    {"attention_combine",
     kAttentionCombineSpirV, sizeof(kAttentionCombineSpirV)},
    {"layernorm", kLayernormSpirV, sizeof(kLayernormSpirV)},
    {"gelu", kGeluSpirV, sizeof(kGeluSpirV)},
    {"bias_add", kBiasAddSpirV, sizeof(kBiasAddSpirV)},
    {"image_patchify", kImagePatchifySpirV, sizeof(kImagePatchifySpirV)},
    {"spatial_merge", kSpatialMergeSpirV, sizeof(kSpatialMergeSpirV)},
    {"delta_step", kDeltaStepSpirV, sizeof(kDeltaStepSpirV)},
    {"mrope", kMropeSpirV, sizeof(kMropeSpirV)},
    {"qgate_split", kQgateSplitSpirV, sizeof(kQgateSplitSpirV)},
    {"add", kAddSpirV, sizeof(kAddSpirV)},
    {"repeat_heads", kRepeatHeadsSpirV, sizeof(kRepeatHeadsSpirV)},
    {"ssm_gate", kSsmGateSpirV, sizeof(kSsmGateSpirV)},
    {"delta_step_heads", kDeltaStepHeadsSpirV, sizeof(kDeltaStepHeadsSpirV)},
    {"silu_mul", kSiluMulSpirV, sizeof(kSiluMulSpirV)},
    {"embedding_f32", kEmbeddingF32SpirV, sizeof(kEmbeddingF32SpirV)},
    {"embedding_bf16", kEmbeddingBf16SpirV, sizeof(kEmbeddingBf16SpirV)},
    {"embedding_q4k", kEmbeddingQ4kSpirV, sizeof(kEmbeddingQ4kSpirV)},
    {"gemm_q4k_batched", kGemmQ4kBatchedSpirV, sizeof(kGemmQ4kBatchedSpirV)},
    {"gemm_mxfp4_batched", kGemmMxfp4BatchedSpirV,
     sizeof(kGemmMxfp4BatchedSpirV)},
    {"gemm_mxfp4_rows", kGemmMxfp4RowsSpirV, sizeof(kGemmMxfp4RowsSpirV)},
    {"gemm_q4k_row", kGemmQ4kRowSpirV, sizeof(kGemmQ4kRowSpirV)},
    {"gemm_q5k_batched", kGemmQ5kBatchedSpirV, sizeof(kGemmQ5kBatchedSpirV)},
    {"gemm_q6k_batched", kGemmQ6kBatchedSpirV, sizeof(kGemmQ6kBatchedSpirV)},
    {"gemm_iq4xs_batched", kGemmIq4XsBatchedSpirV,
     sizeof(kGemmIq4XsBatchedSpirV)},
    {"gemm_q4k_vec", kGemmQ4kVecSpirV, sizeof(kGemmQ4kVecSpirV)},
    {"gemm_q5k_vec", kGemmQ5kVecSpirV, sizeof(kGemmQ5kVecSpirV)},
    {"gemm_q6k_vec", kGemmQ6kVecSpirV, sizeof(kGemmQ6kVecSpirV)},
    {"gemm_iq4xs_vec", kGemmIq4XsVecSpirV, sizeof(kGemmIq4XsVecSpirV)},
    {"gemm_vec_reduce", kGemmVecReduceSpirV, sizeof(kGemmVecReduceSpirV)},
    {"gemm_q4k_rows2", kGemmQ4kRows2SpirV, sizeof(kGemmQ4kRows2SpirV)},
    {"gemm_q4k_rows3", kGemmQ4kRows3SpirV, sizeof(kGemmQ4kRows3SpirV)},
    {"gemm_q4k_rows4", kGemmQ4kRows4SpirV, sizeof(kGemmQ4kRows4SpirV)},
    {"gemm_q5k_rows2", kGemmQ5kRows2SpirV, sizeof(kGemmQ5kRows2SpirV)},
    {"gemm_q5k_rows3", kGemmQ5kRows3SpirV, sizeof(kGemmQ5kRows3SpirV)},
    {"gemm_q5k_rows4", kGemmQ5kRows4SpirV, sizeof(kGemmQ5kRows4SpirV)},
    {"gemm_q6k_rows2", kGemmQ6kRows2SpirV, sizeof(kGemmQ6kRows2SpirV)},
    {"gemm_q6k_rows3", kGemmQ6kRows3SpirV, sizeof(kGemmQ6kRows3SpirV)},
    {"gemm_q6k_rows4", kGemmQ6kRows4SpirV, sizeof(kGemmQ6kRows4SpirV)},
    {"gemm_iq4xs_rows2", kGemmIq4XsRows2SpirV, sizeof(kGemmIq4XsRows2SpirV)},
    {"gemm_iq4xs_rows3", kGemmIq4XsRows3SpirV, sizeof(kGemmIq4XsRows3SpirV)},
    {"gemm_iq4xs_rows4", kGemmIq4XsRows4SpirV, sizeof(kGemmIq4XsRows4SpirV)},
};

int LookupBuiltIn(std::string_view name) {
  for (std::size_t i = 0; i < sizeof(kBuiltInKernels) /
                               sizeof(kBuiltInKernels[0]); ++i) {
    if (kBuiltInKernels[i].name == name) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

}  // namespace

void VulkanCompute::LogError(std::string_view message) const {
  if (diagnostics_ != nullptr) {
    diagnostics_->Error("kernel", message);
  }
}

std::expected<void, StatusCode> VulkanCompute::Init(VkDevice device,
                                                   VkQueue queue,
                                                   log::Diagnostics* diagnostics) {
  if (ready_) {
    return {};
  }
  device_ = device;
  queue_ = queue;
  diagnostics_ = diagnostics;
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  auto result = vkCreateCommandPool(device, &pool_info, nullptr, &pool_);
  if (result != VK_SUCCESS) {
    LogError(std::string("vkCreateCommandPool failed (") +
             std::to_string(static_cast<int>(result)) + ")");
    return std::unexpected(FromVkResult(result));
  }
  // One binding per buffer slot, so kernel i's buffer lands on binding i:
  // GLSL shaders declare each buffer block as its own binding.
  std::array<VkDescriptorSetLayoutBinding, kMaxBoundBuffers> layout_bindings{};
  for (std::size_t i = 0; i < kMaxBoundBuffers; ++i) {
    layout_bindings[i].binding = static_cast<std::uint32_t>(i);
    layout_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layout_bindings[i].descriptorCount = 1;
    layout_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo set_layout_info{};
  set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  set_layout_info.bindingCount = kMaxBoundBuffers;
  set_layout_info.pBindings = layout_bindings.data();
  result = vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr,
                                      &set_layout_);
  if (result != VK_SUCCESS) {
    LogError(std::string("vkCreateDescriptorSetLayout failed (") +
             std::to_string(static_cast<int>(result)) + ")");
    DestroyResources();
    return std::unexpected(FromVkResult(result));
  }
  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = kPushConstantBytes;
  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &set_layout_;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push_range;
  result = vkCreatePipelineLayout(device, &layout_info, nullptr, &layout_);
  if (result != VK_SUCCESS) {
    LogError(std::string("vkCreatePipelineLayout failed (") +
             std::to_string(static_cast<int>(result)) + ")");
    DestroyResources();
    return std::unexpected(FromVkResult(result));
  }
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = kMaxBoundBuffers * kDescriptorSetBudget;
  VkDescriptorPoolCreateInfo pool_create{};
  pool_create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  // Sets are released in bulk by vkResetDescriptorPool after each flush,
  // so individual free is not needed.
  pool_create.maxSets = kDescriptorSetBudget;
  pool_create.poolSizeCount = 1;
  pool_create.pPoolSizes = &pool_size;
  result = vkCreateDescriptorPool(device, &pool_create, nullptr,
                                 &descriptor_pool_);
  if (result != VK_SUCCESS) {
    LogError(std::string("vkCreateDescriptorPool failed (") +
             std::to_string(static_cast<int>(result)) + ")");
    DestroyResources();
    return std::unexpected(FromVkResult(result));
  }
  VkCommandBufferAllocateInfo command_info{};
  command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_info.commandPool = pool_;
  command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_info.commandBufferCount = 1;
  result = vkAllocateCommandBuffers(device, &command_info, &cmd_);
  if (result != VK_SUCCESS) {
    LogError(std::string("vkAllocateCommandBuffers failed (") +
             std::to_string(static_cast<int>(result)) + ")");
    DestroyResources();
    return std::unexpected(FromVkResult(result));
  }
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  // Signaled so the first flush's wait does not block on a fresh fence.
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  result = vkCreateFence(device, &fence_info, nullptr, &fence_);
  if (result != VK_SUCCESS) {
    LogError(std::string("vkCreateFence failed (") +
             std::to_string(static_cast<int>(result)) + ")");
    DestroyResources();
    return std::unexpected(FromVkResult(result));
  }
  ready_ = true;
  return {};
}

void VulkanCompute::DestroyResources() {
  if (fence_ != VK_NULL_HANDLE) {
    vkDestroyFence(device_, fence_, nullptr);
    fence_ = VK_NULL_HANDLE;
  }
  cmd_ = VK_NULL_HANDLE;  // freed with the command pool
  recording_ = false;
  if (descriptor_pool_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    descriptor_pool_ = VK_NULL_HANDLE;
  }
  if (layout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device_, layout_, nullptr);
    layout_ = VK_NULL_HANDLE;
  }
  if (set_layout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    set_layout_ = VK_NULL_HANDLE;
  }
  if (pool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_, pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
  }
}

void VulkanCompute::Shutdown() {
  if (!ready_) {
    return;
  }
  auto flushed = Synchronize();
  (void)flushed;
  vkQueueWaitIdle(queue_);
  DestroyResources();
  ready_ = false;
}

std::expected<std::unique_ptr<VkKernel>, StatusCode>
VulkanCompute::LoadKernel(std::string_view name,
                          std::span<const std::byte> code) {
  if (!ready_) {
    return std::unexpected(StatusCode::DeviceError);
  }
  std::string entry(name);
  if (code.empty()) {
    const int index = LookupBuiltIn(name);
    if (index < 0) {
      LogError(std::string("kernel '") + std::string(name) +
                "' is not a built-in on vulkan; pass the kernel as a "
                "SPIR-V module or add it to the vulkan kernel set");
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    entry = "main";
    code = std::span(kBuiltInKernels[index].code,
                     kBuiltInKernels[index].size);
  } else if (code.size() < 4 ||
             *reinterpret_cast<const std::uint32_t*>(code.data()) !=
                 kSpirvMagic) {
    LogError(std::string("kernel '") + std::string(name) +
             "': bad SPIR-V magic; the code is not a SPIR-V module");
    return std::unexpected(StatusCode::MalformedFile);
  }
  VkShaderModuleCreateInfo module_info{};
  module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  module_info.codeSize = code.size();
  module_info.pCode = reinterpret_cast<const std::uint32_t*>(code.data());
  VkShaderModule module = VK_NULL_HANDLE;
  auto result = vkCreateShaderModule(device_, &module_info, nullptr, &module);
  if (result != VK_SUCCESS) {
    LogError(std::string("kernel '") + std::string(name) +
             "': vkCreateShaderModule failed (" +
             std::to_string(static_cast<int>(result)) +
             "); the driver rejected the SPIR-V module");
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto kernel = std::make_unique<VkKernel>();
  kernel->name = name;
  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = module;
  stage.pName = entry.data();
  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.layout = layout_;
  pipeline_info.stage = stage;
  result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info,
                                   nullptr, &kernel->pipeline);
  vkDestroyShaderModule(device_, module, nullptr);
  if (result != VK_SUCCESS) {
    LogError(std::string("kernel '") + std::string(name) +
             "': vkCreateComputePipelines failed (" +
             std::to_string(static_cast<int>(result)) +
             "); check the entry point name against the SPIR-V module");
    return std::unexpected(StatusCode::MalformedFile);
  }
  return kernel;
}

void VulkanCompute::FreeKernel(VkKernel* kernel) {
  if (kernel == nullptr) {
    return;
  }
  if (kernel->pipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(device_, kernel->pipeline, nullptr);
  }
  delete kernel;
}

std::expected<void, StatusCode> VulkanCompute::EnsureRecording() {
  if (recording_) {
    return {};
  }
  auto result = vkResetCommandBuffer(cmd_, 0);
  if (result != VK_SUCCESS) {
    LogError(std::string("vkResetCommandBuffer failed (") +
             std::to_string(static_cast<int>(result)) + ")");
    return std::unexpected(FromVkResult(result));
  }
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  result = vkBeginCommandBuffer(cmd_, &begin_info);
  if (result != VK_SUCCESS) {
    LogError(std::string("vkBeginCommandBuffer failed (") +
             std::to_string(static_cast<int>(result)) + ")");
    return std::unexpected(FromVkResult(result));
  }
  recording_ = true;
  return {};
}

std::expected<void, StatusCode> VulkanCompute::FlushAndWait() {
  if (!recording_) {
    return {};
  }
  auto result = vkEndCommandBuffer(cmd_);
  if (result != VK_SUCCESS) {
    LogError(std::string("vkEndCommandBuffer failed (") +
             std::to_string(static_cast<int>(result)) + ")");
    recording_ = false;
    return std::unexpected(FromVkResult(result));
  }
  result = vkResetFences(device_, 1, &fence_);
  if (result != VK_SUCCESS) {
    LogError(std::string("vkResetFences failed (") +
             std::to_string(static_cast<int>(result)) + ")");
    recording_ = false;
    return std::unexpected(FromVkResult(result));
  }
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmd_;
  result = vkQueueSubmit(queue_, 1, &submit_info, fence_);
  if (result != VK_SUCCESS) {
    LogError(std::string("vkQueueSubmit failed (") +
             std::to_string(static_cast<int>(result)) + ")");
    recording_ = false;
    return std::unexpected(FromVkResult(result));
  }
  result = vkWaitForFences(device_, 1, &fence_, VK_TRUE, kFenceTimeoutNs);
  recording_ = false;
  if (result == VK_TIMEOUT) {
    LogError("fence wait timed out after 30 s; the device is hung or the "
             "ICD is unresponsive");
    return std::unexpected(StatusCode::DeviceError);
  }
  if (result != VK_SUCCESS) {
    LogError(std::string("vkWaitForFences failed (") +
             std::to_string(static_cast<int>(result)) + ")");
    return std::unexpected(FromVkResult(result));
  }
  // The submission is done, so every descriptor set it used can be released.
  result = vkResetDescriptorPool(device_, descriptor_pool_, 0);
  if (result != VK_SUCCESS) {
    LogError(std::string("vkResetDescriptorPool failed (") +
             std::to_string(static_cast<int>(result)) + ")");
    return std::unexpected(FromVkResult(result));
  }
  return {};
}

std::expected<void, StatusCode> VulkanCompute::Synchronize() {
  return FlushAndWait();
}

std::expected<void, StatusCode> VulkanCompute::RecordCopy(
    VkBuffer src, VkBuffer dst, std::size_t src_offset, std::size_t dst_offset,
    std::size_t bytes) {
  if (!ready_) {
    return std::unexpected(StatusCode::DeviceError);
  }
  auto recording = EnsureRecording();
  if (!recording) {
    return recording;
  }
  // Order this copy after every earlier recorded operation.
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0,
                       nullptr, 0, nullptr);
  VkBufferCopy copy{};
  copy.srcOffset = src_offset;
  copy.dstOffset = dst_offset;
  copy.size = bytes;
  vkCmdCopyBuffer(cmd_, src, dst, 1, &copy);
  return {};
}

std::expected<void, StatusCode> VulkanCompute::LaunchKernel(
    const VkKernel& kernel, const KernelLaunch& launch,
    std::span<const VkBinding> bindings) {
  if (!ready_) {
    return std::unexpected(StatusCode::DeviceError);
  }
  auto recording = EnsureRecording();
  if (!recording) {
    return recording;
  }
  // Order this dispatch after every earlier recorded operation. The
  // deferred command buffer has no implicit inter-dispatch synchronization,
  // so a conservative full barrier keeps every producer/consumer pair
  // ordered.
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0,
                       nullptr, 0, nullptr);
  vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline);
  VkDescriptorSetAllocateInfo set_alloc{};
  set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  set_alloc.descriptorPool = descriptor_pool_;
  set_alloc.descriptorSetCount = 1;
  set_alloc.pSetLayouts = &set_layout_;
  VkDescriptorSet set = VK_NULL_HANDLE;
  auto result = vkAllocateDescriptorSets(device_, &set_alloc, &set);
  if (result == VK_ERROR_OUT_OF_POOL_MEMORY) {
    // The window is larger than the pool: flush what is recorded, then
    // start a new window and retry.
    auto flushed = FlushAndWait();
    if (!flushed) {
      return flushed;
    }
    recording = EnsureRecording();
    if (!recording) {
      return recording;
    }
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline);
    result = vkAllocateDescriptorSets(device_, &set_alloc, &set);
  }
  if (result != VK_SUCCESS) {
    LogError(std::string("kernel ") + kernel.name +
             ": vkAllocateDescriptorSets failed (" +
             std::to_string(static_cast<int>(result)) + ")");
    return std::unexpected(FromVkResult(result));
  }
  // One write per bound buffer: buffer i goes to binding i.
  std::vector<VkDescriptorBufferInfo> buffer_infos;
  buffer_infos.reserve(kMaxBoundBuffers);
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    buffer_infos.push_back(
        VkDescriptorBufferInfo{bindings[i].buffer, 0, bindings[i].size});
  }
  std::vector<VkWriteDescriptorSet> writes;
  writes.reserve(kMaxBoundBuffers);
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = static_cast<std::uint32_t>(i);
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_infos[i];
    writes.push_back(write);
  }
  if (!writes.empty()) {
    vkUpdateDescriptorSets(device_,
                           static_cast<std::uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
  }
  std::uint8_t push_constants[kPushConstantBytes];
  std::memset(push_constants, 0, kPushConstantBytes);
  std::memcpy(push_constants, launch.scalars.data(),
              launch.scalars.size() * sizeof(std::uint64_t));
  vkCmdPushConstants(cmd_, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     kPushConstantBytes, push_constants);
  vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1,
                          &set, 0, nullptr);
  vkCmdDispatch(cmd_, launch.grid_x, launch.grid_y, launch.grid_z);
  return {};
}

}  // namespace tessera::backends::vulkan
