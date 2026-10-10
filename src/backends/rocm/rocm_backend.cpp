#include "tessera/backend.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "backends/rocm/rocm_kernels.hpp"
#include "core/timing.hpp"

namespace tessera::backends::rocm {

namespace {

// Host-visible staging window for a batched upload, the same bound the
// vulkan backend uses. A large checkpoint is copied through one reusable
// pinned buffer of this size, so pinned host memory stays bounded
// regardless of the total weight bytes.
constexpr std::size_t kUploadStagingBytes = 256ull << 20;

StatusCode FromHip(hipError_t error) {
  switch (error) {
    case hipSuccess: return StatusCode::Ok;
    case hipErrorOutOfMemory: return StatusCode::OutOfMemory;
    case hipErrorInvalidValue: return StatusCode::InvalidArgument;
    default: return StatusCode::DeviceError;
  }
}

const char* HipErrorName(hipError_t error) {
  return hipGetErrorName(error);
}

// The kernels compiled into the rocm backend.
struct BuiltInKernel {
  std::string_view name;
  void* function;
};
const BuiltInKernel kBuiltInKernels[] = {
    {"fill", reinterpret_cast<void*>(&FillKernel)},
    {"gemm_q4k", reinterpret_cast<void*>(&GemmQ4KKernel)},
    {"gemm_q4k_row", reinterpret_cast<void*>(&GemmQ4KRowKernel)},
    {"gemm_q4k_vec", reinterpret_cast<void*>(&GemmQ4KVecKernel)},
    {"gemm_q4k_rows2", reinterpret_cast<void*>(&GemmQ4KRowsKernel<2>)},
    {"gemm_q4k_rows3", reinterpret_cast<void*>(&GemmQ4KRowsKernel<3>)},
    {"gemm_q4k_rows4", reinterpret_cast<void*>(&GemmQ4KRowsKernel<4>)},
    {"gemm_q4k_batched", reinterpret_cast<void*>(&GemmQ4KBatchedKernel)},
    {"attention", reinterpret_cast<void*>(&AttentionKernel)},
    {"rope", reinterpret_cast<void*>(&RopeKernel)},
    {"gemm_fp8", reinterpret_cast<void*>(&GemmFp8Kernel)},
    {"gemm_fp8_block", reinterpret_cast<void*>(&GemmFp8BlockKernel)},
    {"gemm_f32", reinterpret_cast<void*>(&GemmF32Kernel)},
    {"gemm_f32_batched", reinterpret_cast<void*>(&GemmF32BatchedKernel)},
    {"gemm_bf16", reinterpret_cast<void*>(&GemmBf16Kernel)},
    {"gemm_bf16_batched", reinterpret_cast<void*>(&GemmBf16BatchedKernel)},
    {"gemm_mxfp4", reinterpret_cast<void*>(&GemmMxFp4Kernel)},
    {"gemm_mxfp4_batched", reinterpret_cast<void*>(&GemmMxFp4BatchedKernel)},
    {"gemm_mxfp4_rows", reinterpret_cast<void*>(&GemmMxFp4RowsKernel)},
#if defined(TESSERA_ROCM_WMMA)
    {"gemm_mxfp4_wmma", reinterpret_cast<void*>(&GemmMxFp4WmmaKernel)},
    {"gemm_mxfp4_wmma_reduce",
     reinterpret_cast<void*>(&GemmMxFp4WmmaReduceKernel)},
    {"gemm_bf16_wmma", reinterpret_cast<void*>(&GemmBf16WmmaKernel)},
    {"mxfp4_row_ref", reinterpret_cast<void*>(&MxFp4RowRefKernel)},
#endif
    {"gemm_q5k", reinterpret_cast<void*>(&GemmQ5KKernel)},
    {"gemm_q5k_vec", reinterpret_cast<void*>(&GemmQ5KVecKernel)},
    {"gemm_q5k_rows2", reinterpret_cast<void*>(&GemmQ5KRowsKernel<2>)},
    {"gemm_q5k_rows3", reinterpret_cast<void*>(&GemmQ5KRowsKernel<3>)},
    {"gemm_q5k_rows4", reinterpret_cast<void*>(&GemmQ5KRowsKernel<4>)},
    {"gemm_q5k_batched", reinterpret_cast<void*>(&GemmQ5KBatchedKernel)},
    {"gemm_q6k", reinterpret_cast<void*>(&GemmQ6KKernel)},
    {"gemm_q6k_vec", reinterpret_cast<void*>(&GemmQ6KVecKernel)},
    {"gemm_q6k_rows2", reinterpret_cast<void*>(&GemmQ6KRowsKernel<2>)},
    {"gemm_q6k_rows3", reinterpret_cast<void*>(&GemmQ6KRowsKernel<3>)},
    {"gemm_q6k_rows4", reinterpret_cast<void*>(&GemmQ6KRowsKernel<4>)},
    {"gemm_q6k_batched", reinterpret_cast<void*>(&GemmQ6KBatchedKernel)},
    {"gemm_q3k", reinterpret_cast<void*>(&GemmQ3KKernel)},
    {"gemm_iq4nl", reinterpret_cast<void*>(&GemmIq4NlKernel)},
    {"gemm_iq4xs", reinterpret_cast<void*>(&GemmIq4XsKernel)},
    {"gemm_iq4xs_vec", reinterpret_cast<void*>(&GemmIq4XsVecKernel)},
    {"gemm_iq4xs_rows2", reinterpret_cast<void*>(&GemmIq4XsRowsKernel<2>)},
    {"gemm_iq4xs_rows3", reinterpret_cast<void*>(&GemmIq4XsRowsKernel<3>)},
    {"gemm_iq4xs_rows4", reinterpret_cast<void*>(&GemmIq4XsRowsKernel<4>)},
    {"gemm_vec_reduce", reinterpret_cast<void*>(&GemmVecReduceKernel)},
    {"gemm_iq4xs_batched", reinterpret_cast<void*>(&GemmIq4XsBatchedKernel)},
    {"gemm_iq3s", reinterpret_cast<void*>(&GemmIq3SKernel)},
    {"gemm_q80", reinterpret_cast<void*>(&GemmQ80Kernel)},
    {"rmsnorm", reinterpret_cast<void*>(&RmsnormKernel)},
    {"sigmoid_gate", reinterpret_cast<void*>(&SigmoidGateKernel)},
    {"l2norm", reinterpret_cast<void*>(&L2NormKernel)},
    {"rmsnorm_gated", reinterpret_cast<void*>(&RmsnormGatedKernel)},
    {"conv1d", reinterpret_cast<void*>(&Conv1dKernel)},
    {"delta_step", reinterpret_cast<void*>(&DeltaStepKernel)},
    {"mrope", reinterpret_cast<void*>(&MropeKernel)},
    {"qgate_split", reinterpret_cast<void*>(&QGateSplitKernel)},
    {"add", reinterpret_cast<void*>(&AddKernel)},
    {"repeat_heads", reinterpret_cast<void*>(&RepeatHeadsKernel)},
    {"conv1d_step", reinterpret_cast<void*>(&Conv1dStepKernel)},
    {"conv1d_state", reinterpret_cast<void*>(&Conv1dStateKernel)},
    {"dflash_conv", reinterpret_cast<void*>(&DflashConvKernel)},
    {"selector_edge_score", reinterpret_cast<void*>(&SelectorEdgeScoreKernel)},
    {"top_k_rows", reinterpret_cast<void*>(&TopKRowsKernel)},
    {"concat_features", reinterpret_cast<void*>(&ConcatFeaturesKernel)},
    {"cast_f32_f16", reinterpret_cast<void*>(&CastF32F16Kernel)},
    {"round_bf16", reinterpret_cast<void*>(&RoundBf16Kernel)},
    {"quantize_q8", reinterpret_cast<void*>(&QuantizeQ8Kernel)},
    {"quantize_fp8", reinterpret_cast<void*>(&QuantizeFp8Kernel)},
    {"quantize_fp8_pack", reinterpret_cast<void*>(&QuantizeFp8PackRowsKernel)},
    {"quantize_fp8_pack_rows",
     reinterpret_cast<void*>(&QuantizeFp8PackRowsKernel)},
    {"attention_fp8", reinterpret_cast<void*>(&AttentionFp8Kernel)},
    {"attention_q8", reinterpret_cast<void*>(&AttentionQ8Kernel)},
    {"attention_split", reinterpret_cast<void*>(&AttentionSplitKernel)},
    {"attention_q8_split",
     reinterpret_cast<void*>(&AttentionQ8SplitKernel)},
    {"attention_fp8_split",
     reinterpret_cast<void*>(&AttentionFp8SplitKernel)},
    {"attention_q4_split",
     reinterpret_cast<void*>(&AttentionQ4SplitKernel)},
    {"attention_combine",
     reinterpret_cast<void*>(&AttentionCombineKernel)},
    {"quantize_q4", reinterpret_cast<void*>(&QuantizeQ4Kernel)},
    {"attention_q4", reinterpret_cast<void*>(&AttentionQ4Kernel)},
    {"layernorm", reinterpret_cast<void*>(&LayerNormKernel)},
    {"gelu", reinterpret_cast<void*>(&GeluKernel)},
    {"bias_add", reinterpret_cast<void*>(&BiasAddKernel)},
    {"spatial_merge", reinterpret_cast<void*>(&SpatialMergeKernel)},
    {"image_patchify", reinterpret_cast<void*>(&ImagePatchifyKernel)},
    {"ssm_gate", reinterpret_cast<void*>(&SsmGateKernel)},
    {"delta_step_heads", reinterpret_cast<void*>(&DeltaStepHeadsKernel)},
    {"silu_mul", reinterpret_cast<void*>(&SiluMulKernel)},
    {"moe_gate", reinterpret_cast<void*>(&MoeGateKernel)},
    {"moe_scale_add", reinterpret_cast<void*>(&MoeScaleAddKernel)},
    {"moe_experts_gate_up_q4k",
     reinterpret_cast<void*>(&MoeExpertsGateUpQ4KKernel)},
    {"moe_experts_down_q4k", reinterpret_cast<void*>(&MoeExpertsDownQ4KKernel)},
    {"moe_experts_down_q6k", reinterpret_cast<void*>(&MoeExpertsDownQ6KKernel)},
    {"embedding_f32", reinterpret_cast<void*>(&EmbeddingF32Kernel)},
    {"embedding_bf16", reinterpret_cast<void*>(&EmbeddingBf16Kernel)},
    {"embedding_q4k", reinterpret_cast<void*>(&EmbeddingQ4KKernel)},
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

class RocmBackend final : public Backend {
 public:
  ~RocmBackend() override {
    ReleaseStaging();
    if (stream_ != nullptr) {
      auto error = hipStreamDestroy(stream_);
      if (error != hipSuccess) {
        LogError(std::string("hipStreamDestroy failed (") +
                 HipErrorName(error) + "); the stream may leak");
      }
      stream_ = nullptr;
    }
  }

  std::string_view Name() const override {
    return "rocm";
  }

  std::string_view DeviceName() const override {
    return device_name_;
  }

  DeviceMemoryInfo MemoryInfo() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
      return {};
    }
    return DeviceMemoryInfo{total_bytes, free_bytes};
  }

  std::expected<void, StatusCode> Init() override {
    if (initialized_) {
      return {};
    }
    int device_count = 0;
    auto error = hipGetDeviceCount(&device_count);
    if (error != hipSuccess) {
      LogError(std::string("hipGetDeviceCount failed (") +
               HipErrorName(error) +
               "); the ROCm runtime is not usable on this host");
      return std::unexpected(FromHip(error));
    }
    if (device_count == 0) {
      LogError("hipGetDeviceCount found 0 devices; check rocm-smi and the "
               "KFD group");
      return std::unexpected(StatusCode::DeviceError);
    }
    if (device_index_ < 0 || device_index_ >= device_count) {
      LogError("requested GPU " + std::to_string(device_index_) + " but only " +
               std::to_string(device_count) + " device(s) are present; pass "
               "--gpu in the range [0, " +
               std::to_string(device_count - 1) +
               "] (defaults to the first GPU)");
      return std::unexpected(StatusCode::InvalidArgument);
    }
    error = hipSetDevice(device_index_);
    if (error != hipSuccess) {
      LogError(std::string("hipSetDevice(") + std::to_string(device_index_) +
               ") failed (" + HipErrorName(error) + "); cannot use that GPU");
      return std::unexpected(FromHip(error));
    }
    hipDeviceProp_t props;
    error = hipGetDeviceProperties(&props, device_index_);
    if (error != hipSuccess) {
      LogError(std::string("hipGetDeviceProperties failed (") +
               HipErrorName(error) + ")");
      return std::unexpected(FromHip(error));
    }
    device_name_ = props.name;
    LogInfo("selected GPU " + std::to_string(device_index_) + " of " +
            std::to_string(device_count) + ": " + device_name_);
    // A dedicated stream with the blocking synchronization policy. The
    // default policy (hipSyncPolicyAuto) actively waits in the runtime on
    // every host synchronization, so a decode step that waits for the GPU
    // burns CPU. hipSyncPolicyBlockingSync sleeps instead. All work is
    // enqueued on this one stream, which also keeps the copies ordered with
    // the kernels.
    auto created = hipStreamCreate(&stream_);
    if (created != hipSuccess) {
      LogError(std::string("hipStreamCreate failed (") + HipErrorName(created) +
               "); decode will synchronize on the default stream");
      stream_ = nullptr;
    } else {
      hipStreamAttrValue attr{};
      attr.syncPolicy = hipSyncPolicyBlockingSync;
      auto set = hipStreamSetAttribute(
          stream_, hipStreamAttributeSynchronizationPolicy, &attr);
      if (set != hipSuccess) {
        LogError(std::string("hipStreamSetAttribute(blocking sync) failed (") +
                 HipErrorName(set) +
                 "); host waits may busy-wait and use more CPU");
      }
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
    if (auto device = SelectDevice(); device != StatusCode::Ok) {
      return std::unexpected(device);
    }
    void* ptr = nullptr;
    auto error =
        kind == MemoryKind::Device
            ? hipMalloc(&ptr, bytes)
            : hipHostMalloc(&ptr, bytes, hipHostMallocDefault);
    if (error != hipSuccess) {
      LogError(std::string(kind == MemoryKind::Device ? "hipMalloc"
                                                     : "hipHostMalloc") +
               " failed for " + std::to_string(bytes) + " bytes (" +
               HipErrorName(error) + "); the device is out of that memory pool");
      return std::unexpected(FromHip(error));
    }
    return AdoptBuffer(ptr, bytes, kind,
                       kind == MemoryKind::HostVisible ? ptr : nullptr);
  }

  void FreeBuffer(MemoryKind kind, void* handle) override {
    (void)SelectDevice();
    auto error =
        kind == MemoryKind::Device ? hipFree(handle) : hipFreeHost(handle);
    if (error != hipSuccess) {
      LogError(std::string(kind == MemoryKind::Device ? "hipFree"
                                                     : "hipFreeHost") +
               " failed (" + HipErrorName(error) + "); the memory may leak");
    }
  }

  std::expected<void, StatusCode> CopyH2D(Buffer& dst,
                                         std::span<const std::byte> src) override {
    if (auto device = SelectDevice(); device != StatusCode::Ok) {
      return std::unexpected(device);
    }
    if (src.size() > dst.Size()) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    auto staging = EnsureStaging(src.size());
    if (!staging) {
      return std::unexpected(staging.error());
    }
    std::memcpy(*staging, src.data(), src.size());
    auto error = hipMemcpyAsync(dst.Handle(), *staging, src.size(),
                                hipMemcpyHostToDevice, stream_);
    if (error == hipSuccess) {
      error = hipStreamSynchronize(stream_);
    }
    if (error != hipSuccess) {
      LogError(std::string("hipMemcpy H2D of ") +
               std::to_string(src.size()) + " bytes failed (" +
               HipErrorName(error) + ")");
      return std::unexpected(FromHip(error));
    }
    return {};
  }

  std::expected<void, StatusCode> CopyH2DBatch(
      std::span<const Backend::HostCopy> copies) override {
    if (auto device = SelectDevice(); device != StatusCode::Ok) {
      return std::unexpected(device);
    }
    for (const Backend::HostCopy& copy : copies) {
      if (copy.dst == nullptr || copy.src.size() > copy.dst->Size()) {
        return std::unexpected(StatusCode::InvalidArgument);
      }
    }
    // Drain the batch through one bounded pinned staging window at a time,
    // then release it. Handing hipMemcpyAsync the pageable weight bytes (the
    // mapped model file) directly grows the runtime's pinned set to the whole
    // checkpoint; a 16 GB load stalled past 5 minutes. A bounded window keeps
    // the pinned set at one window and leaves the source pages reclaimable.
    const auto started = core::PhaseClock::now();
    std::size_t total_bytes = 0;
    std::size_t index = 0;
    while (index < copies.size()) {
      std::size_t staging_size = 0;
      std::size_t end = index;
      while (end < copies.size() &&
             staging_size + copies[end].src.size() <= kUploadStagingBytes) {
        staging_size += copies[end].src.size();
        ++end;
      }
      if (end == index) {  // one copy larger than the whole window
        staging_size = copies[index].src.size();
        end = index + 1;
      }
      auto staging = EnsureStaging(staging_size);
      if (!staging) {
        return std::unexpected(staging.error());
      }
      std::size_t offset = 0;
      for (std::size_t i = index; i < end; ++i) {
        std::memcpy(*staging + offset, copies[i].src.data(),
                    copies[i].src.size());
        offset += copies[i].src.size();
      }
      offset = 0;
      for (std::size_t i = index; i < end; ++i) {
        auto error = hipMemcpyAsync(copies[i].dst->Handle(), *staging + offset,
                                    copies[i].src.size(),
                                    hipMemcpyHostToDevice, stream_);
        if (error != hipSuccess) {
          LogError(std::string("hipMemcpyAsync H2D of ") +
                   std::to_string(copies[i].src.size()) + " bytes failed (" +
                   HipErrorName(error) + ")");
          return std::unexpected(FromHip(error));
        }
        total_bytes += copies[i].src.size();
        offset += copies[i].src.size();
      }
      auto error = hipStreamSynchronize(stream_);
      if (error != hipSuccess) {
        LogError(
            std::string("hipStreamSynchronize after batch upload failed (") +
            HipErrorName(error) + ")");
        return std::unexpected(FromHip(error));
      }
      index = end;
    }
    ReleaseStaging();
    if (diagnostics_ != nullptr) {
      LogInfo(core::FormatTransferSummary(
          total_bytes, copies.size(),
          core::ElapsedMs(started, core::PhaseClock::now())));
    }
    return {};
  }

  std::expected<void, StatusCode> CopyD2H(const Buffer& src, std::byte* dst,
                                         std::size_t bytes) override {
    return CopyD2HAt(src, 0, dst, bytes);
  }

  std::expected<void, StatusCode> CopyD2HAt(
      const Buffer& src, std::size_t offset, std::byte* dst,
      std::size_t bytes) override {
    if (auto device = SelectDevice(); device != StatusCode::Ok) {
      return std::unexpected(device);
    }
    if (offset > src.Size() || bytes > src.Size() - offset) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    // Stage through pinned host memory: a device-to-host copy straight into
    // pageable memory makes hipMemcpyAsync synchronous and busy-waits in the
    // runtime, so every readback goes through the reusable pinned buffer.
    auto staging = EnsureStaging(bytes);
    if (!staging) {
      return std::unexpected(staging.error());
    }
    const auto* base = static_cast<const std::byte*>(src.Handle()) + offset;
    auto error = hipMemcpyAsync(*staging, base, bytes, hipMemcpyDeviceToHost,
                                stream_);
    if (error == hipSuccess) {
      error = hipStreamSynchronize(stream_);
    }
    if (error != hipSuccess) {
      LogError(std::string("hipMemcpy D2H of ") + std::to_string(bytes) +
               " bytes failed (" + HipErrorName(error) + ")");
      return std::unexpected(FromHip(error));
    }
    std::memcpy(dst, *staging, bytes);
    return {};
  }

  std::expected<void, StatusCode> CopyD2D(
      const Buffer& src, std::size_t src_offset, Buffer& dst,
      std::size_t dst_offset, std::size_t bytes) override {
    if (auto device = SelectDevice(); device != StatusCode::Ok) {
      return std::unexpected(device);
    }
    if (src_offset > src.Size() || bytes > src.Size() - src_offset ||
        dst_offset > dst.Size() || bytes > dst.Size() - dst_offset) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    const auto* src_base =
        static_cast<const std::byte*>(src.Handle()) + src_offset;
    auto* dst_base = static_cast<std::byte*>(dst.Handle()) + dst_offset;
    // Asynchronous on the stream: the copies and the kernels are both
    // ordered on the same stream, so the copy still happens before the
    // kernels that read it, but the host does not drain the queue on every
    // call. A synchronous hipMemcpy here serialized the per-row linear
    // prefill (about four copies per row per layer) and the state
    // snapshots.
    auto error = hipMemcpyAsync(dst_base, src_base, bytes,
                                hipMemcpyDeviceToDevice, stream_);
    if (error != hipSuccess) {
      LogError(std::string("hipMemcpy D2D of ") + std::to_string(bytes) +
               " bytes failed (" + HipErrorName(error) + ")");
      return std::unexpected(FromHip(error));
    }
    return {};
  }

  std::expected<std::unique_ptr<Kernel>, StatusCode> LoadKernel(
      std::string_view name, std::span<const std::byte> code) override {
    if (!code.empty()) {
      LogError(std::string("kernel '") + std::string(name) +
               "': compiled kernel modules are not supported on rocm; use a "
               "built-in kernel id");
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    auto index = LookupBuiltIn(name);
    if (index < 0) {
      LogError(std::string("kernel '") + std::string(name) +
               "' is not a built-in on the rocm backend; add the kernel to "
               "the rocm kernel table");
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    return AdoptKernel(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(index) + 1),
        name);
  }

  void FreeKernel(void* handle) override {
    (void)handle;  // built-ins live for the process; nothing to release
  }

  std::expected<void, StatusCode> LaunchKernel(const Kernel& kernel,
                                              const KernelLaunch& launch) override {
    if (auto device = SelectDevice(); device != StatusCode::Ok) {
      return std::unexpected(device);
    }
    auto invalid = ValidateLaunch(launch);
    if (invalid != StatusCode::Ok) {
      return std::unexpected(invalid);
    }
    auto missing = CheckBuiltInArgs(kernel, launch);
    if (missing != StatusCode::Ok) {
      return std::unexpected(missing);
    }
    const auto index = static_cast<std::size_t>(
        reinterpret_cast<std::uintptr_t>(kernel.Handle()) - 1);
    if (index >= sizeof(kBuiltInKernels) / sizeof(kBuiltInKernels[0])) {
      LogError(std::string("kernel '") + std::string(kernel.Id()) +
               ": handle no longer valid; the kernel table changed after "
               "load");
      return std::unexpected(StatusCode::DeviceError);
    }
    std::uint64_t scalar_values[kMaxScalars];
    std::memset(scalar_values, 0, sizeof(scalar_values));
    std::memcpy(scalar_values, launch.scalars.data(),
                launch.scalars.size() * sizeof(std::uint64_t));
    // Fixed-size stack storage: ValidateLaunch already bounds the counts,
    // and this keeps the per-launch CPU cost off the heap (the decode
    // submits on the order of 500 launches per token).
    void* buffer_pointers[kMaxBoundBuffers];
    std::size_t buffer_count = 0;
    for (const auto* buffer : launch.buffers) {
      if (buffer == nullptr) {
        return std::unexpected(StatusCode::InvalidArgument);
      }
      buffer_pointers[buffer_count++] = buffer->Handle();
    }
    void* arg_pointers[kMaxBoundBuffers + kMaxScalars];
    std::size_t arg_count = 0;
    for (std::size_t i = 0; i < buffer_count; ++i) {
      arg_pointers[arg_count++] = &buffer_pointers[i];
    }
    for (std::size_t i = 0; i < launch.scalars.size(); ++i) {
      arg_pointers[arg_count++] = &scalar_values[i];
    }
    hipLaunchConfig_t config{};
    config.gridDim = dim3(launch.grid_x, launch.grid_y, launch.grid_z);
    config.blockDim = dim3(launch.block_x, launch.block_y, launch.block_z);
    config.dynamicSmemBytes = 0;
    config.stream = stream_;
    auto error = hipLaunchKernelExC(
        &config, kBuiltInKernels[index].function, arg_pointers);
    if (error != hipSuccess) {
      LogError(std::string("kernel ") + std::string(kernel.Id()) +
               " launch failed (" + HipErrorName(error) + ")");
      return std::unexpected(FromHip(error));
    }
    return {};
  }

  void Synchronize() override {
    (void)SelectDevice();
    auto error = hipStreamSynchronize(stream_);
    if (error != hipSuccess) {
      LogError(std::string("hipStreamSynchronize failed (") +
               HipErrorName(error) +
               "); the device did not reach a quiescent state");
    }
  }

 private:
  // HIP's current device is per-thread; Init selects it only on the
  // creating thread, so the serve loader and other workers start on the
  // default device. Every device entry point re-selects it here, and the
  // thread-local cache keeps the hot path to one comparison.
  StatusCode SelectDevice() {
    static thread_local int selected = -1;
    if (selected == device_index_) {
      return StatusCode::Ok;
    }
    auto error = hipSetDevice(device_index_);
    if (error != hipSuccess) {
      LogError(std::string("hipSetDevice(") + std::to_string(device_index_) +
               ") failed on this thread (" + HipErrorName(error) +
               "); the backend cannot touch that GPU here");
      return FromHip(error);
    }
    selected = device_index_;
    return StatusCode::Ok;
  }

  // A reusable pinned host staging buffer. A device<->host copy to pageable
  // memory makes hipMemcpyAsync synchronous and busy-waits in the runtime,
  // so every host readback and upload goes through pinned memory instead.
  // The buffer grows to the largest copy seen and is reused.
  std::expected<std::byte*, StatusCode> EnsureStaging(std::size_t bytes) {
    if (bytes == 0) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    if (staging_ == nullptr || staging_bytes_ < bytes) {
      ReleaseStaging();
      void* pinned = nullptr;
      auto error = hipHostMalloc(&pinned, bytes, hipHostMallocDefault);
      if (error != hipSuccess) {
        LogError(std::string("hipHostMalloc failed for the ") +
                 std::to_string(bytes) + " byte staging buffer (" +
                 HipErrorName(error) + ")");
        return std::unexpected(FromHip(error));
      }
      staging_ = static_cast<std::byte*>(pinned);
      staging_bytes_ = bytes;
    }
    return staging_;
  }

  // Free the reusable staging buffer. A batched upload releases it when
  // done, so a loaded model does not hold a large pinned block.
  void ReleaseStaging() {
    if (staging_ == nullptr) {
      return;
    }
    auto freed = hipHostFree(staging_);
    if (freed != hipSuccess) {
      LogError(std::string("hipHostFree of the staging buffer failed (") +
               HipErrorName(freed) + "); the memory may leak");
    }
    staging_ = nullptr;
    staging_bytes_ = 0;
  }

  std::string device_name_;
  // The stream every kernel and copy uses; carries the blocking sync policy.
  hipStream_t stream_ = nullptr;
  // Pinned host staging for device<->host copies (see EnsureStaging).
  std::byte* staging_ = nullptr;
  std::size_t staging_bytes_ = 0;
  bool initialized_ = false;
};

// Best-effort device name list; empty when the runtime is unavailable.
std::vector<std::string> ListDeviceNames() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) {
    return {};
  }
  std::vector<std::string> names;
  names.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, i) == hipSuccess) {
      names.emplace_back(props.name);
    } else {
      names.emplace_back("unknown");
    }
  }
  return names;
}

}  // namespace tessera::backends::rocm

namespace tessera {

std::unique_ptr<Backend> CreateBackend() {
  return std::make_unique<backends::rocm::RocmBackend>();
}

std::expected<std::vector<std::string>, StatusCode> ListGpuNames() {
  std::vector<std::string> names = backends::rocm::ListDeviceNames();
  if (names.empty()) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return names;
}

}  // namespace tessera
