#include "tessera/backend.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "backends/rocm/rocm_kernels.hpp"

namespace tessera::backends::rocm {

namespace {

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
    {"gemm_q5k", reinterpret_cast<void*>(&GemmQ5KKernel)},
    {"gemm_q5k_batched", reinterpret_cast<void*>(&GemmQ5KBatchedKernel)},
    {"gemm_q6k", reinterpret_cast<void*>(&GemmQ6KKernel)},
    {"gemm_q6k_batched", reinterpret_cast<void*>(&GemmQ6KBatchedKernel)},
    {"gemm_q3k", reinterpret_cast<void*>(&GemmQ3KKernel)},
    {"gemm_iq4nl", reinterpret_cast<void*>(&GemmIq4NlKernel)},
    {"gemm_iq4xs", reinterpret_cast<void*>(&GemmIq4XsKernel)},
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
    {"concat_features", reinterpret_cast<void*>(&ConcatFeaturesKernel)},
    {"cast_f32_f16", reinterpret_cast<void*>(&CastF32F16Kernel)},
    {"round_bf16", reinterpret_cast<void*>(&RoundBf16Kernel)},
    {"quantize_q8", reinterpret_cast<void*>(&QuantizeQ8Kernel)},
    {"quantize_fp8", reinterpret_cast<void*>(&QuantizeFp8Kernel)},
    {"quantize_fp8_pack", reinterpret_cast<void*>(&QuantizeFp8PackKernel)},
    {"attention_fp8", reinterpret_cast<void*>(&AttentionFp8Kernel)},
    {"attention_q8", reinterpret_cast<void*>(&AttentionQ8Kernel)},
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
  std::string_view Name() const override {
    return "rocm";
  }

  std::string_view DeviceName() const override {
    return device_name_;
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
               "--gpu in the range [0, " + std::to_string(device_count - 1) +
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
    if (src.size() > dst.Size()) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    auto error = hipMemcpy(dst.Handle(), src.data(), src.size(),
                          hipMemcpyHostToDevice);
    if (error != hipSuccess) {
      LogError(std::string("hipMemcpy H2D of ") +
               std::to_string(src.size()) + " bytes failed (" +
               HipErrorName(error) + ")");
      return std::unexpected(FromHip(error));
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
    if (offset > src.Size() || bytes > src.Size() - offset) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    const auto* base = static_cast<const std::byte*>(src.Handle()) + offset;
    auto error =
        hipMemcpy(dst, base, bytes, hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
      LogError(std::string("hipMemcpy D2H of ") + std::to_string(bytes) +
               " bytes failed (" + HipErrorName(error) + ")");
      return std::unexpected(FromHip(error));
    }
    return {};
  }

  std::expected<void, StatusCode> CopyD2D(
      const Buffer& src, std::size_t src_offset, Buffer& dst,
      std::size_t dst_offset, std::size_t bytes) override {
    if (src_offset > src.Size() || bytes > src.Size() - src_offset ||
        dst_offset > dst.Size() || bytes > dst.Size() - dst_offset) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    const auto* src_base =
        static_cast<const std::byte*>(src.Handle()) + src_offset;
    auto* dst_base = static_cast<std::byte*>(dst.Handle()) + dst_offset;
    // Asynchronous on the null stream: the copies and the kernels are both
    // ordered on the null stream, so the copy still happens before the
    // kernels that read it, but the host does not drain the queue on every
    // call. A synchronous hipMemcpy here serialized the per-row linear
    // prefill (about four copies per row per layer) and the state
    // snapshots.
    auto error = hipMemcpyAsync(dst_base, src_base, bytes,
                                hipMemcpyDeviceToDevice, nullptr);
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
    config.stream = nullptr;
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
    auto error = hipDeviceSynchronize();
    if (error != hipSuccess) {
      LogError(std::string("hipDeviceSynchronize failed (") +
               HipErrorName(error) +
               "); the device did not reach a quiescent state");
    }
  }

 private:
  std::string device_name_;
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
