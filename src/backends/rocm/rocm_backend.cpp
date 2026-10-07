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
    {"attention", reinterpret_cast<void*>(&AttentionKernel)},
    {"rope", reinterpret_cast<void*>(&RopeKernel)},
    {"gemm_fp8", reinterpret_cast<void*>(&GemmFp8Kernel)},
    {"gemm_mxfp4", reinterpret_cast<void*>(&GemmMxFp4Kernel)},
    {"gemm_q5k", reinterpret_cast<void*>(&GemmQ5KKernel)},
    {"gemm_q6k", reinterpret_cast<void*>(&GemmQ6KKernel)},
    {"gemm_q3k", reinterpret_cast<void*>(&GemmQ3KKernel)},
    {"gemm_iq4nl", reinterpret_cast<void*>(&GemmIq4NlKernel)},
    {"gemm_iq4xs", reinterpret_cast<void*>(&GemmIq4XsKernel)},
    {"gemm_iq3s", reinterpret_cast<void*>(&GemmIq3SKernel)},
    {"rmsnorm", reinterpret_cast<void*>(&RmsnormKernel)},
    {"sigmoid_gate", reinterpret_cast<void*>(&SigmoidGateKernel)},
    {"conv1d", reinterpret_cast<void*>(&Conv1dKernel)},
    {"delta_step", reinterpret_cast<void*>(&DeltaStepKernel)},
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
    error = hipSetDevice(0);
    if (error != hipSuccess) {
      LogError(std::string("hipSetDevice(0) failed (") + HipErrorName(error) +
               "); cannot use GPU 0");
      return std::unexpected(FromHip(error));
    }
    hipDeviceProp_t props;
    error = hipGetDeviceProperties(&props, 0);
    if (error != hipSuccess) {
      LogError(std::string("hipGetDeviceProperties failed (") +
               HipErrorName(error) + ")");
      return std::unexpected(FromHip(error));
    }
    device_name_ = props.name;
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
    // Arg pointers must outlive the launch call, so they are set after
    // buffer_pointers has finished growing (no reallocation in between).
    std::vector<void*> buffer_pointers;
    buffer_pointers.reserve(kMaxBoundBuffers);
    for (const auto* buffer : launch.buffers) {
      if (buffer == nullptr) {
        return std::unexpected(StatusCode::InvalidArgument);
      }
      buffer_pointers.push_back(buffer->Handle());
    }
    std::vector<void*> arg_pointers;
    arg_pointers.reserve(kMaxBoundBuffers + kMaxScalars);
    for (std::size_t i = 0; i < buffer_pointers.size(); ++i) {
      arg_pointers.push_back(&buffer_pointers[i]);
    }
    for (std::size_t i = 0; i < launch.scalars.size(); ++i) {
      arg_pointers.push_back(&scalar_values[i]);
    }
    hipLaunchConfig_t config{};
    config.gridDim = dim3(launch.grid_x, launch.grid_y, launch.grid_z);
    config.blockDim = dim3(launch.block_x, launch.block_y, launch.block_z);
    config.dynamicSmemBytes = 0;
    config.stream = nullptr;
    auto error = hipLaunchKernelExC(
        &config, kBuiltInKernels[index].function, arg_pointers.data());
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

}  // namespace tessera::backends::rocm

namespace tessera {

std::unique_ptr<Backend> CreateBackend() {
  return std::make_unique<backends::rocm::RocmBackend>();
}

}  // namespace tessera
