#include "tessera/backend.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

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

// Built-in "fill": writes scalar 0 to every int32 element (scalar 1 is
// the element count). Contract: CheckBuiltInArgs (tessera API).
__global__ void FillKernel(int* out, unsigned long long value,
                          unsigned long long count) {
  unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = static_cast<int>(value);
  }
}

// IEEE binary16 (little-endian) -> fp32, exact (device port of the
// core Fp16ToFloat in src/core/numerics/quant.cpp).
__host__ __device__ float Fp16ToFloatDev(std::uint16_t half) {
  const std::uint32_t sign = half >> 15;
  const std::uint32_t exp = (half >> 10) & 0x1F;
  const std::uint32_t mant = half & 0x3FF;
  float value;
  if (exp == 0) {
    value = static_cast<float>(mant) * 0x1p-24f;
  } else if (exp == 31) {
    value = mant != 0
        ? std::numeric_limits<float>::quiet_NaN()
        : std::numeric_limits<float>::infinity();
  } else {
    value = (1.0f + static_cast<float>(mant) / 1024.0f) *
            std::ldexp(1.0f, static_cast<int>(exp) - 15);
  }
  return sign != 0 ? -value : value;
}

// The 6-bit scale/min pair of sub-block j from the 12-byte packing
// (device port of the core GetScaleMin; the vulkan kernel uses the
// same formula).
__host__ __device__ void GetScaleMinDev(std::size_t j,
                                        const unsigned char* scales,
                                        std::uint8_t* scale,
                                        std::uint8_t* min) {
  if (j < 4) {
    *scale = scales[j] & 63;
    *min = scales[j + 4] & 63;
  } else {
    *scale = (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4);
    *min = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
  }
}

// Built-in "gemm_q4k": buffer 0 is the activation A (fp32, m x k),
// buffer 1 the quantized weights W (Q4_K, n x k), buffer 2 the output
// C (fp32, m x n); scalar 0 is k, scalar 1 n, scalar 2 m. One
// workgroup per 256 output elements (block_x = 256).
__global__ void GemmQ4KKernel(const float* a, const unsigned char* w,
                              float* c, unsigned long long k,
                              unsigned long long n,
                              unsigned long long m) {
  unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= m * n) {
    return;
  }
  const unsigned long long row_a = idx / n;
  const unsigned long long row_w = idx % n;
  const unsigned long long blocks = k / 256;
  float acc = 0.0f;
  for (unsigned long long b = 0; b < blocks; ++b) {
    const unsigned char* base =
        w + (row_w * blocks + b) * 144;
    std::uint16_t d_bits = 0;
    std::uint16_t dm_bits = 0;
    std::memcpy(&d_bits, base, 2);
    std::memcpy(&dm_bits, base + 2, 2);
    const float d = Fp16ToFloatDev(d_bits);
    const float dm = Fp16ToFloatDev(dm_bits);
    const unsigned char* scales = base + 4;
    const unsigned char* qs = base + 16;
    for (unsigned long long grp = 0; grp < 4; ++grp) {
      std::uint8_t sc0, mn0, sc1, mn1;
      GetScaleMinDev(2 * grp, scales, &sc0, &mn0);
      GetScaleMinDev(2 * grp + 1, scales, &sc1, &mn1);
      const float a0 = d * static_cast<float>(sc0);
      const float o0 = dm * static_cast<float>(mn0);
      const float a1 = d * static_cast<float>(sc1);
      const float o1 = dm * static_cast<float>(mn1);
      for (unsigned long long l = 0; l < 32; ++l) {
        const std::uint8_t q = qs[32 * grp + l];
        const unsigned long long t = b * 256 + grp * 64 + l;
        acc += a[row_a * k + t] * (a0 * static_cast<float>(q & 15) - o0);
        acc += a[row_a * k + t + 32] *
               (a1 * static_cast<float>(q >> 4) - o1);
      }
    }
  }
  c[idx] = acc;
}

// The kernels compiled into the rocm backend.
struct BuiltInKernel {
  std::string_view name;
  void* function;
};
const BuiltInKernel kBuiltInKernels[] = {
    {"fill", reinterpret_cast<void*>(&FillKernel)},
    {"gemm_q4k", reinterpret_cast<void*>(&GemmQ4KKernel)},
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
    if (bytes > src.Size()) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    auto error =
        hipMemcpy(dst, src.Handle(), bytes, hipMemcpyDeviceToHost);
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
