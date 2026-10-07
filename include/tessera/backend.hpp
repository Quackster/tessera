#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "tessera/log.hpp"
#include "tessera/types.hpp"

namespace tessera {

class Buffer;    // forward declaration (defined below)
class Backend;   // forward declaration (defined below)
class Kernel;    // forward declaration (defined below)

// Where a buffer's memory lives.
enum class MemoryKind : int {
  Device = 0,     // accelerator memory
  HostVisible = 1,  // pinned host memory, mappable by the accelerator
};

// Kernel argument binding limits: the layout both backends expose to a
// kernel (descriptor bindings and scalar/push-constant arguments).
constexpr std::size_t kMaxBoundBuffers = 16;
constexpr std::size_t kMaxScalars = 8;

// One kernel launch request. grid_* is the workgroup count per axis on
// both backends. block_* is the workgroup size: honored on rocm
// (capped at the device limit); on vulkan the workgroup size is fixed
// when the kernel is compiled, so block_* is ignored there.
// Arguments bind in kernel parameter order: device buffers first
// (parameters 0..N-1), then 64-bit scalar values.
struct KernelLaunch {
  std::uint32_t grid_x = 1;
  std::uint32_t grid_y = 1;
  std::uint32_t grid_z = 1;
  std::uint32_t block_x = 256;
  std::uint32_t block_y = 1;
  std::uint32_t block_z = 1;
  // Bound buffers, in kernel parameter order (buffers before scalars).
  std::vector<const Buffer*> buffers;
  // 64-bit scalar arguments, in kernel parameter order.
  std::vector<std::uint64_t> scalars;
};

// RAII accelerator buffer allocated through a Backend.
// The handle is opaque to the core; only the allocating backend knows its
// meaning (a VkDeviceMemory record, a hipMalloc pointer, ...).
class Buffer {
 public:
  Buffer() = default;
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;
  ~Buffer();
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  [[nodiscard]] std::size_t Size() const;
  [[nodiscard]] MemoryKind Kind() const;
  // Opaque accelerator handle; pass to this buffer's backend only.
  [[nodiscard]] void* Handle() const;
  // Valid when Kind() == MemoryKind::HostVisible.
  [[nodiscard]] void* HostMap() const;
  // The backend that allocated this buffer.
  [[nodiscard]] Backend& Owner() const;

   private:
  friend class Backend;
  Buffer(Backend& backend, void* handle, std::size_t size, MemoryKind kind,
         void* host_map);
  void FreeIfOwned() noexcept;
  Backend* backend_ = nullptr;
  void* handle_ = nullptr;
  std::size_t size_ = 0;
  MemoryKind kind_ = MemoryKind::Device;
  void* host_map_ = nullptr;
};

// RAII handle for a compiled kernel loaded through Backend::LoadKernel.
// Destroying the Kernel releases the compiled artifact (a VkPipeline on
// vulkan; nothing for rocm built-ins).
//
// Usage:
//   auto kernel = backend.LoadKernel("fill", {});
//   if (!kernel) return 1;
//   KernelLaunch launch;  // buffers + scalars, in parameter order
//   auto result = backend.LaunchKernel(*kernel, launch);
class Kernel {
 public:
  Kernel() = default;
  Kernel(Kernel&& other) noexcept;
  Kernel& operator=(Kernel&& other) noexcept;
  ~Kernel();
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

  // The name used at load time ("fill", ...).
  [[nodiscard]] std::string_view Id() const;
  // The backend that loaded this kernel.
  [[nodiscard]] Backend& Owner() const;
  // Opaque accelerator handle; pass to this kernel's backend only.
  [[nodiscard]] void* Handle() const;

 private:
  friend class Backend;
  Kernel(Backend& backend, void* handle, std::string id);
  void FreeIfOwned() noexcept;
  Backend* backend_ = nullptr;
  void* handle_ = nullptr;
  std::string id_;
};

// Shared launch validation: grid/block axes must be nonzero and the
// argument counts must fit the contract limits.
[[nodiscard]] inline StatusCode ValidateLaunch(const KernelLaunch& launch) {
  if (launch.grid_x == 0 || launch.grid_y == 0 || launch.grid_z == 0 ||
      launch.block_x == 0 || launch.block_y == 0 || launch.block_z == 0) {
    return StatusCode::InvalidArgument;
  }
  if (launch.buffers.size() > kMaxBoundBuffers ||
      launch.scalars.size() > kMaxScalars) {
    return StatusCode::InvalidArgument;
  }
  return StatusCode::Ok;
}

// Built-in kernel argument contract, identical on every backend.
// "fill": buffer 0 is an int32 array; scalar 0 is the value, scalar 1
// the element count.
// "gemm_q4k": buffer 0 is the activation A (fp32, m x k), buffer 1 the
// quantized weights W (Q4_K, n x k rows of k/256 blocks), buffer 2 the
// output C (fp32, m x n); scalar 0 is k (a positive multiple of
// kQ4KBlockElements), scalar 1 is n, scalar 2 is m. The dispatch is
// ceil(m * n / 256) workgroups of 256.
// "rope": buffer 0 holds rows x heads x head_dim fp32 rotated in place
// (NeoX pairing over rope_dim per head); scalars are rows, heads,
// head_dim, rope_dim, pos_base, and the fp32 theta bits. head_dim and
// rope_dim are even, rope_dim <= head_dim. The dispatch is
// ceil(rows * heads * (rope_dim / 2) / 256) workgroups of 256.
// "attention": buffers are q (m x heads*head_dim fp32), k and v
// (n x kv_heads*head_dim each), out (m x heads*head_dim); scalars are
// m, n, heads, kv_heads, head_dim, q_base. Query row i sits at
// position q_base + i and attends keys 0..pos (clamped to n - 1) with
// scale 1/sqrt(head_dim); head h reads kv head h / (heads/kv_heads).
// The dispatch is ceil(m * heads * head_dim / 256) workgroups of 256.
// "gemm_fp8": buffer 0 is A (fp32, m x k), buffer 1 the FP8 E4M3
// weights W (n x k bytes), buffer 2 one fp32 scale per row of W,
// buffer 3 the output C (fp32, m x n); scalars are m, n, k. The
// dispatch is ceil(m * n / 256) workgroups of 256.
// "gemm_mxfp4": buffer 0 is A (fp32, m x k), buffer 1 the MXFP4
// weights W (n x k/2 bytes, two E2M1 nibbles per byte), buffer 2 the
// E8M0 scales (n x k/32 bytes), buffer 3 the output C (fp32, m x n);
// scalars are m, n, k with k a positive multiple of 32. The dispatch
// is ceil(m * n / 256) workgroups of 256.
// "gemm_q5k", "gemm_q6k", "gemm_q3k", "gemm_iq4nl", "gemm_iq4xs",
// "gemm_iq3s": buffer 0 is A (fp32, m x k), buffer 1 the quantized
// weights W, buffer 2 the output C (fp32, m x n); scalars are m, n,
// k with k a positive multiple of 256 (32 for iq4nl). The dispatch
// is ceil(m * n / 256) workgroups of 256.
// "rmsnorm": buffer 0 is X (fp32, rows x cols), buffer 1 the weight W
// (fp32, cols), buffer 2 the output Y (fp32, rows x cols); scalars
// are rows, cols, and the fp32 epsilon bits. Y = X / sqrt(mean(X^2) +
// eps) * W per row with sequential accumulation. The dispatch is
// ceil(rows * cols / 256) workgroups of 256.
// "sigmoid_gate": buffers are A and G (fp32, n each) and the output O
// (fp32, n); scalar 0 is n. O = A * sigmoid(G) elementwise. The
// dispatch is ceil(n / 256) workgroups of 256.
// "conv1d": buffer 0 is X (fp32, channels x length), buffer 1 the
// weights W (fp32, channels x width), buffer 2 the output Y (fp32,
// channels x length); scalars are channels, length, width. Y is the
// causal depthwise convolution with sequential accumulation. The
// dispatch is ceil(channels * length / 256) workgroups of 256.
[[nodiscard]] inline StatusCode CheckBuiltInArgs(const Kernel& kernel,
                                                 const KernelLaunch& launch) {
  if (kernel.Id() == "fill" &&
      (launch.buffers.size() != 1 || launch.scalars.size() != 2)) {
    return StatusCode::InvalidArgument;
  }
  if (kernel.Id() == "gemm_q4k") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t k = launch.scalars[0];
    if (k == 0 || k % kQ4KBlockElements != 0 || launch.scalars[1] == 0 ||
        launch.scalars[2] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "rope") {
    if (launch.buffers.size() != 1 || launch.scalars.size() != 6) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t head_dim = launch.scalars[2];
    const std::uint64_t rope_dim = launch.scalars[3];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || head_dim == 0 ||
        rope_dim == 0 || rope_dim > head_dim || (head_dim % 2) != 0 ||
        (rope_dim % 2) != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "attention") {
    if (launch.buffers.size() != 4 || launch.scalars.size() != 6) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t heads = launch.scalars[2];
    const std::uint64_t kv_heads = launch.scalars[3];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || heads == 0 ||
        kv_heads == 0 || launch.scalars[4] == 0 ||
        (heads % kv_heads) != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gemm_fp8") {
    if (launch.buffers.size() != 4 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 ||
        launch.scalars[2] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gemm_mxfp4") {
    if (launch.buffers.size() != 4 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t k = launch.scalars[2];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || k == 0 ||
        k % 32 != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gemm_q5k" || kernel.Id() == "gemm_q6k" ||
      kernel.Id() == "gemm_q3k" || kernel.Id() == "gemm_iq4xs" ||
      kernel.Id() == "gemm_iq3s") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t k = launch.scalars[2];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || k == 0 ||
        k % 256 != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gemm_iq4nl") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t k = launch.scalars[2];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || k == 0 ||
        k % 32 != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "rmsnorm") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "sigmoid_gate") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 1) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "conv1d") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 ||
        launch.scalars[2] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  return StatusCode::Ok;
}

// Abstract compute backend. Exactly one instance per Engine; the concrete
// backend (vulkan or rocm) is selected at configure time via
// TESSERA_BACKEND. All device work in the core goes through this
// interface; no vendor type may cross it.
class Backend {
 public:
  virtual ~Backend() = default;

  // "vulkan" or "rocm".
  [[nodiscard]] virtual std::string_view Name() const = 0;
  // Human-readable device name, for diagnostics.
  [[nodiscard]] virtual std::string_view DeviceName() const = 0;

  // Point the backend at the engine's diagnostics channel (called once
  // after construction; before Init).
  virtual void SetDiagnostics(log::Diagnostics* diagnostics);

  // One-time device initialization. Returns Ok when already done.
  virtual std::expected<void, StatusCode> Init() = 0;

  // Allocate `bytes` of `kind` memory (0 bytes is InvalidArgument).
  //
  // Usage:
  //   auto result = backend.AllocateBuffer(4096, MemoryKind::Device);
  //   if (result) result->CopyH2D(*buffer, host_data);
  virtual std::expected<std::unique_ptr<Buffer>, StatusCode> AllocateBuffer(
      std::size_t bytes, MemoryKind kind) = 0;

  // Wrap an allocated handle into a Buffer. Backend implementations call
  // this after a successful allocation; the Buffer frees the handle
  // through this backend when destroyed.
  std::unique_ptr<Buffer> AdoptBuffer(void* handle, std::size_t bytes,
                                     MemoryKind kind, void* host_map);

  // Release a buffer. Called by Buffer's destructor only.
  virtual void FreeBuffer(MemoryKind kind, void* handle) = 0;

  // Wrap a loaded kernel into a Kernel. Backend implementations call this
  // after a successful LoadKernel.
  std::unique_ptr<Kernel> AdoptKernel(void* handle, std::string_view id);

  // Copy host bytes into `dst` (bounds-checked against the buffer size).
  virtual std::expected<void, StatusCode> CopyH2D(
      Buffer& dst, std::span<const std::byte> src) = 0;
  // Copy `bytes` from `src` into host memory (bounds-checked).
  virtual std::expected<void, StatusCode> CopyD2H(
      const Buffer& src, std::byte* dst, std::size_t bytes) = 0;
  // Copy `bytes` from `src` at `offset` into host memory
  // (bounds-checked). Reads a slice without staging the whole buffer
  // (embedding rows, single cache lines).
  virtual std::expected<void, StatusCode> CopyD2HAt(
      const Buffer& src, std::size_t offset, std::byte* dst,
      std::size_t bytes) = 0;

  // Load a compiled kernel. Vulkan: `code` is a SPIR-V module and `name`
  // its entry point; an empty code resolves a built-in (today "fill" and
  // "gemm_q4k"). ROCm: built-ins only; a non-empty code is
  // UnsupportedFeature. A name the backend does not implement is
  // UnsupportedFeature.
  //
  // Usage:
  //   auto kernel = backend.LoadKernel("fill", {});
  //   if (!kernel) return 1;
  virtual std::expected<std::unique_ptr<Kernel>, StatusCode> LoadKernel(
      std::string_view name, std::span<const std::byte> code) = 0;

  // Release a loaded kernel. Called by the Kernel destructor only.
  virtual void FreeKernel(void* handle) = 0;

  // Launch `kernel`; arguments bind in kernel parameter order (buffers
  // first, then scalars). Shape checks: ValidateLaunch, CheckBuiltInArgs.
  virtual std::expected<void, StatusCode> LaunchKernel(
      const Kernel& kernel, const KernelLaunch& launch) = 0;

  // Block the host until all previously submitted device work is done.
  virtual void Synchronize() = 0;

 protected:
  void LogWarn(std::string_view message) const;
  void LogError(std::string_view message) const;
  log::Diagnostics* diagnostics_ = nullptr;
};

// Construct the configured backend. Returns nullptr only when the build
// has no backend (a misconfiguration the engine reports as an error).
[[nodiscard]] std::unique_ptr<Backend> CreateBackend();

}  // namespace tessera
