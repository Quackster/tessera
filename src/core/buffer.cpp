#include "tessera/backend.hpp"

#include <span>
#include <vector>

namespace tessera {

Buffer::Buffer(Backend& backend, void* handle, std::size_t size, MemoryKind kind,
               void* host_map)
    : backend_(&backend), handle_(handle), size_(size), kind_(kind),
      host_map_(host_map) {}

Buffer::Buffer(Buffer&& other) noexcept
    : backend_(other.backend_), handle_(other.handle_), size_(other.size_),
      kind_(other.kind_), host_map_(other.host_map_) {
  other.backend_ = nullptr;
  other.handle_ = nullptr;
  other.size_ = 0;
  other.host_map_ = nullptr;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    FreeIfOwned();
    backend_ = other.backend_;
    handle_ = other.handle_;
    size_ = other.size_;
    kind_ = other.kind_;
    host_map_ = other.host_map_;
    other.backend_ = nullptr;
    other.handle_ = nullptr;
    other.size_ = 0;
    other.host_map_ = nullptr;
  }
  return *this;
}

Buffer::~Buffer() {
  FreeIfOwned();
}

void Buffer::FreeIfOwned() noexcept {
  if (backend_ != nullptr && handle_ != nullptr) {
    backend_->FreeBuffer(kind_, handle_);
    backend_ = nullptr;
    handle_ = nullptr;
    host_map_ = nullptr;
  }
}

std::size_t Buffer::Size() const {
  return size_;
}

MemoryKind Buffer::Kind() const {
  return kind_;
}

void* Buffer::Handle() const {
  return handle_;
}

void* Buffer::HostMap() const {
  return host_map_;
}

Backend& Buffer::Owner() const {
  return *backend_;
}

Kernel::Kernel(Backend& backend, void* handle, std::string id)
    : backend_(&backend), handle_(handle), id_(std::move(id)) {}

Kernel::Kernel(Kernel&& other) noexcept
    : backend_(other.backend_), handle_(other.handle_),
      id_(std::move(other.id_)) {
  other.backend_ = nullptr;
  other.handle_ = nullptr;
}

Kernel& Kernel::operator=(Kernel&& other) noexcept {
  if (this != &other) {
    FreeIfOwned();
    backend_ = other.backend_;
    handle_ = other.handle_;
    id_ = std::move(other.id_);
    other.backend_ = nullptr;
    other.handle_ = nullptr;
  }
  return *this;
}

Kernel::~Kernel() {
  FreeIfOwned();
}

void Kernel::FreeIfOwned() noexcept {
  if (backend_ != nullptr && handle_ != nullptr) {
    backend_->FreeKernel(handle_);
    backend_ = nullptr;
    handle_ = nullptr;
  }
}

std::string_view Kernel::Id() const {
  return id_;
}

void* Kernel::Handle() const {
  return handle_;
}

Backend& Kernel::Owner() const {
  return *backend_;
}

std::unique_ptr<Buffer> Backend::AdoptBuffer(void* handle, std::size_t bytes,
                                            MemoryKind kind, void* host_map) {
  return std::unique_ptr<Buffer>(
      new Buffer(*this, handle, bytes, kind, host_map));
}

std::unique_ptr<Kernel> Backend::AdoptKernel(void* handle,
                                            std::string_view id) {
  return std::unique_ptr<Kernel>(
      new Kernel(*this, handle, std::string(id)));
}

void Backend::SetDiagnostics(log::Diagnostics* diagnostics) {
  diagnostics_ = diagnostics;
}

const log::Diagnostics* Backend::Diagnostics() const { return diagnostics_; }

void Backend::SetDeviceIndex(int index) {
  device_index_ = index;
}

int Backend::DeviceIndex() const { return device_index_; }

void Backend::SetCachePhase(CachePhase phase) {
  cache_phase_ = phase;
}

CachePhase Backend::cache_phase() const { return cache_phase_; }

std::expected<void, StatusCode> Backend::CopyPeerFrom(Buffer& dst,
                                                      const Buffer& src,
                                                      std::size_t bytes) {
  if (bytes == 0) {
    return {};
  }
  if (bytes > dst.Size() || bytes > src.Size()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<std::byte> stage(bytes);
  auto down = src.Owner().CopyD2HAt(src, 0, stage.data(), bytes);
  if (!down) {
    return down;
  }
  return CopyH2DAt(dst, 0, std::span<const std::byte>(stage.data(), bytes));
}

void Backend::LogInfo(std::string_view message) const {
  if (diagnostics_ != nullptr) {
    diagnostics_->Info("backend", message);
  }
}

void Backend::LogWarn(std::string_view message) const {
  if (diagnostics_ != nullptr) {
    diagnostics_->Warn("backend", message);
  }
}

void Backend::LogError(std::string_view message) const {
  if (diagnostics_ != nullptr) {
    diagnostics_->Error("backend", message);
  }
}

std::expected<void, StatusCode> CopyAcrossDevices(
    const Buffer& src, std::size_t src_offset, Buffer& dst,
    std::size_t dst_offset, std::size_t bytes) {
  if (src_offset > src.Size() || bytes > src.Size() - src_offset ||
      dst_offset > dst.Size() || bytes > dst.Size() - dst_offset) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (bytes == 0) {
    return {};
  }
  // A same-device copy stays on the device; a cross-device copy uses a
  // direct peer transfer when the buffers start at offset 0, else stages
  // through host memory (correct for any device pair).
  if (&src.Owner() == &dst.Owner()) {
    return src.Owner().CopyD2D(src, src_offset, dst, dst_offset, bytes);
  }
  if (src_offset == 0 && dst_offset == 0) {
    return dst.Owner().CopyPeerFrom(dst, src, bytes);
  }
  std::vector<std::byte> stage(bytes);
  auto down = src.Owner().CopyD2HAt(src, src_offset, stage.data(), bytes);
  if (!down) {
    return down;
  }
  return dst.Owner().CopyH2DAt(
      dst, dst_offset, std::span<const std::byte>(stage.data(), bytes));
}

std::expected<void, StatusCode> AllReduceSum(std::span<Buffer* const> buffers,
                                             std::size_t bytes) {
  if (buffers.empty() || (bytes % 4) != 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (const Buffer* buffer : buffers) {
    if (buffer == nullptr || bytes > buffer->Size()) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
  }
  if (bytes == 0 || buffers.size() == 1) {
    return {};
  }
  const std::size_t count = bytes / 4;
  std::vector<float> sum(count, 0.0f);
  std::vector<float> part(count);
  for (Buffer* buffer : buffers) {
    auto down = buffer->Owner().CopyD2H(
        *buffer, reinterpret_cast<std::byte*>(part.data()), bytes);
    if (!down) {
      return down;
    }
    for (std::size_t i = 0; i < count; ++i) {
      sum[i] += part[i];
    }
  }
  const std::span<const std::byte> packed(
      reinterpret_cast<const std::byte*>(sum.data()), bytes);
  for (Buffer* buffer : buffers) {
    auto up = buffer->Owner().CopyH2D(*buffer, packed);
    if (!up) {
      return up;
    }
  }
  return {};
}

}  // namespace tessera
