#include "tessera/backend.hpp"

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

void Backend::SetDeviceIndex(int index) {
  device_index_ = index;
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

}  // namespace tessera
