#include "core/files.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <utility>

namespace tessera::core {

std::expected<std::vector<std::byte>, StatusCode> ReadFile(
    const std::filesystem::path& path) {
  std::error_code ec;
  if (!std::filesystem::exists(path, ec) || ec) {
    return std::unexpected(StatusCode::FileNotFound);
  }
  if (!std::filesystem::is_regular_file(path, ec) || ec) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return std::unexpected(StatusCode::FileNotFound);
  }
  stream.seekg(0, std::ios::end);
  const std::streamoff size = stream.tellg();
  stream.seekg(0, std::ios::beg);
  if (size < 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::vector<std::byte> data(static_cast<std::size_t>(size));
  if (size > 0 &&
      !stream.read(reinterpret_cast<char*>(data.data()), size)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return data;
}

std::expected<void, StatusCode> WriteFile(
    const std::filesystem::path& path, std::span<const std::byte> data) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec) || ec) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (!data.empty()) {
    stream.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size()));
  }
  stream.flush();
  if (!stream) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return {};
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    Reset();
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

MappedFile::~MappedFile() { Reset(); }

void MappedFile::Reset() {
  if (data_ != nullptr) {
    ::munmap(const_cast<std::byte*>(data_), size_);
    data_ = nullptr;
    size_ = 0;
  }
}

std::expected<MappedFile, StatusCode> MappedFile::Open(
    const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return std::unexpected(StatusCode::FileNotFound);
  }
  struct stat info {};
  if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
    ::close(fd);
    return std::unexpected(StatusCode::MalformedFile);
  }
  MappedFile result;
  result.size_ = static_cast<std::size_t>(info.st_size);
  if (result.size_ != 0) {
    void* address = ::mmap(nullptr, result.size_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (address == MAP_FAILED) {
      ::close(fd);
      return std::unexpected(StatusCode::MalformedFile);
    }
    result.data_ = static_cast<const std::byte*>(address);
  }
  ::close(fd);
  return result;
}

}  // namespace tessera::core
