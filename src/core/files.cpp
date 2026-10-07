#include "core/files.hpp"

#include <fstream>

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

}  // namespace tessera::core
