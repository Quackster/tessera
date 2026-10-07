#include "core/files.hpp"
#include "core/loaders/safetensors.hpp"

#include <algorithm>
#include <cstring>

namespace tessera::core {

namespace {

// Boundary checks for the container (never configurable).
constexpr std::size_t kSafetensorsMinSize = 10;  // 8-byte prefix + '{}'
constexpr std::uint64_t kMaxSafetensorsHeaderLen = 1ull << 30;

}  // namespace

std::expected<SafetensorsFileCheck, StatusCode> InspectSafetensorsFile(
    const std::filesystem::path& path) {
  auto data = ReadFile(path);
  if (!data) {
    return std::unexpected(data.error());
  }
  if (data->size() < kSafetensorsMinSize) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::uint64_t header_len = 0;
  std::memcpy(&header_len, data->data(), 8);  // little-endian on all targets
  if (header_len == 0 || header_len > kMaxSafetensorsHeaderLen) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (data->size() < 8 + static_cast<std::size_t>(header_len)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (data->at(8) != static_cast<std::byte>('{')) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return SafetensorsFileCheck{data->size(), header_len};
}

std::expected<MxFp4Layout, StatusCode> InspectMxFp4Directory(
    const std::filesystem::path& dir) {
  std::error_code ec;
  if (!std::filesystem::exists(dir, ec) || ec) {
    return std::unexpected(StatusCode::FileNotFound);
  }
  if (!std::filesystem::is_directory(dir, ec) || ec) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (!std::filesystem::exists(dir / "config.json", ec) || ec) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  // Deterministic choice: lexicographically first *.safetensors.
  std::vector<std::filesystem::path> weights;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (!entry.is_regular_file(ec) || ec) continue;
    if (entry.path().extension() == ".safetensors") {
      weights.push_back(entry.path());
    }
  }
  if (ec) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (weights.empty()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (weights.size() > 1) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::sort(weights.begin(), weights.end());
  auto check = InspectSafetensorsFile(weights.front());
  if (!check) {
    return std::unexpected(check.error());
  }
  return MxFp4Layout{
      weights.front().string(), check->header_len, check->file_size};
}

}  // namespace tessera::core
