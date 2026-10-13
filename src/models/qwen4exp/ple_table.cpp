#include "models/qwen4exp/ple_table.hpp"

#include <string_view>

#include "core/loaders/gguf.hpp"
#include "core/numerics/quant.hpp"

namespace tessera::models::qwen4exp {

namespace {
constexpr std::string_view kTableName = "per_layer_token_embd.weight";
}  // namespace

std::expected<std::unique_ptr<PleTable>, StatusCode> PleTable::Open(
    const std::string& path) {
  auto mapped = core::MappedFile::Open(path);
  if (!mapped) {
    return std::unexpected(mapped.error());
  }
  auto file = std::make_shared<core::MappedFile>(std::move(*mapped));
  auto parsed = core::ParseGguf(file->bytes());
  if (!parsed) {
    return std::unexpected(parsed.error());
  }
  for (std::size_t i = 0; i < parsed->tensors.size(); ++i) {
    const TensorEntry& entry = parsed->tensors[i];
    if (entry.name != kTableName) {
      continue;
    }
    if (entry.dtype != DType::IQ4_NL || entry.shape.rank != 2 ||
        entry.shape.dims[0] != kPleRowElements) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    const std::uint64_t rows = entry.shape.dims[1];
    const std::uint64_t begin = parsed->tensor_data_start + parsed->tensor_offsets[i];
    const std::uint64_t bytes = rows * kPleRowBytes;
    if (begin > file->bytes().size() ||
        bytes > file->bytes().size() - begin) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    auto table = std::unique_ptr<PleTable>(new PleTable());
    const std::span<const std::byte> tensor = file->bytes().subspan(begin, bytes);
    table->file_ = std::move(file);
    table->data_ = tensor;
    table->rows_ = static_cast<std::size_t>(rows);
    return table;
  }
  return std::unexpected(StatusCode::FileNotFound);
}

std::expected<void, StatusCode> PleTable::Gather(
    std::span<const std::uint32_t> rows, std::span<float> out) const {
  if (out.size() != rows.size() * kPleRowElements) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (rows[i] >= rows_) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    const std::byte* row = data_.data() + static_cast<std::size_t>(rows[i]) * kPleRowBytes;
    for (std::size_t b = 0; b < kPleRowElements / 32; ++b) {
      core::DequantizeIQ4NL(
          std::span<const std::byte>(row + b * 18, 18),
          std::span<float>(out.data() + i * kPleRowElements + b * 32, 32));
    }
  }
  return {};
}

}  // namespace tessera::models::qwen4exp
