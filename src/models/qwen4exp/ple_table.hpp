#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>

#include "core/files.hpp"
#include "tessera/types.hpp"

// The PLE n-gram table (`per_layer_token_embd.weight`, IQ4_NL) read a row
// at a time from its GGUF file (the model's second shard). One row is
// `embedding_length_per_layer_input` = 160 values = 90 bytes (five IQ4_NL
// blocks of 32). The file is memory-mapped, so a token's sixteen rows
// fault in only the pages it touches and the 28.8 GB table never has to
// fit in RAM. Ported from ../Strata include/strata/kernels/ngram.hpp
// (the row gather and decode).

namespace tessera::models::qwen4exp {

// One 160-value row is five 32-value IQ4_NL blocks.
constexpr std::size_t kPleRowElements = 160;
constexpr std::size_t kPleRowBytes = (kPleRowElements / 32) * 18;  // 90

class PleTable {
 public:
  PleTable() = default;
  PleTable(const PleTable&) = delete;
  PleTable& operator=(const PleTable&) = delete;
  PleTable(PleTable&&) = delete;
  PleTable& operator=(PleTable&&) = delete;

  // Map `path` and locate the table tensor. FileNotFound / MalformedFile /
  // UnsupportedFeature on a bad path, a missing tensor, a wrong type or an
  // inconsistent shape.
  [[nodiscard]] static std::expected<std::unique_ptr<PleTable>, StatusCode>
  Open(const std::string& path);

  [[nodiscard]] std::size_t Rows() const { return rows_; }

  // Decode the rows named by `rows` into `out`, one 160-value row per
  // index, head-slowest (row i occupies out[i*160, (i+1)*160)). An
  // out-of-range index or a size mismatch is InvalidArgument.
  //
  // Usage:
  //   if (auto g = table->Gather(rows16, emb); !g) return g.error();
  [[nodiscard]] std::expected<void, StatusCode> Gather(
      std::span<const std::uint32_t> rows, std::span<float> out) const;

 private:
  std::shared_ptr<core::MappedFile> file_;
  std::span<const std::byte> data_;
  std::size_t rows_ = 0;
};

}  // namespace tessera::models::qwen4exp
