#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "tessera/types.hpp"

namespace tessera::spec {

// The DFlash2 draft configuration (config.json): the transformer geometry
// plus the dflash_config block (dynamic convolution, candidate selector,
// target layer ids and the draft block).
struct DFlash2Config {
  std::size_t hidden_size = 0;
  std::size_t num_layers = 0;
  std::size_t num_heads = 0;
  std::size_t num_kv_heads = 0;
  std::size_t head_dim = 0;
  std::size_t intermediate_size = 0;
  std::size_t vocab_size = 0;
  double rms_norm_eps = 0.0;
  double rope_theta = 0.0;
  std::size_t sliding_window = 0;
  std::size_t block_size = 0;
  std::size_t conv_group_size = 0;
  std::size_t conv_kernel_size = 0;
  std::uint32_t mask_token_id = 0;
  std::size_t selector_rank = 0;
  std::size_t selector_top_k = 0;
  std::vector<std::uint32_t> target_layer_ids;
  std::vector<std::string> layer_types;
};

// Parse a DFlash2 draft config.json. MalformedFile when a required field
// is missing, not of the right type, zero, or inconsistent (for example
// conv_group_size not dividing hidden_size, or the layer_types length not
// matching num_hidden_layers).
[[nodiscard]] std::expected<DFlash2Config, StatusCode> ParseDFlash2Config(
    std::string_view text);

// Read `dir/config.json` and parse it. FileNotFound for a missing file,
// MalformedFile for a bad field.
[[nodiscard]] std::expected<DFlash2Config, StatusCode> LoadDFlash2Config(
    const std::filesystem::path& dir);

}  // namespace tessera::spec
