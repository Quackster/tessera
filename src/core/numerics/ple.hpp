#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "tessera/types.hpp"

// The PLE block's post-ops: everything after the key and value projections.
// The block reads the gathered n-gram rows, projects a key and a value,
// gates the value per residual stream, normalizes it, runs a dilated
// causal convolution over a short normalized history, and adds the result
// back into the residual stack. Ported from ../Strata
// src/kernels/cuda/native_ple_postops.cu.
//
// Layouts (device data): `key`, `gated`, `normalized`, `conv` and the
// history are hc*n_embd wide; `value` is n_embd; `gate` is hc; the
// history is `history_rows` rows per channel, channel-major
// (history[channel*history_rows + row]); the conv taps are f16 stored two
// bytes each, four per channel.

namespace tessera::core {

// Geometry of one PLE post-ops call. `n_embd` is the value width and the
// per-stream norm width; `hc` is the residual stream count; `history_rows`
// is (conv_kernel - 1) * ngram_size; `conv_taps` is conv_kernel.
struct PlePostopsShapes {
  std::size_t n_embd = 0;
  std::size_t hc = 0;
  std::size_t history_rows = 0;
  std::size_t conv_taps = 0;
};

// Host reference. `projected_key` and `hidden` are hc*n_embd, `value` is
// n_embd, `history` and `history_out` are (hc*n_embd) * history_rows
// (one channel per stack element, channel-major); `norm_key`,
// `norm_query`, `norm_conv` are hc*n_embd f32; `conv1d` is
// hc*n_embd*conv_taps f16. `result` (hc*n_embd) is `hidden` plus the
// gated value plus the activated convolution; `gate` (hc) is the per-
// stream gate. All spans must match; a mismatch is InvalidArgument.
[[nodiscard]] std::expected<void, StatusCode> PlePostopsRef(
    std::span<const float> projected_key, std::span<const float> hidden,
    std::span<const float> value, std::span<const float> history,
    std::span<const float> norm_key, std::span<const float> norm_query,
    std::span<const float> norm_conv, std::span<const std::byte> conv1d,
    const PlePostopsShapes& s, double eps, std::span<float> result,
    std::span<float> gate, std::span<float> history_out);

}  // namespace tessera::core
