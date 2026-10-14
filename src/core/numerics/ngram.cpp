#include "core/numerics/ngram.hpp"

namespace tessera::core {

namespace {

// Sentinel predecessor (llama.cpp LLAMA_TOKEN_NULL). Token 0 is a real
// token, so the test is `t < 0`, not `t == 0`.
constexpr std::int32_t kTokenNull = -1;

}  // namespace

std::uint64_t NgramMixed(std::span<const std::int64_t> context,
                         std::span<const std::uint64_t> mult) {
  if (context.empty() || mult.size() < context.size()) {
    return 0;
  }
  std::uint64_t mixed =
      static_cast<std::uint64_t>(context[0]) * mult[0];
  for (std::size_t j = 1; j < context.size(); ++j) {
    mixed ^= static_cast<std::uint64_t>(context[j]) * mult[j];
  }
  return mixed;
}

std::expected<void, StatusCode> NgramRows(
    std::span<const std::int32_t> tokens, std::span<const std::int32_t> prev,
    const PleParams& ple, std::span<std::uint32_t> out) {
  const std::size_t ngram = ple.ngram_size;
  const std::size_t heads_per = ple.heads_per_ngram;
  if (ngram < 2 || heads_per == 0 ||
      ple.layer_multipliers.size() != ngram ||
      ple.head_offsets.size() != heads_per * (ngram - 1) ||
      ple.head_vocab_sizes.size() != ple.head_offsets.size()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t heads = heads_per * (ngram - 1);
  const std::size_t n_prev = ngram - 1;
  if (out.size() != tokens.size() * heads) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (prev.size() != tokens.size() * n_prev) {
    return std::unexpected(StatusCode::InvalidArgument);
  }

  for (std::size_t i = 0; i < tokens.size(); ++i) {
    std::int64_t context[/*ngram*/ 8] = {};
    if (ngram > 8) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    context[0] = tokens[i];
    bool cut = false;
    for (std::size_t s = 1; s < ngram; ++s) {
      // `prev` is OLDEST FIRST, so predecessor s back is entry
      // (n_prev - s): s = 1 reads the newest.
      const std::int32_t t =
          cut ? kTokenNull : prev[i * n_prev + (n_prev - s)];
      cut = cut || t < 0 || t == static_cast<std::int32_t>(ple.eos_token_id);
      context[s] = cut ? static_cast<std::int64_t>(ple.eos_token_id) : t;
    }
    for (std::size_t n = 2; n <= ngram; ++n) {
      const std::uint64_t mixed =
          NgramMixed(std::span<const std::int64_t>(context, n),
                     std::span<const std::uint64_t>(ple.layer_multipliers));
      const std::size_t base = (n - 2) * heads_per;
      for (std::size_t g = 0; g < heads_per; ++g) {
        const std::size_t h = base + g;
        out[i * heads + h] = static_cast<std::uint32_t>(
            mixed % ple.head_vocab_sizes[h] + ple.head_offsets[h]);
      }
    }
  }
  return {};
}

}  // namespace tessera::core
