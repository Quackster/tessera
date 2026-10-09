#include "spec/dflash2_candidates.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

namespace tessera::spec {

std::expected<void, StatusCode> DraftCandidates(
    std::span<const float> logits, std::span<std::uint32_t> ids,
    std::span<float> unary, std::size_t rows, std::size_t vocab,
    std::size_t top_k) {
  if (rows == 0 || vocab == 0 || top_k == 0 || top_k > vocab) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (logits.size() != rows * vocab || ids.size() != rows * top_k ||
      unary.size() != rows * top_k) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<std::uint32_t> row_ids(vocab);
  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = logits.data() + r * vocab;
    // Reset every element each row: keeping the vector at size `vocab` (rather
    // than resizing to `top_k`) is what lets the next row rank the whole
    // vocabulary. Sorting only the `top_k` prefix leaves the rest untouched.
    std::iota(row_ids.begin(), row_ids.end(), 0);
    const auto greater = [row](std::uint32_t a, std::uint32_t b) {
      return row[a] > row[b];
    };
    std::nth_element(row_ids.begin(), row_ids.begin() + top_k, row_ids.end(),
                     greater);
    std::sort(row_ids.begin(), row_ids.begin() + top_k, greater);
    for (std::size_t k = 0; k < top_k; ++k) {
      ids[r * top_k + k] = row_ids[k];
      unary[r * top_k + k] = row[row_ids[k]];
    }
  }
  return {};
}

}  // namespace tessera::spec
