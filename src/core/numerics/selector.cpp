#include "core/numerics/selector.hpp"

#include <cmath>

namespace tessera::core {

std::expected<void, StatusCode> SelectorEdgeScoreRef(
    std::span<const float> predecessor_codebook,
    std::span<const float> successor_codebook, std::span<const float> hidden,
    std::span<const std::int32_t> candidate_ids,
    std::span<const std::int32_t> anchor_ids, std::span<const float> unary,
    std::span<float> out, std::size_t batch, std::size_t seq,
    std::size_t top_k, std::size_t rank, std::size_t vocab) {
  if (batch == 0 || seq == 0 || top_k == 0 || rank == 0 || vocab == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t positions = batch * seq;
  if (predecessor_codebook.size() != vocab * rank ||
      successor_codebook.size() != vocab * rank ||
      hidden.size() != positions * rank ||
      candidate_ids.size() != positions * top_k ||
      anchor_ids.size() != positions || unary.size() != positions * top_k ||
      out.size() != positions * top_k * top_k) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t pos = 0; pos < positions; ++pos) {
    const std::size_t l = pos % seq;
    const std::size_t cand_row = pos * top_k;
    const std::size_t hidden_row = pos * rank;
    for (std::size_t p = 0; p < top_k; ++p) {
      const std::int32_t pred_id =
          l == 0 ? anchor_ids[pos] : candidate_ids[cand_row - top_k + p];
      const float* pred = &predecessor_codebook[
          static_cast<std::size_t>(pred_id) * rank];
      const float* hid = &hidden[hidden_row];
      for (std::size_t c = 0; c < top_k; ++c) {
        const std::int32_t succ_id = candidate_ids[cand_row + c];
        const float* succ =
            &successor_codebook[static_cast<std::size_t>(succ_id) * rank];
        float dot = 0.0f;
        for (std::size_t r = 0; r < rank; ++r) {
          dot = std::fma(pred[r] * hid[r], succ[r], dot);
        }
        out[(pos * top_k + p) * top_k + c] = unary[cand_row + p] + dot;
      }
    }
  }
  return {};
}

}  // namespace tessera::core
