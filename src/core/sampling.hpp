#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "tessera/engine.hpp"

namespace tessera::core {

// Draw one token from a logits row. Applies the penalties over `history`
// (each history token at most once for the presence penalty), then
// temperature, min_p, top_k and top_p, and samples. Deterministic for a
// given `rng` state. Returns the argmax when temperature is 0 or the
// filtered set is empty. `logits` must be nonempty.
[[nodiscard]] std::uint32_t SampleToken(
    std::span<const float> logits, const SamplingOptions& options,
    std::span<const std::uint32_t> history, std::mt19937_64& rng);

// The same sampling distribution as SampleToken, materialized densely over
// the vocabulary (zero outside the kept set, normalized inside). A point
// mass at the argmax when temperature is 0.
[[nodiscard]] std::vector<double> BuildDenseDistribution(
    std::span<const float> logits, const SamplingOptions& options,
    std::span<const std::uint32_t> history);

// Draw one token from a dense probability vector.
[[nodiscard]] std::uint32_t SampleFromDense(const std::vector<double>& probs,
                                            std::mt19937_64& rng);

// The outcome of one rejection step: `accepted` is the draft token kept,
// otherwise `token` is the replacement drawn from the residual.
struct RejectionResult {
  bool accepted = false;
  std::uint32_t token = 0;
};

// One rejection step of speculative sampling. `target` and `draft` are the
// target and draft distributions over the vocabulary (each normalized or
// zero). The drafted token `x` is accepted with probability
// min(1, target(x)/draft(x)); on rejection the replacement is drawn from
// the residual max(0, target - draft). Lossless for any draft distribution:
// the emitted token is distributed as the target.
[[nodiscard]] RejectionResult RejectionStep(
    std::span<const double> target, std::span<const double> draft,
    std::uint32_t x, std::mt19937_64& rng);

// The outcome of a sampled speculative verification: the leading drafts
// that matched the target's drawn token, and the target's drawn token at
// the first mismatch (or the bonus draw when every draft matched).
struct SampledMatch {
  std::size_t accepted = 0;
  std::uint32_t next_token = 0;
};

// Accept the leading drafts that match the token drawn from the target at
// each position. `row(i)` returns the vocab-wide logits that score draft[i]
// for i < draft.size(); `bonus()` returns the logits for the extra token
// after every draft matched. `history` is the penalty history before the
// verified window; each accepted token extends it. `rng` advances once per
// scored row up to the first mismatch, plus the bonus draw when all
// matched, so the emitted prefix equals plain sampling for the same seed.
template <typename RowFn, typename BonusFn>
SampledMatch AcceptSampledPrefix(std::span<const std::uint32_t> draft,
                                 const SamplingOptions& options,
                                 std::span<const std::uint32_t> history,
                                 std::mt19937_64& rng, RowFn row,
                                 BonusFn bonus) {
  std::vector<std::uint32_t> hist(history.begin(), history.end());
  SampledMatch match;
  bool all = true;
  for (std::size_t i = 0; i < draft.size(); ++i) {
    const std::uint32_t y = SampleToken(row(i), options, hist, rng);
    if (y != draft[i]) {
      match.next_token = y;
      all = false;
      break;
    }
    ++match.accepted;
    hist.push_back(y);
  }
  if (all) {
    match.next_token = SampleToken(bonus(), options, hist, rng);
  }
  return match;
}

// Rejection-sampling verification (speculative sampling). `prob_row(i, hist)`
// returns the target's dense distribution at scored row i, `draft_row(i)` the
// draft's dense distribution the token `draft[i]` was drawn from, and
// `bonus(hist)` the target's distribution for the extra token after every
// draft matched. Each accepted draft extends the penalty history. Lossless:
// the emitted prefix is distributed as plain sampling of the target.
template <typename ProbFn, typename DraftFn, typename BonusFn>
SampledMatch AcceptRejectionPrefix(std::span<const std::uint32_t> draft,
                                   std::span<const std::uint32_t> history,
                                   std::mt19937_64& rng, ProbFn prob_row,
                                   DraftFn draft_row, BonusFn bonus) {
  std::vector<std::uint32_t> hist(history.begin(), history.end());
  SampledMatch match;
  bool all = true;
  for (std::size_t i = 0; i < draft.size(); ++i) {
    const std::vector<double> target = prob_row(i, hist);
    const std::span<const double> draft_dist = draft_row(i);
    const RejectionResult step =
        RejectionStep(target, draft_dist, draft[i], rng);
    if (!step.accepted) {
      match.next_token = step.token;
      all = false;
      break;
    }
    ++match.accepted;
    hist.push_back(draft[i]);
  }
  if (all) {
    match.next_token = SampleFromDense(bonus(hist), rng);
  }
  return match;
}

}  // namespace tessera::core
