#include "core/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace tessera::core {

namespace {

std::uint32_t ArgMax(std::span<const float> values) {
  std::uint32_t best = 0;
  for (std::size_t i = 1; i < values.size(); ++i) {
    if (values[i] > values[best]) {
      best = static_cast<std::uint32_t>(i);
    }
  }
  return best;
}

}  // namespace

std::uint32_t SampleToken(std::span<const float> logits,
                          const SamplingOptions& options,
                          std::span<const std::uint32_t> history,
                          std::mt19937_64& rng) {
  std::vector<float> scores(logits.begin(), logits.end());
  if (options.repetition_penalty != 1.0f) {
    for (const std::uint32_t token : history) {
      if (token >= scores.size()) {
        continue;
      }
      scores[token] = scores[token] > 0.0f
                          ? scores[token] / options.repetition_penalty
                          : scores[token] * options.repetition_penalty;
    }
  }
  if (options.presence_penalty != 0.0f) {
    std::vector<std::uint32_t> seen(history.begin(), history.end());
    std::sort(seen.begin(), seen.end());
    seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
    for (const std::uint32_t token : seen) {
      if (token < scores.size()) {
        scores[token] -= options.presence_penalty;
      }
    }
  }
  if (options.temperature <= 0.0f) {
    return ArgMax(scores);
  }
  for (float& value : scores) {
    value /= options.temperature;
  }
  const float max_logit = *std::max_element(scores.begin(), scores.end());
  if (options.min_p > 0.0f) {
    const float threshold = max_logit + std::log(options.min_p);
    for (float& value : scores) {
      if (value < threshold) {
        value = -std::numeric_limits<float>::infinity();
      }
    }
  }
  std::vector<std::uint32_t> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  const std::size_t limit =
      (options.top_k > 0 &&
       static_cast<std::size_t>(options.top_k) < indices.size())
          ? static_cast<std::size_t>(options.top_k)
          : indices.size();
  const auto greater = [&scores](std::uint32_t a, std::uint32_t b) {
    return scores[a] > scores[b];
  };
  if (limit < indices.size()) {
    std::nth_element(indices.begin(), indices.begin() + limit, indices.end(),
                     greater);
    indices.resize(limit);
  }
  std::sort(indices.begin(), indices.end(), greater);
  // Softmax over the retained set, then the nucleus cut.
  double denom = 0.0;
  for (const std::uint32_t i : indices) {
    denom += std::exp(static_cast<double>(scores[i] - max_logit));
  }
  if (!(denom > 0.0)) {
    return ArgMax(scores);
  }
  std::vector<double> weights(indices.size());
  double cumulative = 0.0;
  std::size_t kept = 0;
  for (std::size_t pos = 0; pos < indices.size(); ++pos) {
    weights[pos] = std::exp(static_cast<double>(scores[indices[pos]] -
                                                max_logit)) /
                   denom;
    cumulative += weights[pos];
    ++kept;
    if (options.top_p < 1.0f && cumulative >= options.top_p) {
      break;
    }
  }
  weights.resize(kept);
  std::discrete_distribution<std::size_t> distribution(weights.begin(),
                                                       weights.end());
  return indices[distribution(rng)];
}

}  // namespace tessera::core
