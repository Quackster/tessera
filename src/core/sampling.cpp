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

// The logits with the repetition and presence penalties applied over
// `history`. Shared by every sampling entry point so the score transform
// has one implementation.
std::vector<float> PenalizedScores(std::span<const float> logits,
                                   const SamplingOptions& options,
                                   std::span<const std::uint32_t> history) {
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
  return scores;
}

// The kept token ids and their normalized weights after temperature, min_p,
// top_k and top_p. `scores` is modified in place. Empty only when the row
// is empty; a temperature-0 caller must not reach here (it takes the argmax
// without consuming the rng).
struct KeptSet {
  std::vector<std::uint32_t> ids;
  std::vector<double> probs;
};

KeptSet TruncateAndNormalize(std::vector<float>& scores,
                             const SamplingOptions& options) {
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
  KeptSet kept;
  if (!(denom > 0.0)) {
    kept.ids.push_back(ArgMax(scores));
    kept.probs.push_back(1.0);
    return kept;
  }
  double cumulative = 0.0;
  for (std::size_t pos = 0; pos < indices.size(); ++pos) {
    const double weight =
        std::exp(static_cast<double>(scores[indices[pos]] - max_logit)) /
        denom;
    cumulative += weight;
    kept.ids.push_back(indices[pos]);
    kept.probs.push_back(weight);
    if (options.top_p < 1.0f && cumulative >= options.top_p) {
      break;
    }
  }
  return kept;
}

}  // namespace

std::uint32_t SampleToken(std::span<const float> logits,
                          const SamplingOptions& options,
                          std::span<const std::uint32_t> history,
                          std::mt19937_64& rng) {
  std::vector<float> scores = PenalizedScores(logits, options, history);
  if (options.temperature <= 0.0f) {
    return ArgMax(scores);
  }
  KeptSet kept = TruncateAndNormalize(scores, options);
  std::discrete_distribution<std::size_t> distribution(kept.probs.begin(),
                                                       kept.probs.end());
  return kept.ids[distribution(rng)];
}

std::vector<double> BuildDenseDistribution(
    std::span<const float> logits, const SamplingOptions& options,
    std::span<const std::uint32_t> history) {
  std::vector<float> scores = PenalizedScores(logits, options, history);
  std::vector<double> dense(logits.size(), 0.0);
  if (options.temperature <= 0.0f) {
    dense[ArgMax(scores)] = 1.0;
    return dense;
  }
  KeptSet kept = TruncateAndNormalize(scores, options);
  for (std::size_t i = 0; i < kept.ids.size(); ++i) {
    dense[kept.ids[i]] = kept.probs[i];
  }
  return dense;
}

std::uint32_t SampleFromDense(const std::vector<double>& probs,
                              std::mt19937_64& rng) {
  std::discrete_distribution<std::size_t> distribution(probs.begin(),
                                                       probs.end());
  return static_cast<std::uint32_t>(distribution(rng));
}

RejectionResult RejectionStep(std::span<const double> target,
                              std::span<const double> draft, std::uint32_t x,
                              std::mt19937_64& rng) {
  RejectionResult result;
  const double draft_x = x < draft.size() ? draft[x] : 0.0;
  const double target_x = x < target.size() ? target[x] : 0.0;
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  const double draw = uniform(rng);
  if (draft_x > 0.0 && draw < std::min(1.0, target_x / draft_x)) {
    result.accepted = true;
    result.token = x;
    return result;
  }
  // Residual max(0, target - draft), renormalized; sample the replacement.
  std::vector<double> residual(target.size(), 0.0);
  double sum = 0.0;
  for (std::size_t v = 0; v < target.size(); ++v) {
    const double draft_v = v < draft.size() ? draft[v] : 0.0;
    const double delta = target[v] - draft_v;
    if (delta > 0.0) {
      residual[v] = delta;
      sum += delta;
    }
  }
  if (!(sum > 0.0)) {
    residual.assign(target.begin(), target.end());
  }
  result.accepted = false;
  result.token = SampleFromDense(residual, rng);
  return result;
}

}  // namespace tessera::core
