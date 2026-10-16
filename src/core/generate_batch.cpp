#include "tessera/engine.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "core/generate_helpers.hpp"
#include "core/sampling.hpp"

namespace tessera {

namespace {

// One in-flight sequence: its own cache, the prompt fed chunk by chunk,
// the tokens fed so far, the tokens produced, the resolved stop set, its
// RNG and the pending token. Greedy sequences keep the device argmax;
// sampled ones draw from their logits.
struct BatchSequence {
  core::DecodeCache cache;
  std::vector<std::uint32_t> prompt;
  std::size_t prefilled = 0;  // prompt rows committed to the cache
  std::size_t chunk = 0;      // prefill rows per scheduler round
  std::vector<std::uint32_t> history;
  std::vector<std::uint32_t> produced;
  std::vector<std::uint32_t> stops;
  std::size_t budget = 0;   // completion tokens allowed
  std::uint32_t next = 0;   // the token ready to emit
  bool has_next = false;    // next holds a token (prefill finished)
  bool sample = false;
  SamplingOptions sampling;
  std::mt19937_64 rng;
  bool active = false;
};

// Pick the next token: the per-sequence seeded draw when sampling, the
// device argmax otherwise.
std::uint32_t Pick(BatchSequence& sequence, std::span<const float> logits) {
  return PickToken(logits, sequence.sample, sequence.sampling,
                   sequence.history, sequence.rng);
}

}  // namespace

std::expected<std::vector<std::vector<std::uint32_t>>, StatusCode>
Engine::GenerateBatch(Model& model, std::span<const GenerateOptions> requests) {
  std::vector<BatchSequence> sequences(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const GenerateOptions& options = requests[i];
    BatchSequence& sequence = sequences[i];
    sequence.cache.kv_type = options.kv_type;
    sequence.cache.kv_offload = options.kv_offload;
    sequence.cache.tuning.mxfp4_split_target = options.mxfp4_split_target;
    sequence.cache.tuning.mxfp4_split_cap = options.mxfp4_split_cap;
    sequence.cache.tuning.attention_split = options.attention_split;
    sequence.sample = options.sample;
    sequence.sampling = options.sampling;
    sequence.rng.seed(options.seed);
    std::vector<std::uint32_t> prompt = options.prompt_tokens;
    if (prompt.empty()) {
      prompt.push_back(options.first_token);
    }
    if (prompt.size() > model.MaxContextLength()) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    sequence.budget =
        model.EffectiveMaxTokens(prompt.size(), options.max_completion_tokens);
    sequence.stops = StopSet(model, options);
    sequence.prompt = std::move(prompt);
    sequence.chunk = ResolvePrefillChunkTokens(
        prefill_chunk_tokens_, options.prefill_chunk_tokens,
        model.MaxContextLength());
  }
  // Scheduler rounds, Sarathi-style: every round first advances each
  // unfinished prefill by one chunk, then steps every sequence ready to
  // decode. A long prompt prefills chunk by chunk while short sequences
  // already decode, so no sequence stalls behind another's full prefill;
  // each sequence's output still equals a separate Generate. The greedy
  // decode advances are batched into one multi-sequence forward when the
  // architecture supports it; sampled ones draw per sequence.
  bool any_work = true;
  while (any_work) {
    any_work = false;
    for (std::size_t i = 0; i < sequences.size(); ++i) {
      BatchSequence& sequence = sequences[i];
      if (sequence.prefilled >= sequence.prompt.size()) {
        continue;
      }
      any_work = true;
      const std::size_t step = sequence.chunk > 0
                                   ? sequence.chunk
                                   : sequence.prompt.size() - sequence.prefilled;
      const std::size_t len =
          std::min(step, sequence.prompt.size() - sequence.prefilled);
      auto logits = core::PrefillTokens(
          *backend_, model, sequence.cache,
          std::span<const std::uint32_t>(sequence.prompt).subspan(
              sequence.prefilled, len),
          nullptr, nullptr, sequence.chunk, nullptr);
      if (!logits) {
        diagnostics_.Warn("engine", std::string("batch prefill failed: ") +
                                        std::string(ToString(logits.error())));
        return std::unexpected(logits.error());
      }
      sequence.prefilled += len;
      if (const auto& hook = requests[i].prefill_progress; hook) {
        hook(sequence.prefilled, sequence.prompt.size());
      }
      if (sequence.prefilled >= sequence.prompt.size()) {
        sequence.history = sequence.prompt;
        sequence.next = Pick(sequence, *logits);
        sequence.has_next = true;
        sequence.active = sequence.budget > 0;
      }
    }
    std::vector<std::size_t> greedy_advance;
    std::vector<std::size_t> sampled_advance;
    for (std::size_t i = 0; i < sequences.size(); ++i) {
      BatchSequence& sequence = sequences[i];
      if (!sequence.active) {
        continue;
      }
      any_work = true;
      if (core::detail::IsStopToken(sequence.next, sequence.stops) ||
          sequence.produced.size() >= sequence.budget) {
        sequence.active = false;
        continue;
      }
      sequence.produced.push_back(sequence.next);
      sequence.history.push_back(sequence.next);
      if (sequence.produced.size() >= sequence.budget) {
        sequence.active = false;
        continue;
      }
      (sequence.sample ? sampled_advance : greedy_advance).push_back(i);
    }
    if (!greedy_advance.empty()) {
      std::vector<core::DecodeCache*> caches;
      std::vector<std::uint32_t> tokens_in;
      caches.reserve(greedy_advance.size());
      tokens_in.reserve(greedy_advance.size());
      for (std::size_t i : greedy_advance) {
        caches.push_back(&sequences[i].cache);
        tokens_in.push_back(sequences[i].next);
      }
      auto advanced = AdvanceGreedyCaches(*backend_, model, diagnostics_,
                                          caches, tokens_in);
      if (!advanced) {
        return std::unexpected(advanced.error());
      }
      for (std::size_t k = 0; k < greedy_advance.size(); ++k) {
        sequences[greedy_advance[k]].next = (*advanced)[k];
      }
    }
    for (std::size_t i : sampled_advance) {
      auto logits =
          core::DecodeLogits(*backend_, model, sequences[i].cache,
                             sequences[i].next);
      if (!logits) {
        diagnostics_.Warn("engine", std::string("batch decode failed: ") +
                                        std::string(ToString(logits.error())));
        return std::unexpected(logits.error());
      }
      sequences[i].next = Pick(sequences[i], *logits);
    }
  }
  std::vector<std::vector<std::uint32_t>> output;
  output.reserve(sequences.size());
  for (BatchSequence& sequence : sequences) {
    output.push_back(std::move(sequence.produced));
  }
  return output;
}

}  // namespace tessera
