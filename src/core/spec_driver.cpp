#include "core/spec_driver.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/generate_helpers.hpp"
#include "core/think_budget.hpp"
#include "core/timing.hpp"

// The speculative decode driver (see spec_driver.hpp). This is the one
// token loop for greedy and speculative decode; the target seam supplies
// the forwards (AGENTS.md rule 2).

namespace tessera::core {

namespace {

// Milliseconds per second, for prompt tok/s from fractional timers.
constexpr double kMillisPerSecond = 1000.0;

// Describe a logits or token step error. A non-finite logit means a corrupt
// weight read or device buffer, not a generic step failure.
std::string StepError(StatusCode code) {
  if (code == StatusCode::NonFinite) {
    return "non-finite logits (corrupt model weights or device memory); "
           "stopping generation";
  }
  return std::string(ToString(code));
}

}  // namespace

std::expected<GenerateOutcome, StatusCode> RunDecode(
    SpecTarget& target, SpeculativeStrategy* strategy,
    const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token) {
  log::Diagnostics& diagnostics = target.diagnostics();
  Model& model = target.model();
  if (options.sample &&
      ValidateSamplingOptions(options.sampling, diagnostics) !=
          StatusCode::Ok) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<std::uint32_t> prompt = options.prompt_tokens;
  if (prompt.empty()) {
    prompt.push_back(options.first_token);
  }
  const std::size_t max_completion_tokens =
      model.EffectiveMaxTokens(prompt.size(), options.max_completion_tokens);
  if (max_completion_tokens == 0) {
    return GenerateOutcome{};
  }
  if (prompt.size() > model.MaxContextLength()) {
    diagnostics.Warn("engine", std::string("prompt of ") +
                                     std::to_string(prompt.size()) +
                                     " token(s) exceeds the " +
                                     std::to_string(model.MaxContextLength()) +
                                     " token context; reject without running "
                                     "prefill so the device stays alive");
    return std::unexpected(StatusCode::InvalidArgument);
  }
  // A sampled request must not run through a greedy-only drafter: the greedy
  // accept rule would decode the sampled request as greedy. Drop the
  // strategy for this generation and sample normally.
  if (strategy != nullptr && options.sample && !strategy->SupportsSampling()) {
    diagnostics.Info("engine", "request samples: strategy '" +
                                   std::string(strategy->Name()) +
                                   "' is greedy only; decoding without it");
    strategy = nullptr;
  }
  if (auto prepared = target.Prepare(strategy, options); !prepared) {
    diagnostics.Warn("engine", "strategy prepare failed: " +
                                   std::string(ToString(prepared.error())));
    return std::unexpected(prepared.error());
  }
  std::vector<float> hidden;
  std::vector<float> first_logits;
  double compute_ms = 0.0;
  const auto prefill_started = std::chrono::steady_clock::now();
  auto prefilled =
      target.Prefill(prompt, options, &hidden, &first_logits, &compute_ms);
  if (!prefilled) {
    diagnostics.Warn("engine", std::string("prefill failed: ") +
                                   StepError(prefilled.error()));
    return std::unexpected(prefilled.error());
  }
  const double prefill_ms_exact =
      core::ElapsedMsDouble(prefill_started, std::chrono::steady_clock::now());
  const long long prefill_ms = static_cast<long long>(prefill_ms_exact);
  if (options.progress_every > 0 && prefill_ms > 0) {
    const long long prompt_tps =
        static_cast<long long>(prompt.size()) * 1000 / prefill_ms;
    diagnostics.Info("engine",
                     std::string("prefill: ") + std::to_string(prompt.size()) +
                         " prompt token(s) in " + std::to_string(prefill_ms) +
                         " ms (" + std::to_string(prompt_tps) +
                         " prompt tok/s)");
    if (compute_ms > 0.0) {
      const long long compute_ms_int =
          static_cast<long long>(std::llround(compute_ms));
      const long long compute_tps = static_cast<long long>(
          std::llround(prompt.size() * kMillisPerSecond / compute_ms));
      diagnostics.Info("engine",
                       std::string("prefill compute: ") +
                           std::to_string(prompt.size()) +
                           " prompt token(s) in " + std::to_string(compute_ms_int) +
                           " ms (" + std::to_string(compute_tps) +
                           " prompt tok/s)");
    }
  }
  const auto decode_started = std::chrono::steady_clock::now();
  std::mt19937_64 rng(options.seed);
  std::vector<std::uint32_t> history = prompt;
  const std::vector<std::uint32_t> stops = StopSet(model, options);
  const auto pick = [&](std::span<const float> logits) {
    return PickToken(logits, options.sample, options.sampling, history, rng);
  };
  std::size_t block = strategy != nullptr ? strategy->DraftBlock() : 0;
  if (options.draft_tokens > 0 && options.draft_tokens < block) {
    block = options.draft_tokens;
  }
  std::vector<std::uint32_t> drafts(block);
  std::uint32_t next = pick(first_logits);
  std::uint64_t position = prompt.size();
  // A folding drafter drafts before the target sees the anchor, pairing the
  // anchor token with the prefill's final hidden (the hidden at one position
  // before). Seed it here; the loop re-records it after each verify.
  if (strategy != nullptr && block > 0 && strategy->NeedsPrefillAnchor() &&
      !hidden.empty()) {
    auto seeded = strategy->OnAnchor(target.backend(), model, target.cache(),
                                     next, position, hidden);
    if (!seeded) {
      return std::unexpected(seeded.error());
    }
  }
  std::size_t produced = 0;
  std::size_t spec_proposed = 0;
  std::size_t spec_accepted = 0;
  std::size_t spec_steps = 0;
  std::size_t loop_steps = 0;
  bool stopped = false;
  bool aborted = false;
  auto [think, think_close] =
      ResolveThinkBudget(model, options.max_thinking_tokens);
  if (options.max_thinking_tokens > 0 && !think.engaged()) {
    diagnostics.Warn("engine", "thinking budget ignored: the think tags do "
                               "not resolve on this model; thinking stays "
                               "unlimited");
  }
  for (std::uint32_t id : prompt) {
    think.noteEmitted(id);
  }
  const auto emit = [&](std::uint32_t token) {
    think.noteEmitted(token);
    return on_token(token);
  };
  // Feed and emit the think-close ids, keeping the target, the drafter
  // context, the sampling history and the position in step.
  auto force_close = [&]()
      -> std::expected<std::optional<std::uint32_t>, StatusCode> {
    std::vector<float> close_logits;
    for (std::uint32_t id : think_close) {
      std::expected<std::vector<float>, StatusCode> logits;
      logits = target.LogitsStep(id, &hidden);
      if (!logits) {
        diagnostics.Warn("engine", std::string("think force-close failed: ") +
                                       StepError(logits.error()));
        return std::unexpected(logits.error());
      }
      if (strategy != nullptr && block > 0) {
        auto anchored = strategy->OnAnchor(target.backend(), model,
                                           target.cache(), id, position,
                                           hidden);
        if (!anchored) {
          return std::unexpected(anchored.error());
        }
        if (strategy->FoldsAnchor()) {
          auto committed = strategy->Commit(target.backend(), model,
                                            target.cache(), 0);
          if (!committed) {
            return std::unexpected(committed.error());
          }
        }
      }
      think.noteEmitted(id);
      if (!on_token(id)) {
        aborted = true;
        return std::optional<std::uint32_t>{};
      }
      history.push_back(id);
      ++produced;
      if (produced >= max_completion_tokens) {
        return std::optional<std::uint32_t>{};
      }
      ++position;
      close_logits = std::move(*logits);
    }
    return pick(close_logits);
  };
  while (produced < max_completion_tokens) {
    ++loop_steps;
    if (think.closeOwed()) {
      diagnostics.Info("engine", "thinking budget (" +
                                     std::to_string(
                                         options.max_thinking_tokens) +
                                     " token(s)) exceeded: think block "
                                     "force-closed, decoding continues");
      auto forced = force_close();
      if (!forced) {
        return std::unexpected(forced.error());
      }
      if (!*forced) {
        break;
      }
      next = **forced;
    }
    if (core::detail::IsStopToken(next, stops)) {
      stopped = true;
      break;
    }
    if (!emit(next)) {
      aborted = true;
      break;
    }
    ++produced;
    if (produced >= max_completion_tokens) {
      break;
    }
    history.push_back(next);
    // A folding strategy rides the anchor into the verify batch, so the trunk
    // is read once per step. The drafter context already excludes this anchor:
    // OnAnchor runs after the verify and appends the anchor it produced.
    if (strategy != nullptr && block > 0 && strategy->FoldsAnchor()) {
      auto count = strategy->Draft(target.backend(), model, target.cache(), {},
                                   next, drafts);
      if (!count) {
        return std::unexpected(count.error());
      }
      if (*count == 0) {
        // No usable proposal: decode the anchor alone and take its greedy
        // token. The drafter context still gains the anchor below.
        auto logits = target.LogitsStep(next, &hidden);
        if (!logits) {
          return std::unexpected(logits.error());
        }
        const std::uint32_t picked = pick(*logits);
        auto anchored = strategy->OnAnchor(target.backend(), model,
                                           target.cache(), picked,
                                           position + 1, hidden);
        if (!anchored) {
          return std::unexpected(anchored.error());
        }
        auto committed = strategy->Commit(target.backend(), model,
                                          target.cache(), 0);
        if (!committed) {
          return std::unexpected(committed.error());
        }
        next = picked;
        ++position;
        continue;
      }
      std::span<const std::uint32_t> proposal(drafts.data(), *count);
      auto verify = target.Verify(proposal, {}, next, &hidden);
      if (!verify) {
        return std::unexpected(verify.error());
      }
      auto anchored = strategy->OnAnchor(
          target.backend(), model, target.cache(), verify->next_token,
          position + 1 + verify->accepted, hidden);
      if (!anchored) {
        return std::unexpected(anchored.error());
      }
      auto committed = strategy->Commit(target.backend(), model,
                                        target.cache(), verify->accepted);
      if (!committed) {
        return std::unexpected(committed.error());
      }
      spec_proposed += *count;
      spec_accepted += verify->accepted;
      ++spec_steps;
      for (std::size_t i = 0; i < verify->accepted; ++i) {
        if (produced >= max_completion_tokens) {
          break;
        }
        if (core::detail::IsStopToken(drafts[i], stops)) {
          stopped = true;
          break;
        }
        if (!emit(drafts[i])) {
          aborted = true;
          break;
        }
        history.push_back(drafts[i]);
        ++produced;
      }
      if (stopped || aborted || produced >= max_completion_tokens) {
        break;
      }
      next = verify->next_token;
      position += 1 + verify->accepted;
      continue;
    }
    // Greedy without a drafter: argmax on the device, so the whole
    // vocabulary is not downloaded and the hidden state is not read back.
    if (strategy == nullptr && !options.sample) {
      auto token = target.GreedyStep(next);
      if (!token) {
        diagnostics.Warn("engine", std::string("generation step failed: ") +
                                       StepError(token.error()));
        return std::unexpected(token.error());
      }
      next = *token;
      ++position;
      continue;
    }
    // Advance the target by the just-emitted token. `hidden` describes its
    // position, which the drafter chains from; without a drafter nothing
    // reads it.
    std::vector<float>* hidden_out = strategy != nullptr ? &hidden : nullptr;
    auto logits = target.LogitsStep(next, hidden_out);
    if (!logits) {
      diagnostics.Warn("engine", std::string("generation step failed: ") +
                                     StepError(logits.error()));
      return std::unexpected(logits.error());
    }
    if (strategy != nullptr && block > 0) {
      auto anchored = strategy->OnAnchor(target.backend(), model,
                                         target.cache(), next, position,
                                         hidden);
      if (!anchored) {
        return std::unexpected(anchored.error());
      }
      auto count = strategy->Draft(target.backend(), model, target.cache(),
                                   *logits, next, drafts);
      if (!count) {
        return std::unexpected(count.error());
      }
      if (*count > 0) {
        std::span<const std::uint32_t> proposal(drafts.data(), *count);
        auto verify = target.Verify(proposal, *logits, std::nullopt, &hidden);
        if (!verify) {
          return std::unexpected(verify.error());
        }
        auto committed = strategy->Commit(target.backend(), model,
                                          target.cache(), verify->accepted);
        if (!committed) {
          return std::unexpected(committed.error());
        }
        spec_proposed += *count;
        spec_accepted += verify->accepted;
        ++spec_steps;
        for (std::size_t i = 0; i < verify->accepted; ++i) {
          if (produced >= max_completion_tokens) {
            break;
          }
          if (core::detail::IsStopToken(drafts[i], stops)) {
            stopped = true;
            break;
          }
          if (!emit(drafts[i])) {
            stopped = true;
            break;
          }
          history.push_back(drafts[i]);
          ++produced;
        }
        if (stopped || produced >= max_completion_tokens) {
          break;
        }
        next = verify->next_token;
        position += 1 + verify->accepted;
        continue;
      }
    }
    next = pick(*logits);
    ++position;
  }
  if (stopped) {
    diagnostics.Info("engine", std::string("stopped at declared stop token ") +
                                    std::to_string(next));
  }
  if (spec_steps > 0) {
    diagnostics.Info("engine",
                     "speculative: accepted " + std::to_string(spec_accepted) +
                         " of " + std::to_string(spec_proposed) +
                         " draft token(s) over " + std::to_string(spec_steps) +
                         " step(s)");
  }
  const double decode_ms = static_cast<double>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - decode_started)
          .count());
  target.ReportDiagnostics(loop_steps);
  if (strategy != nullptr) {
    strategy->ReportCaches(diagnostics, "engine");
  }
  if (produced > 0) {
    const long long decode_ms_ll = static_cast<long long>(decode_ms);
    const long long decode_tps =
        decode_ms_ll > 0
            ? static_cast<long long>(produced) * 1000 / decode_ms_ll
            : 0;
    diagnostics.Info(
        "engine", std::string("decode: ") + std::to_string(produced) +
                      " token(s) in " + std::to_string(decode_ms_ll) + " ms (" +
                      std::to_string(decode_tps) + " tok/s, " +
                      std::to_string(decode_ms_ll /
                                     static_cast<long long>(produced)) +
                      " ms/token)");
  }
  const FinishReason reason = stopped    ? FinishReason::Stop
                              : aborted  ? FinishReason::Aborted
                                         : FinishReason::Length;
  GenerateOutcome outcome;
  outcome.produced = produced;
  outcome.reason = reason;
  outcome.prompt_tokens = prompt.size();
  outcome.prefill_ms = prefill_ms_exact;
  outcome.prefill_compute_ms = compute_ms;
  outcome.decode_ms = decode_ms;
  return outcome;
}

}  // namespace tessera::core
