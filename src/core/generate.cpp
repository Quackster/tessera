#include "tessera/engine.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "core/generate_helpers.hpp"
#include "core/profile.hpp"
#include "core/sampling.hpp"
#include "core/think_budget.hpp"

namespace tessera {

std::expected<std::vector<std::uint32_t>, StatusCode> Engine::GenerateDraft(
    Model& model, const GenerateOptions& options,
    const std::string& draft_path) {
  auto strategy = CreateDFlash2Strategy();
  auto attached =
      strategy->Attach(StrategyOptions{draft_path, options.draft_tokens});
  if (!attached) {
    return std::unexpected(attached.error());
  }
  auto previous = std::move(speculative_);
  speculative_ = std::move(strategy);
  auto produced = Generate(model, options);
  speculative_ = std::move(previous);
  return produced;
}

std::expected<std::vector<std::uint32_t>, StatusCode>
Engine::GenerateSpeculative(Model& model, const GenerateOptions& options) {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  if (!config->hybrid) {
    diagnostics_.Warn("engine", "GenerateSpeculative: MTP needs a hybrid model");
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  auto strategy = CreateMtpStrategy();
  auto attached =
      strategy->Attach(StrategyOptions{"", options.draft_tokens});
  if (!attached) {
    return std::unexpected(attached.error());
  }
  auto previous = std::move(speculative_);
  speculative_ = std::move(strategy);
  auto produced = Generate(model, options);
  speculative_ = std::move(previous);
  return produced;
}

std::expected<std::uint32_t, StatusCode> Engine::MtpDraft(
    Model& model, std::uint32_t token) {
  core::DecodeCache cache;
  std::vector<float> hidden;
  auto step = core::DecodeStep(*backend_, model, cache, token, &hidden);
  if (!step) {
    return std::unexpected(step.error());
  }
  return core::MtpDraftStep(*backend_, model, cache, hidden, *step, 1);
}

std::expected<GenerateOutcome, StatusCode> Engine::GenerateStreaming(
    Model& model, const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token) {
  if (options.sample) {
    const SamplingOptions& p = options.sampling;
    if (!(p.temperature >= 0.0f) || !(p.top_p > 0.0f && p.top_p <= 1.0f) ||
        p.top_k < 0 || !(p.min_p >= 0.0f && p.min_p < 1.0f) ||
        p.repetition_penalty <= 0.0f) {
      diagnostics_.Warn("engine", "invalid sampling parameters; using defaults "
                                  "is not attempted, request rejected");
      return std::unexpected(StatusCode::InvalidArgument);
    }
  }
  core::DecodeCache cache;
  cache.kv_type = options.kv_type;
  cache.tuning.mxfp4_split_target = options.mxfp4_split_target;
  cache.tuning.mxfp4_split_cap = options.mxfp4_split_cap;
  // Opt-in decode phase profiler (TESSERA_PROFILE=1). The profile lives for
  // the whole generation and is reported after the decode summary.
  core::Profile profile(core::DecodeProfilingDeep());
  const bool profiling = core::DecodeProfilingEnabled();
  cache.profile = profiling ? &profile : nullptr;
  std::vector<std::uint32_t> prompt = options.prompt_tokens;
  if (prompt.empty()) {
    prompt.push_back(options.first_token);
  }
  // A zero count fills the remaining context.
  const std::size_t max_completion_tokens =
      model.EffectiveMaxTokens(prompt.size(), options.max_completion_tokens);
  if (max_completion_tokens == 0) {
    return GenerateOutcome{};
  }
  if (prompt.size() > model.MaxContextLength()) {
    diagnostics_.Warn("engine", std::string("prompt of ") +
                                     std::to_string(prompt.size()) +
                                     " token(s) exceeds the " +
                                     std::to_string(model.MaxContextLength()) +
                                     " token context; reject without running "
                                     "prefill so the device stays alive");
    return std::unexpected(StatusCode::InvalidArgument);
  }
  // The speculative strategy (null on the plain greedy path). The engine owns
  // the target and the accept rule; the strategy owns the draft.
  SpeculativeStrategy* strategy = speculative_.get();
  std::vector<std::size_t> capture_layers;
  std::vector<Buffer*> capture_buffers;
  if (strategy != nullptr) {
    auto prepared = strategy->Prepare(*backend_, model);
    if (!prepared) {
      diagnostics_.Warn("engine", "strategy prepare failed: " +
                                      std::string(ToString(prepared.error())));
      return std::unexpected(prepared.error());
    }
    const std::span<const std::size_t> layers = strategy->CaptureLayers();
    capture_layers.assign(layers.begin(), layers.end());
    const std::span<Buffer* const> buffers = strategy->CaptureBuffers();
    capture_buffers.assign(buffers.begin(), buffers.end());
  }
  const bool capturing = !capture_layers.empty();
  const std::vector<std::size_t>* capture_layers_ptr =
      capturing ? &capture_layers : nullptr;
  std::vector<Buffer*>* capture_ptr = capturing ? &capture_buffers : nullptr;

  // The prefill clock starts after strategy preparation, so a one-time draft
  // weight load is not billed to the prompt.
  const auto prefill_started = std::chrono::steady_clock::now();
  const std::size_t prefill_chunk = ResolvePrefillChunkTokens(
      prefill_chunk_tokens_, options.prefill_chunk_tokens,
      model.MaxContextLength());
  if (options.progress_every > 0) {
    diagnostics_.Info("engine", std::string("prefill: ") +
                                    std::to_string(prompt.size()) +
                                    " prompt token(s), chunk " +
                                    std::to_string(prefill_chunk));
  }
  std::vector<float> hidden;
  std::vector<float> first_logits;
  // Chunk progress for a long prefill: one line per progress_every
  // chunk reports plus the final row count, so a multi-minute prefill
  // shows movement instead of silence. Zero disables the reports; the
  // request hook still fires for stream progress.
  std::size_t progress_calls = 0;
  core::PrefillProgress progress = [&](std::size_t done, std::size_t total) {
    ++progress_calls;
    if (options.progress_every > 0 &&
        (done == total || progress_calls % options.progress_every == 1)) {
      diagnostics_.Info("engine", std::string("prefill: ") +
                                      std::to_string(done) + "/" +
                                      std::to_string(total) + " rows");
    }
    if (options.prefill_progress) {
      options.prefill_progress(done, total);
    }
  };
  const core::PrefillProgress* progress_ptr =
      (options.progress_every > 0 || options.prefill_progress) ? &progress
                                                               : nullptr;
  if (capturing) {
    auto prefilled = PrefillCapturing(*backend_, model, cache, *strategy,
                                      prompt, prefill_chunk, capture_layers,
                                      capture_buffers, hidden, first_logits,
                                      progress_ptr);
    if (!prefilled) {
      diagnostics_.Warn("engine", std::string("prefill failed: ") +
                                      std::string(ToString(prefilled.error())));
      return std::unexpected(prefilled.error());
    }
  } else {
    auto logits = core::PrefillTokens(*backend_, model, cache, prompt, &hidden,
                                      nullptr, prefill_chunk, progress_ptr);
    if (!logits) {
      diagnostics_.Warn("engine", std::string("prefill failed: ") +
                                      std::string(ToString(logits.error())));
      return std::unexpected(logits.error());
    }
    first_logits = std::move(*logits);
  }

  const long long prefill_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - prefill_started)
          .count();
  if (options.progress_every > 0 && prefill_ms > 0) {
    const long long prompt_tps =
        static_cast<long long>(prompt.size()) * 1000 / prefill_ms;
    diagnostics_.Info("engine",
                      std::string("prefill: ") + std::to_string(prompt.size()) +
                          " prompt token(s) in " + std::to_string(prefill_ms) +
                          " ms (" + std::to_string(prompt_tps) +
                          " prompt tok/s)");
  }
  const auto decode_started = std::chrono::steady_clock::now();
  std::mt19937_64 rng(options.seed);
  std::vector<std::uint32_t> history = prompt;
  const std::vector<std::uint32_t> stops = StopSet(model, options);
  const auto pick = [&](std::span<const float> logits) {
    return options.sample
               ? core::SampleToken(logits, options.sampling, history, rng)
               : core::detail::ArgMax(logits);
  };
  std::size_t block = strategy != nullptr ? strategy->DraftBlock() : 0;
  if (options.draft_tokens > 0 && options.draft_tokens < block) {
    block = options.draft_tokens;
  }
  std::vector<std::uint32_t> drafts(block);
  std::uint32_t next = pick(first_logits);
  std::uint64_t position = prompt.size();
  // A folding drafter drafts before the target sees the anchor, pairing the
  // anchor token with the prefill's final hidden (the hidden at one
  // position before). Seed it here; the loop re-records it after each
  // verify. DFlash2 records its own context during prefill and ignores the
  // values.
  if (strategy != nullptr && block > 0 && strategy->NeedsPrefillAnchor() &&
      !hidden.empty()) {
    auto seeded =
        strategy->OnAnchor(*backend_, model, cache, next, position, hidden);
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
  // Thinking budget: resolve the tags once and scan the prompt, so a
  // prompt trailing an open think block starts inside thinking.
  auto [think, think_close] =
      ResolveThinkBudget(model, options.max_thinking_tokens);
  if (options.max_thinking_tokens > 0 && !think.engaged()) {
    diagnostics_.Warn("engine", "thinking budget ignored: the think tags do "
                                "not resolve on this model; thinking stays "
                                "unlimited");
  }
  for (std::uint32_t id : prompt) {
    think.noteEmitted(id);
  }
  // Every emitted token funnels through here, so the budget sees
  // anchors and accepted drafts alike. Forcing happens at loop top so
  // the close markup feeds before the pending token is consumed.
  const auto emit = [&](std::uint32_t token) {
    think.noteEmitted(token);
    return on_token(token);
  };
  // Feed and emit the think-close ids, keeping the target cache, the
  // drafter context, the sampling history and the position in step.
  // Returns the re-picked next token, or nullopt to end the turn (peer
  // abort or completion budget spent on the close markup).
  auto force_close = [&]()
      -> std::expected<std::optional<std::uint32_t>, StatusCode> {
    std::vector<float> close_logits;
    for (std::uint32_t id : think_close) {
      // Forced markup bypasses the stop check like any protocol token.
      std::expected<std::vector<float>, StatusCode> logits;
      {
        core::PhaseScope scope(cache.profile, core::Phase::kAnchor);
        logits = core::DecodeLogits(*backend_, model, cache, id, &hidden,
                                    capture_layers_ptr, capture_ptr);
      }
      if (!logits) {
        diagnostics_.Warn("engine", std::string("think force-close failed: ") +
                                        std::string(ToString(logits.error())));
        return std::unexpected(logits.error());
      }
      if (strategy != nullptr && block > 0) {
        // Mirror the anchor bookkeeping of the decode paths below: the
        // folding path commits the lone anchor, the plain one only
        // records it.
        auto anchored = strategy->OnAnchor(*backend_, model, cache, id,
                                           position, hidden);
        if (!anchored) {
          return std::unexpected(anchored.error());
        }
        if (strategy->FoldsAnchor()) {
          auto committed = strategy->Commit(*backend_, model, cache, 0);
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
      diagnostics_.Info("engine", "thinking budget (" +
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
      std::expected<std::size_t, StatusCode> count;
      {
        core::PhaseScope scope(cache.profile, core::Phase::kDraft);
        count = strategy->Draft(*backend_, model, cache, {}, next, drafts);
      }
      if (!count) {
        return std::unexpected(count.error());
      }
      if (*count == 0) {
        // No usable proposal: decode the anchor alone and take its greedy
        // token. The drafter context still gains the anchor below.
        std::expected<std::vector<float>, StatusCode> logits;
        {
          core::PhaseScope scope(cache.profile, core::Phase::kAnchor);
          logits = core::DecodeLogits(*backend_, model, cache, next, &hidden,
                                      capture_layers_ptr, capture_ptr);
        }
        if (!logits) {
          return std::unexpected(logits.error());
        }
        const std::uint32_t picked = pick(*logits);
        auto anchored = strategy->OnAnchor(*backend_, model, cache, picked,
                                           position + 1, hidden);
        if (!anchored) {
          return std::unexpected(anchored.error());
        }
        // A folding strategy defers its context append to Commit, so commit
        // the anchor alone (zero accepted drafts) to keep the context in step.
        auto committed = strategy->Commit(*backend_, model, cache, 0);
        if (!committed) {
          return std::unexpected(committed.error());
        }
        next = picked;
        ++position;
        continue;
      }
      std::span<const std::uint32_t> proposal(drafts.data(), *count);
      std::expected<DraftVerification, StatusCode> verify;
      {
        core::PhaseScope scope(cache.profile, core::Phase::kVerify);
        // The anchor leads the batch and its row plus the committed drafts
        // advance the cache; `hidden` receives the last committed row's
        // residual hidden, which the drafter pairs with the next anchor
        // (the reference MTP contract: token at p with the hidden at p-1).
        verify = core::VerifyDraft(*backend_, model, cache, proposal, {}, next,
                                   &hidden, capture_layers_ptr, capture_ptr);
      }
      if (!verify) {
        return std::unexpected(verify.error());
      }
      // Record the next anchor for the drafter: the token the last
      // committed row selects, at its absolute position, with that row's
      // hidden (the hidden at one position before the anchor).
      auto anchored = strategy->OnAnchor(
          *backend_, model, cache, verify->next_token,
          position + 1 + verify->accepted, hidden);
      if (!anchored) {
        return std::unexpected(anchored.error());
      }
      auto committed =
          strategy->Commit(*backend_, model, cache, verify->accepted);
      if (!committed) {
        return std::unexpected(committed.error());
      }
      spec_proposed += *count;
      spec_accepted += verify->accepted;
      ++spec_steps;
      if (profiling) {
        // Per-step draft trace: the accepted draft (or the target's
        // replacement on a reject), so an acceptance problem is
        // visible token by token.
        diagnostics_.Info(
            "engine", "draft step " + std::to_string(spec_steps) +
                          ": proposed " + std::to_string(drafts[0]) +
                          (verify->accepted > 0
                               ? " accepted"
                               : " rejected, target picked " +
                                     std::to_string(verify->next_token)));
      }
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
    // vocabulary is not downloaded and the hidden state is not read back
    // (nothing consumes it).
    if (strategy == nullptr && !options.sample) {
      auto token = core::DecodeToken(*backend_, model, cache, next);
      if (!token) {
        diagnostics_.Warn(
            "engine", std::string("generation step failed: ") +
                          std::string(ToString(token.error())));
        return std::unexpected(token.error());
      }
      next = *token;
      ++position;
      continue;
    }
    // Advance the target by the just-emitted token. `hidden` describes its
    // position, which the drafter chains from; without a drafter nothing
    // reads it, so it is not downloaded.
    std::vector<float>* hidden_out = strategy != nullptr ? &hidden : nullptr;
    std::expected<std::vector<float>, StatusCode> logits;
    {
      core::PhaseScope scope(cache.profile, core::Phase::kAnchor);
      logits = core::DecodeLogits(*backend_, model, cache, next, hidden_out,
                                  capture_layers_ptr, capture_ptr);
    }
    if (!logits) {
      diagnostics_.Warn(
          "engine", std::string("generation step failed: ") +
                        std::string(ToString(logits.error())));
      return std::unexpected(logits.error());
    }
    if (strategy != nullptr && block > 0) {
      auto anchored = strategy->OnAnchor(*backend_, model, cache, next,
                                         position, hidden);
      if (!anchored) {
        return std::unexpected(anchored.error());
      }
      std::expected<std::size_t, StatusCode> count;
      {
        core::PhaseScope scope(cache.profile, core::Phase::kDraft);
        count =
            strategy->Draft(*backend_, model, cache, *logits, next, drafts);
      }
      if (!count) {
        return std::unexpected(count.error());
      }
      if (*count > 0) {
        std::span<const std::uint32_t> proposal(drafts.data(), *count);
        std::expected<DraftVerification, StatusCode> verify;
        {
          core::PhaseScope scope(cache.profile, core::Phase::kVerify);
          verify = core::VerifyDraft(*backend_, model, cache, proposal,
                                     *logits, std::nullopt, &hidden,
                                     capture_layers_ptr, capture_ptr);
        }
        if (!verify) {
          return std::unexpected(verify.error());
        }
        auto committed = strategy->Commit(*backend_, model, cache,
                                          verify->accepted);
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
    diagnostics_.Info("engine", std::string("stopped at declared stop token ") +
                                    std::to_string(next));
  }
  if (spec_steps > 0) {
    diagnostics_.Info("engine",
                      "speculative: accepted " + std::to_string(spec_accepted) +
                          " of " + std::to_string(spec_proposed) +
                          " draft token(s) over " + std::to_string(spec_steps) +
                          " step(s)");
  }
  const double decode_ms = static_cast<double>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - decode_started)
          .count());
  if (profiling) {
    profile.Report(diagnostics_, "engine", loop_steps);
  }
  if (produced > 0) {
    const long long decode_ms_ll = static_cast<long long>(decode_ms);
    const long long decode_tps =
        decode_ms_ll > 0
            ? static_cast<long long>(produced) * 1000 / decode_ms_ll
            : 0;
    diagnostics_.Info(
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
  outcome.prefill_ms = static_cast<double>(prefill_ms);
  outcome.decode_ms = decode_ms;
  return outcome;
}

std::expected<std::vector<std::uint32_t>, StatusCode> Engine::Generate(
    Model& model, const GenerateOptions& options) {
  std::vector<std::uint32_t> produced;
  produced.reserve(options.max_completion_tokens);
  auto streamed = GenerateStreaming(
      model, options, [&produced](std::uint32_t token) {
        produced.push_back(token);
        return true;
      });
  if (!streamed) {
    return std::unexpected(streamed.error());
  }
  return produced;
}

}  // namespace tessera
