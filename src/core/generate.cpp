#include "tessera/engine.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/profile.hpp"
#include "core/single_device_target.hpp"
#include "core/spec_driver.hpp"

// The engine's generation entry points. GenerateStreaming builds the
// single-device decode target and hands the token loop to the decode driver
// (core::RunDecode), so the greedy and speculative paths share one loop.

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
  core::DecodeCache cache;
  cache.kv_type = options.kv_type;
  cache.kv_offload = options.kv_offload;
  cache.tuning.mxfp4_split_target = options.mxfp4_split_target;
  cache.tuning.mxfp4_split_cap = options.mxfp4_split_cap;
  cache.tuning.attention_split = options.attention_split;
  // Opt-in decode phase profiler (TESSERA_PROFILE=1). The profile lives for
  // the whole generation and is reported through the target's diagnostics.
  core::Profile profile(core::DecodeProfilingDeep());
  const bool profiling = core::DecodeProfilingEnabled();
  cache.profile = profiling ? &profile : nullptr;
  core::SingleDeviceTarget target(*backend_, model, cache, diagnostics_,
                                  prefill_chunk_tokens_,
                                  profiling ? &profile : nullptr,
                                  /*single_token_prefill=*/false,
                                  /*supports_sampled_verify=*/true);
  return core::RunDecode(target, speculative_.get(), options, on_token);
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
