#include "tessera/pipeline.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/decode_internal.hpp"
#include "core/pipeline_impl.hpp"
#include "core/timing.hpp"
#include "tessera/architecture.hpp"
#include "tessera/backend.hpp"
#include "tessera/engine.hpp"
#include "tessera/log.hpp"
#include "tessera/model.hpp"
#include "tessera/tokenizer.hpp"
#include "tessera/types.hpp"

// The single-sequence pipeline path: build the layer-split stages, run one
// prompt through them, and time the prefill and the decode. The overlapped
// multi-sequence path lives in pipeline_many.cpp.

namespace tessera {

namespace {

// Concurrent serve turns the pipeline keeps sequences for. Each sequence
// keeps its own KV cache, so the per-stage cache memory multiplies by this
// count at a long context.
constexpr std::size_t kMaxConcurrentTurns = 2;

}  // namespace

std::expected<GenerateOutcome, StatusCode> Pipeline::Impl::Run(
    std::span<const std::uint32_t> prompt, const GenerateOptions& options,
    std::vector<std::uint32_t>* produced) {
  if (prompt.empty() || prompt.size() > context) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  produced->clear();
  ResetSequence(DefaultSequence(), options);
  Activate(DefaultSequence());
  GenerateOutcome outcome;
  outcome.prompt_tokens = prompt.size();
  std::expected<std::vector<float>, StatusCode> logits;
  const auto prefill_started = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    const bool last = (i + 1 == prompt.size());
    if (last) {
      logits = Step(prompt[i], true);
    } else {
      auto step = Step(prompt[i], false);
      if (!step) {
        return std::unexpected(step.error());
      }
    }
  }
  outcome.prefill_ms = core::ElapsedMsDouble(prefill_started,
                                             std::chrono::steady_clock::now());
  if (!logits) {
    return std::unexpected(logits.error());
  }
  // The intermittent non-finite fault (a corrupt row) degenerates to token
  // 0; stop instead of looping on `!`, like the non-pipeline decode path.
  if (!core::detail::AllFinite(*logits)) {
    outcome.reason = FinishReason::Aborted;
    return outcome;
  }
  const std::size_t budget =
      options.max_completion_tokens == 0 ? context - prompt.size()
                                         : options.max_completion_tokens;
  std::uint32_t next = pipeline_internal::ArgMax(*logits);
  bool stopped = false;
  bool aborted = false;
  const auto decode_started = std::chrono::steady_clock::now();
  for (std::size_t count = 0; count < budget; ++count) {
    if (pipeline_internal::IsStop(next, stops)) {
      stopped = true;
      break;
    }
    produced->push_back(next);
    auto step = Step(next, true);
    if (!step) {
      return std::unexpected(step.error());
    }
    if (!core::detail::AllFinite(*step)) {
      aborted = true;
      break;
    }
    next = pipeline_internal::ArgMax(*step);
  }
  outcome.decode_ms = core::ElapsedMsDouble(decode_started,
                                            std::chrono::steady_clock::now());
  outcome.produced = produced->size();
  outcome.reason = aborted  ? FinishReason::Aborted
                   : stopped ? FinishReason::Stop
                             : FinishReason::Length;
  ReleaseSequenceCaches(DefaultSequence());
  return outcome;
}

Pipeline::Pipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline&& other) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&& other) noexcept = default;

std::expected<std::unique_ptr<Pipeline>, StatusCode> Pipeline::Create(
    const PipelineOptions& options) {
  if (options.devices.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto config = Model::ProbeConfig(options.model_path);
  if (!config) {
    return std::unexpected(config.error());
  }
  const std::size_t layers = config->layers;
  const std::size_t stage_count = options.devices.size();
  auto ranges = pipeline_internal::StageLayerRanges(
      layers, stage_count, options.stage_layers);
  if (!ranges) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto impl = std::make_unique<Impl>();
  // The boundary handoff carries the architecture residual stream, not the
  // single hidden: a gated residual stack hands off every stream. A
  // single-stream model returns hidden_dim, so its buffers are unchanged.
  impl->hidden_dim = config->ResidualWidth();
  impl->context = options.context;
  impl->kv_type = options.kv_type;
  impl->stages.reserve(stage_count);
  for (std::size_t i = 0; i < stage_count; ++i) {
    // Contiguous ranges, balanced unless the caller supplied counts.
    const std::size_t first = (*ranges)[i].first;
    const std::size_t last = (*ranges)[i].second;
    EngineOptions engine_options;
    engine_options.device_index = options.devices[i];
    auto engine = Engine::Create(engine_options);
    if (!engine) {
      return std::unexpected(engine.error());
    }
    ModelOptions model_options;
    model_options.path = options.model_path;
    model_options.max_context_length = options.context;
    model_options.gpu_layers = options.gpu_layers;
    model_options.shard.enabled = true;
    model_options.shard.first_layer = first;
    model_options.shard.last_layer = last;
    model_options.shard.first_stage = (i == 0);
    model_options.shard.last_stage = (i + 1 == stage_count);
    model_options.shard.last_stage_embedding = options.last_stage_embedding;
    // A fetched MTP head loads on the last stage only, which owns it.
    if (i + 1 == stage_count) {
      model_options.mtp_path = options.mtp_path;
    }
    auto model = (*engine)->LoadModel(model_options);
    if (!model) {
      return std::unexpected(model.error());
    }
    Impl::Stage stage;
    stage.engine = std::move(*engine);
    stage.model = std::move(*model);
    stage.boundary_layer = last - 1;
    stage.first_layer = first;
    stage.last_layer = last;
    impl->stages.push_back(std::move(stage));
  }
  const std::span<const std::uint32_t> stops =
      impl->stages.back().model->StopTokens();
  impl->stops.assign(stops.begin(), stops.end());
  // One default sequence for the single-turn run and calibrate paths, plus
  // the pool the serve path draws concurrent turns from.
  for (std::size_t i = 0; i <= kMaxConcurrentTurns; ++i) {
    auto sequence = impl->NewSequence();
    if (!sequence) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    impl->sequences.push_back(std::move(sequence));
  }
  impl->Activate(impl->DefaultSequence());
  return std::unique_ptr<Pipeline>(new Pipeline(std::move(impl)));
}

std::expected<std::vector<std::uint32_t>, StatusCode> Pipeline::Generate(
    std::span<const std::uint32_t> prompt, std::size_t max_tokens) {
  GenerateOptions options;
  options.max_completion_tokens = max_tokens;
  options.kv_type = impl_->kv_type;
  std::vector<std::uint32_t> produced;
  auto outcome = impl_->HasStrategy()
                     ? impl_->RunSpeculative(prompt, options, &produced)
                     : impl_->Run(prompt, options, &produced);
  if (!outcome) {
    return std::unexpected(outcome.error());
  }
  const log::Diagnostics& log = impl_->stages.back().engine->Diagnostics();
  const long long prefill_ms = static_cast<long long>(outcome->prefill_ms);
  if (prefill_ms > 0) {
    const long long prompt_tps =
        static_cast<long long>(prompt.size()) * 1000 / prefill_ms;
    log.Info("pipeline",
             std::string("prefill: ") + std::to_string(prompt.size()) +
                 " prompt token(s) in " + std::to_string(prefill_ms) + " ms (" +
                 std::to_string(prompt_tps) + " prompt tok/s)");
  }
  const long long decode_ms = static_cast<long long>(outcome->decode_ms);
  if (decode_ms > 0 && !produced.empty()) {
    const long long decode_tps =
        static_cast<long long>(produced.size()) * 1000 / decode_ms;
    log.Info("pipeline",
             std::string("decode: ") + std::to_string(produced.size()) +
                 " token(s) in " + std::to_string(decode_ms) + " ms (" +
                 std::to_string(decode_tps) + " tok/s, " +
                 std::to_string(decode_ms /
                                static_cast<long long>(produced.size())) +
                 " ms/token)");
  }
  return produced;
}

std::expected<GenerateOutcome, StatusCode> Pipeline::GenerateTimed(
    std::span<const std::uint32_t> prompt, const GenerateOptions& options) {
  std::vector<std::uint32_t> produced;
  auto outcome = impl_->HasStrategy()
                     ? impl_->RunSpeculative(prompt, options, &produced)
                     : impl_->Run(prompt, options, &produced);
  if (!outcome) {
    return std::unexpected(outcome.error());
  }
  return *outcome;
}

std::size_t Pipeline::StageCount() const { return impl_->stages.size(); }

Engine& Pipeline::StageEngine(std::size_t i) {
  return *impl_->stages.at(i).engine;
}

Model& Pipeline::StageModel(std::size_t i) {
  return *impl_->stages.at(i).model;
}

std::vector<std::pair<std::size_t, std::size_t>> Pipeline::StageRanges()
    const {
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  ranges.reserve(impl_->stages.size());
  for (const Impl::Stage& stage : impl_->stages) {
    ranges.emplace_back(stage.first_layer, stage.last_layer);
  }
  return ranges;
}

const Tokenizer* Pipeline::GetTokenizer() const {
  return impl_->stages.empty() ? nullptr
                               : impl_->stages.front().model->GetTokenizer();
}

std::expected<std::string, StatusCode> Pipeline::ChatPrompt(
    std::string_view user_text, bool enable_thinking,
    std::string_view reasoning_effort) const {
  if (impl_->stages.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return impl_->stages.front().model->ChatPrompt(user_text, enable_thinking,
                                                 reasoning_effort);
}

}  // namespace tessera
