#include "tessera/spec_split.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/spec_driver.hpp"
#include "tessera/architecture.hpp"
#include "tessera/backend.hpp"
#include "tessera/engine.hpp"
#include "tessera/log.hpp"
#include "tessera/model.hpp"
#include "tessera/speculative.hpp"

namespace tessera {

namespace {

// One decode request on the split layout, behind the shared decode driver.
// The strategy runs against the draft device, every target-side op runs
// against the target device, and two cache objects carry this request's
// state. backend()/model()/cache() expose the drafter's route: the driver
// passes them only to strategy calls and to model metadata that is
// identical across the checkpoint's routes. The driver runs on one thread,
// so the target is externally synchronized and needs no locking.
class SpecSplitTarget final : public core::SpecTarget {
 public:
  SpecSplitTarget(Backend& target_backend, Model& target,
                  core::DecodeCache& target_cache, Backend& draft_backend,
                  Model& draft, core::DecodeCache& draft_cache,
                  log::Diagnostics& diagnostics, std::size_t prefill_chunk)
      : target_backend_(target_backend),
        target_(target),
        target_cache_(target_cache),
        draft_backend_(draft_backend),
        draft_(draft),
        draft_cache_(draft_cache),
        diagnostics_(diagnostics),
        prefill_chunk_(prefill_chunk) {}

  log::Diagnostics& diagnostics() override { return diagnostics_; }
  Backend& backend() override { return draft_backend_; }
  Model& model() override { return draft_; }
  core::DecodeCache& cache() override { return draft_cache_; }

  std::expected<void, StatusCode> Prepare(
      SpeculativeStrategy* strategy, const GenerateOptions& options) override {
    if (strategy == nullptr) {
      return {};
    }
    // The strategy lives on the draft device, so every prepare call runs
    // against the head shard there; a fresh draft cache needs its state.
    auto prepared = strategy->Prepare(draft_backend_, draft_);
    if (!prepared) {
      return std::unexpected(prepared.error());
    }
    strategy->SetDraftContext(options.draft_context);
    const Architecture* arch = draft_.Arch();
    if (arch == nullptr) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    // Allocate the head's decode state without running the trunk, which
    // the draft device does not hold.
    auto ready = arch->EnsureReady(draft_backend_, draft_, draft_cache_);
    if (!ready && ready.error() != StatusCode::UnsupportedFeature) {
      return std::unexpected(ready.error());
    }
    return {};
  }

  std::expected<void, StatusCode> Prefill(std::span<const std::uint32_t> prompt,
                                          const GenerateOptions& options,
                                          std::vector<float>* hidden,
                                          std::vector<float>* logits,
                                          double* compute_ms) override {
    core::PrefillProgress progress = [&](std::size_t done, std::size_t total) {
      if (options.prefill_progress) {
        options.prefill_progress(done, total);
      }
    };
    const core::PrefillProgress* progress_ptr =
        options.prefill_progress ? &progress : nullptr;
    auto prefilled = core::PrefillTokens(
        target_backend_, target_, target_cache_, prompt, hidden, nullptr,
        prefill_chunk_, progress_ptr, compute_ms);
    if (!prefilled) {
      return std::unexpected(prefilled.error());
    }
    *logits = std::move(*prefilled);
    return {};
  }

  std::expected<std::uint32_t, StatusCode> GreedyStep(
      std::uint32_t token) override {
    return core::DecodeToken(target_backend_, target_, target_cache_, token);
  }

  std::expected<std::vector<float>, StatusCode> LogitsStep(
      std::uint32_t token, std::vector<float>* hidden) override {
    return core::DecodeLogits(target_backend_, target_, target_cache_, token,
                              hidden, nullptr, nullptr);
  }

  std::expected<DraftVerification, StatusCode> Verify(
      std::span<const std::uint32_t> draft,
      std::span<const float> prefix_logits, std::optional<std::uint32_t> anchor,
      std::vector<float>* hidden) override {
    return core::VerifyDraft(target_backend_, target_, target_cache_, draft,
                             prefix_logits, anchor, hidden, nullptr, nullptr);
  }

  void ReportDiagnostics(std::size_t loop_steps) override {
    (void)loop_steps;
    if (target_cache_.device != nullptr) {
      target_cache_.device->gemms.Report(diagnostics_, "spec-split",
                                         "decode gemm");
      target_cache_.device->embed_kernels.Report(diagnostics_, "spec-split",
                                                 "decode embedding");
    }
    if (target_cache_.arch != nullptr) {
      target_cache_.arch->ReportCaches(diagnostics_, "spec-split");
    }
  }

 private:
  Backend& target_backend_;
  Model& target_;
  core::DecodeCache& target_cache_;
  Backend& draft_backend_;
  Model& draft_;
  core::DecodeCache& draft_cache_;
  log::Diagnostics& diagnostics_;
  std::size_t prefill_chunk_ = 0;
};

}  // namespace

struct SpecSplit::Impl {
  std::unique_ptr<Engine> target_engine;
  std::unique_ptr<Model> target;
  std::unique_ptr<core::DecodeCache> target_cache;
  std::unique_ptr<Engine> draft_engine;
  std::unique_ptr<Model> draft;
  std::unique_ptr<core::DecodeCache> draft_cache;
  std::unique_ptr<SpeculativeStrategy> strategy;
  std::vector<std::uint32_t> stops;
  std::size_t context = 0;
  std::size_t prefill_chunk = 0;
  KvCacheType kv_type = KvCacheType::F32;
  bool kv_offload = false;
};

SpecSplit::SpecSplit(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

SpecSplit::~SpecSplit() = default;
SpecSplit::SpecSplit(SpecSplit&& other) noexcept = default;
SpecSplit& SpecSplit::operator=(SpecSplit&& other) noexcept = default;

std::expected<std::unique_ptr<SpecSplit>, StatusCode> SpecSplit::Create(
    const SpecSplitOptions& options) {
  if (options.devices.size() < 2) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto probed = Model::ProbeConfig(options.model_path);
  if (!probed) {
    return std::unexpected(probed.error());
  }
  const std::size_t trunk_layers = probed->layers;
  auto impl = std::make_unique<Impl>();
  impl->context = options.context;
  impl->prefill_chunk = options.prefill_chunk;
  impl->kv_type = options.kv_type;
  impl->kv_offload = options.kv_offload;
  // Target first, then draft: one model load at a time, so only one
  // checkpoint is in host memory at once.
  {
    EngineOptions engine_options;
    engine_options.device_index = options.devices[0];
    auto engine = Engine::Create(engine_options);
    if (!engine) {
      return std::unexpected(engine.error());
    }
    impl->target_engine = std::move(*engine);
    ModelOptions model_options;
    model_options.path = options.model_path;
    model_options.max_context_length = options.context;
    model_options.load_mtp_head = true;
    model_options.gpu_layers = options.gpu_layers;
    auto model = impl->target_engine->LoadModel(model_options);
    if (!model) {
      return std::unexpected(model.error());
    }
    impl->target = std::move(*model);
  }
  {
    EngineOptions engine_options;
    engine_options.device_index = options.devices[1];
    auto engine = Engine::Create(engine_options);
    if (!engine) {
      return std::unexpected(engine.error());
    }
    impl->draft_engine = std::move(*engine);
    // Head-only shard: an empty trunk range with both stage flags keeps
    // the embedding, the MTP block group and the output head.
    ModelOptions model_options;
    model_options.path = options.model_path;
    model_options.max_context_length = options.context;
    model_options.load_mtp_head = true;
    // The draft device reads the separately fetched MTP head; the target
    // does not, so only this load carries the path.
    model_options.mtp_path = options.mtp_path;
    model_options.shard.enabled = true;
    model_options.shard.first_layer = trunk_layers;
    model_options.shard.last_layer = trunk_layers;
    model_options.shard.first_stage = true;
    model_options.shard.last_stage = true;
    auto model = impl->draft_engine->LoadModel(model_options);
    if (!model) {
      return std::unexpected(model.error());
    }
    impl->draft = std::move(*model);
  }
  if (impl->draft->Arch() == nullptr) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  impl->strategy = CreateMtpStrategy();
  if (auto attached =
          impl->strategy->Attach(StrategyOptions{"", options.draft_block});
      !attached) {
    return std::unexpected(attached.error());
  }
  const std::span<const std::uint32_t> stops = impl->target->StopTokens();
  impl->stops.assign(stops.begin(), stops.end());
  return std::unique_ptr<SpecSplit>(new SpecSplit(std::move(impl)));
}

std::expected<GenerateOutcome, StatusCode> SpecSplit::GenerateStreaming(
    std::span<const std::uint32_t> prompt, const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token,
    const std::function<bool()>& cancelled) {
  if (!on_token) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  Impl& p = *impl_;
  if (prompt.empty() || prompt.size() > p.context) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  // Fresh caches per turn: a turn starts at position zero, so one split
  // can serve many sequential requests. The per-call Prepare allocates the
  // head's decode state on the fresh draft cache.
  p.target_cache = std::make_unique<core::DecodeCache>();
  p.target_cache->kv_type = options.kv_type;
  p.target_cache->kv_offload = options.kv_offload;
  p.draft_cache = std::make_unique<core::DecodeCache>();
  SpecSplitTarget target(p.target_engine->Owner(), *p.target, *p.target_cache,
                         p.draft_engine->Owner(), *p.draft, *p.draft_cache,
                         p.target_engine->Diagnostics(), p.prefill_chunk);
  const GenerateOptions framed = core::WithPrompt(options, prompt);
  // The driver runs on one thread, so a cancel between tokens is safe.
  const auto gated = [&](std::uint32_t token) {
    if (cancelled && cancelled()) {
      return false;
    }
    return on_token(token);
  };
  return core::RunDecode(target, p.strategy.get(), framed, gated);
}

std::expected<std::vector<std::uint32_t>, StatusCode> SpecSplit::Generate(
    std::span<const std::uint32_t> prompt, std::size_t max_tokens) {
  GenerateOptions options;
  options.max_completion_tokens = max_tokens;
  options.kv_type = impl_->kv_type;
  options.kv_offload = impl_->kv_offload;
  std::vector<std::uint32_t> produced;
  auto outcome =
      GenerateStreaming(prompt, options, [&produced](std::uint32_t token) {
        produced.push_back(token);
        return true;
      });
  if (!outcome) {
    return std::unexpected(outcome.error());
  }
  return produced;
}

const Tokenizer* SpecSplit::GetTokenizer() const {
  return impl_->target == nullptr ? nullptr : impl_->target->GetTokenizer();
}

Engine& SpecSplit::TargetEngine() {
  return *impl_->target_engine;
}

Model& SpecSplit::TargetModel() {
  return *impl_->target;
}

std::expected<std::string, StatusCode> SpecSplit::ChatPrompt(
    std::string_view user_text, bool enable_thinking,
    std::string_view reasoning_effort) const {
  if (impl_->target == nullptr) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return impl_->target->ChatPrompt(user_text, enable_thinking,
                                   reasoning_effort);
}

}  // namespace tessera
