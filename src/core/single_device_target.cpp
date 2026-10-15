#include "core/single_device_target.hpp"

#include <cstddef>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/generate_helpers.hpp"
#include "tessera/architecture.hpp"

// The single-device SpecTarget (see single_device_target.hpp). Every op is
// the core decode call the engine used before the seam, so the decode
// output is unchanged.

namespace tessera::core {

SingleDeviceTarget::SingleDeviceTarget(Backend& backend, Model& model,
                                       DecodeCache& cache,
                                       log::Diagnostics& diagnostics,
                                       std::size_t prefill_chunk,
                                       Profile* profile,
                                       bool single_token_prefill)
    : backend_(backend),
      model_(model),
      cache_(cache),
      diagnostics_(diagnostics),
      prefill_chunk_(prefill_chunk),
      profile_(profile),
      single_token_prefill_(single_token_prefill) {}

std::expected<void, StatusCode> SingleDeviceTarget::Prepare(
    SpeculativeStrategy* strategy, const GenerateOptions& options) {
  strategy_ = strategy;
  if (strategy != nullptr) {
    auto prepared = strategy->Prepare(backend_, model_);
    if (!prepared) {
      return std::unexpected(prepared.error());
    }
    // A request may override the draft context window; set it before the
    // prefill so the capture tail matches.
    strategy->SetDraftContext(options.draft_context);
    const std::span<const std::size_t> layers = strategy->CaptureLayers();
    capture_layers_.assign(layers.begin(), layers.end());
    const std::span<Buffer* const> buffers = strategy->CaptureBuffers();
    capture_buffers_.assign(buffers.begin(), buffers.end());
  }
  // Hoist the one-time trunk init (kernel loads, scratch) out of the
  // prefill window. Idempotent: a later batched forward skips it.
  if (const Architecture* arch = model_.Arch(); arch != nullptr) {
    auto ready = arch->EnsureReady(backend_, model_, cache_);
    if (!ready && ready.error() != StatusCode::UnsupportedFeature) {
      diagnostics_.Warn("engine", std::string("trunk init failed: ") +
                                      std::string(ToString(ready.error())));
      return std::unexpected(ready.error());
    }
  }
  return {};
}

std::expected<void, StatusCode> SingleDeviceTarget::Prefill(
    std::span<const std::uint32_t> prompt, const GenerateOptions& options,
    std::vector<float>* hidden, std::vector<float>* logits,
    double* compute_ms) {
  if (single_token_prefill_) {
    // One token per forward: the tensor-parallel rank path, where the
    // collective forward (all-reduce) runs per token. The last token scores
    // the logits and its hidden feeds the drafter.
    const Architecture* arch = model_.Arch();
    if (arch == nullptr) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    // A capturing strategy (DFlash2) reads the target hidden at its layers,
    // so the rank path passes the capture buffers when the drafter asked for
    // them. MTP captures nothing and passes null.
    const std::vector<std::size_t>* layers =
        capture_layers_.empty() ? nullptr : &capture_layers_;
    std::vector<Buffer*>* buffers =
        capture_layers_.empty() ? nullptr : &capture_buffers_;
    for (std::size_t i = 0; i < prompt.size(); ++i) {
      const bool last = (i + 1 == prompt.size());
      if (last) {
        auto scored = arch->Logits(backend_, model_, cache_, prompt[i], hidden,
                                   layers, buffers, nullptr);
        if (!scored) {
          return std::unexpected(scored.error());
        }
        *logits = std::move(*scored);
      } else {
        auto forwarded = arch->Forward(backend_, model_, cache_, prompt[i],
                                       nullptr, layers, buffers, nullptr);
        if (!forwarded) {
          return std::unexpected(forwarded.error());
        }
      }
      if (options.prefill_progress) {
        options.prefill_progress(i + 1, prompt.size());
      }
    }
    return {};
  }
  const std::size_t prefill_chunk = ResolvePrefillChunkTokens(
      prefill_chunk_, options.prefill_chunk_tokens, model_.MaxContextLength());
  if (options.progress_every > 0) {
    diagnostics_.Info("engine", std::string("prefill: ") +
                                    std::to_string(prompt.size()) +
                                    " prompt token(s), chunk " +
                                    std::to_string(prefill_chunk));
  }
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
  if (!capture_layers_.empty() && strategy_ != nullptr) {
    return PrefillCapturing(backend_, model_, cache_, *strategy_, prompt,
                            prefill_chunk, capture_layers_, capture_buffers_,
                            *hidden, *logits, progress_ptr, compute_ms);
  }
  auto prefilled = core::PrefillTokens(backend_, model_, cache_, prompt, hidden,
                                       nullptr, prefill_chunk, progress_ptr,
                                       compute_ms);
  if (!prefilled) {
    return std::unexpected(prefilled.error());
  }
  *logits = std::move(*prefilled);
  return {};
}

std::expected<std::uint32_t, StatusCode> SingleDeviceTarget::GreedyStep(
    std::uint32_t token) {
  return core::DecodeToken(backend_, model_, cache_, token);
}

std::expected<std::vector<float>, StatusCode> SingleDeviceTarget::LogitsStep(
    std::uint32_t token, std::vector<float>* hidden) {
  core::PhaseScope scope(profile_, core::Phase::kAnchor);
  const std::vector<std::size_t>* layers =
      capture_layers_.empty() ? nullptr : &capture_layers_;
  std::vector<Buffer*>* buffers =
      capture_layers_.empty() ? nullptr : &capture_buffers_;
  return core::DecodeLogits(backend_, model_, cache_, token, hidden, layers,
                            buffers);
}

std::expected<DraftVerification, StatusCode> SingleDeviceTarget::Verify(
    std::span<const std::uint32_t> draft,
    std::span<const float> prefix_logits, std::optional<std::uint32_t> anchor,
    std::vector<float>* hidden) {
  core::PhaseScope scope(profile_, core::Phase::kVerify);
  const std::vector<std::size_t>* layers =
      capture_layers_.empty() ? nullptr : &capture_layers_;
  std::vector<Buffer*>* buffers =
      capture_layers_.empty() ? nullptr : &capture_buffers_;
  return core::VerifyDraft(backend_, model_, cache_, draft, prefix_logits,
                           anchor, hidden, layers, buffers);
}

void SingleDeviceTarget::ReportDiagnostics(std::size_t loop_steps) {
  if (profile_ != nullptr) {
    profile_->Report(diagnostics_, "engine", loop_steps);
  }
  // Kernel-cache hit rates per phase: a low rate means the same kernels
  // reload every token instead of once.
  if (cache_.device != nullptr) {
    cache_.device->gemms.Report(diagnostics_, "engine", "decode gemm");
    cache_.device->embed_kernels.Report(diagnostics_, "engine",
                                        "decode embedding");
  }
  if (cache_.arch != nullptr) {
    cache_.arch->ReportCaches(diagnostics_, "engine");
  }
}

}  // namespace tessera::core
