#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "core/pipeline_impl.hpp"
#include "core/spec_driver.hpp"
#include "tessera/architecture.hpp"
#include "tessera/backend.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/speculative.hpp"

// Speculative decoding for the pipeline route. The decode driver runs on one
// thread; the target drives the stages in order, so the drafter runs on the
// last stage and every target forward runs across all stages. A
// hidden-conditioned drafter (DFlash2) captures target layers that may live
// on other stages, so every captured layer is copied to the drafter's stage.

namespace tessera {

// The single-thread pipeline decode target.
struct Pipeline::Impl::Target : public core::SpecTarget {
  explicit Target(Impl& impl) : impl_(impl) {}

  log::Diagnostics& diagnostics() override {
    return impl_.stages.back().engine->Diagnostics();
  }
  Backend& backend() override { return impl_.stages.back().engine->Owner(); }
  Model& model() override { return *impl_.stages.back().model; }
  core::DecodeCache& cache() override { return *impl_.stages.back().cache; }

  std::expected<void, StatusCode> Prepare(
      SpeculativeStrategy* strategy, const GenerateOptions& options) override {
    impl_.ResetCaches(options);
    strategy_ = strategy;
    if (strategy != nullptr) {
      auto prepared = strategy->Prepare(backend(), model());
      if (!prepared) {
        return std::unexpected(prepared.error());
      }
      strategy->SetDraftContext(options.draft_context);
      const std::span<const std::size_t> layers = strategy->CaptureLayers();
      capture_layers_.assign(layers.begin(), layers.end());
      const std::span<Buffer* const> buffers = strategy->CaptureBuffers();
      capture_buffers_.assign(buffers.begin(), buffers.end());
    }
    return {};
  }

  std::expected<void, StatusCode> Prefill(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      std::vector<float>* hidden, std::vector<float>* logits,
      double* compute_ms) override {
    (void)compute_ms;
    // A capturing drafter (DFlash2) reads the target hidden at its layers
    // during prefill, so it needs the batched capturing prefill. MTP
    // captures nothing and uses the one-token step below.
    if (strategy_ != nullptr && !capture_layers_.empty()) {
      return impl_.PrefillCapturing(*strategy_, prompt, options, hidden, logits,
                                    &position_);
    }
    for (std::size_t i = 0; i < prompt.size(); ++i) {
      const bool last = (i + 1 == prompt.size());
      auto out = impl_.Step(prompt[i], last, last ? hidden : nullptr);
      if (!out) {
        return std::unexpected(out.error());
      }
      if (last) {
        *logits = std::move(*out);
      }
      if (options.prefill_progress) {
        options.prefill_progress(i + 1, prompt.size());
      }
    }
    position_ = prompt.size();
    return {};
  }

  std::expected<std::uint32_t, StatusCode> GreedyStep(
      std::uint32_t token) override {
    auto out = impl_.Step(token, true, nullptr);
    if (!out) {
      return std::unexpected(out.error());
    }
    ++position_;
    return core::detail::ArgMax(*out);
  }

  std::expected<std::vector<float>, StatusCode> LogitsStep(
      std::uint32_t token, std::vector<float>* hidden) override {
    if (capture_layers_.empty()) {
      // MTP: the plain one-token cross-stage step, unchanged.
      auto out = impl_.Step(token, true, hidden);
      if (!out) {
        return std::unexpected(out.error());
      }
      ++position_;
      return std::move(*out);
    }
    const std::vector<std::size_t>* cap = &capture_layers_;
    std::vector<Buffer*>* cap_out = &capture_buffers_;
    std::vector<float> logits;
    std::vector<float> hidden_all;
    auto status = impl_.ForwardRows(std::span<const std::uint32_t>(&token, 1),
                                    /*snapshot_state=*/false, cap, cap_out,
                                    /*logits_all=*/true, &logits,
                                    hidden != nullptr ? &hidden_all : nullptr);
    if (!status) {
      return std::unexpected(status.error());
    }
    if (hidden != nullptr) {
      *hidden = std::move(hidden_all);
    }
    ++position_;
    return logits;
  }

  std::expected<DraftVerification, StatusCode> Verify(
      std::span<const std::uint32_t> draft,
      std::span<const float> prefix_logits, std::optional<std::uint32_t> anchor,
      std::vector<float>* hidden) override {
    const std::size_t prefix = position_;
    const std::vector<std::size_t>* cap =
        capture_layers_.empty() ? nullptr : &capture_layers_;
    std::vector<Buffer*>* cap_out =
        capture_layers_.empty() ? nullptr : &capture_buffers_;
    auto verified = impl_.VerifyBatch(draft, prefix_logits, anchor, hidden,
                                      prefix, cap, cap_out);
    if (!verified) {
      return std::unexpected(verified.error());
    }
    position_ += (anchor.has_value() ? 1 : 0) + verified->accepted;
    return verified;
  }

  void ReportDiagnostics(std::size_t loop_steps) override {
    (void)loop_steps;
  }

 private:
  Impl& impl_;
  std::size_t position_ = 0;
  SpeculativeStrategy* strategy_ = nullptr;
  std::vector<std::size_t> capture_layers_;
  std::vector<Buffer*> capture_buffers_;
};

std::expected<void, StatusCode> Pipeline::Impl::ForwardRows(
    std::span<const std::uint32_t> tokens, bool snapshot_state,
    const std::vector<std::size_t>* capture_layers,
    const std::vector<Buffer*>* capture_out, bool logits_all,
    std::vector<float>* logits_out, std::vector<float>* hidden_all_out) {
  const std::size_t n = stages.size();
  const std::size_t rows = tokens.size();
  if (rows == 0) {
    return {};
  }
  const std::size_t hidden = hidden_dim;
  const bool capturing = capture_layers != nullptr && !capture_layers->empty();
  // The owning stage and its slot for each captured target layer.
  std::vector<std::size_t> owner(capturing ? capture_layers->size() : 0, n);
  std::vector<std::size_t> slot(capturing ? capture_layers->size() : 0, 0);
  if (capturing) {
    for (std::size_t i = 0; i < capture_layers->size(); ++i) {
      for (std::size_t s = 0; s < n; ++s) {
        if ((*capture_layers)[i] >= stages[s].first_layer &&
            (*capture_layers)[i] < stages[s].last_layer) {
          owner[i] = s;
          break;
        }
      }
      if (owner[i] == n) {
        return std::unexpected(StatusCode::InvalidArgument);
      }
    }
  }
  // Boundary buffers (rows x hidden) and the last stage's final hidden.
  std::vector<std::unique_ptr<Buffer>> outgoing(n);
  std::vector<std::unique_ptr<Buffer>> incoming(n);
  for (std::size_t i = 0; i + 1 < n; ++i) {
    auto out = stages[i].engine->Owner().AllocateBuffer(rows * hidden * 4,
                                                        MemoryKind::Device);
    auto in = stages[i + 1].engine->Owner().AllocateBuffer(
        rows * hidden * 4, MemoryKind::Device);
    if (!out || !in) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    outgoing[i] = std::move(*out);
    incoming[i + 1] = std::move(*in);
  }
  auto last_made = stages.back().engine->Owner().AllocateBuffer(
      rows * hidden * 4, MemoryKind::Device);
  if (!last_made) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  std::unique_ptr<Buffer> last_hidden = std::move(*last_made);
  // Per-stage capture lists: the boundary layer first, then the strategy
  // layers this stage owns (captured into a temp, or the caller's buffer
  // when this stage is the drafter's).
  std::vector<std::vector<std::size_t>> stage_layers(n);
  std::vector<std::vector<Buffer*>> stage_ptrs(n);
  std::vector<std::vector<std::unique_ptr<Buffer>>> temps(n);
  for (std::size_t s = 0; s < n; ++s) {
    stage_layers[s].push_back(stages[s].boundary_layer);
  }
  if (capturing) {
    for (std::size_t i = 0; i < capture_layers->size(); ++i) {
      const std::size_t s = owner[i];
      if (s + 1 == n) {
        // The drafter's stage captures straight into the caller's buffer.
        slot[i] = stage_ptrs[s].size();
        stage_layers[s].push_back((*capture_layers)[i]);
        stage_ptrs[s].push_back((*capture_out)[i]);
      } else {
        auto buf = stages[s].engine->Owner().AllocateBuffer(
            rows * hidden * 4, MemoryKind::Device);
        if (!buf) {
          return std::unexpected(StatusCode::OutOfMemory);
        }
        temps[s].push_back(std::move(*buf));
        slot[i] = stage_ptrs[s].size();
        stage_layers[s].push_back((*capture_layers)[i]);
        stage_ptrs[s].push_back(temps[s].back().get());
      }
    }
  }
  // Boundary capture buffer per stage (last stage -> last_hidden).
  for (std::size_t s = 0; s < n; ++s) {
    Buffer* boundary = (s + 1 == n) ? last_hidden.get() : outgoing[s].get();
    stage_ptrs[s].insert(stage_ptrs[s].begin(), boundary);
  }
  std::vector<float> flat;
  for (std::size_t i = 0; i < n; ++i) {
    Stage& stage = stages[i];
    const Architecture* arch = stage.model->Arch();
    if (arch == nullptr) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    const bool last = (i + 1 == n);
    std::vector<std::uint32_t> placeholder;
    std::span<const std::uint32_t> stage_tokens = tokens;
    if (i > 0) {
      placeholder.assign(rows, 0);
      stage_tokens = placeholder;
    }
    const Buffer* embeddings = i > 0 ? incoming[i].get() : nullptr;
    auto status = arch->ForwardBatch(
        stage.engine->Owner(), *stage.model, *stage.cache, stage_tokens,
        (logits_all && last) ? &flat : nullptr, nullptr, (logits_all && last),
        embeddings, &stage_layers[i], &stage_ptrs[i], snapshot_state);
    if (!status) {
      return std::unexpected(status.error());
    }
    if (!last) {
      auto copied = CopyAcrossDevices(*outgoing[i], 0, *incoming[i + 1], 0,
                                      rows * hidden * 4);
      if (!copied) {
        return std::unexpected(copied.error());
      }
    }
  }
  // Copy the layers another stage owns into the drafter's buffers.
  if (capturing) {
    for (std::size_t i = 0; i < capture_layers->size(); ++i) {
      const std::size_t s = owner[i];
      if (s + 1 == n) {
        continue;
      }
      auto copied = CopyAcrossDevices(*temps[s][slot[i]], 0, *(*capture_out)[i],
                                      0, rows * hidden * 4);
      if (!copied) {
        return std::unexpected(copied.error());
      }
    }
  }
  if (logits_out != nullptr) {
    *logits_out = std::move(flat);
  }
  if (hidden_all_out != nullptr) {
    hidden_all_out->resize(rows * hidden);
    if (!stages.back().engine->Owner().CopyD2H(
            *last_hidden,
            reinterpret_cast<std::byte*>(hidden_all_out->data()),
            rows * hidden * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  return {};
}

std::expected<DraftVerification, StatusCode> Pipeline::Impl::VerifyBatch(
    std::span<const std::uint32_t> draft, std::span<const float> prefix_logits,
    std::optional<std::uint32_t> anchor, std::vector<float>* hidden_out,
    std::size_t prefix_position, const std::vector<std::size_t>* capture_layers,
    const std::vector<Buffer*>* capture_out) {
  const std::size_t lead = anchor.has_value() ? 1 : 0;
  std::vector<std::uint32_t> tokens;
  tokens.reserve(draft.size() + lead);
  if (lead == 1) {
    tokens.push_back(*anchor);
  }
  tokens.insert(tokens.end(), draft.begin(), draft.end());
  const std::size_t rows = tokens.size();
  if (rows == 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto config = stages.back().model->Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const std::size_t hidden = hidden_dim;
  const std::size_t vocab = config->vocab_size;
  std::vector<float> flat;
  std::vector<float> hidden_all;
  auto forwarded = ForwardRows(tokens, /*snapshot_state=*/true, capture_layers,
                               capture_out, /*logits_all=*/true, &flat,
                               &hidden_all);
  if (!forwarded) {
    return std::unexpected(forwarded.error());
  }
  DraftVerification result;
  std::size_t accepted = 0;
  const auto logits_row = [&](std::size_t row) {
    return std::span<const float>(flat.data() + row * vocab, vocab);
  };
  if (lead == 1) {
    for (std::size_t i = 0; i < draft.size(); ++i) {
      if (core::detail::ArgMax(logits_row(i)) != draft[i]) {
        break;
      }
      ++accepted;
    }
    result.accepted = accepted;
    result.next_token = core::detail::ArgMax(logits_row(accepted));
  } else {
    std::span<const float> last = prefix_logits;
    for (std::size_t i = 0; i < draft.size(); ++i) {
      if (core::detail::ArgMax(last) != draft[i]) {
        break;
      }
      ++accepted;
      last = logits_row(i + lead);
    }
    result.accepted = accepted;
    result.logits.assign(last.begin(), last.end());
    result.next_token = core::detail::ArgMax(result.logits);
  }
  if (hidden_out != nullptr && (accepted > 0 || lead == 1)) {
    const std::size_t row = lead == 1 ? accepted : accepted - 1;
    hidden_out->assign(hidden_all.begin() + row * hidden,
                       hidden_all.begin() + (row + 1) * hidden);
  }
  const std::size_t keep = prefix_position + lead + accepted;
  for (Stage& stage : stages) {
    const Architecture* arch = stage.model->Arch();
    if (arch == nullptr) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    auto truncated = arch->TruncateTarget(stage.engine->Owner(), *stage.model,
                                          *stage.cache, keep);
    if (!truncated) {
      return std::unexpected(truncated.error());
    }
  }
  return result;
}

std::expected<void, StatusCode> Pipeline::Impl::PrefillCapturing(
    SpeculativeStrategy& strategy, std::span<const std::uint32_t> prompt,
    const GenerateOptions& options, std::vector<float>* hidden_out,
    std::vector<float>* logits_out, std::size_t* position_out) {
  const std::size_t hidden = hidden_dim;
  const std::span<const std::size_t> capture_span = strategy.CaptureLayers();
  const std::vector<std::size_t> capture_layers(capture_span.begin(),
                                                capture_span.end());
  const std::size_t tail = strategy.PrefillCaptureTail();
  const std::size_t keep =
      (tail == 0 || tail >= prompt.size()) ? prompt.size() : tail;
  const std::size_t head = prompt.size() - keep;
  const std::size_t chunk =
      ResolvePrefillChunkTokens(0, options.prefill_chunk_tokens, 0);
  Backend& last_backend = stages.back().engine->Owner();
  Model& last_model = *stages.back().model;
  core::DecodeCache& last_cache = *stages.back().cache;
  // A strategy without a batched append must not use this path.
  {
    auto probe =
        strategy.AppendPrefill(last_backend, last_model, last_cache, {}, 0, {});
    if (!probe) {
      return std::unexpected(probe.error());
    }
  }
  const auto run_chunk = [&](std::size_t off, std::size_t len, bool capture,
                             bool last) -> std::expected<void, StatusCode> {
    const std::size_t rows = len;
    // Chunk-sized capture buffers on the drafter's stage.
    std::vector<std::unique_ptr<Buffer>> store;
    std::vector<Buffer*> buffers;
    for (std::size_t i = 0; i < capture_layers.size(); ++i) {
      auto buf = stages.back().engine->Owner().AllocateBuffer(
          rows * hidden * 4, MemoryKind::Device);
      if (!buf) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      store.push_back(std::move(*buf));
      buffers.push_back(store.back().get());
    }
    std::vector<float> logits;
    std::vector<float> hidden_all;
    auto status = ForwardRows(
        prompt.subspan(off, rows), /*snapshot_state=*/false,
        capture ? &capture_layers : nullptr, capture ? &buffers : nullptr,
        last, last ? &logits : nullptr, (last && hidden_out != nullptr)
                                           ? &hidden_all
                                           : nullptr);
    if (!status) {
      return status;
    }
    if (capture) {
      auto appended = strategy.AppendPrefill(last_backend, last_model,
                                             last_cache, prompt.subspan(off, rows),
                                             off, buffers);
      if (!appended) {
        return std::unexpected(appended.error());
      }
    }
    if (last) {
      if (logits_out != nullptr) {
        *logits_out = std::move(logits);
      }
      if (hidden_out != nullptr && hidden_all.size() >= hidden) {
        hidden_out->assign(hidden_all.end() - hidden, hidden_all.end());
      }
    }
    return {};
  };
  for (std::size_t off = 0; off < head;) {
    const std::size_t len = std::min(chunk, head - off);
    if (auto status = run_chunk(off, len, false, false); !status) {
      return status;
    }
    if (options.prefill_progress) {
      options.prefill_progress(off + len, prompt.size());
    }
    off += len;
  }
  for (std::size_t off = head; off < prompt.size();) {
    const std::size_t len = std::min(chunk, prompt.size() - off);
    const bool last = (off + len == prompt.size());
    if (auto status = run_chunk(off, len, true, last); !status) {
      return status;
    }
    if (options.prefill_progress) {
      options.prefill_progress(off + len, prompt.size());
    }
    off += len;
  }
  if (position_out != nullptr) {
    *position_out = prompt.size();
  }
  return {};
}

std::expected<GenerateOutcome, StatusCode> Pipeline::Impl::RunSpeculative(
    std::span<const std::uint32_t> prompt, const GenerateOptions& options,
    std::vector<std::uint32_t>* produced) {
  if (prompt.empty() || prompt.size() > context) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  produced->clear();
  Target target(*this);
  auto outcome = core::RunDecode(
      target, strategy.get(), options,
      [produced](std::uint32_t token) {
        produced->push_back(token);
        return true;
      });
  if (!outcome) {
    return std::unexpected(outcome.error());
  }
  return *outcome;
}

std::expected<void, StatusCode> Pipeline::AttachSpeculative(
    const std::function<std::unique_ptr<SpeculativeStrategy>()>& factory,
    const StrategyOptions& options) {
  if (!factory) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto strategy = factory();
  if (strategy == nullptr) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto attached = strategy->Attach(options);
  if (!attached) {
    return std::unexpected(attached.error());
  }
  impl_->strategy = std::move(strategy);
  return {};
}

}  // namespace tessera
