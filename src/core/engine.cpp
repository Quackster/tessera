#include "tessera/engine.hpp"

#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "core/sampling.hpp"

namespace tessera {

namespace {
// The stop set for one request: the model's declared stop tokens first, then
// the caller's extras from GenerateOptions, without duplicates.
std::vector<std::uint32_t> StopSet(const Model& model,
                                   const GenerateOptions& options) {
  const std::span<const std::uint32_t> declared = model.StopTokens();
  std::vector<std::uint32_t> stops(declared.begin(), declared.end());
  for (std::uint32_t token : options.stop_tokens) {
    if (std::find(stops.begin(), stops.end(), token) == stops.end()) {
      stops.push_back(token);
    }
  }
  return stops;
}
}  // namespace

namespace {

// The draft context of a hidden-conditioned drafter only needs the
// prompt tail (DFlash2 keeps its recent window); earlier positions
// prefill in chunk-sized forwards without capture, and the tail runs
// chunked with engine-owned capture buffers plus one strategy append
// per chunk. Only a strategy without a batched append keeps the
// per-token loop. Every chunk stays within the per-launch attention
// work budget (see core::detail::ClampPrefillRows); the cache is empty
// (prefill starts a request), so the keys a chunk's last row sees end
// at its absolute end offset. Null progress disables chunk reports.
std::expected<void, StatusCode> PrefillCapturing(
    Backend& backend, Model& model, core::DecodeCache& cache,
    SpeculativeStrategy& strategy, std::span<const std::uint32_t> prompt,
    std::size_t chunk, const std::vector<std::size_t>& capture_layers,
    std::vector<Buffer*>& strategy_buffers, std::vector<float>& hidden_out,
    std::vector<float>& logits_out, const core::PrefillProgress* progress) {
  const std::size_t tail = strategy.PrefillCaptureTail();
  const std::size_t keep =
      (tail == 0 || tail >= prompt.size()) ? prompt.size() : tail;
  const std::size_t head = prompt.size() - keep;
  const Architecture* arch = model.Arch();
  // An empty call probes support: Ok selects the batched tail below,
  // UnsupportedFeature the per-token loop, anything else fails fast.
  bool batched = arch != nullptr;
  if (batched) {
    auto probe = strategy.AppendPrefill(backend, model, cache, {}, 0, {});
    if (!probe && probe.error() == StatusCode::UnsupportedFeature) {
      batched = false;
    } else if (!probe) {
      return std::unexpected(probe.error());
    }
  }
  if (!batched) {
    std::vector<Buffer*>* capture_ptr =
        capture_layers.empty() ? nullptr : &strategy_buffers;
    const std::vector<std::size_t>* layers_ptr =
        capture_layers.empty() ? nullptr : &capture_layers;
    for (std::size_t i = 0; i < prompt.size(); ++i) {
      const bool last = i + 1 == prompt.size();
      if (last) {
        auto logits = core::DecodeLogits(backend, model, cache, prompt[i],
                                         &hidden_out, layers_ptr, capture_ptr);
        if (!logits) {
          return std::unexpected(logits.error());
        }
        logits_out = std::move(*logits);
      } else {
        auto forwarded = core::DecodeForward(backend, model, cache, prompt[i],
                                            &hidden_out, nullptr, layers_ptr,
                                            capture_ptr);
        if (!forwarded) {
          return std::unexpected(forwarded.error());
        }
      }
      auto anchored =
          strategy.OnAnchor(backend, model, cache, prompt[i], i, hidden_out);
      if (!anchored) {
        return std::unexpected(anchored.error());
      }
      if (progress != nullptr) {
        (*progress)(i + 1, prompt.size());
      }
    }
    return {};
  }
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  const std::size_t hidden = config->hidden_dim;
  const std::size_t heads = config->attention.heads;
  const std::size_t head_dim = config->attention.head_dim;
  // One forward per chunk, each within the attention work budget. The
  // base is the keys committed before the region (absolute offset).
  const std::vector<std::size_t> head_lens =
      core::detail::PrefillLens(head, chunk, /*base=*/0, heads, head_dim);
  std::size_t off = 0;
  for (const std::size_t len : head_lens) {
    auto status = arch->ForwardBatch(backend, model, cache,
                                     prompt.subspan(off, len), nullptr,
                                     nullptr, /*all_logits=*/false, nullptr);
    if (!status) {
      return std::unexpected(status.error());
    }
    off += len;
    if (progress != nullptr) {
      (*progress)(off, prompt.size());
    }
  }
  const std::size_t rows = std::min(chunk, keep);
  std::vector<std::unique_ptr<Buffer>> cap_store;
  std::vector<Buffer*> cap_ptrs;
  for (std::size_t i = 0; i < capture_layers.size(); ++i) {
    auto made = backend.AllocateBuffer(rows * hidden * 4, MemoryKind::Device);
    if (!made) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    cap_ptrs.push_back(made->get());
    cap_store.push_back(std::move(*made));
  }
  const std::vector<std::size_t> tail_lens =
      core::detail::PrefillLens(keep, chunk, /*base=*/head, heads, head_dim);
  std::size_t tail_done = 0;
  for (const std::size_t len : tail_lens) {
    const std::size_t off = head + tail_done;
    const bool last = off + len == prompt.size();
    std::vector<float> chunk_logits;
    auto status = arch->ForwardBatch(
        backend, model, cache, prompt.subspan(off, len),
        last ? &chunk_logits : nullptr, last ? &hidden_out : nullptr,
        /*all_logits=*/false, nullptr, &capture_layers, &cap_ptrs);
    if (!status) {
      return std::unexpected(status.error());
    }
    auto appended = strategy.AppendPrefill(backend, model, cache,
                                           prompt.subspan(off, len), off,
                                           cap_ptrs);
    if (!appended) {
      return std::unexpected(appended.error());
    }
    if (last) {
      logits_out = std::move(chunk_logits);
    }
    tail_done += len;
    if (progress != nullptr) {
      (*progress)(head + tail_done, prompt.size());
    }
  }
  return {};
}

}  // namespace

Engine::Engine(std::unique_ptr<Backend> backend, log::Diagnostics diagnostics,
               std::size_t prefill_chunk_tokens)
    : backend_(std::move(backend)),
      diagnostics_(std::move(diagnostics)),
      prefill_chunk_tokens_(prefill_chunk_tokens) {}

std::expected<std::unique_ptr<Engine>, StatusCode> Engine::Create(
    const EngineOptions& options) {
  log::Diagnostics log = options.diagnostics;
  auto backend = CreateBackend();
  if (!backend) {
    log.Error("engine", "CreateBackend returned null; the build has no backend");
    return std::unexpected(StatusCode::DeviceError);
  }
  auto engine = std::unique_ptr<Engine>(
      new Engine(std::move(backend), log, options.prefill_chunk_tokens));
  // The backend logs through the engine's diagnostics copy.
  engine->backend_->SetDiagnostics(&engine->diagnostics_);
  engine->backend_->SetDeviceIndex(options.device_index);
  auto init = engine->backend_->Init();
  if (!init) {
    engine->diagnostics_.Error(
        "engine", std::string("backend init failed: ") +
                      std::string(ToString(init.error())) +
                      "; engine not started");
    return std::unexpected(init.error());
  }
  engine->diagnostics_.Info(
      "engine", std::string("engine started on ") +
                   std::string(engine->backend_->Name()) + " (device " +
                   std::to_string(options.device_index) + ": " +
                   std::string(engine->backend_->DeviceName()) + ")");
  return engine;
}

std::expected<std::unique_ptr<Model>, StatusCode> Engine::LoadModel(
    const ModelOptions& options) {
  auto model = Model::Load(*backend_, options);
  if (!model) {
    diagnostics_.Warn(
        "model", std::string("load failed for '") + options.path + "': " +
                     std::string(ToString(model.error())) +
                     "; check the path and file format");
    return model;
  }
  auto& loaded = *model;
  const char* format_name =
      loaded->Format() == ModelFormat::Gguf ? "gguf" : "mxfp4";
  diagnostics_.Info(
      "model", std::string("loaded ") + format_name + " model with " +
                   std::to_string(loaded->Tensors().size()) + " tensors");
  return model;
}

std::expected<std::vector<std::uint32_t>, StatusCode>
Engine::GenerateMultimodal(Model& model, const GenerateOptions& options,
                           std::span<const float> image_embeddings,
                           std::size_t image_tokens,
                           std::uint32_t image_token_id) {
  auto config = model.Config();
  if (!config) {
    return std::unexpected(config.error());
  }
  if (image_embeddings.size() != image_tokens * config->hidden_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const auto started = std::chrono::steady_clock::now();
  const auto elapsed_ms = [&started]() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - started)
        .count();
  };
  core::DecodeCache cache;
  cache.kv_type = options.kv_type;
  std::vector<std::uint32_t> prompt = options.prompt_tokens;
  if (prompt.empty()) {
    prompt.push_back(options.first_token);
  }
  // A zero count fills the remaining context.
  const std::size_t max_tokens =
      model.EffectiveMaxTokens(prompt.size(), options.max_tokens);
  if (max_tokens == 0) {
    return std::vector<std::uint32_t>{};
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
  const std::size_t prefill_chunk = ResolvePrefillChunkTokens(
      prefill_chunk_tokens_, options.prefill_chunk_tokens,
      model.MaxContextLength());
  if (options.progress_every > 0) {
    diagnostics_.Info("engine", std::string("multimodal prefill: ") +
                                    std::to_string(prompt.size()) +
                                    " token(s), " +
                                    std::to_string(image_tokens) + " image, chunk " +
                                    std::to_string(prefill_chunk));
  }
  const std::size_t hidden = config->hidden_dim;
  auto embed = core::detail::NeedWeightAny(model, "token_embd.weight");
  if (!embed) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto embedding_buffer =
      backend_->AllocateBuffer(prompt.size() * hidden * 4, MemoryKind::Device);
  if (!embedding_buffer) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  // Assemble one embedding per row: the image rows replace the placeholder
  // token's gathered row. Then prefill in chunk-sized forwards.
  std::vector<float> host(prompt.size() * hidden);
  std::vector<float> row(hidden);
  std::size_t used_images = 0;
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    float* dst = host.data() + i * hidden;
    if (prompt[i] == image_token_id && used_images < image_tokens) {
      std::copy(image_embeddings.begin() + used_images * hidden,
                image_embeddings.begin() + (used_images + 1) * hidden, dst);
      ++used_images;
    } else {
      auto gathered = core::detail::GatherEmbedding(*backend_, **embed,
                                                    prompt[i], hidden, row);
      if (!gathered) {
        return std::unexpected(gathered.error());
      }
      std::copy(row.begin(), row.end(), dst);
    }
  }
  if (!backend_->CopyH2D(**embedding_buffer,
                         std::span<const std::byte>(
                             reinterpret_cast<const std::byte*>(host.data()),
                             host.size() * 4))) {
    return std::unexpected(StatusCode::DeviceError);
  }
  auto first = core::PrefillTokens(*backend_, model, cache, prompt, nullptr,
                                    embedding_buffer->get(), prefill_chunk);
  if (!first) {
    return std::unexpected(first.error());
  }
  std::vector<float> current = std::move(*first);
  std::vector<std::uint32_t> produced;
  produced.reserve(max_tokens);
  std::mt19937_64 rng(options.seed);
  std::vector<std::uint32_t> history = prompt;
  const auto pick = [&](std::span<const float> logits) {
    return options.sample
               ? core::SampleToken(logits, options.sampling, history, rng)
               : core::detail::ArgMax(logits);
  };
  const std::vector<std::uint32_t> stops = StopSet(model, options);
  std::uint32_t next = pick(current);
  bool stopped = false;
  while (produced.size() < max_tokens) {
    if (core::detail::IsStopToken(next, stops)) {
      stopped = true;
      break;
    }
    produced.push_back(next);
    if (produced.size() >= max_tokens) {
      break;
    }
    history.push_back(next);
    auto logits = core::DecodeLogits(*backend_, model, cache, next);
    if (!logits) {
      return std::unexpected(logits.error());
    }
    next = pick(*logits);
  }
  if (stopped) {
    diagnostics_.Info("engine", std::string("multimodal: stopped at declared "
                                            "stop token ") +
                                    std::to_string(next));
  }
  if (!produced.empty()) {
    const long long ms = elapsed_ms();
    diagnostics_.Info(
        "engine",
        std::string("generated ") + std::to_string(produced.size()) +
            " token(s) in " + std::to_string(ms) + " ms (" +
            std::to_string(ms / static_cast<long long>(produced.size())) +
            " ms/token)");
  }
  return produced;
}

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

std::expected<void, StatusCode> Engine::AttachSpeculative(
    std::unique_ptr<SpeculativeStrategy>&& strategy) {
  if (!strategy) {
    diagnostics_.Warn("engine", "AttachSpeculative: null strategy");
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (speculative_) {
    diagnostics_.Warn("engine",
                     "AttachSpeculative: a strategy is already attached");
    return std::unexpected(StatusCode::InvalidArgument);
  }
  speculative_ = std::move(strategy);
  diagnostics_.Info(
      "engine", std::string("speculative strategy '") +
                   std::string(speculative_->Name()) + "' attached");
  return {};
}

std::expected<std::size_t, StatusCode> Engine::GenerateStreaming(
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
  std::vector<std::uint32_t> prompt = options.prompt_tokens;
  if (prompt.empty()) {
    prompt.push_back(options.first_token);
  }
  // A zero count fills the remaining context.
  const std::size_t max_tokens =
      model.EffectiveMaxTokens(prompt.size(), options.max_tokens);
  if (max_tokens == 0) {
    return 0;
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
  // shows movement instead of silence. Zero disables the reports.
  std::size_t progress_calls = 0;
  core::PrefillProgress progress = [&](std::size_t done, std::size_t total) {
    ++progress_calls;
    if (done == total || progress_calls % options.progress_every == 1) {
      diagnostics_.Info("engine", std::string("prefill: ") +
                                      std::to_string(done) + "/" +
                                      std::to_string(total) + " rows");
    }
  };
  const core::PrefillProgress* progress_ptr =
      options.progress_every > 0 ? &progress : nullptr;
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
  std::size_t produced = 0;
  std::size_t spec_proposed = 0;
  std::size_t spec_accepted = 0;
  std::size_t spec_steps = 0;
  bool stopped = false;
  while (produced < max_tokens) {
    if (core::detail::IsStopToken(next, stops)) {
      stopped = true;
      break;
    }
    if (!on_token(next)) {
      break;
    }
    ++produced;
    if (produced >= max_tokens) {
      break;
    }
    history.push_back(next);
    // A folding strategy rides the anchor into the verify batch, so the trunk
    // is read once per step. The drafter context already excludes this anchor:
    // OnAnchor runs after the verify and appends the anchor it produced.
    if (strategy != nullptr && block > 0 && strategy->FoldsAnchor()) {
      auto count = strategy->Draft(*backend_, model, cache, {}, next, drafts);
      if (!count) {
        return std::unexpected(count.error());
      }
      if (*count <= 1) {
        // No usable proposal: decode the anchor alone and take its greedy
        // token. The drafter context still gains the anchor below.
        auto logits = core::DecodeLogits(*backend_, model, cache, next, &hidden,
                                         capture_layers_ptr, capture_ptr);
        if (!logits) {
          return std::unexpected(logits.error());
        }
        auto anchored = strategy->OnAnchor(*backend_, model, cache, next,
                                           position, hidden);
        if (!anchored) {
          return std::unexpected(anchored.error());
        }
        // A folding strategy defers its context append to Commit, so commit
        // the anchor alone (zero accepted drafts) to keep the context in step.
        auto committed = strategy->Commit(*backend_, model, cache, 0);
        if (!committed) {
          return std::unexpected(committed.error());
        }
        next = pick(*logits);
        ++position;
        continue;
      }
      std::span<const std::uint32_t> proposal(drafts.data(), *count);
      auto verify =
          core::VerifyDraft(*backend_, model, cache, proposal, {}, next,
                            /*hidden_out=*/nullptr, capture_layers_ptr,
                            capture_ptr);
      if (!verify) {
        return std::unexpected(verify.error());
      }
      auto anchored = strategy->OnAnchor(*backend_, model, cache, next,
                                         position, {});
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
      for (std::size_t i = 0; i < verify->accepted; ++i) {
        if (produced >= max_tokens) {
          break;
        }
        if (core::detail::IsStopToken(drafts[i], stops)) {
          stopped = true;
          break;
        }
        if (!on_token(drafts[i])) {
          stopped = true;
          break;
        }
        history.push_back(drafts[i]);
        ++produced;
      }
      if (stopped || produced >= max_tokens) {
        break;
      }
      next = verify->next_token;
      position += 1 + verify->accepted;
      continue;
    }
    // Advance the target by the just-emitted token. `hidden` describes its
    // position, which the drafter chains from.
    auto logits = core::DecodeLogits(*backend_, model, cache, next, &hidden,
                                     capture_layers_ptr, capture_ptr);
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
      auto count =
          strategy->Draft(*backend_, model, cache, *logits, next, drafts);
      if (!count) {
        return std::unexpected(count.error());
      }
      if (*count > 0) {
        std::span<const std::uint32_t> proposal(drafts.data(), *count);
        auto verify = core::VerifyDraft(*backend_, model, cache, proposal,
                                        *logits, std::nullopt, &hidden,
                                        capture_layers_ptr, capture_ptr);
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
          if (produced >= max_tokens) {
            break;
          }
          if (core::detail::IsStopToken(drafts[i], stops)) {
            stopped = true;
            break;
          }
          if (!on_token(drafts[i])) {
            stopped = true;
            break;
          }
          history.push_back(drafts[i]);
          ++produced;
        }
        if (stopped || produced >= max_tokens) {
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
  if (produced > 0) {
    const long long decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() -
                                    decode_started)
                                    .count();
    const long long decode_tps =
        decode_ms > 0
            ? static_cast<long long>(produced) * 1000 / decode_ms
            : 0;
    diagnostics_.Info(
        "engine", std::string("decode: ") + std::to_string(produced) +
                      " token(s) in " + std::to_string(decode_ms) + " ms (" +
                      std::to_string(decode_tps) + " tok/s, " +
                      std::to_string(decode_ms /
                                     static_cast<long long>(produced)) +
                      " ms/token)");
  }
  return produced;
}


std::expected<std::vector<std::uint32_t>, StatusCode> Engine::Generate(
    Model& model, const GenerateOptions& options) {
  std::vector<std::uint32_t> produced;
  produced.reserve(options.max_tokens);
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

Backend& Engine::Owner() {
  return *backend_;
}

const SpeculativeStrategy* Engine::Speculative() const {
  return speculative_.get();
}

log::Diagnostics& Engine::Diagnostics() {
  return diagnostics_;
}

std::size_t Engine::PrefillChunkTokens() const {
  return prefill_chunk_tokens_;
}

}  // namespace tessera
