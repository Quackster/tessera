#include "tessera/context_parallel.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/generate_helpers.hpp"
#include "core/single_device_target.hpp"
#include "core/spec_driver.hpp"
#include "core/think_budget.hpp"
#include "tessera/architecture.hpp"
#include "tessera/backend.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"

namespace tessera {

namespace {

bool IsStop(std::uint32_t token, std::span<const std::uint32_t> stops) {
  for (const std::uint32_t stop : stops) {
    if (stop == token) {
      return true;
    }
  }
  return false;
}

}  // namespace

struct ContextParallel::Impl {
  // One rank: its engine, its full model, its decode cache with only its
  // key chunk, and its attention staging buffers (see CpContext).
  struct Rank {
    std::unique_ptr<Engine> engine;
    std::unique_ptr<Model> model;
    std::unique_ptr<core::DecodeCache> cache;
    std::unique_ptr<Buffer> partial;
    std::unique_ptr<Buffer> attn;
    // The rank's speculative strategy (null when none is attached).
    std::unique_ptr<SpeculativeStrategy> strategy;
  };
  std::vector<Rank> ranks;
  core::TpBarrier barrier;
  std::vector<std::unique_ptr<core::CpContext>> contexts;
  std::size_t context = 0;
  KvCacheType kv_type = KvCacheType::F32;
  bool kv_offload = false;
  std::vector<std::uint32_t> stops;

  // Fresh decode caches for every rank, so one generation starts at position
  // zero and the request's speed tuning and KV type take effect (the
  // architecture state copies tuning when it is first built). The CP context
  // is reattached to each new cache.
  void ResetCaches(const GenerateOptions& options) {
    for (std::size_t r = 0; r < ranks.size(); ++r) {
      auto cache = std::make_unique<core::DecodeCache>();
      cache->kv_type = options.kv_type;
      cache->kv_offload = options.kv_offload;
      cache->tuning.mxfp4_split_target = options.mxfp4_split_target;
      cache->tuning.mxfp4_split_cap = options.mxfp4_split_cap;
      cache->tuning.attention_split = options.attention_split;
      cache->cp = contexts[r].get();
      ranks[r].cache = std::move(cache);
    }
  }

  // Greedy generation across the ranks: one Step per prefill token, then
  // one Step per decoded token. `on_token` streams every produced token
  // (the rank-0 sequence, which equals every rank's) and returns false to
  // stop; `cancelled` is polled between decode steps. The outcome carries
  // the produced count, the finish reason and the prefill/decode wall
  // times.
  std::expected<GenerateOutcome, StatusCode> Run(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled);

  // True when a speculative strategy is attached on the ranks.
  bool HasStrategy() const {
    for (const Rank& rank : ranks) {
      if (rank.strategy != nullptr) {
        return true;
      }
    }
    return false;
  }

  // One speculative generation across the ranks: every rank runs the decode
  // driver on its own target and strategy; the collective attention keeps the
  // ranks in lockstep and rank 0's tokens win. A rank cannot leave a
  // collective mid-step, so a false from `on_token` only stops the stream
  // and the decode runs to its stop token or budget; `cancelled` is polled
  // before the turn only.
  std::expected<GenerateOutcome, StatusCode> RunSpeculative(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled);

  // Run one token through every rank on its own thread; the ranks
  // rendezvous at each full-attention layer to combine the per-rank
  // partials. When `want_token` rank 0 also runs the head and returns
  // the greedy token (the device argmax, no vocab download); the other
  // ranks skip the head.
  std::expected<std::uint32_t, StatusCode> Step(std::uint32_t token,
                                                bool want_token) {
    std::expected<std::uint32_t, StatusCode> result = 0;
    std::vector<std::expected<void, StatusCode>> forwards(ranks.size());
    std::vector<std::thread> threads;
    threads.reserve(ranks.size());
    for (std::size_t r = 0; r < ranks.size(); ++r) {
      threads.emplace_back([this, r, token, want_token, &result, &forwards] {
        Impl::Rank& rank = ranks[r];
        const Architecture* arch = rank.model->Arch();
        if (arch == nullptr) {
          forwards[r] = std::unexpected(StatusCode::UnsupportedFeature);
          return;
        }
        Backend& backend = rank.engine->Owner();
        if (r == 0 && want_token) {
          result = arch->GreedyToken(backend, *rank.model, *rank.cache, token,
                                     nullptr, nullptr, nullptr, nullptr);
          forwards[r] = result ? std::expected<void, StatusCode>{}
                               : std::unexpected(result.error());
          return;
        }
        forwards[r] = arch->Forward(backend, *rank.model, *rank.cache, token,
                                    nullptr, nullptr, nullptr, nullptr);
      });
    }
    for (std::thread& thread : threads) {
      thread.join();
    }
    for (const auto& forward : forwards) {
      if (!forward) {
        return std::unexpected(forward.error());
      }
    }
    if (!want_token) {
      return 0;
    }
    return result;
  }
};

std::expected<GenerateOutcome, StatusCode> ContextParallel::Impl::Run(
    std::span<const std::uint32_t> prompt, const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token,
    const std::function<bool()>& cancelled) {
  if (prompt.empty() || prompt.size() > context) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  ResetCaches(options);
  log::Diagnostics& log = ranks[0].engine->Diagnostics();
  GenerateOutcome outcome;
  outcome.prompt_tokens = prompt.size();
  std::uint32_t next = 0;
  const auto prefill_started = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    const bool last = (i + 1 == prompt.size());
    auto token = Step(prompt[i], last);
    if (!token) {
      return std::unexpected(token.error());
    }
    if (last) {
      next = *token;
    }
  }
  outcome.prefill_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - prefill_started)
                           .count();
  const std::size_t budget = options.max_completion_tokens == 0
                                 ? context - prompt.size()
                                 : options.max_completion_tokens;
  // Thinking budget, the same host-side state machine as the single-device
  // driver. Rank 0 owns the tokenizer tags; every rank sees the same ids
  // through the lockstep Step calls.
  auto [think, think_close] =
      ResolveThinkBudget(*ranks[0].model, options.max_thinking_tokens);
  if (options.max_thinking_tokens > 0 && !think.engaged()) {
    log.Warn("cp",
             "thinking budget ignored: the think tags do not resolve on "
             "this model; thinking stays unlimited");
  }
  for (std::uint32_t id : prompt) {
    think.noteEmitted(id);
  }
  std::size_t produced = 0;
  bool stopped = false;
  bool aborted = false;
  const auto emit = [&](std::uint32_t token) {
    think.noteEmitted(token);
    ++produced;
    return on_token(token);
  };
  // Forward and emit the think-close ids; the last step's argmax is the
  // next anchor. Mirrors the single-device driver's force-close.
  auto force_close = [&]() -> std::expected<std::uint32_t, StatusCode> {
    std::uint32_t after = 0;
    for (std::uint32_t id : think_close) {
      auto token = Step(id, true);
      if (!token) {
        return std::unexpected(token.error());
      }
      after = *token;
      if (!emit(id)) {
        aborted = true;
        return after;
      }
      if (produced >= budget) {
        return after;
      }
    }
    return after;
  };
  const auto decode_started = std::chrono::steady_clock::now();
  while (produced < budget) {
    if (cancelled && cancelled()) {
      aborted = true;
      break;
    }
    if (think.closeOwed()) {
      log.Info("cp", "thinking budget (" +
                         std::to_string(options.max_thinking_tokens) +
                         " token(s)) exceeded: think block force-closed, "
                         "decoding continues");
      auto forced = force_close();
      if (!forced) {
        return std::unexpected(forced.error());
      }
      if (aborted || produced >= budget) {
        break;
      }
      next = *forced;
    }
    if (IsStop(next, stops)) {
      stopped = true;
      break;
    }
    if (!emit(next)) {
      aborted = true;
      break;
    }
    if (produced >= budget) {
      break;
    }
    auto token = Step(next, true);
    if (!token) {
      return std::unexpected(token.error());
    }
    next = *token;
  }
  outcome.decode_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - decode_started)
                          .count();
  outcome.produced = produced;
  outcome.reason = stopped   ? FinishReason::Stop
                   : aborted ? FinishReason::Aborted
                             : FinishReason::Length;
  return outcome;
}

std::expected<GenerateOutcome, StatusCode>
ContextParallel::Impl::RunSpeculative(
    std::span<const std::uint32_t> prompt, const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token,
    const std::function<bool()>& cancelled) {
  if (prompt.empty() || prompt.size() > context) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (cancelled && cancelled()) {
    GenerateOutcome outcome;
    outcome.prompt_tokens = prompt.size();
    outcome.reason = FinishReason::Aborted;
    return outcome;
  }
  ResetCaches(options);
  for (Rank& rank : ranks) {
    if (rank.strategy == nullptr) {
      continue;
    }
    auto prepared = rank.strategy->Prepare(rank.engine->Owner(), *rank.model);
    if (!prepared) {
      return std::unexpected(prepared.error());
    }
    rank.strategy->SetDraftContext(options.draft_context);
  }
  std::vector<std::expected<GenerateOutcome, StatusCode>> outcomes(
      ranks.size(), std::unexpected(StatusCode::DeviceError));
  std::vector<std::thread> threads;
  threads.reserve(ranks.size());
  // The prompt arrives separately from the options; fold it in once so every
  // rank's driver prefills it instead of the single first_token.
  const GenerateOptions framed = core::WithPrompt(options, prompt);
  // The ranks stay in lockstep through the collective forwards, so rank 0
  // streams every rank's token. A false from `on_token` cannot break the
  // rank out of the driver without stranding the others in a collective:
  // it only stops the stream, and the decode runs to its stop token or
  // budget.
  std::atomic<bool> stream_stopped{false};
  for (std::size_t r = 0; r < ranks.size(); ++r) {
    threads.emplace_back([this, r, &framed, &on_token, &stream_stopped,
                          &outcomes] {
      Rank& rank = ranks[r];
      core::SingleDeviceTarget target(
          rank.engine->Owner(), *rank.model, *rank.cache,
          rank.engine->Diagnostics(), 0, nullptr,
          // The batched prefill, now that ForwardBatch combines across the
          // ranks: it captures the target hidden for a DFlash2 drafter and
          // drives the strategy's context append. MTP captures nothing.
          /*single_token_prefill=*/false);
      const bool emit = (r == 0);
      auto streamed = [emit, &on_token, &stream_stopped](std::uint32_t token) {
        if (emit && !stream_stopped.load(std::memory_order_relaxed) &&
            !on_token(token)) {
          stream_stopped.store(true, std::memory_order_relaxed);
        }
        return true;
      };
      outcomes[r] =
          core::RunDecode(target, rank.strategy.get(), framed, streamed);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  for (const auto& outcome : outcomes) {
    if (!outcome) {
      return std::unexpected(outcome.error());
    }
  }
  return outcomes[0];
}

ContextParallel::ContextParallel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

ContextParallel::~ContextParallel() = default;
ContextParallel::ContextParallel(ContextParallel&& other) noexcept = default;
ContextParallel& ContextParallel::operator=(ContextParallel&& other) noexcept =
    default;

std::expected<std::unique_ptr<ContextParallel>, StatusCode>
ContextParallel::Create(const ContextParallelOptions& options) {
  if (options.devices.size() < 2) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto config = Model::ProbeConfig(options.model_path);
  if (!config) {
    return std::unexpected(config.error());
  }
  const std::size_t heads = config->attention.heads;
  const std::size_t head_dim = config->attention.head_dim;
  if (heads == 0 || head_dim == 0 || head_dim > 256) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  auto impl = std::make_unique<Impl>();
  impl->context = options.context;
  impl->kv_type = options.kv_type;
  impl->kv_offload = options.kv_offload;
  impl->barrier.world = static_cast<int>(options.devices.size());
  const int world = static_cast<int>(options.devices.size());
  const std::size_t partial_bytes = heads * (head_dim + 2) * 4;
  const std::size_t attn_bytes = heads * head_dim * 4;
  impl->ranks.reserve(options.devices.size());
  for (std::size_t r = 0; r < options.devices.size(); ++r) {
    EngineOptions engine_options;
    engine_options.device_index = options.devices[r];
    auto engine = Engine::Create(engine_options);
    if (!engine) {
      return std::unexpected(engine.error());
    }
    ModelOptions model_options;
    model_options.path = options.model_path;
    model_options.max_context_length = options.context;
    auto model = (*engine)->LoadModel(model_options);
    if (!model) {
      return std::unexpected(model.error());
    }
    Impl::Rank rank;
    rank.engine = std::move(*engine);
    rank.model = std::move(*model);
    auto partial =
        rank.engine->Owner().AllocateBuffer(partial_bytes, MemoryKind::Device);
    auto attn =
        rank.engine->Owner().AllocateBuffer(attn_bytes, MemoryKind::Device);
    if (!partial || !attn) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    rank.partial = std::move(*partial);
    rank.attn = std::move(*attn);
    rank.cache = std::make_unique<core::DecodeCache>();
    rank.cache->kv_type = options.kv_type;
    rank.cache->kv_offload = options.kv_offload;
    impl->ranks.push_back(std::move(rank));
  }
  std::vector<Buffer*> partials;
  std::vector<Buffer*> attns;
  partials.reserve(impl->ranks.size());
  attns.reserve(impl->ranks.size());
  for (Impl::Rank& rank : impl->ranks) {
    partials.push_back(rank.partial.get());
    attns.push_back(rank.attn.get());
  }
  for (std::size_t r = 0; r < impl->ranks.size(); ++r) {
    auto context = std::make_unique<core::CpContext>();
    context->enabled = true;
    context->rank = static_cast<int>(r);
    context->world = world;
    // Contiguous key chunks over the configured context.
    context->k0 = options.context * r / impl->ranks.size();
    context->k1 = options.context * (r + 1) / impl->ranks.size();
    context->partial = partials;
    context->attn = attns;
    context->barrier = &impl->barrier;
    impl->contexts.push_back(std::move(context));
    impl->ranks[r].cache->cp = impl->contexts.back().get();
  }
  const std::span<const std::uint32_t> stops =
      impl->ranks[0].model->StopTokens();
  impl->stops.assign(stops.begin(), stops.end());
  return std::unique_ptr<ContextParallel>(new ContextParallel(std::move(impl)));
}

std::expected<void, StatusCode> ContextParallel::AttachSpeculative(
    const std::function<std::unique_ptr<SpeculativeStrategy>()>& factory,
    const StrategyOptions& options) {
  if (!factory) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (Impl::Rank& rank : impl_->ranks) {
    auto strategy = factory();
    if (strategy == nullptr) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    auto attached = strategy->Attach(options);
    if (!attached) {
      return std::unexpected(attached.error());
    }
    rank.strategy = std::move(strategy);
  }
  return {};
}

std::expected<GenerateOutcome, StatusCode> ContextParallel::GenerateStreaming(
    std::span<const std::uint32_t> prompt, const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token,
    const std::function<bool()>& cancelled) {
  if (!on_token) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return impl_->HasStrategy()
             ? impl_->RunSpeculative(prompt, options, on_token, cancelled)
             : impl_->Run(prompt, options, on_token, cancelled);
}

std::expected<std::vector<std::uint32_t>, StatusCode> ContextParallel::Generate(
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
  if (!produced.empty() && outcome->decode_ms > 0.0) {
    const long long tps = static_cast<long long>(
        static_cast<double>(produced.size()) * 1000.0 / outcome->decode_ms);
    impl_->ranks[0].engine->Diagnostics().Info(
        "cp", std::string("decode: ") + std::to_string(produced.size()) +
                  " token(s) in " +
                  std::to_string(static_cast<long long>(outcome->decode_ms)) +
                  " ms (" + std::to_string(tps) +
                  " tok/s, ranks=" + std::to_string(impl_->ranks.size()) + ")");
  }
  return produced;
}

std::expected<GenerateOutcome, StatusCode> ContextParallel::GenerateTimed(
    std::span<const std::uint32_t> prompt, const GenerateOptions& options) {
  return GenerateStreaming(prompt, options, [](std::uint32_t) { return true; });
}

std::size_t ContextParallel::RankCount() const {
  return impl_->ranks.size();
}

Engine& ContextParallel::RankEngine(std::size_t i) {
  return *impl_->ranks.at(i).engine;
}

Model& ContextParallel::RankModel(std::size_t i) {
  return *impl_->ranks.at(i).model;
}

const Tokenizer* ContextParallel::GetTokenizer() const {
  return impl_->ranks.empty() ? nullptr
                              : impl_->ranks.front().model->GetTokenizer();
}

std::expected<std::string, StatusCode> ContextParallel::ChatPrompt(
    std::string_view user_text, bool enable_thinking,
    std::string_view reasoning_effort) const {
  if (impl_->ranks.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return impl_->ranks.front().model->ChatPrompt(user_text, enable_thinking,
                                                reasoning_effort);
}

}  // namespace tessera
