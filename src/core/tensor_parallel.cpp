#include "tessera/tensor_parallel.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/single_device_target.hpp"
#include "core/spec_driver.hpp"
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

// Rows one batched tensor-parallel forward handles at once: a speculative
// verify scores the anchor plus up to the checkpoint's draft block. The rank
// partial and the rank-0 reduce staging buffers are sized for this many
// hidden rows, so the batched all-reduce covers a whole verify batch.
constexpr std::size_t kTpBatchRows = 16;

}  // namespace

struct TensorParallel::Impl {
  // One rank: its engine, its sliced model, its decode cache and the FFN
  // down-projection partial output the all-reduce sums.
  struct Rank {
    std::unique_ptr<Engine> engine;
    std::unique_ptr<Model> model;
    std::unique_ptr<core::DecodeCache> cache;
    std::unique_ptr<Buffer> ffn_partial;
    // The rank's speculative strategy (null when none is attached). Each
    // rank runs its own instance on its own sharded model.
    std::unique_ptr<SpeculativeStrategy> strategy;
  };
  std::vector<Rank> ranks;
  core::TpBarrier barrier;
  std::vector<std::unique_ptr<core::TpContext>> contexts;
  // Staging buffer on rank 0's device for the peer all-reduce.
  std::unique_ptr<Buffer> reduce_scratch;
  std::size_t hidden_dim = 0;
  std::size_t context = 0;
  KvCacheType kv_type = KvCacheType::F32;
  std::vector<std::uint32_t> stops;

  // Fresh decode caches for every rank, so one generation starts at position
  // zero and the request's speed tuning and KV type take effect (the
  // architecture state copies tuning when it is first built). The TP context
  // is reattached to each new cache.
  void ResetCaches(const GenerateOptions& options) {
    for (std::size_t r = 0; r < ranks.size(); ++r) {
      auto cache = std::make_unique<core::DecodeCache>();
      cache->kv_type = options.kv_type;
      cache->tuning.mxfp4_split_target = options.mxfp4_split_target;
      cache->tuning.mxfp4_split_cap = options.mxfp4_split_cap;
      cache->tuning.attention_split = options.attention_split;
      cache->tp = contexts[r].get();
      ranks[r].cache = std::move(cache);
    }
  }

  // Greedy generation with `options` tuning, filling `produced`; the outcome
  // carries the produced count and the prefill/decode wall times.
  std::expected<GenerateOutcome, StatusCode> Run(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      std::vector<std::uint32_t>* produced);

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
  // driver on its own target and its own strategy instance, so a
  // tensor-parallel (sharded) draft head works. The collective forwards keep
  // the ranks in lockstep; rank 0's tokens and outcome are the result.
  std::expected<GenerateOutcome, StatusCode> RunSpeculative(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      std::vector<std::uint32_t>* produced);

  // Run one token through every rank on its own thread; the ranks rendezvous
  // at each FFN and attention all-reduce. When `want_token` rank 0 also runs
  // the head and returns the greedy token (the device argmax, no vocab
  // download); the other ranks skip the head.
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

std::expected<GenerateOutcome, StatusCode> TensorParallel::Impl::Run(
    std::span<const std::uint32_t> prompt, const GenerateOptions& options,
    std::vector<std::uint32_t>* produced) {
  if (prompt.empty() || prompt.size() > context) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  produced->clear();
  ResetCaches(options);
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
  const std::size_t budget =
      options.max_completion_tokens == 0 ? context - prompt.size()
                                         : options.max_completion_tokens;
  bool stopped = false;
  const auto decode_started = std::chrono::steady_clock::now();
  for (std::size_t count = 0; count < budget; ++count) {
    if (IsStop(next, stops)) {
      stopped = true;
      break;
    }
    produced->push_back(next);
    auto token = Step(next, true);
    if (!token) {
      return std::unexpected(token.error());
    }
    next = *token;
  }
  outcome.decode_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - decode_started)
                          .count();
  outcome.produced = produced->size();
  outcome.reason = stopped ? FinishReason::Stop : FinishReason::Length;
  return outcome;
}

std::expected<GenerateOutcome, StatusCode>
TensorParallel::Impl::RunSpeculative(std::span<const std::uint32_t> prompt,
                                     const GenerateOptions& options,
                                     std::vector<std::uint32_t>* produced) {
  if (prompt.empty() || prompt.size() > context) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  produced->clear();
  ResetCaches(options);
  // Every rank prepares its own strategy on its own backend and model, so a
  // sharded (tensor-parallel) draft head stays on its own rank.
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
  for (std::size_t r = 0; r < ranks.size(); ++r) {
    threads.emplace_back([this, r, prompt, &options, produced, &outcomes] {
      Rank& rank = ranks[r];
      core::SingleDeviceTarget target(
          rank.engine->Owner(), *rank.model, *rank.cache,
          rank.engine->Diagnostics(), 0, nullptr,
          /*single_token_prefill=*/true);
      // Rank 0 owns the produced token list; the ranks stay in lockstep
      // through the collective forwards, so every rank sees the same tokens.
      const bool emit = (r == 0);
      auto on_token = [emit, produced](std::uint32_t token) {
        if (emit && produced != nullptr) {
          produced->push_back(token);
        }
        return true;
      };
      outcomes[r] =
          core::RunDecode(target, rank.strategy.get(), options, on_token);
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

TensorParallel::TensorParallel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TensorParallel::~TensorParallel() = default;
TensorParallel::TensorParallel(TensorParallel&& other) noexcept = default;
TensorParallel& TensorParallel::operator=(TensorParallel&& other) noexcept =
    default;

std::expected<std::unique_ptr<TensorParallel>, StatusCode>
TensorParallel::Create(const TensorParallelOptions& options) {
  if (options.devices.size() < 2) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto config = Model::ProbeConfig(options.model_path);
  if (!config) {
    return std::unexpected(config.error());
  }
  auto impl = std::make_unique<Impl>();
  impl->hidden_dim = config->hidden_dim;
  impl->context = options.context;
  impl->kv_type = options.kv_type;
  impl->barrier.world = static_cast<int>(options.devices.size());
  const int world = static_cast<int>(options.devices.size());
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
    model_options.tensor.enabled = true;
    model_options.tensor.rank = static_cast<int>(r);
    model_options.tensor.world = world;
    auto model = (*engine)->LoadModel(model_options);
    if (!model) {
      return std::unexpected(model.error());
    }
    Impl::Rank rank;
    rank.engine = std::move(*engine);
    rank.model = std::move(*model);
    auto buffer = rank.engine->Owner().AllocateBuffer(
        kTpBatchRows * impl->hidden_dim * 4, MemoryKind::Device);
    if (!buffer) {
      return std::unexpected(buffer.error());
    }
    rank.ffn_partial = std::move(*buffer);
    rank.cache = std::make_unique<core::DecodeCache>();
    impl->ranks.push_back(std::move(rank));
  }
  std::vector<Buffer*> partials;
  partials.reserve(impl->ranks.size());
  for (Impl::Rank& rank : impl->ranks) {
    partials.push_back(rank.ffn_partial.get());
  }
  // Rank 0 stages each peer's partial here before accumulating it.
  auto scratch = impl->ranks[0].engine->Owner().AllocateBuffer(
      kTpBatchRows * impl->hidden_dim * 4, MemoryKind::Device);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  impl->reduce_scratch = std::move(*scratch);
  for (std::size_t r = 0; r < impl->ranks.size(); ++r) {
    auto context = std::make_unique<core::TpContext>();
    context->enabled = true;
    context->rank = static_cast<int>(r);
    context->world = world;
    context->ffn_partial = partials;
    context->reduce_scratch = impl->reduce_scratch.get();
    context->barrier = &impl->barrier;
    impl->contexts.push_back(std::move(context));
    impl->ranks[r].cache->tp = impl->contexts.back().get();
  }
  const std::span<const std::uint32_t> stops =
      impl->ranks[0].model->StopTokens();
  impl->stops.assign(stops.begin(), stops.end());
  return std::unique_ptr<TensorParallel>(new TensorParallel(std::move(impl)));
}

std::expected<void, StatusCode> TensorParallel::AttachSpeculative(
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

std::expected<std::vector<std::uint32_t>, StatusCode> TensorParallel::Generate(
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
  if (!produced.empty() && outcome->decode_ms > 0.0) {
    const long long tps = static_cast<long long>(
        static_cast<double>(produced.size()) * 1000.0 / outcome->decode_ms);
    impl_->ranks[0].engine->Diagnostics().Info(
        "tp", std::string("decode: ") + std::to_string(produced.size()) +
                  " token(s) in " + std::to_string(static_cast<long long>(
                      outcome->decode_ms)) + " ms (" + std::to_string(tps) +
                  " tok/s, ranks=" + std::to_string(impl_->ranks.size()) + ")");
  }
  return produced;
}

std::expected<GenerateOutcome, StatusCode> TensorParallel::GenerateTimed(
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

std::size_t TensorParallel::RankCount() const { return impl_->ranks.size(); }

Engine& TensorParallel::RankEngine(std::size_t i) {
  return *impl_->ranks.at(i).engine;
}

Model& TensorParallel::RankModel(std::size_t i) {
  return *impl_->ranks.at(i).model;
}

const Tokenizer* TensorParallel::GetTokenizer() const {
  return impl_->ranks.empty() ? nullptr
                              : impl_->ranks.front().model->GetTokenizer();
}

std::expected<std::string, StatusCode> TensorParallel::ChatPrompt(
    std::string_view user_text, bool enable_thinking,
    std::string_view reasoning_effort) const {
  if (impl_->ranks.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return impl_->ranks.front().model->ChatPrompt(user_text, enable_thinking,
                                                reasoning_effort);
}

}  // namespace tessera
