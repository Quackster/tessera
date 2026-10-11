#include "tessera/context_parallel.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/decode.hpp"
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
  };
  std::vector<Rank> ranks;
  core::TpBarrier barrier;
  std::vector<std::unique_ptr<core::CpContext>> contexts;
  std::size_t context = 0;
  std::vector<std::uint32_t> stops;

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
    auto partial = rank.engine->Owner().AllocateBuffer(partial_bytes,
                                                       MemoryKind::Device);
    auto attn = rank.engine->Owner().AllocateBuffer(attn_bytes,
                                                    MemoryKind::Device);
    if (!partial || !attn) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    rank.partial = std::move(*partial);
    rank.attn = std::move(*attn);
    rank.cache = std::make_unique<core::DecodeCache>();
    rank.cache->kv_type = options.kv_type;
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
  return std::unique_ptr<ContextParallel>(
      new ContextParallel(std::move(impl)));
}

std::expected<std::vector<std::uint32_t>, StatusCode> ContextParallel::Generate(
    std::span<const std::uint32_t> prompt, std::size_t max_tokens) {
  Impl& p = *impl_;
  if (prompt.empty() || prompt.size() > p.context) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::uint32_t next = 0;
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    const bool last = (i + 1 == prompt.size());
    auto token = p.Step(prompt[i], last);
    if (!token) {
      return std::unexpected(token.error());
    }
    if (last) {
      next = *token;
    }
  }
  const std::size_t budget =
      max_tokens == 0 ? p.context - prompt.size() : max_tokens;
  const auto decode_started = std::chrono::steady_clock::now();
  std::vector<std::uint32_t> produced;
  for (std::size_t count = 0; count < budget; ++count) {
    if (IsStop(next, p.stops)) {
      break;
    }
    produced.push_back(next);
    auto token = p.Step(next, true);
    if (!token) {
      return std::unexpected(token.error());
    }
    next = *token;
  }
  if (!produced.empty()) {
    const long long decode_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - decode_started)
            .count();
    const long long tps =
        decode_ms > 0
            ? static_cast<long long>(produced.size()) * 1000 / decode_ms
            : 0;
    p.ranks[0].engine->Diagnostics().Info(
        "cp", std::string("decode: ") + std::to_string(produced.size()) +
                  " token(s) in " + std::to_string(decode_ms) + " ms (" +
                  std::to_string(tps) + " tok/s, ranks=" +
                  std::to_string(p.ranks.size()) + ")");
  }
  return produced;
}

std::size_t ContextParallel::RankCount() const { return impl_->ranks.size(); }

const Tokenizer* ContextParallel::GetTokenizer() const {
  return impl_->ranks.empty() ? nullptr
                              : impl_->ranks.front().model->GetTokenizer();
}

std::expected<std::string, StatusCode> ContextParallel::ChatPrompt(
    std::string_view user_text, bool enable_thinking) const {
  if (impl_->ranks.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return impl_->ranks.front().model->ChatPrompt(user_text, enable_thinking);
}

}  // namespace tessera
