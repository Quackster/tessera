#include "tessera/spec_split.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "tessera/architecture.hpp"
#include "tessera/backend.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/speculative.hpp"

namespace tessera {

namespace detail = ::tessera::core::detail;

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
    auto model = impl->target_engine->LoadModel(model_options);
    if (!model) {
      return std::unexpected(model.error());
    }
    impl->target = std::move(*model);
    impl->target_cache = std::make_unique<core::DecodeCache>();
    impl->target_cache->kv_type = options.kv_type;
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
    impl->draft_cache = std::make_unique<core::DecodeCache>();
  }
  impl->strategy = CreateMtpStrategy();
  if (auto attached = impl->strategy->Attach(
          StrategyOptions{"", options.draft_block});
      !attached) {
    return std::unexpected(attached.error());
  }
  if (auto prepared = impl->strategy->Prepare(impl->draft_engine->Owner(),
                                              *impl->draft);
      !prepared) {
    return std::unexpected(prepared.error());
  }
  // Allocate the draft decode state without running the trunk, which the
  // draft device does not hold.
  const Architecture* arch = impl->draft->Arch();
  if (arch == nullptr) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  if (auto ready = arch->EnsureReady(impl->draft_engine->Owner(), *impl->draft,
                                    *impl->draft_cache);
      !ready) {
    return std::unexpected(ready.error());
  }
  const std::span<const std::uint32_t> stops = impl->target->StopTokens();
  impl->stops.assign(stops.begin(), stops.end());
  return std::unique_ptr<SpecSplit>(new SpecSplit(std::move(impl)));
}

std::expected<std::vector<std::uint32_t>, StatusCode> SpecSplit::Generate(
    std::span<const std::uint32_t> prompt, std::size_t max_tokens) {
  Impl& p = *impl_;
  if (prompt.empty() || prompt.size() > p.context) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  Backend& target_backend = p.target_engine->Owner();
  Backend& draft_backend = p.draft_engine->Owner();
  Model& target = *p.target;
  Model& draft = *p.draft;
  core::DecodeCache& target_cache = *p.target_cache;
  core::DecodeCache& draft_cache = *p.draft_cache;
  SpeculativeStrategy& strategy = *p.strategy;
  // Target prefill; the last position's hidden seeds the first anchor.
  std::vector<float> hidden;
  auto prefill = core::PrefillTokens(target_backend, target, target_cache,
                                    prompt, &hidden, nullptr,
                                    p.prefill_chunk, nullptr);
  if (!prefill) {
    return std::unexpected(prefill.error());
  }
  std::uint32_t next = detail::ArgMax(*prefill);
  std::uint64_t position = prompt.size();
  const std::size_t block = strategy.DraftBlock();
  if (block == 0) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  if (auto seeded =
          strategy.OnAnchor(draft_backend, draft, draft_cache, next, position,
                            hidden);
      !seeded) {
    return std::unexpected(seeded.error());
  }
  std::vector<std::uint32_t> drafts(block);
  std::vector<std::uint32_t> produced;
  const std::size_t budget =
      max_tokens == 0 ? p.context - prompt.size() : max_tokens;
  bool stopped = false;
  std::size_t spec_proposed = 0;
  std::size_t spec_accepted = 0;
  std::size_t spec_steps = 0;
  while (!stopped && produced.size() < budget) {
    if (detail::IsStopToken(next, p.stops)) {
      break;
    }
    produced.push_back(next);
    auto count =
        strategy.Draft(draft_backend, draft, draft_cache, {}, next, drafts);
    if (!count) {
      return std::unexpected(count.error());
    }
    if (*count == 0) {
      // No usable proposal: decode the anchor alone and take its greedy
      // token on the target.
      auto logits =
          core::DecodeLogits(target_backend, target, target_cache, next,
                             &hidden, nullptr, nullptr);
      if (!logits) {
        return std::unexpected(logits.error());
      }
      const std::uint32_t picked = detail::ArgMax(*logits);
      if (auto anchored = strategy.OnAnchor(draft_backend, draft, draft_cache,
                                            picked, position + 1, hidden);
          !anchored) {
        return std::unexpected(anchored.error());
      }
      if (auto committed =
              strategy.Commit(draft_backend, draft, draft_cache, 0);
          !committed) {
        return std::unexpected(committed.error());
      }
      next = picked;
      ++position;
      continue;
    }
    std::span<const std::uint32_t> proposal(drafts.data(), *count);
    // The anchor leads the batch; `hidden` receives the last committed
    // row's residual hidden for the next anchor.
    auto verify = core::VerifyDraft(target_backend, target, target_cache,
                                    proposal, {}, next, &hidden, nullptr,
                                    nullptr);
    if (!verify) {
      return std::unexpected(verify.error());
    }
    if (auto anchored = strategy.OnAnchor(
            draft_backend, draft, draft_cache, verify->next_token,
            position + 1 + verify->accepted, hidden);
        !anchored) {
      return std::unexpected(anchored.error());
    }
    if (auto committed = strategy.Commit(draft_backend, draft, draft_cache,
                                         verify->accepted);
        !committed) {
      return std::unexpected(committed.error());
    }
    spec_proposed += *count;
    spec_accepted += verify->accepted;
    ++spec_steps;
    for (std::size_t i = 0; i < verify->accepted; ++i) {
      if (produced.size() >= budget) {
        break;
      }
      if (detail::IsStopToken(drafts[i], p.stops)) {
        stopped = true;
        break;
      }
      produced.push_back(drafts[i]);
    }
    if (stopped || produced.size() >= budget) {
      break;
    }
    next = verify->next_token;
    position += 1 + verify->accepted;
  }
  if (spec_steps > 0) {
    p.target_engine->Diagnostics().Info(
        "spec-split", std::string("accepted ") +
                          std::to_string(spec_accepted) + " of " +
                          std::to_string(spec_proposed) + " draft token(s) " +
                          "over " + std::to_string(spec_steps) + " step(s)");
  }
  return produced;
}

const Tokenizer* SpecSplit::GetTokenizer() const {
  return impl_->target == nullptr ? nullptr
                                  : impl_->target->GetTokenizer();
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
