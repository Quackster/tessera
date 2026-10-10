#include "tessera/engine.hpp"

#include <chrono>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "core/generate_helpers.hpp"
#include "core/sampling.hpp"

namespace tessera {

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
  cache.tuning.mxfp4_split_target = options.mxfp4_split_target;
  cache.tuning.mxfp4_split_cap = options.mxfp4_split_cap;
  cache.tuning.attention_split = options.attention_split;
  std::vector<std::uint32_t> prompt = options.prompt_tokens;
  if (prompt.empty()) {
    prompt.push_back(options.first_token);
  }
  // A zero count fills the remaining context.
  const std::size_t max_completion_tokens =
      model.EffectiveMaxTokens(prompt.size(), options.max_completion_tokens);
  if (max_completion_tokens == 0) {
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
  core::PrefillProgress mm_progress = [&](std::size_t done,
                                           std::size_t total) {
    if (options.prefill_progress) {
      options.prefill_progress(done, total);
    }
  };
  const core::PrefillProgress* mm_progress_ptr =
      options.prefill_progress ? &mm_progress : nullptr;
  auto first = core::PrefillTokens(*backend_, model, cache, prompt, nullptr,
                                   embedding_buffer->get(), prefill_chunk,
                                   mm_progress_ptr);
  if (!first) {
    return std::unexpected(first.error());
  }
  std::vector<float> current = std::move(*first);
  std::vector<std::uint32_t> produced;
  produced.reserve(max_completion_tokens);
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
  auto [think, think_close] =
      ResolveThinkBudget(model, options.max_thinking_tokens);
  if (options.max_thinking_tokens > 0 && !think.engaged()) {
    diagnostics_.Warn("engine", "thinking budget ignored: the think tags do "
                                "not resolve on this model; thinking stays "
                                "unlimited");
  }
  for (std::uint32_t id : prompt) {
    think.noteEmitted(id);
  }
  while (produced.size() < max_completion_tokens) {
    if (think.closeOwed()) {
      diagnostics_.Info("engine", "thinking budget (" +
                                       std::to_string(
                                           options.max_thinking_tokens) +
                                       " token(s)) exceeded: think block "
                                       "force-closed, decoding continues");
      bool done = false;
      std::vector<float> close_logits;
      for (std::uint32_t id : think_close) {
        // Forced markup bypasses the stop check like any protocol token.
        auto logits = core::DecodeLogits(*backend_, model, cache, id);
        if (!logits) {
          return std::unexpected(logits.error());
        }
        think.noteEmitted(id);
        produced.push_back(id);
        if (produced.size() >= max_completion_tokens) {
          done = true;
          break;
        }
        history.push_back(id);
        close_logits = std::move(*logits);
      }
      if (done) {
        break;
      }
      next = pick(close_logits);
    }
    if (core::detail::IsStopToken(next, stops)) {
      stopped = true;
      break;
    }
    think.noteEmitted(next);
    produced.push_back(next);
    if (produced.size() >= max_completion_tokens) {
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

}  // namespace tessera
