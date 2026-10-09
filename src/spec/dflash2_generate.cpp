#include "spec/dflash2_generate.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "core/loaders/safetensors.hpp"
#include "spec/dflash2_config.hpp"
#include "spec/dflash2_candidates.hpp"
#include "spec/dflash2_drafter.hpp"
#include "spec/dflash2_mask.hpp"
#include "spec/dflash2_selector.hpp"

namespace tessera::spec {

std::expected<std::vector<std::uint32_t>, StatusCode> GenerateDFlash2(
    Backend& backend, Model& target, const GenerateOptions& options,
    const std::string& draft_path, std::span<const std::uint32_t> stop_tokens,
    const log::Diagnostics* log) {
  auto config = target.Config();
  if (!config || !config->hybrid) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  auto draft_config = LoadDFlash2Config(draft_path);
  if (!draft_config) {
    return std::unexpected(draft_config.error());
  }
  auto drafter = DFlash2Drafter::Create(backend, draft_path, *draft_config);
  if (!drafter) {
    return std::unexpected(drafter.error());
  }
  const auto started = std::chrono::steady_clock::now();
  const DeviceTensor* embed = nullptr;
  const DeviceTensor* output = nullptr;
  for (const DeviceTensor& weight : target.Weights()) {
    if (weight.manifest.name == "token_embd.weight") {
      embed = &weight;
    } else if (weight.manifest.name == "output.weight") {
      output = &weight;
    }
  }
  if (embed == nullptr || output == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::unordered_map<int, std::unique_ptr<Kernel>> gemms;
  auto head_gemm = core::detail::GemmFor(backend, gemms, output->manifest.dtype);
  if (!head_gemm) {
    return std::unexpected(head_gemm.error());
  }
  const std::size_t hidden = config->hidden_dim;
  const std::size_t vocab = draft_config->vocab_size;
  // vLLM's draft block holds `1 + num_speculative_tokens` query rows: one
  // anchor plus the mask rows. The config's block_size is that total, so the
  // mask count is one less (the README's block size 8 means 7 draft tokens).
  std::size_t block = draft_config->block_size - 1;
  if (block == 0) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (options.draft_tokens > 0 && options.draft_tokens < block) {
    block = options.draft_tokens;
  }
  const std::size_t n = draft_config->target_layer_ids.size();
  // The draft conditions on the prefix up to the anchor, capped by the
  // checkpoint's sliding window (vLLM's context), so a short sequence uses
  // its whole prefix. TESSERA_DFLASH2_CTX overrides the cap for experiments;
  // a checkpoint window of 0 means unlimited.
  std::size_t ctx_window = draft_config->sliding_window;
  if (const char* env = std::getenv("TESSERA_DFLASH2_CTX"); env != nullptr) {
    const int v = std::atoi(env);
    if (v > 0) {
      ctx_window = static_cast<std::size_t>(v);
    }
  }
  // The cache holds the context rows plus the current anchor, so the cap is
  // one more than the context window (0 keeps every position).
  drafter->SetContextLimit(ctx_window == 0 ? 0 : ctx_window + 1);
  const std::size_t query_rows = block + 1;
  std::vector<std::size_t> capture_layers(draft_config->target_layer_ids.begin(),
                                          draft_config->target_layer_ids.end());
  std::vector<std::unique_ptr<Buffer>> capture_storage;
  std::vector<Buffer*> captures;
  std::vector<std::unique_ptr<Buffer>> verify_storage;
  std::vector<Buffer*> verify_captures;
  for (std::size_t i = 0; i < n; ++i) {
    auto buffer = backend.AllocateBuffer(hidden * 4, MemoryKind::Device);
    auto batch = backend.AllocateBuffer(block * hidden * 4, MemoryKind::Device);
    if (!buffer || !batch) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    capture_storage.push_back(std::move(*buffer));
    captures.push_back(capture_storage.back().get());
    verify_storage.push_back(std::move(*batch));
    verify_captures.push_back(verify_storage.back().get());
  }
  auto logits =
      backend.AllocateBuffer(query_rows * vocab * 4, MemoryKind::Device);
  auto draft_hidden =
      backend.AllocateBuffer(query_rows * hidden * 4, MemoryKind::Device);
  auto sel_hidden =
      backend.AllocateBuffer(block * hidden * 4, MemoryKind::Device);
  if (!logits || !draft_hidden || !sel_hidden) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  auto selector_gemm = backend.LoadKernel("gemm_f32", {});
  auto selector_kernel = backend.LoadKernel("selector_edge_score", {});
  core::DecodeCache cache;
  cache.kv_type = options.kv_type;
  std::vector<std::uint32_t> prompt = options.prompt_tokens;
  if (prompt.empty()) {
    prompt.push_back(options.first_token);
  }
  // A zero count fills the remaining context.
  const std::size_t max_tokens =
      target.EffectiveMaxTokens(prompt.size(), options.max_tokens);
  if (max_tokens == 0) {
    return std::vector<std::uint32_t>{};
  }
  std::vector<float> hidden_state;
  const std::vector<const Buffer*> capture_ptrs(captures.begin(),
                                                captures.end());
  const auto append_capture = [&]() {
    return drafter->AppendContext(backend, capture_ptrs, 1);
  };
  // Capture the aux hidden of every prompt position: vLLM's draft context
  // is the whole prefix up to the anchor, so the early prompt positions are
  // context too.
  for (std::size_t i = 0; i + 1 < prompt.size(); ++i) {
    auto forward = core::DecodeForward(backend, target, cache, prompt[i],
                                       &hidden_state, nullptr, &capture_layers,
                                       &captures);
    if (!forward) {
      return std::unexpected(forward.error());
    }
    if (auto appended = append_capture(); !appended) {
      return std::unexpected(appended.error());
    }
  }
  auto first = core::DecodeLogits(backend, target, cache, prompt.back(),
                                  &hidden_state, &capture_layers, &captures);
  if (!first) {
    return std::unexpected(first.error());
  }
  if (auto appended = append_capture(); !appended) {
    return std::unexpected(appended.error());
  }
  std::vector<std::uint32_t> produced;
  produced.reserve(max_tokens);
  std::size_t proposed = 0;
  std::size_t accepted = 0;
  std::size_t steps = 0;
  std::uint32_t next = core::detail::ArgMax(*first);
  bool stopped = false;
  while (produced.size() < max_tokens) {
    if (core::detail::IsStopToken(next, stop_tokens)) {
      stopped = true;
      break;
    }
    produced.push_back(next);
    if (produced.size() >= max_tokens) {
      break;
    }
    auto current = core::DecodeLogits(backend, target, cache, next,
                                      &hidden_state, &capture_layers, &captures);
    if (!current) {
      return std::unexpected(current.error());
    }
    if (auto appended = append_capture(); !appended) {
      return std::unexpected(appended.error());
    }
    // The anchor (bonus) is the last committed token; the context is the
    // prefix before it, so the draft uses all but the last cache row.
    const std::size_t ctx = drafter->ContextRows() - 1;
    if (ctx == 0) {
      return std::unexpected(StatusCode::DeviceError);
    }
    auto query = QueryEmbeddings(backend, *embed, next,
                                 draft_config->mask_token_id, block, hidden);
    if (!query) {
      return std::unexpected(query.error());
    }
    // The draft attends over the cached context (absolute positions
    // ContextBase()..) and the query block (anchor first, then the masks).
    auto run = drafter->Run(backend, **query, *output->device, **head_gemm,
                            **logits, query_rows, ctx, drafter->ContextBase(),
                            vocab, draft_hidden->get());
    if (!run) {
      return std::unexpected(run.error());
    }
    backend.Synchronize();
    std::vector<float> draft_logits(query_rows * vocab);
    if (!backend.CopyD2H(**logits,
                         reinterpret_cast<std::byte*>(draft_logits.data()),
                         draft_logits.size() * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    // Only the mask rows (one after the anchor) predict draft tokens.
    std::span<const float> mask_logits(draft_logits);
    mask_logits = mask_logits.subspan(vocab);
    std::vector<std::uint32_t> draft_tokens(block);
    const DraftSelectorWeights& selector = drafter->Weights().Selector();
    if (selector.projection != nullptr && selector_gemm.has_value() &&
        selector_kernel.has_value()) {
      const std::size_t topk = draft_config->selector_top_k;
      const std::size_t rank = draft_config->selector_rank;
      std::vector<std::uint32_t> cand_ids(block * topk);
      std::vector<float> unary(block * topk);
      auto candidates =
          DraftCandidates(mask_logits, std::span<std::uint32_t>(cand_ids),
                          std::span<float>(unary), block, vocab, topk);
      if (!candidates) {
        return std::unexpected(candidates.error());
      }
      std::vector<std::int32_t> anchors(block, static_cast<std::int32_t>(next));
      auto upload = [&](const void* data, std::size_t bytes) {
        auto buffer = backend.AllocateBuffer(bytes, MemoryKind::Device);
        if (buffer) {
          backend.CopyH2D(**buffer, std::span<const std::byte>(
                                       reinterpret_cast<const std::byte*>(data),
                                       bytes));
        }
        return buffer;
      };
      auto cand_buf = upload(cand_ids.data(), cand_ids.size() * 4);
      auto anchor_buf = upload(anchors.data(), anchors.size() * 4);
      auto unary_buf = upload(unary.data(), unary.size() * 4);
      auto proj_buf = backend.AllocateBuffer(block * rank * 4,
                                             MemoryKind::Device);
      auto scores_buf = backend.AllocateBuffer(block * topk * topk * 4,
                                               MemoryKind::Device);
      if (!cand_buf || !anchor_buf || !unary_buf || !proj_buf || !scores_buf) {
        return std::unexpected(StatusCode::OutOfMemory);
      }
      // The selector conditions on the draft hidden at the mask rows, which
      // start one row after the anchor.
      if (!backend.CopyD2D(**draft_hidden, hidden * 4, **sel_hidden, 0,
                           block * hidden * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      auto scored = DraftSelectorDevice(
          backend, **selector_gemm, **selector_kernel, **proj_buf,
          **sel_hidden, *selector.projection, *selector.predecessor,
          *selector.successor, **cand_buf, **anchor_buf, **unary_buf,
          **scores_buf, block, hidden, rank, vocab, topk);
      if (!scored) {
        return std::unexpected(scored.error());
      }
      backend.Synchronize();
      std::vector<float> scores(block * topk * topk);
      if (!backend.CopyD2H(**scores_buf,
                           reinterpret_cast<std::byte*>(scores.data()),
                           scores.size() * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
      std::size_t predecessor = 0;
      for (std::size_t r = 0; r < block; ++r) {
        const float* row = scores.data() + (r * topk + predecessor) * topk;
        std::size_t best = 0;
        for (std::size_t c = 1; c < topk; ++c) {
          if (row[c] > row[best]) {
            best = c;
          }
        }
        draft_tokens[r] = cand_ids[r * topk + best];
        predecessor = best;
      }
    } else {
      for (std::size_t r = 0; r < block; ++r) {
        std::span<const float> row(mask_logits.data() + r * vocab, vocab);
        draft_tokens[r] = core::detail::ArgMax(row);
      }
    }
    auto verify = core::VerifyDraft(backend, target, cache, draft_tokens,
                                    *current, &hidden_state, &capture_layers,
                                    &verify_captures);
    if (!verify) {
      return std::unexpected(verify.error());
    }
    // Keep the accepted positions in the context; the rejected rows are
    // dropped with the cache rollback.
    if (verify->accepted > 0) {
      const std::vector<const Buffer*> verify_ptrs(verify_captures.begin(),
                                                   verify_captures.end());
      if (auto appended = drafter->AppendContext(backend, verify_ptrs,
                                                 verify->accepted);
          !appended) {
        return std::unexpected(appended.error());
      }
    }
    ++steps;
    proposed += draft_tokens.size();
    accepted += verify->accepted;
    for (std::size_t i = 0; i < verify->accepted; ++i) {
      if (core::detail::IsStopToken(draft_tokens[i], stop_tokens)) {
        stopped = true;
        break;
      }
      if (produced.size() >= max_tokens) {
        break;
      }
      produced.push_back(draft_tokens[i]);
    }
    if (stopped || produced.size() >= max_tokens) {
      break;
    }
    // The bonus token is emitted by the loop top on the next iteration.
    next = verify->next_token;
  }
  if (stopped && log != nullptr) {
    log->Info("spec dflash2", "stopped at a declared stop token");
  }
  if (log != nullptr) {
    log->Info("spec dflash2",
              "accepted " + std::to_string(accepted) + " of " +
                  std::to_string(proposed) + " draft token(s) over " +
                  std::to_string(steps) + " step(s)");
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    const std::string per_token =
        produced.empty()
            ? std::string()
            : " (" + std::to_string(elapsed / produced.size()) + " ms/token)";
    log->Info("spec dflash2", "generated " + std::to_string(produced.size()) +
                                  " token(s) in " + std::to_string(elapsed) +
                                  " ms" + per_token);
  }
  return produced;
}

}  // namespace tessera::spec
