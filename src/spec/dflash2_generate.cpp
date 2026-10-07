#include "spec/dflash2_generate.hpp"

#include <cstring>
#include <memory>

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
    const std::string& draft_path) {
  if (options.max_tokens == 0) {
    return std::vector<std::uint32_t>{};
  }
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
  std::size_t block = draft_config->block_size;
  if (options.draft_tokens > 0 && options.draft_tokens < block) {
    block = options.draft_tokens;
  }
  const std::size_t n = draft_config->target_layer_ids.size();
  std::vector<std::size_t> capture_layers(draft_config->target_layer_ids.begin(),
                                          draft_config->target_layer_ids.end());
  std::vector<std::unique_ptr<Buffer>> capture_storage;
  std::vector<Buffer*> captures;
  for (std::size_t i = 0; i < n; ++i) {
    auto buffer = backend.AllocateBuffer(hidden * 4, MemoryKind::Device);
    if (!buffer) {
      return std::unexpected(StatusCode::OutOfMemory);
    }
    capture_storage.push_back(std::move(*buffer));
    captures.push_back(capture_storage.back().get());
  }
  auto aux = backend.AllocateBuffer(n * hidden * 4, MemoryKind::Device);
  auto logits = backend.AllocateBuffer(block * vocab * 4, MemoryKind::Device);
  auto draft_hidden = backend.AllocateBuffer(block * hidden * 4,
                                             MemoryKind::Device);
  if (!aux || !logits || !draft_hidden) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  auto selector_gemm = backend.LoadKernel("gemm_f32", {});
  auto selector_kernel = backend.LoadKernel("selector_edge_score", {});
  core::DecodeCache cache;
  cache.kv_f16 = options.kv_f16;
  std::vector<std::uint32_t> prompt = options.prompt_tokens;
  if (prompt.empty()) {
    prompt.push_back(options.first_token);
  }
  for (std::size_t i = 0; i + 1 < prompt.size(); ++i) {
    auto forward = core::DecodeForward(backend, target, cache, prompt[i]);
    if (!forward) {
      return std::unexpected(forward.error());
    }
  }
  std::vector<float> hidden_state;
  auto first = core::DecodeLogits(backend, target, cache, prompt.back(),
                                  &hidden_state, &capture_layers, &captures);
  if (!first) {
    return std::unexpected(first.error());
  }
  std::vector<std::uint32_t> produced;
  produced.reserve(options.max_tokens);
  std::uint32_t next = core::detail::ArgMax(*first);
  while (produced.size() < options.max_tokens) {
    produced.push_back(next);
    if (produced.size() >= options.max_tokens) {
      break;
    }
    auto current = core::DecodeLogits(backend, target, cache, next,
                                      &hidden_state, &capture_layers, &captures);
    if (!current) {
      return std::unexpected(current.error());
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (!backend.CopyD2D(*captures[i], 0, **aux, i * hidden * 4, hidden * 4)) {
        return std::unexpected(StatusCode::DeviceError);
      }
    }
    auto mask = MaskEmbeddings(backend, *embed, draft_config->mask_token_id,
                               block, hidden);
    if (!mask) {
      return std::unexpected(mask.error());
    }
    auto run = drafter->Run(backend, **mask, **aux, *output->device, **head_gemm,
                            **logits, block, 1, vocab, draft_hidden->get());
    if (!run) {
      return std::unexpected(run.error());
    }
    backend.Synchronize();
    std::vector<float> draft_logits(block * vocab);
    if (!backend.CopyD2H(**logits,
                         reinterpret_cast<std::byte*>(draft_logits.data()),
                         draft_logits.size() * 4)) {
      return std::unexpected(StatusCode::DeviceError);
    }
    std::vector<std::uint32_t> draft_tokens(block);
    const DraftSelectorWeights& selector = drafter->Weights().Selector();
    if (selector.projection != nullptr && selector_gemm.has_value() &&
        selector_kernel.has_value()) {
      const std::size_t topk = draft_config->selector_top_k;
      const std::size_t rank = draft_config->selector_rank;
      std::vector<std::uint32_t> cand_ids(block * topk);
      std::vector<float> unary(block * topk);
      auto candidates = DraftCandidates(std::span<const float>(draft_logits),
                                        std::span<std::uint32_t>(cand_ids),
                                        std::span<float>(unary), block, vocab,
                                        topk);
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
      auto scored = DraftSelectorDevice(
          backend, **selector_gemm, **selector_kernel, **proj_buf,
          **draft_hidden, *selector.projection, *selector.predecessor,
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
        std::span<const float> row(draft_logits.data() + r * vocab, vocab);
        draft_tokens[r] = core::detail::ArgMax(row);
      }
    }
    auto verify = core::VerifyDraft(backend, target, cache, draft_tokens,
                                    *current, &hidden_state);
    if (!verify) {
      return std::unexpected(verify.error());
    }
    for (std::size_t i = 0; i < verify->accepted; ++i) {
      if (produced.size() >= options.max_tokens) {
        break;
      }
      produced.push_back(draft_tokens[i]);
    }
    // The bonus token is emitted by the loop top on the next iteration.
    next = verify->next_token;
  }
  return produced;
}

}  // namespace tessera::spec
