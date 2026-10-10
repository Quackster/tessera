#include "core/generate_helpers.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "core/think_budget.hpp"

namespace tessera {

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

// Think tags for a thinking budget: the tokenizer's special ids when
// declared, else the BPE encoding of the markers. A disengaged budget
// (plus empty close ids) when the budget is zero, the model has no
// tokenizer, or the tags do not resolve.
std::pair<core::ThinkBudget, std::vector<std::uint32_t>> ResolveThinkBudget(
    const Model& model, std::size_t budget) {
  core::ThinkBudget think;
  std::vector<std::uint32_t> close;
  if (budget == 0) {
    return {think, close};
  }
  const Tokenizer* tokenizer = model.GetTokenizer();
  std::vector<std::uint32_t> open;
  if (tokenizer != nullptr) {
    const auto open_special =
        tokenizer->SpecialTokenId(core::kThinkOpenTag);
    const auto close_special =
        tokenizer->SpecialTokenId(core::kThinkCloseTag);
    if (open_special && close_special) {
      open.push_back(*open_special);
      close.push_back(*close_special);
    } else {
      auto open_ids = tokenizer->Encode(core::kThinkOpenTag);
      auto close_ids = tokenizer->Encode(core::kThinkCloseTag);
      if (open_ids && close_ids && !open_ids->empty() &&
          !close_ids->empty()) {
        open = std::move(*open_ids);
        close = std::move(*close_ids);
      }
    }
  }
  if (!open.empty() && !close.empty()) {
    think.engage(open, close, budget);
  }
  return {think, close};
}

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

}  // namespace tessera
