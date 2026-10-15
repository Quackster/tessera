#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <vector>

#include "core/decode.hpp"
#include "core/spec_driver.hpp"
#include "tessera/architecture.hpp"
#include "tessera/backend.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/pipeline.hpp"
#include "tessera/speculative.hpp"
#include "tessera/types.hpp"

// Private definition of Pipeline::Impl, shared by pipeline.cpp (the
// single-sequence path) and pipeline_many.cpp (the overlapped
// multi-sequence path). Not part of the public API.

namespace tessera {

namespace pipeline_internal {

// The greedy index of the largest value (the last stage's logits).
inline std::uint32_t ArgMax(std::span<const float> values) {
  std::uint32_t best = 0;
  float best_value = values[0];
  for (std::size_t i = 1; i < values.size(); ++i) {
    if (values[i] > best_value) {
      best_value = values[i];
      best = static_cast<std::uint32_t>(i);
    }
  }
  return best;
}

// True when `token` is one of `stops`.
inline bool IsStop(std::uint32_t token, std::span<const std::uint32_t> stops) {
  for (const std::uint32_t stop : stops) {
    if (stop == token) {
      return true;
    }
  }
  return false;
}

}  // namespace pipeline_internal

struct Pipeline::Impl {
  // One pipeline stage: its own engine, sharded model and decode cache, plus
  // the buffers that carry the boundary hidden to and from the neighbours.
  struct Stage {
    std::unique_ptr<Engine> engine;
    std::unique_ptr<Model> model;
    std::unique_ptr<core::DecodeCache> cache;
    std::unique_ptr<Buffer> incoming;  // null on the first stage
    std::unique_ptr<Buffer> outgoing;  // null on the last stage
    // The last trunk layer this stage owns; the boundary capture reads the
    // residual hidden after it.
    std::size_t boundary_layer = 0;
    // Owned trunk layer range [first_layer, last_layer) for budgeting.
    std::size_t first_layer = 0;
    std::size_t last_layer = 0;
  };

  std::vector<Stage> stages;
  std::size_t hidden_dim = 0;
  std::size_t context = 0;
  KvCacheType kv_type = KvCacheType::F32;
  std::vector<std::uint32_t> stops;
  // The speculative strategy (null when none is attached). The drafter runs
  // on the last stage, which owns the MTP head.
  std::unique_ptr<SpeculativeStrategy> strategy;
  // The single-thread decode target over the stages (defined in
  // pipeline_spec.cpp). Nested so it can reach the stage list.
  struct Target;

  // True when a speculative strategy is attached.
  [[nodiscard]] bool HasStrategy() const { return strategy != nullptr; }

  // One speculative generation: the decode driver runs on one thread and the
  // target drives the stages in order (the drafter on the last stage).
  std::expected<GenerateOutcome, StatusCode> RunSpeculative(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      std::vector<std::uint32_t>* produced);

  // Batched cross-stage verify of [anchor, drafts...]: every stage runs a
  // ForwardBatch over the rows, the last stage scores the logits and the
  // accepted prefix is kept, rolling each stage back with TruncateTarget.
  // `prefix_position` is the committed row count before the batch.
  std::expected<DraftVerification, StatusCode> VerifyBatch(
      std::span<const std::uint32_t> draft,
      std::span<const float> prefix_logits, std::optional<std::uint32_t> anchor,
      std::vector<float>* hidden_out, std::size_t prefix_position,
      const std::vector<std::size_t>* capture_layers = nullptr,
      const std::vector<Buffer*>* capture_out = nullptr);

  // Run one batched cross-stage forward over `tokens` (rows) with the
  // boundary handoff. `capture_layers`, when non-null, are target block ids
  // captured on their owning stages and copied to `capture_out` (one
  // rows*hidden buffer per layer, on the last stage). `logits_all` fills
  // `logits_out` with rows*vocab. `hidden_all_out`, when non-null, receives
  // every row's final hidden (rows*hidden). Shared by the prefill, the
  // verify, and the single-token step.
  std::expected<void, StatusCode> ForwardRows(
      std::span<const std::uint32_t> tokens, bool snapshot_state,
      const std::vector<std::size_t>* capture_layers,
      const std::vector<Buffer*>* capture_out, bool logits_all,
      std::vector<float>* logits_out, std::vector<float>* hidden_all_out);

  // Batched cross-stage prefill with target-hidden capture for a
  // hidden-conditioned drafter (DFlash2): each chunk runs ForwardRows and
  // strategy.AppendPrefill appends the captured rows. `position_out` is set
  // to the prompt length. MTP does not use this path.
  std::expected<void, StatusCode> PrefillCapturing(
      SpeculativeStrategy& strategy, std::span<const std::uint32_t> prompt,
      const GenerateOptions& options, std::vector<float>* hidden_out,
      std::vector<float>* logits_out, std::size_t* position_out);

  // Fresh caches for every stage, so a new sequence starts at position zero
  // and the request's speed tuning and KV type take effect (the architecture
  // state copies tuning when it is first built).
  void ResetCaches(const GenerateOptions& options) {
    for (Stage& stage : stages) {
      stage.cache = std::make_unique<core::DecodeCache>();
      stage.cache->kv_type = options.kv_type;
      stage.cache->tuning.mxfp4_split_target = options.mxfp4_split_target;
      stage.cache->tuning.mxfp4_split_cap = options.mxfp4_split_cap;
      stage.cache->tuning.attention_split = options.attention_split;
    }
  }

  // Greedy generation with `options` tuning, filling `produced`; the outcome
  // carries the produced count and the prefill/decode wall times.
  std::expected<GenerateOutcome, StatusCode> Run(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      std::vector<std::uint32_t>* produced);

  // Run `token` through every stage, returning the last stage's logits
  // when `want_logits`; otherwise only the trunk runs and the logits
  // result is empty. The hidden handoff copies each stage's boundary
  // residual into the next stage's incoming buffer.
  std::expected<std::vector<float>, StatusCode> Step(
      std::uint32_t token, bool want_logits,
      std::vector<float>* hidden_out = nullptr) {
    const std::size_t n = stages.size();
    for (std::size_t i = 0; i < n; ++i) {
      Stage& stage = stages[i];
      const Architecture* arch = stage.model->Arch();
      if (arch == nullptr) {
        return std::unexpected(StatusCode::UnsupportedFeature);
      }
      Backend& backend = stage.engine->Owner();
      const Buffer* incoming = stage.incoming.get();
      if (i + 1 == n) {
        if (!want_logits) {
          auto forwarded = arch->Forward(backend, *stage.model, *stage.cache,
                                         token, hidden_out, nullptr, nullptr,
                                         incoming);
          if (!forwarded) {
            return std::unexpected(forwarded.error());
          }
          return std::vector<float>{};
        }
        return arch->Logits(backend, *stage.model, *stage.cache, token,
                            hidden_out, nullptr, nullptr, incoming);
      }
      std::vector<std::size_t> capture_layers{stage.boundary_layer};
      std::vector<Buffer*> capture{stage.outgoing.get()};
      auto forwarded =
          arch->Forward(backend, *stage.model, *stage.cache, token, nullptr,
                        &capture_layers, &capture, incoming);
      if (!forwarded) {
        return std::unexpected(forwarded.error());
      }
      Stage& next = stages[i + 1];
      auto copied = CopyAcrossDevices(*stage.outgoing, 0, *next.incoming, 0,
                                      hidden_dim * 4);
      if (!copied) {
        return std::unexpected(copied.error());
      }
    }
    return std::unexpected(StatusCode::DeviceError);
  }
};

}  // namespace tessera
