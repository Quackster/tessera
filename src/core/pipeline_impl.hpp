#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
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

// Contiguous trunk layer ranges, one [first, last) per stage. An empty
// `stage_layers` splits `total_layers` as evenly as the count allows;
// otherwise it holds each stage's layer count and must have one entry per
// stage, each at least one, summing to `total_layers`. nullopt on a bad
// count.
//
// Usage:
//   auto ranges = StageLayerRanges(layers, devices.size(), stage_layers);
inline std::optional<std::vector<std::pair<std::size_t, std::size_t>>>
StageLayerRanges(std::size_t total_layers, std::size_t stage_count,
                 std::span<const std::size_t> stage_layers) {
  if (stage_count == 0 || total_layers < stage_count) {
    return std::nullopt;
  }
  std::vector<std::pair<std::size_t, std::size_t>> ranges(stage_count);
  if (!stage_layers.empty()) {
    if (stage_layers.size() != stage_count) {
      return std::nullopt;
    }
    std::size_t sum = 0;
    for (const std::size_t count : stage_layers) {
      if (count == 0) {
        return std::nullopt;
      }
      sum += count;
    }
    if (sum != total_layers) {
      return std::nullopt;
    }
    std::size_t first = 0;
    for (std::size_t i = 0; i < stage_count; ++i) {
      ranges[i] = {first, first + stage_layers[i]};
      first += stage_layers[i];
    }
    return ranges;
  }
  for (std::size_t i = 0; i < stage_count; ++i) {
    const std::size_t first = total_layers * i / stage_count;
    const std::size_t last = total_layers * (i + 1) / stage_count;
    if (last <= first) {
      return std::nullopt;
    }
    ranges[i] = {first, last};
  }
  return ranges;
}

}  // namespace pipeline_internal

struct Pipeline::Impl {
  // One pipeline stage: its own engine, sharded model and decode cache, plus
  // the buffers that carry the boundary hidden to and from the neighbours.
  struct Stage {
    std::unique_ptr<Engine> engine;
    std::unique_ptr<Model> model;
    // The active turn's cache and boundary buffers. They are owned by the
    // SequenceState that Activate() points the stages at, so several turns
    // share the weights and the device while each keeps its own KV and
    // boundary state. Null until the first Activate.
    core::DecodeCache* cache = nullptr;
    Buffer* incoming = nullptr;  // null on the first stage
    Buffer* outgoing = nullptr;  // null on the last stage
    // The last trunk layer this stage owns; the boundary capture reads the
    // residual hidden after it.
    std::size_t boundary_layer = 0;
    // Owned trunk layer range [first_layer, last_layer) for budgeting.
    std::size_t first_layer = 0;
    std::size_t last_layer = 0;
  };

  // One turn's private state: a cache per stage, the boundary handoff
  // buffers, and its own draft strategy. The pipeline shares the weights and
  // the device; at most one sequence is active at a time (the serve path
  // holds the device lock and yields between tokens), so the stage pointers
  // carry one sequence's state at a time.
  struct SequenceState {
    std::vector<std::unique_ptr<core::DecodeCache>> caches;
    std::vector<std::unique_ptr<Buffer>> incoming;  // null on stage 0
    std::vector<std::unique_ptr<Buffer>> outgoing;  // null on the last stage
    std::unique_ptr<SpeculativeStrategy> strategy;
    bool in_use = false;
  };

  std::vector<Stage> stages;
  std::size_t hidden_dim = 0;
  std::size_t context = 0;
  KvCacheType kv_type = KvCacheType::F32;
  std::vector<std::uint32_t> stops;
  // One sequence per concurrent turn; [0] is the single-turn default the
  // CLI run and calibrate paths use. The rest are the serve pool.
  std::vector<std::unique_ptr<SequenceState>> sequences;
  // The attached draft strategy factory and its options. AttachSpeculative
  // builds one strategy per sequence from them; a sequence built later (a new
  // pool entry) gets its own too.
  std::function<std::unique_ptr<SpeculativeStrategy>()> strategy_factory;
  StrategyOptions strategy_options;
  // Serializes all pipeline device work. A turn holds it for its whole
  // generation and releases it between tokens (see GenerateStreamingOn), so
  // concurrent turns interleave instead of running one to completion.
  mutable std::mutex device;
  // Guards the sequence pool's in_use flags and its wait queue.
  std::mutex pool;
  std::condition_variable pool_cv;
  // The single-thread decode target over the stages (defined in
  // pipeline_spec.cpp). Nested so it can reach the stage list.
  struct Target;

  // True when a speculative strategy factory is attached.
  [[nodiscard]] bool HasStrategy() const {
    return strategy_factory != nullptr;
  }

  // Point every stage at `sequence`'s cache and boundary buffers. Call while
  // the device lock is held, before driving the stages for that turn.
  void Activate(SequenceState& sequence) {
    for (std::size_t i = 0; i < stages.size(); ++i) {
      stages[i].cache = sequence.caches[i].get();
      stages[i].incoming =
          sequence.incoming.empty() ? nullptr : sequence.incoming[i].get();
      stages[i].outgoing =
          sequence.outgoing.empty() ? nullptr : sequence.outgoing[i].get();
    }
  }

  // Build one sequence: a cache per stage, the boundary buffers, and a draft
  // strategy from the attached factory when one is set. Null on allocation
  // failure.
  std::unique_ptr<SequenceState> NewSequence();

  // Take a free sequence from the pool, blocking until one is free. Returns
  // the default sequence [0] never; the pool starts at index 1. Null when
  // `cancelled` reports a cancel while the caller waits.
  SequenceState* AcquireSequence(const std::function<bool()>& cancelled = {});

  // Return a sequence taken by AcquireSequence to the pool.
  void ReleaseSequence(SequenceState& sequence);

  // The single-turn default sequence the CLI run and calibrate paths use.
  SequenceState& DefaultSequence() { return *sequences.front(); }

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

  // Fresh caches for `sequence`, so a new turn starts at position zero and
  // the request's speed tuning and KV type take effect (the architecture
  // state copies tuning when it is first built).
  void ResetSequence(SequenceState& sequence, const GenerateOptions& options) {
    for (std::unique_ptr<core::DecodeCache>& cache : sequence.caches) {
      cache = std::make_unique<core::DecodeCache>();
      cache->kv_type = options.kv_type;
      cache->tuning.mxfp4_split_target = options.mxfp4_split_target;
      cache->tuning.mxfp4_split_cap = options.mxfp4_split_cap;
      cache->tuning.attention_split = options.attention_split;
    }
  }

  // Drain every stage's device work. The stages share per-model state across
  // turns: the MoE expert staging views and their cache live on the Model, not
  // on a turn's cache. A turn may hand the device lock to another turn only
  // after its own device work completes, or the next turn overwrites the
  // shared staging while the previous turn's kernels still read it. The same
  // drain is required before freeing a turn's caches, whose buffers in-flight
  // kernels still reference.
  void SynchronizeStages() {
    for (Stage& stage : stages) {
      if (stage.engine != nullptr) {
        stage.engine->Owner().Synchronize();
      }
    }
  }

  // Free `sequence`'s per-turn caches so an idle sequence holds no KV or
  // draft state. The next turn on it rebuilds them in ResetSequence. The
  // boundary buffers stay (small, and reused every turn).
  void ReleaseSequenceCaches(SequenceState& sequence) {
    // Drain before freeing: the last step's kernels and copies are
    // asynchronous and still reference these buffers, and the next turn's
    // allocations reuse the freed addresses.
    SynchronizeStages();
    for (std::unique_ptr<core::DecodeCache>& cache : sequence.caches) {
      cache.reset();
    }
    // Point the stages at null so a stale pointer never outlives the cache.
    Activate(sequence);
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
      const Buffer* incoming = stage.incoming;
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
      std::vector<Buffer*> capture{stage.outgoing};
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
