#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <vector>

#include "core/decode.hpp"
#include "core/profile.hpp"
#include "core/spec_driver.hpp"
#include "tessera/backend.hpp"
#include "tessera/log.hpp"
#include "tessera/model.hpp"
#include "tessera/speculative.hpp"
#include "tessera/types.hpp"

// The single-device decode target: the engine's one backend, model and
// decode cache, with the drafter's capture buffers held here. It implements
// the SpecTarget seam for Engine::GenerateStreaming, so the one decode
// driver runs unchanged over one device.

namespace tessera::core {

class SingleDeviceTarget : public SpecTarget {
 public:
  SingleDeviceTarget(Backend& backend, Model& model, DecodeCache& cache,
                     log::Diagnostics& diagnostics, std::size_t prefill_chunk,
                     Profile* profile);
  SingleDeviceTarget(const SingleDeviceTarget&) = delete;
  SingleDeviceTarget& operator=(const SingleDeviceTarget&) = delete;

  [[nodiscard]] log::Diagnostics& diagnostics() override {
    return diagnostics_;
  }
  [[nodiscard]] Backend& backend() override { return backend_; }
  [[nodiscard]] Model& model() override { return model_; }
  [[nodiscard]] DecodeCache& cache() override { return cache_; }

  std::expected<void, StatusCode> Prepare(
      SpeculativeStrategy* strategy, const GenerateOptions& options) override;
  std::expected<void, StatusCode> Prefill(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      std::vector<float>* hidden, std::vector<float>* logits,
      double* compute_ms) override;
  std::expected<std::uint32_t, StatusCode> GreedyStep(
      std::uint32_t token) override;
  std::expected<std::vector<float>, StatusCode> LogitsStep(
      std::uint32_t token, std::vector<float>* hidden) override;
  std::expected<DraftVerification, StatusCode> Verify(
      std::span<const std::uint32_t> draft,
      std::span<const float> prefix_logits,
      std::optional<std::uint32_t> anchor,
      std::vector<float>* hidden) override;
  void ReportDiagnostics(std::size_t loop_steps) override;

 private:
  Backend& backend_;
  Model& model_;
  DecodeCache& cache_;
  log::Diagnostics& diagnostics_;
  std::size_t prefill_chunk_ = 0;
  Profile* profile_ = nullptr;
  SpeculativeStrategy* strategy_ = nullptr;
  std::vector<std::size_t> capture_layers_;
  std::vector<Buffer*> capture_buffers_;
};

}  // namespace tessera::core
