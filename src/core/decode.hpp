#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <vector>

#include "tessera/backend.hpp"
#include "tessera/model.hpp"

namespace tessera::core {

// Per-step state for single-token decoding: the host key/value cache
// per layer plus the loaded kernels. The cache grows one row per step;
// kernels load once and are reused.
struct DecodeCache {
  struct LayerCache {
    std::vector<float> k;
    std::vector<float> v;
  };
  std::vector<LayerCache> layers;
  std::unique_ptr<Kernel> gemm_kernel;
  std::unique_ptr<Kernel> rope_kernel;
  std::unique_ptr<Kernel> attention_kernel;
};

// One decoder step for a vanilla transformer (llama.cpp GGUF tensor
// naming): embed the token, run every block (QKV projections on the
// device, RoPE, causal attention over the cache, gated MLP), project
// to logits and return the argmax token. Projections must be Q4_K and
// norms/embeddings F32, else UnsupportedFeature; a missing tensor is
// MalformedFile. Deterministic: greedy sampling, fixed op order.
[[nodiscard]] std::expected<std::uint32_t, StatusCode> DecodeStep(
    Backend& backend, const Model& model, DecodeCache& cache,
    std::uint32_t token);

}  // namespace tessera::core
