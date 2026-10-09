#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "models/qwen3_5/state.hpp"
#include "spec/dflash2_mask.hpp"
#include "test_helpers.hpp"
#include "tessera/engine.hpp"
#include "tessera/types.hpp"

namespace tessera::testing {

// A tiny gated-attention hybrid GGUF: one full-attention block
// (interval 1 so every layer is full attention), hidden 256, 8 heads
// over 8 kv groups of dim 32, ffn 256, vocab 32. Projections are Q4_K,
// vectors F32; tensor names follow the gated-attention convention
// (fused attn_q, QK norms, post_attention_norm).
inline std::filesystem::path WriteGatedHybridFixture(const std::string& name) {
  std::mt19937 rng(21);
  struct Spec {
    const char* tensor;
    std::uint32_t type;
    std::vector<std::uint64_t> dims;
    std::size_t rows;  // Q4_K row count (0 for F32)
    std::size_t cols;  // Q4_K row length (0 for F32)
  };
  const std::vector<Spec> specs = {
      {"token_embd.weight", 12, {256, 32}, 32, 256},
      {"output_norm.weight", 0, {256}, 0, 0},
      {"output.weight", 12, {256, 32}, 32, 256},
      {"blk.0.attn_norm.weight", 0, {256}, 0, 0},
      {"blk.0.attn_q.weight", 12, {256, 512}, 512, 256},
      {"blk.0.attn_k.weight", 12, {256, 256}, 256, 256},
      {"blk.0.attn_v.weight", 12, {256, 256}, 256, 256},
      {"blk.0.attn_output.weight", 12, {256, 256}, 256, 256},
      {"blk.0.attn_q_norm.weight", 0, {32}, 0, 0},
      {"blk.0.attn_k_norm.weight", 0, {32}, 0, 0},
      {"blk.0.post_attention_norm.weight", 0, {256}, 0, 0},
      {"blk.0.ffn_gate.weight", 12, {256, 256}, 256, 256},
      {"blk.0.ffn_up.weight", 12, {256, 256}, 256, 256},
      {"blk.0.ffn_down.weight", 12, {256, 256}, 256, 256},
  };
  GgufBuilder builder;
  builder.Header(0x46554747, 3, specs.size(), 19);
  builder.KvString("general.name", "tiny-gated");
  builder.KvString("general.architecture", "qwen35");
  builder.KvU32("qwen35.block_count", 1);
  builder.KvU32("qwen35.embedding_length", 256);
  builder.KvU32("qwen35.feed_forward_length", 256);
  builder.KvF32("qwen35.attention.layer_norm_rms_epsilon", 1e-5f);
  builder.KvU32("qwen35.attention.head_count", 8);
  builder.KvU32("qwen35.attention.head_count_kv", 8);
  builder.KvU32("qwen35.attention.key_length", 32);
  builder.KvU32("qwen35.attention.value_length", 32);
  builder.KvU32("qwen35.rope.dimension_count", 32);
  builder.KvF32("qwen35.rope.freq_base", 10000.0f);
  builder.PushString("qwen35.rope.dimension_sections");
  builder.PushU32(9);
  builder.PushU32(4);
  builder.PushU64(4);
  for (std::uint32_t s : {4u, 4u, 4u, 0u}) {
    builder.PushU32(s);
  }
  builder.KvU32("qwen35.ssm.conv_kernel", 1);
  builder.KvU32("qwen35.ssm.state_size", 1);
  builder.KvU32("qwen35.ssm.group_count", 1);
  builder.KvU32("qwen35.ssm.time_step_rank", 1);
  builder.KvU32("qwen35.ssm.inner_size", 1);
  builder.KvU32("qwen35.full_attention_interval", 1);
  std::uint64_t offset = 0;
  for (const auto& spec : specs) {
    const std::uint64_t placed = offset;
    std::size_t bytes = 0;
    if (spec.type == 0) {
      std::size_t numel = 1;
      for (auto d : spec.dims) {
        numel *= static_cast<std::size_t>(d);
      }
      bytes = numel * 4;
    } else {
      bytes = spec.rows * (spec.cols / 256) * 144;
    }
    offset = (offset + bytes + 31) & ~31u;
    if (spec.dims.size() == 1) {
      builder.Tensor(spec.tensor, 1, {spec.dims[0]}, spec.type, placed);
    } else {
      builder.Tensor(spec.tensor, 2, {spec.dims[0], spec.dims[1]}, spec.type,
                     placed);
    }
  }
  builder.PadTo(((builder.bytes.size() + 31) & ~31u));
  for (const auto& spec : specs) {
    if (spec.type == 0) {
      std::size_t numel = 1;
      for (auto d : spec.dims) {
        numel *= static_cast<std::size_t>(d);
      }
      const bool is_norm =
          std::string(spec.tensor).find("norm") != std::string::npos;
      for (std::size_t i = 0; i < numel; ++i) {
        builder.PushF32(is_norm ? 1.0f : DrawValue(rng));
      }
    } else {
      std::vector<float> values(spec.rows * spec.cols);
      for (auto& v : values) {
        v = DrawValue(rng);
      }
      auto block = QuantizeRows(values, spec.rows, spec.cols);
      for (auto b : block) {
        builder.bytes.push_back(b);
      }
    }
    builder.PadTo(((builder.bytes.size() + 31) & ~31u));
  }
  auto dir = FreshTempDir("tessera_tests_gated_hybrid");
  auto path = dir / name;
  WriteBytes(path, builder.bytes);
  return path;
}

// A tiny all-linear-attention hybrid GGUF: one gated-delta block
// (interval 2 so layer 0 is linear attention), hidden 256, key_dim 16
// (2 key heads of 8), value_dim 256 (4 value heads of 64), conv width
// 4, ffn 256, vocab 32. Projections are Q4_K, vectors F32.
inline std::filesystem::path WriteLinearHybridFixture(
    const std::string& name) {
  std::mt19937 rng(22);
  struct Spec {
    const char* tensor;
    std::uint32_t type;
    std::vector<std::uint64_t> dims;
    std::size_t rows;
    std::size_t cols;
  };
  const std::vector<Spec> specs = {
      {"token_embd.weight", 0, {256, 32}, 0, 0},
      {"output_norm.weight", 0, {256}, 0, 0},
      {"output.weight", 12, {256, 32}, 32, 256},
      {"blk.0.attn_norm.weight", 0, {256}, 0, 0},
      {"blk.0.attn_qkv.weight", 12, {256, 288}, 288, 256},
      {"blk.0.attn_gate.weight", 12, {256, 256}, 256, 256},
      {"blk.0.ssm_conv1d.weight", 0, {4, 288}, 0, 0},
      {"blk.0.ssm_alpha.weight", 12, {256, 4}, 4, 256},
      {"blk.0.ssm_beta.weight", 12, {256, 4}, 4, 256},
      {"blk.0.ssm_a", 0, {4}, 0, 0},
      {"blk.0.ssm_dt.bias", 0, {4}, 0, 0},
      {"blk.0.ssm_norm.weight", 0, {64}, 0, 0},
      {"blk.0.ssm_out.weight", 12, {256, 256}, 256, 256},
      {"blk.0.post_attention_norm.weight", 0, {256}, 0, 0},
      {"blk.0.ffn_gate.weight", 12, {256, 256}, 256, 256},
      {"blk.0.ffn_up.weight", 12, {256, 256}, 256, 256},
      {"blk.0.ffn_down.weight", 12, {256, 256}, 256, 256},
  };
  GgufBuilder builder;
  builder.Header(0x46554747, 3, specs.size(), 19);
  builder.KvString("general.name", "tiny-linear");
  builder.KvString("general.architecture", "qwen35");
  builder.KvU32("qwen35.block_count", 1);
  builder.KvU32("qwen35.embedding_length", 256);
  builder.KvU32("qwen35.feed_forward_length", 256);
  builder.KvF32("qwen35.attention.layer_norm_rms_epsilon", 1e-5f);
  builder.KvU32("qwen35.attention.head_count", 8);
  builder.KvU32("qwen35.attention.head_count_kv", 8);
  builder.KvU32("qwen35.attention.key_length", 32);
  builder.KvU32("qwen35.attention.value_length", 32);
  builder.KvU32("qwen35.rope.dimension_count", 32);
  builder.KvF32("qwen35.rope.freq_base", 10000.0f);
  builder.PushString("qwen35.rope.dimension_sections");
  builder.PushU32(9);
  builder.PushU32(4);
  builder.PushU64(4);
  for (std::uint32_t s : {4u, 4u, 4u, 0u}) {
    builder.PushU32(s);
  }
  builder.KvU32("qwen35.ssm.conv_kernel", 4);
  builder.KvU32("qwen35.ssm.state_size", 8);
  builder.KvU32("qwen35.ssm.group_count", 2);
  builder.KvU32("qwen35.ssm.time_step_rank", 4);
  builder.KvU32("qwen35.ssm.inner_size", 256);
  builder.KvU32("qwen35.full_attention_interval", 2);
  std::uint64_t offset = 0;
  for (const auto& spec : specs) {
    const std::uint64_t placed = offset;
    std::size_t bytes = 0;
    if (spec.type == 0) {
      std::size_t numel = 1;
      for (auto d : spec.dims) {
        numel *= static_cast<std::size_t>(d);
      }
      bytes = numel * 4;
    } else {
      bytes = spec.rows * (spec.cols / 256) * 144;
    }
    offset = (offset + bytes + 31) & ~31u;
    if (spec.dims.size() == 1) {
      builder.Tensor(spec.tensor, 1, {spec.dims[0]}, spec.type, placed);
    } else {
      builder.Tensor(spec.tensor, 2, {spec.dims[0], spec.dims[1]}, spec.type,
                     placed);
    }
  }
  builder.PadTo(((builder.bytes.size() + 31) & ~31u));
  for (const auto& spec : specs) {
    if (spec.type == 0) {
      std::size_t numel = 1;
      for (auto d : spec.dims) {
        numel *= static_cast<std::size_t>(d);
      }
      const bool is_norm =
          std::string(spec.tensor).find("norm") != std::string::npos;
      for (std::size_t i = 0; i < numel; ++i) {
        builder.PushF32(is_norm ? 1.0f : DrawValue(rng));
      }
    } else {
      std::vector<float> values(spec.rows * spec.cols);
      for (auto& v : values) {
        v = DrawValue(rng);
      }
      auto block = QuantizeRows(values, spec.rows, spec.cols);
      for (auto b : block) {
        builder.bytes.push_back(b);
      }
    }
    builder.PadTo(((builder.bytes.size() + 31) & ~31u));
  }
  auto dir = FreshTempDir("tessera_tests_linear_hybrid");
  auto path = dir / name;
  WriteBytes(path, builder.bytes);
  return path;
}


}  // namespace tessera::testing

using tessera::Engine;
using tessera::ModelOptions;
using tessera::testing::MakeEngineOrSkip;
using tessera::testing::WriteGatedHybridFixture;
using tessera::testing::WriteLinearHybridFixture;

namespace {

// A hybrid model decodes two steps and a fresh cache replays the first.
void ExpectDeterministicDecode(const std::string& path) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{path, 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto config = (*model)->Config();
  ASSERT_TRUE(config.has_value()) << tessera::ToString(config.error());
  EXPECT_TRUE(config->hybrid);
  tessera::core::DecodeCache cache;
  auto first = tessera::core::DecodeStep(engine->Owner(), **model, cache, 0);
  ASSERT_TRUE(first.has_value()) << tessera::ToString(first.error());
  auto second =
      tessera::core::DecodeStep(engine->Owner(), **model, cache, *first);
  ASSERT_TRUE(second.has_value()) << tessera::ToString(second.error());
  tessera::core::DecodeCache replay;
  auto again = tessera::core::DecodeStep(engine->Owner(), **model, replay, 0);
  ASSERT_TRUE(again.has_value()) << tessera::ToString(again.error());
  EXPECT_EQ(*again, *first);
}

}  // namespace

TEST(HybridDecodeTest, GatedFullAttentionDecodesDeterministically) {
  ExpectDeterministicDecode(WriteGatedHybridFixture("gated.gguf").string());
}

TEST(HybridDecodeTest, LinearAttentionDecodesDeterministically) {
  ExpectDeterministicDecode(WriteLinearHybridFixture("linear.gguf").string());
}

namespace {

// Greedy tokens from token 0 for `n` steps (fixed-seed fixtures).
std::vector<std::uint32_t> GreedyTokens(const std::string& path,
                                        std::size_t n) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{path, 1024});
  EXPECT_TRUE(model.has_value()) << tessera::ToString(model.error());
  std::vector<std::uint32_t> tokens;
  if (!model.has_value()) {
    return tokens;
  }
  tessera::core::DecodeCache cache;
  std::uint32_t next = 0;
  for (std::size_t i = 0; i < n; ++i) {
    auto step = tessera::core::DecodeStep(engine->Owner(), **model, cache, next);
    EXPECT_TRUE(step.has_value()) << tessera::ToString(step.error());
    if (!step.has_value()) {
      break;
    }
    next = *step;
    tokens.push_back(next);
  }
  return tokens;
}

// Greedy decode driven by speculative verification with a fixed draft
// pair. The output must not depend on the draft.
std::vector<std::uint32_t> SpeculativeTokens(const std::string& path,
                                             std::size_t n,
                                             std::uint32_t draft_token) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{path, 1024});
  EXPECT_TRUE(model.has_value()) << tessera::ToString(model.error());
  std::vector<std::uint32_t> out;
  if (!model.has_value()) {
    return out;
  }
  tessera::core::DecodeCache cache;
  auto current = tessera::core::DecodeLogits(engine->Owner(), **model, cache, 0);
  EXPECT_TRUE(current.has_value()) << tessera::ToString(current.error());
  if (!current.has_value()) {
    return out;
  }
  while (out.size() < n) {
    const std::vector<std::uint32_t> draft = {draft_token, draft_token};
    auto verify = tessera::core::VerifyDraft(engine->Owner(), **model, cache,
                                             draft, *current);
    EXPECT_TRUE(verify.has_value()) << tessera::ToString(verify.error());
    if (!verify.has_value()) {
      break;
    }
    for (std::size_t i = 0; i < verify->accepted && out.size() < n; ++i) {
      out.push_back(draft[i]);
    }
    if (out.size() >= n) {
      break;
    }
    out.push_back(verify->next_token);
    auto next = tessera::core::DecodeLogits(engine->Owner(), **model, cache,
                                            verify->next_token);
    EXPECT_TRUE(next.has_value()) << tessera::ToString(next.error());
    if (!next.has_value()) {
      break;
    }
    current = std::move(*next);
  }
  return out;
}

}  // namespace

// Baseline pinning: the fixed-seed fixtures decode to these exact greedy
// tokens on both backends (fp32 sequential accumulation, identical op
// order). A drift here means a kernel or scheduler change.
TEST(HybridDecodeTest, GatedBaselineIsPinned) {
  const auto got = GreedyTokens(WriteGatedHybridFixture("gated.gguf").string(),
                                8);
  const std::vector<std::uint32_t> want = {1, 0, 26, 18, 17, 3, 3, 3};
  EXPECT_EQ(got, want);
}

TEST(HybridDecodeTest, LinearBaselineIsPinned) {
  const auto got =
      GreedyTokens(WriteLinearHybridFixture("linear.gguf").string(), 8);
  const std::vector<std::uint32_t> want = {21, 28, 24, 6, 9, 15, 18, 24};
  EXPECT_EQ(got, want);
}

namespace {

std::uint32_t RowArgMax(const std::vector<float>& row) {
  std::uint32_t best = 0;
  for (std::size_t i = 1; i < row.size(); ++i) {
    if (row[i] > row[best]) {
      best = static_cast<std::uint32_t>(i);
    }
  }
  return best;
}

}  // namespace

// ScoreTokens is the verifier vocabulary: each row is the full logits
// distribution after the prefix ending at that token. Its argmaxes must
// follow the pinned greedy sequence, each row must be vocab sized, and a
// single DecodeLogits step must agree with the first row exactly.
TEST(HybridDecodeTest, ScoreTokensMatchesGreedy) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(
      ModelOptions{WriteGatedHybridFixture("gated.gguf").string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto cfg = (*model)->Config();
  ASSERT_TRUE(cfg.has_value()) << tessera::ToString(cfg.error());
  const std::vector<std::uint32_t> prefix = {0, 1, 0, 26};
  auto rows = tessera::core::ScoreTokens(engine->Owner(), **model, prefix);
  ASSERT_TRUE(rows.has_value()) << tessera::ToString(rows.error());
  ASSERT_EQ(rows->size(), prefix.size());
  const std::vector<std::uint32_t> want = {1, 0, 26, 18};
  for (std::size_t i = 0; i < rows->size(); ++i) {
    EXPECT_EQ((*rows)[i].size(), cfg->vocab_size);
    EXPECT_EQ(RowArgMax((*rows)[i]), want[i]);
  }
  tessera::core::DecodeCache cache;
  auto one = tessera::core::DecodeLogits(engine->Owner(), **model, cache, 0);
  ASSERT_TRUE(one.has_value()) << tessera::ToString(one.error());
  EXPECT_EQ(*one, (*rows)[0]);
  EXPECT_EQ(RowArgMax(*one), 1u);

  auto empty = tessera::core::ScoreTokens(
      engine->Owner(), **model, std::span<const std::uint32_t>{});
  ASSERT_TRUE(empty.has_value());
  EXPECT_TRUE(empty->empty());
}

// Batched scoring must equal the sequential per-token scoring on both
// the gated (full attention) and the stateful linear hybrid fixtures.
TEST(HybridDecodeTest, BatchedLogitsMatchSequential) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  const std::vector<std::string> fixtures = {
      WriteGatedHybridFixture("batch-gated.gguf").string(),
      WriteLinearHybridFixture("batch-linear.gguf").string()};
  const std::vector<std::uint32_t> tokens = {0, 1, 0, 26};
  for (const std::string& path : fixtures) {
    auto model = engine->LoadModel(ModelOptions{path, 1024});
    ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
    tessera::core::DecodeCache batched_cache;
    auto batched = tessera::core::DecodeLogitsBatch(
        engine->Owner(), **model, batched_cache, tokens);
    ASSERT_TRUE(batched.has_value()) << tessera::ToString(batched.error());
    ASSERT_EQ(batched->size(), tokens.size());
    tessera::core::DecodeCache seq_cache;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      auto one = tessera::core::DecodeLogits(engine->Owner(), **model,
                                             seq_cache, tokens[i]);
      ASSERT_TRUE(one.has_value()) << tessera::ToString(one.error());
      ASSERT_EQ((*batched)[i].size(), one->size());
      float max_abs = 0.0f;
      for (std::size_t j = 0; j < one->size(); ++j) {
        max_abs = std::max(max_abs, std::abs((*batched)[i][j] - (*one)[j]));
      }
      EXPECT_LE(max_abs, 1e-3f)
          << "fixture " << path << " row " << i << " max_abs " << max_abs;
    }
  }
}

// Batched prefill must equal the sequential per-token prefill: the last
// logits and the retained hidden match, on both fixtures.
TEST(HybridDecodeTest, BatchedPrefillMatchesSequential) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  const std::vector<std::string> fixtures = {
      WriteGatedHybridFixture("prefill-gated.gguf").string(),
      WriteLinearHybridFixture("prefill-linear.gguf").string()};
  const std::vector<std::uint32_t> tokens = {0, 1, 0, 26, 5};
  for (const std::string& path : fixtures) {
    auto model = engine->LoadModel(ModelOptions{path, 1024});
    ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
    tessera::core::DecodeCache batched;
    std::vector<float> batched_hidden;
    auto fast = tessera::core::PrefillTokens(engine->Owner(), **model, batched,
                                             tokens, &batched_hidden);
    ASSERT_TRUE(fast.has_value()) << tessera::ToString(fast.error());
    tessera::core::DecodeCache seq;
    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
      ASSERT_TRUE(tessera::core::DecodeForward(engine->Owner(), **model, seq,
                                               tokens[i])
                      .has_value());
    }
    std::vector<float> seq_hidden;
    auto slow = tessera::core::DecodeLogits(engine->Owner(), **model, seq,
                                            tokens.back(), &seq_hidden);
    ASSERT_TRUE(slow.has_value()) << tessera::ToString(slow.error());
    // The batched path projects through the tiled GEMM and the sequential
    // path through the coalesced GEMV, whose reductions associate the fp
    // sums differently, so the two agree to fp reassociation rather than
    // bit for bit.
    constexpr float kPrefillConsistency = 1e-2f;
    ASSERT_EQ(fast->size(), slow->size());
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < slow->size(); ++i) {
      max_abs = std::max(max_abs, std::abs((*fast)[i] - (*slow)[i]));
    }
    EXPECT_LE(max_abs, kPrefillConsistency)
        << path << " logits max_abs " << max_abs;
    ASSERT_EQ(batched_hidden.size(), seq_hidden.size());
    float hidden_abs = 0.0f;
    for (std::size_t i = 0; i < seq_hidden.size(); ++i) {
      hidden_abs =
          std::max(hidden_abs, std::abs(batched_hidden[i] - seq_hidden[i]));
    }
    EXPECT_LE(hidden_abs, kPrefillConsistency)
        << path << " hidden max_abs " << hidden_abs;
  }
}

// Chunked prefill must equal one forward: the same last logits and
// hidden, and the same follow-up decode (the cache state carries
// across chunk boundaries). A 220k-token prompt on the 8-bit KV path
// prefills in bounded memory only through chunking.
TEST(HybridDecodeTest, ChunkedPrefillMatchesSingleForward) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  const std::vector<std::string> fixtures = {
      WriteGatedHybridFixture("chunk-gated.gguf").string(),
      WriteLinearHybridFixture("chunk-linear.gguf").string()};
  const std::vector<std::uint32_t> tokens = {0, 1, 0, 26, 5, 3, 7, 2};
  for (const std::string& path : fixtures) {
    auto model = engine->LoadModel(ModelOptions{path, 1024});
    ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
    tessera::core::DecodeCache single;
    std::vector<float> single_hidden;
    auto single_logits = tessera::core::PrefillTokens(engine->Owner(), **model,
                                                      single, tokens,
                                                      &single_hidden);
    ASSERT_TRUE(single_logits.has_value())
        << tessera::ToString(single_logits.error());
    tessera::core::DecodeCache chunked;
    std::vector<float> chunked_hidden;
    auto chunked_logits = tessera::core::PrefillTokens(
        engine->Owner(), **model, chunked, tokens, &chunked_hidden,
        nullptr, /*chunk_tokens=*/3);
    ASSERT_TRUE(chunked_logits.has_value())
        << tessera::ToString(chunked_logits.error());
    // Chunked forwards reuse the same kernels per chunk, so they agree
    // with one forward far more closely than the sequential path does;
    // keep the same bound as the batched-vs-sequential test.
    constexpr float kChunkConsistency = 1e-2f;
    ASSERT_EQ(chunked_logits->size(), single_logits->size());
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < single_logits->size(); ++i) {
      max_abs = std::max(max_abs,
                         std::abs((*chunked_logits)[i] - (*single_logits)[i]));
    }
    EXPECT_LE(max_abs, kChunkConsistency)
        << path << " logits max_abs " << max_abs;
    ASSERT_EQ(chunked_hidden.size(), single_hidden.size());
    float hidden_abs = 0.0f;
    for (std::size_t i = 0; i < single_hidden.size(); ++i) {
      hidden_abs = std::max(hidden_abs,
                            std::abs(chunked_hidden[i] - single_hidden[i]));
    }
    EXPECT_LE(hidden_abs, kChunkConsistency)
        << path << " hidden max_abs " << hidden_abs;
    // The caches must agree past the boundary: one more decode step
    // from each cache scores the same distribution.
    auto next_single =
        tessera::core::DecodeLogits(engine->Owner(), **model, single, 4);
    auto next_chunked =
        tessera::core::DecodeLogits(engine->Owner(), **model, chunked, 4);
    ASSERT_TRUE(next_single.has_value())
        << tessera::ToString(next_single.error());
    ASSERT_TRUE(next_chunked.has_value())
        << tessera::ToString(next_chunked.error());
    ASSERT_EQ(next_chunked->size(), next_single->size());
    float next_abs = 0.0f;
    for (std::size_t i = 0; i < next_single->size(); ++i) {
      next_abs = std::max(next_abs,
                          std::abs((*next_chunked)[i] - (*next_single)[i]));
    }
    EXPECT_LE(next_abs, kChunkConsistency)
        << path << " follow-up max_abs " << next_abs;
  }
}
// logits and the per-layer linear state snapshots by need, not by every
// prompt row. Prefill scores one row and never rolls back; a following
// verification then grows the state history and still accepts the draft.
// A 112-token prefill used to ask for tens of GB and fail with
// out_of_memory.
TEST(HybridDecodeTest, BatchedPrefillThenVerifySizesScratchByNeed) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  const std::vector<std::uint32_t> prompt = {0, 1, 0, 1, 0, 1, 0, 1};
  const std::vector<std::string> fixtures = {
      WriteGatedHybridFixture("need-gated.gguf").string(),
      WriteLinearHybridFixture("need-linear.gguf").string()};
  for (const std::string& path : fixtures) {
    auto model = engine->LoadModel(ModelOptions{path, 1024});
    ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
    // Sequential reference: the greedy draft after the prompt.
    tessera::core::DecodeCache ref;
    for (std::size_t i = 0; i + 1 < prompt.size(); ++i) {
      ASSERT_TRUE(tessera::core::DecodeForward(engine->Owner(), **model, ref,
                                               prompt[i])
                      .has_value());
    }
    auto ref_logits = tessera::core::DecodeLogits(engine->Owner(), **model, ref,
                                                  prompt.back());
    ASSERT_TRUE(ref_logits.has_value()) << tessera::ToString(ref_logits.error());
    std::vector<std::uint32_t> draft;
    std::vector<float> lg = *ref_logits;
    for (int i = 0; i < 3; ++i) {
      draft.push_back(RowArgMax(lg));
      auto nxt = tessera::core::DecodeLogits(engine->Owner(), **model, ref,
                                             draft.back());
      ASSERT_TRUE(nxt.has_value());
      lg = *nxt;
    }
    // Batched prefill of the whole prompt.
    tessera::core::DecodeCache cache;
    auto prefill =
        tessera::core::PrefillTokens(engine->Owner(), **model, cache, prompt);
    ASSERT_TRUE(prefill.has_value()) << tessera::ToString(prefill.error());
    {
      auto& s = tessera::models::qwen3_5::State(cache);
      ASSERT_NE(s.batch, nullptr) << path;
      EXPECT_EQ(s.batch->logits_capacity, 1u) << path;
      EXPECT_FALSE(s.batch->snapshot_states) << path;
      EXPECT_TRUE(s.batch->state_hist.empty()) << path;
    }
    auto v = tessera::core::VerifyDraft(engine->Owner(), **model, cache, draft,
                                        *prefill);
    ASSERT_TRUE(v.has_value()) << tessera::ToString(v.error());
    EXPECT_EQ(v->accepted, draft.size()) << path;
    auto& s = tessera::models::qwen3_5::State(cache);
    ASSERT_NE(s.batch, nullptr) << path;
    EXPECT_EQ(s.batch->logits_capacity, draft.size()) << path;
    EXPECT_TRUE(s.batch->snapshot_states) << path;
    EXPECT_FALSE(s.batch->state_hist.empty()) << path;
  }
}

// Batched prefill honours a caller-supplied embedding per row (the
// multimodal image path): it must equal a sequential prefill that feeds
// the same embeddings one row at a time.
TEST(HybridDecodeTest, BatchedPrefillEmbeddingsMatchSequential) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(
      ModelOptions{WriteGatedHybridFixture("emb-gated.gguf").string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto cfg = (*model)->Config();
  ASSERT_TRUE(cfg.has_value());
  const std::size_t hidden = cfg->hidden_dim;
  const std::vector<std::uint32_t> tokens = {0, 1, 0, 26, 5};
  const std::size_t rows = tokens.size();
  std::vector<float> emb(rows * hidden);
  for (std::size_t i = 0; i < emb.size(); ++i) {
    emb[i] = 0.0007f * static_cast<float>(i) - 0.5f;
  }
  auto emb_buf =
      engine->Owner().AllocateBuffer(rows * hidden * 4, tessera::MemoryKind::Device);
  ASSERT_TRUE(emb_buf.has_value());
  ASSERT_TRUE(engine->Owner()
                  .CopyH2D(**emb_buf,
                           std::span<const std::byte>(
                               reinterpret_cast<const std::byte*>(emb.data()),
                               emb.size() * 4))
                  .has_value());
  tessera::core::DecodeCache batched;
  auto fast = tessera::core::PrefillTokens(engine->Owner(), **model, batched,
                                           tokens, nullptr, emb_buf->get());
  ASSERT_TRUE(fast.has_value()) << tessera::ToString(fast.error());
  tessera::core::DecodeCache seq;
  auto row_buf =
      engine->Owner().AllocateBuffer(hidden * 4, tessera::MemoryKind::Device);
  ASSERT_TRUE(row_buf.has_value());
  for (std::size_t i = 0; i + 1 < rows; ++i) {
    ASSERT_TRUE(engine->Owner()
                    .CopyD2D(**emb_buf, i * hidden * 4, **row_buf, 0,
                             hidden * 4)
                    .has_value());
    ASSERT_TRUE(tessera::core::DecodeForward(engine->Owner(), **model, seq,
                                             tokens[i], nullptr,
                                             row_buf->get())
                    .has_value());
  }
  ASSERT_TRUE(engine->Owner()
                  .CopyD2D(**emb_buf, (rows - 1) * hidden * 4, **row_buf, 0,
                           hidden * 4)
                  .has_value());
  auto slow = tessera::core::DecodeLogits(engine->Owner(), **model, seq,
                                          tokens.back(), nullptr, nullptr,
                                          nullptr, row_buf->get());
  ASSERT_TRUE(slow.has_value()) << tessera::ToString(slow.error());
  ASSERT_EQ(fast->size(), slow->size());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < slow->size(); ++i) {
    max_abs = std::max(max_abs, std::abs((*fast)[i] - (*slow)[i]));
  }
  EXPECT_LE(max_abs, 1e-3f) << "logits max_abs " << max_abs;
}

// Chunked prefill with caller embeddings must equal one forward: the
// staging slice per chunk feeds the same rows.
TEST(HybridDecodeTest, ChunkedPrefillEmbeddingsMatchSingle) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(
      ModelOptions{WriteGatedHybridFixture("emb-chunk-gated.gguf").string(),
                   1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto cfg = (*model)->Config();
  ASSERT_TRUE(cfg.has_value());
  const std::size_t hidden = cfg->hidden_dim;
  const std::vector<std::uint32_t> tokens = {0, 1, 0, 26, 5};
  const std::size_t rows = tokens.size();
  std::vector<float> emb(rows * hidden);
  for (std::size_t i = 0; i < emb.size(); ++i) {
    emb[i] = 0.0007f * static_cast<float>(i) - 0.5f;
  }
  auto emb_buf =
      engine->Owner().AllocateBuffer(rows * hidden * 4, tessera::MemoryKind::Device);
  ASSERT_TRUE(emb_buf.has_value());
  ASSERT_TRUE(engine->Owner()
                  .CopyH2D(**emb_buf,
                           std::span<const std::byte>(
                               reinterpret_cast<const std::byte*>(emb.data()),
                               emb.size() * 4))
                  .has_value());
  tessera::core::DecodeCache single;
  auto want = tessera::core::PrefillTokens(engine->Owner(), **model, single,
                                           tokens, nullptr, emb_buf->get());
  ASSERT_TRUE(want.has_value()) << tessera::ToString(want.error());
  tessera::core::DecodeCache chunked;
  auto got = tessera::core::PrefillTokens(engine->Owner(), **model, chunked,
                                          tokens, nullptr, emb_buf->get(),
                                          /*chunk_tokens=*/3);
  ASSERT_TRUE(got.has_value()) << tessera::ToString(got.error());
  ASSERT_EQ(got->size(), want->size());
  float max_abs = 0.0f;
  for (std::size_t i = 0; i < want->size(); ++i) {
    max_abs = std::max(max_abs, std::abs((*got)[i] - (*want)[i]));
  }
  EXPECT_LE(max_abs, 1e-3f) << "logits max_abs " << max_abs;
}

// Batched verification must be output preserving: for any draft the
// accepted tokens plus the bonus token equal the plain greedy sequence,
// and the rolled-back cache continues greedily from the accepted prefix.
TEST(HybridDecodeTest, BatchedVerifyPreservesGreedy) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  const std::vector<std::string> fixtures = {
      WriteGatedHybridFixture("verify-gated.gguf").string(),
      WriteLinearHybridFixture("verify-linear.gguf").string()};
  const std::vector<std::uint32_t> prefix = {0, 1, 0, 26};
  for (const std::string& path : fixtures) {
    auto model = engine->LoadModel(ModelOptions{path, 1024});
    ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
    auto cfg = (*model)->Config();
    ASSERT_TRUE(cfg.has_value());
    const std::uint32_t vocab = static_cast<std::uint32_t>(cfg->vocab_size);
    const auto prime = [&](tessera::core::DecodeCache& cache,
                           std::vector<float>& logits) {
      for (std::size_t i = 0; i + 1 < prefix.size(); ++i) {
        EXPECT_TRUE(tessera::core::DecodeLogits(engine->Owner(), **model, cache,
                                                prefix[i])
                        .has_value());
      }
      auto pl = tessera::core::DecodeLogits(engine->Owner(), **model, cache,
                                            prefix.back());
      EXPECT_TRUE(pl.has_value());
      logits = *pl;
    };
    // Greedy continuation reference.
    std::vector<float> pl;
    tessera::core::DecodeCache ref_cache;
    prime(ref_cache, pl);
    std::vector<std::uint32_t> greedy;
    {
      std::vector<float> lg = pl;
      for (int i = 0; i < 6; ++i) {
        const std::uint32_t t = RowArgMax(lg);
        greedy.push_back(t);
        auto nxt =
            tessera::core::DecodeLogits(engine->Owner(), **model, ref_cache, t);
        EXPECT_TRUE(nxt.has_value());
        lg = *nxt;
      }
    }
    const std::size_t kDraft = 4;
    for (int corrupt = -1; corrupt < static_cast<int>(kDraft); ++corrupt) {
      std::vector<std::uint32_t> draft(greedy.begin(),
                                       greedy.begin() + kDraft);
      if (corrupt >= 0) {
        draft[corrupt] = (draft[corrupt] + 1) % vocab;
      }
      tessera::core::DecodeCache cache;
      std::vector<float> logits;
      prime(cache, logits);
      auto v = tessera::core::VerifyDraft(engine->Owner(), **model, cache,
                                          draft, logits);
      ASSERT_TRUE(v.has_value()) << tessera::ToString(v.error());
      const std::size_t want_accepted =
          corrupt < 0 ? kDraft : static_cast<std::size_t>(corrupt);
      EXPECT_EQ(v->accepted, want_accepted) << path << " corrupt " << corrupt;
      EXPECT_EQ(v->next_token, greedy[want_accepted]);
      // The cache continues greedily from the accepted prefix.
      std::vector<float> lg = v->logits;
      EXPECT_EQ(RowArgMax(lg), greedy[want_accepted]);
      auto nxt = tessera::core::DecodeLogits(engine->Owner(), **model, cache,
                                             v->next_token);
      ASSERT_TRUE(nxt.has_value());
      EXPECT_EQ(RowArgMax(*nxt), greedy[want_accepted + 1]);
    }
  }
}

// Greedy speculative decoding must be output preserving: for any draft,
// accepted tokens plus the bonus token equal the plain greedy sequence.
// A fixed wrong draft exercises the all-rejected path; a draft that
// starts with the correct token exercises acceptance. Covers the
// full-attention and the stateful linear fixtures.
TEST(HybridDecodeTest, SpeculationMatchesGreedy) {
  const std::string gated = WriteGatedHybridFixture("gated.gguf").string();
  const std::vector<std::uint32_t> want_gated = {1, 0, 26, 18, 17, 3, 3, 3};
  EXPECT_EQ(SpeculativeTokens(gated, 8, 31), want_gated);
  EXPECT_EQ(SpeculativeTokens(gated, 8, 1), want_gated);
  const std::string linear = WriteLinearHybridFixture("linear.gguf").string();
  const std::vector<std::uint32_t> want_linear = {21, 28, 24, 6, 9, 15, 18, 24};
  EXPECT_EQ(SpeculativeTokens(linear, 8, 31), want_linear);
  EXPECT_EQ(SpeculativeTokens(linear, 8, 5), want_linear);
}

// GenerateSpeculative must agree with Generate. On the tiny hybrids there
// is no MTP head, so the drafter is unavailable and every step falls back
// to the target's greedy token; this exercises the loop and the fallback.
TEST(HybridDecodeTest, SpeculativeFallbackMatchesGreedy) {
  for (const std::string path :
       {WriteGatedHybridFixture("gated.gguf").string(),
        WriteLinearHybridFixture("linear.gguf").string()}) {
    std::unique_ptr<Engine> engine;
    MakeEngineOrSkip(engine);
    auto model = engine->LoadModel(ModelOptions{path, 1024});
    ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
    tessera::GenerateOptions options;
    options.max_tokens = 6;
    options.first_token = 0;
    auto greedy = engine->Generate(**model, options);
    ASSERT_TRUE(greedy.has_value()) << tessera::ToString(greedy.error());
    auto spec = engine->GenerateSpeculative(**model, options);
    ASSERT_TRUE(spec.has_value()) << tessera::ToString(spec.error());
    EXPECT_EQ(*spec, *greedy);
  }
}

// With the real MTP head the drafter runs and is verified; speculation
// must still reproduce the greedy sequence (27B target; TESSERA_TEST_MODEL).
TEST(HybridDecodeTest, SpeculativeMatchesGreedyOnModel) {
  const char* path = std::getenv("TESSERA_TEST_MODEL");
  if (path == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_MODEL not set";
  }
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{path, 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  tessera::GenerateOptions options;
  options.max_tokens = 4;
  if (const char* count = std::getenv("TESSERA_SPEC_TOKENS");
      count != nullptr) {
    const int value = std::atoi(count);
    if (value > 0) {
      options.max_tokens = static_cast<std::size_t>(value);
    }
  }
  options.first_token = 0;
  auto greedy = engine->Generate(**model, options);
  ASSERT_TRUE(greedy.has_value()) << tessera::ToString(greedy.error());
  auto spec = engine->GenerateSpeculative(**model, options);
  ASSERT_TRUE(spec.has_value()) << tessera::ToString(spec.error());
  EXPECT_EQ(*spec, *greedy);
}

// The forward-only step leaves the cache in the same state as a full step,
// so the next token's logits match.
TEST(HybridDecodeTest, ForwardMatchesStepState) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(
      ModelOptions{WriteGatedHybridFixture("gated.gguf").string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  tessera::core::DecodeCache full;
  auto first = tessera::core::DecodeStep(engine->Owner(), **model, full, 0);
  ASSERT_TRUE(first.has_value()) << tessera::ToString(first.error());
  tessera::core::DecodeCache fwd;
  auto forward = tessera::core::DecodeForward(engine->Owner(), **model, fwd, 0);
  ASSERT_TRUE(forward.has_value()) << tessera::ToString(forward.error());
  auto from_full =
      tessera::core::DecodeLogits(engine->Owner(), **model, full, *first);
  auto from_fwd =
      tessera::core::DecodeLogits(engine->Owner(), **model, fwd, *first);
  ASSERT_TRUE(from_full.has_value() && from_fwd.has_value());
  EXPECT_EQ(*from_full, *from_fwd);
}

// The MTP head drafts a token from the backbone hidden state (27B target;
// path via TESSERA_TEST_MODEL). Deterministic across fresh caches.
TEST(HybridDecodeTest, MtpDraftWhenModelProvided) {
  const char* path = std::getenv("TESSERA_TEST_MODEL");
  if (path == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_MODEL not set";
  }
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{path, 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto cfg = (*model)->Config();
  ASSERT_TRUE(cfg.has_value()) << tessera::ToString(cfg.error());
  tessera::core::DecodeCache cache;
  std::vector<float> hidden;
  auto step =
      tessera::core::DecodeStep(engine->Owner(), **model, cache, 0, &hidden);
  ASSERT_TRUE(step.has_value()) << tessera::ToString(step.error());
  ASSERT_EQ(hidden.size(), cfg->hidden_dim);
  auto draft = tessera::core::MtpDraftStep(engine->Owner(), **model, cache,
                                           hidden, *step, 1);
  ASSERT_TRUE(draft.has_value()) << tessera::ToString(draft.error());
  EXPECT_LT(*draft, cfg->vocab_size);
  tessera::core::DecodeCache replay;
  std::vector<float> hidden2;
  auto step2 =
      tessera::core::DecodeStep(engine->Owner(), **model, replay, 0, &hidden2);
  ASSERT_TRUE(step2.has_value()) << tessera::ToString(step2.error());
  auto draft2 = tessera::core::MtpDraftStep(engine->Owner(), **model, replay,
                                            hidden2, *step2, 1);
  ASSERT_TRUE(draft2.has_value()) << tessera::ToString(draft2.error());
  EXPECT_EQ(*draft2, *draft);
}

// The forward can copy the residual-stream hidden after selected layers
// (the DFlash2 target hidden states). On the one-layer gated fixture the
// captured layer-0 hidden equals the final hidden.
TEST(HybridDecodeTest, ForwardCapturesLayerHidden) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(
      ModelOptions{WriteGatedHybridFixture("gated.gguf").string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto cfg = (*model)->Config();
  ASSERT_TRUE(cfg.has_value()) << tessera::ToString(cfg.error());
  const std::size_t hidden = cfg->hidden_dim;
  auto buffer = engine->Owner().AllocateBuffer(hidden * 4,
                                               tessera::MemoryKind::Device);
  ASSERT_TRUE(buffer.has_value());
  std::vector<std::size_t> layers = {0};
  std::vector<tessera::Buffer*> captures = {buffer->get()};
  std::vector<float> hidden_out;
  tessera::core::DecodeCache cache;
  auto forward = tessera::core::DecodeLogits(
      engine->Owner(), **model, cache, 0, &hidden_out, &layers, &captures);
  ASSERT_TRUE(forward.has_value()) << tessera::ToString(forward.error());
  engine->Owner().Synchronize();
  std::vector<float> captured(hidden);
  engine->Owner().CopyD2H(**buffer,
                          reinterpret_cast<std::byte*>(captured.data()),
                          captured.size() * 4);
  EXPECT_EQ(captured, hidden_out);
}

// The 1+N draft queries put the anchor row first and tile the mask rows
// after it; an out-of-range anchor is rejected.
TEST(HybridDecodeTest, QueryEmbeddingsAnchorThenMask) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(
      ModelOptions{WriteGatedHybridFixture("gated.gguf").string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto cfg = (*model)->Config();
  ASSERT_TRUE(cfg.has_value()) << tessera::ToString(cfg.error());
  const std::size_t hidden = cfg->hidden_dim;
  const tessera::DeviceTensor* embed = nullptr;
  for (const auto& weight : (*model)->Weights()) {
    if (weight.manifest.name == "token_embd.weight") {
      embed = &weight;
    }
  }
  ASSERT_NE(embed, nullptr);
  constexpr std::size_t kMasks = 2;
  auto buffer = tessera::spec::QueryEmbeddings(engine->Owner(), *embed, 1, 0,
                                               kMasks, hidden);
  ASSERT_TRUE(buffer.has_value()) << tessera::ToString(buffer.error());
  engine->Owner().Synchronize();
  std::vector<float> queries((1 + kMasks) * hidden);
  engine->Owner().CopyD2H(**buffer,
                          reinterpret_cast<std::byte*>(queries.data()),
                          queries.size() * 4);
  std::vector<float> anchor(hidden);
  std::vector<float> mask(hidden);
  ASSERT_TRUE(tessera::core::detail::GatherEmbedding(engine->Owner(), *embed, 1,
                                                     hidden, anchor)
                  .has_value());
  ASSERT_TRUE(tessera::core::detail::GatherEmbedding(engine->Owner(), *embed, 0,
                                                     hidden, mask)
                  .has_value());
  for (std::size_t c = 0; c < hidden; ++c) {
    EXPECT_EQ(queries[c], anchor[c]);
  }
  for (std::size_t r = 1; r <= kMasks; ++r) {
    for (std::size_t c = 0; c < hidden; ++c) {
      EXPECT_EQ(queries[r * hidden + c], mask[c]);
    }
  }
  auto bad = tessera::spec::QueryEmbeddings(engine->Owner(), *embed, 999, 0,
                                            kMasks, hidden);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), tessera::StatusCode::InvalidArgument);
}

// The fp16 KV cache path decodes deterministically (a fresh cache with the
// same option replays the first step).
TEST(HybridDecodeTest, Fp16KvDecodesDeterministically) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(
      ModelOptions{WriteGatedHybridFixture("gated.gguf").string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  tessera::core::DecodeCache cache;
  cache.kv_type = tessera::KvCacheType::F16;
  auto first = tessera::core::DecodeStep(engine->Owner(), **model, cache, 0);
  ASSERT_TRUE(first.has_value()) << tessera::ToString(first.error());
  auto second =
      tessera::core::DecodeStep(engine->Owner(), **model, cache, *first);
  ASSERT_TRUE(second.has_value()) << tessera::ToString(second.error());
  tessera::core::DecodeCache replay;
  replay.kv_type = tessera::KvCacheType::F16;
  auto again = tessera::core::DecodeStep(engine->Owner(), **model, replay, 0);
  ASSERT_TRUE(again.has_value()) << tessera::ToString(again.error());
  EXPECT_EQ(*again, *first);
}

// The q8 KV cache path decodes deterministically.
TEST(HybridDecodeTest, Q8KvDecodesDeterministically) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(
      ModelOptions{WriteGatedHybridFixture("gated.gguf").string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  tessera::core::DecodeCache cache;
  cache.kv_type = tessera::KvCacheType::Q8;
  auto first = tessera::core::DecodeStep(engine->Owner(), **model, cache, 0);
  ASSERT_TRUE(first.has_value()) << tessera::ToString(first.error());
  auto second =
      tessera::core::DecodeStep(engine->Owner(), **model, cache, *first);
  ASSERT_TRUE(second.has_value()) << tessera::ToString(second.error());
  tessera::core::DecodeCache replay;
  replay.kv_type = tessera::KvCacheType::Q8;
  auto again = tessera::core::DecodeStep(engine->Owner(), **model, replay, 0);
  ASSERT_TRUE(again.has_value()) << tessera::ToString(again.error());
  EXPECT_EQ(*again, *first);
}

// The q4 KV cache path decodes deterministically.
TEST(HybridDecodeTest, Q4KvDecodesDeterministically) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(
      ModelOptions{WriteGatedHybridFixture("gated.gguf").string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  tessera::core::DecodeCache cache;
  cache.kv_type = tessera::KvCacheType::Q4;
  auto first = tessera::core::DecodeStep(engine->Owner(), **model, cache, 0);
  ASSERT_TRUE(first.has_value()) << tessera::ToString(first.error());
  auto second =
      tessera::core::DecodeStep(engine->Owner(), **model, cache, *first);
  ASSERT_TRUE(second.has_value()) << tessera::ToString(second.error());
  tessera::core::DecodeCache replay;
  replay.kv_type = tessera::KvCacheType::Q4;
  auto again = tessera::core::DecodeStep(engine->Owner(), **model, replay, 0);
  ASSERT_TRUE(again.has_value()) << tessera::ToString(again.error());
  EXPECT_EQ(*again, *first);
}

// The fp8 KV cache path decodes deterministically.
TEST(HybridDecodeTest, Fp8KvDecodesDeterministically) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(
      ModelOptions{WriteGatedHybridFixture("gated.gguf").string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  tessera::core::DecodeCache cache;
  cache.kv_type = tessera::KvCacheType::FP8;
  auto first = tessera::core::DecodeStep(engine->Owner(), **model, cache, 0);
  ASSERT_TRUE(first.has_value()) << tessera::ToString(first.error());
  auto second =
      tessera::core::DecodeStep(engine->Owner(), **model, cache, *first);
  ASSERT_TRUE(second.has_value()) << tessera::ToString(second.error());
  tessera::core::DecodeCache replay;
  replay.kv_type = tessera::KvCacheType::FP8;
  auto again = tessera::core::DecodeStep(engine->Owner(), **model, replay, 0);
  ASSERT_TRUE(again.has_value()) << tessera::ToString(again.error());
  EXPECT_EQ(*again, *first);
}

// A precomputed embedding buffer (an image token) is equivalent to the
// gathered embedding of the same token.
TEST(HybridDecodeTest, FeedEmbeddingMatchesToken) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(
      ModelOptions{WriteGatedHybridFixture("gated.gguf").string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto cfg = (*model)->Config();
  ASSERT_TRUE(cfg.has_value()) << tessera::ToString(cfg.error());
  const std::size_t hidden = cfg->hidden_dim;
  const tessera::DeviceTensor* embed = nullptr;
  for (const auto& weight : (*model)->Weights()) {
    if (weight.manifest.name == "token_embd.weight") {
      embed = &weight;
    }
  }
  ASSERT_NE(embed, nullptr);
  std::vector<float> row(hidden);
  ASSERT_TRUE(tessera::core::detail::GatherEmbedding(engine->Owner(), *embed, 0,
                                                     hidden, row)
                  .has_value());
  auto buffer = engine->Owner().AllocateBuffer(hidden * 4,
                                               tessera::MemoryKind::Device);
  ASSERT_TRUE(buffer.has_value());
  engine->Owner().CopyH2D(**buffer, std::span<const std::byte>(
                                       reinterpret_cast<const std::byte*>(
                                           row.data()),
                                       row.size() * 4));

  tessera::core::DecodeCache token_cache;
  auto a = tessera::core::DecodeForward(engine->Owner(), **model, token_cache,
                                        0);
  ASSERT_TRUE(a.has_value()) << tessera::ToString(a.error());
  auto logits_a = tessera::core::DecodeLogits(engine->Owner(), **model,
                                              token_cache, 1);
  ASSERT_TRUE(logits_a.has_value()) << tessera::ToString(logits_a.error());

  tessera::core::DecodeCache embed_cache;
  auto b = tessera::core::DecodeForward(engine->Owner(), **model, embed_cache,
                                        0, nullptr, buffer->get());
  ASSERT_TRUE(b.has_value()) << tessera::ToString(b.error());
  auto logits_b = tessera::core::DecodeLogits(engine->Owner(), **model,
                                              embed_cache, 1);
  ASSERT_TRUE(logits_b.has_value()) << tessera::ToString(logits_b.error());
  EXPECT_EQ(*logits_a, *logits_b);
}

// Multimodal generation feeds image embeddings at the placeholder tokens and
// is deterministic.
TEST(HybridDecodeTest, MultimodalGenerate) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(
      ModelOptions{WriteGatedHybridFixture("gated.gguf").string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto cfg = (*model)->Config();
  ASSERT_TRUE(cfg.has_value()) << tessera::ToString(cfg.error());
  const std::size_t hidden = cfg->hidden_dim;
  std::mt19937 rng(120);
  std::vector<float> embeddings(2 * hidden);
  for (auto& v : embeddings) v = tessera::testing::DrawValue(rng) * 0.1f;
  tessera::GenerateOptions options;
  options.max_tokens = 4;
  options.prompt_tokens = {0, 0, 5, 7};
  auto first = engine->GenerateMultimodal(**model, options, embeddings, 2, 0);
  ASSERT_TRUE(first.has_value()) << tessera::ToString(first.error());
  EXPECT_EQ(first->size(), 4u);
  auto again = engine->GenerateMultimodal(**model, options, embeddings, 2, 0);
  ASSERT_TRUE(again.has_value()) << tessera::ToString(again.error());
  EXPECT_EQ(*first, *again);
}
