#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/decode.hpp"
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
  builder.KvString("general.architecture", "test-hybrid");
  builder.KvU32("test-hybrid.block_count", 1);
  builder.KvU32("test-hybrid.embedding_length", 256);
  builder.KvU32("test-hybrid.feed_forward_length", 256);
  builder.KvF32("test-hybrid.attention.layer_norm_rms_epsilon", 1e-5f);
  builder.KvU32("test-hybrid.attention.head_count", 8);
  builder.KvU32("test-hybrid.attention.head_count_kv", 8);
  builder.KvU32("test-hybrid.attention.key_length", 32);
  builder.KvU32("test-hybrid.attention.value_length", 32);
  builder.KvU32("test-hybrid.rope.dimension_count", 32);
  builder.KvF32("test-hybrid.rope.freq_base", 10000.0f);
  builder.PushString("test-hybrid.rope.dimension_sections");
  builder.PushU32(9);
  builder.PushU32(4);
  builder.PushU64(4);
  for (std::uint32_t s : {4u, 4u, 4u, 0u}) {
    builder.PushU32(s);
  }
  builder.KvU32("test-hybrid.ssm.conv_kernel", 1);
  builder.KvU32("test-hybrid.ssm.state_size", 1);
  builder.KvU32("test-hybrid.ssm.group_count", 1);
  builder.KvU32("test-hybrid.ssm.time_step_rank", 1);
  builder.KvU32("test-hybrid.ssm.inner_size", 1);
  builder.KvU32("test-hybrid.full_attention_interval", 1);
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
  builder.KvString("general.architecture", "test-hybrid");
  builder.KvU32("test-hybrid.block_count", 1);
  builder.KvU32("test-hybrid.embedding_length", 256);
  builder.KvU32("test-hybrid.feed_forward_length", 256);
  builder.KvF32("test-hybrid.attention.layer_norm_rms_epsilon", 1e-5f);
  builder.KvU32("test-hybrid.attention.head_count", 8);
  builder.KvU32("test-hybrid.attention.head_count_kv", 8);
  builder.KvU32("test-hybrid.attention.key_length", 32);
  builder.KvU32("test-hybrid.attention.value_length", 32);
  builder.KvU32("test-hybrid.rope.dimension_count", 32);
  builder.KvF32("test-hybrid.rope.freq_base", 10000.0f);
  builder.PushString("test-hybrid.rope.dimension_sections");
  builder.PushU32(9);
  builder.PushU32(4);
  builder.PushU64(4);
  for (std::uint32_t s : {4u, 4u, 4u, 0u}) {
    builder.PushU32(s);
  }
  builder.KvU32("test-hybrid.ssm.conv_kernel", 4);
  builder.KvU32("test-hybrid.ssm.state_size", 8);
  builder.KvU32("test-hybrid.ssm.group_count", 2);
  builder.KvU32("test-hybrid.ssm.time_step_rank", 4);
  builder.KvU32("test-hybrid.ssm.inner_size", 256);
  builder.KvU32("test-hybrid.full_attention_interval", 2);
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
  const std::vector<std::uint32_t> want = {5, 5, 31, 0, 30, 4, 19, 19};
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
  const std::vector<std::uint32_t> want_linear = {5, 5, 31, 0, 30, 4, 19, 19};
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
// must still reproduce the greedy sequence (27B target; TESSERA_TEST_GGUF).
TEST(HybridDecodeTest, SpeculativeMatchesGreedyOnModel) {
  const char* path = std::getenv("TESSERA_TEST_GGUF");
  if (path == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_GGUF not set";
  }
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{path, 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  tessera::GenerateOptions options;
  options.max_tokens = 4;
  options.first_token = 0;
  auto greedy = engine->Generate(**model, options);
  ASSERT_TRUE(greedy.has_value()) << tessera::ToString(greedy.error());
  auto spec = engine->GenerateSpeculative(**model, options);
  ASSERT_TRUE(spec.has_value()) << tessera::ToString(spec.error());
  EXPECT_EQ(*spec, *greedy);
}

// The MTP head drafts a token from the backbone hidden state (27B target;
// path via TESSERA_TEST_GGUF). Deterministic across fresh caches.
TEST(HybridDecodeTest, MtpDraftWhenModelProvided) {
  const char* path = std::getenv("TESSERA_TEST_GGUF");
  if (path == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_GGUF not set";
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
