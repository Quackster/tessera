#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>

#include <memory>

#include "test_helpers.hpp"
#include "spec/dflash2_config.hpp"
#include "spec/dflash2_weights.hpp"
#include "spec/dflash2_drafter.hpp"
#include "tessera/engine.hpp"
#include "tessera/speculative.hpp"
#include "tessera/types.hpp"

using tessera::spec::DFlash2Config;
using tessera::spec::LoadDFlash2Config;
using tessera::spec::ParseDFlash2Config;
using tessera::CreateDFlash2Strategy;
using tessera::StrategyOptions;
using tessera::StatusCode;
using tessera::testing::FreshTempDir;
using tessera::testing::WritePlaceholderConfig;
using tessera::testing::WritePlaceholderWeights;

namespace {

// A valid draft checkpoint: config.json + one weight file.
std::filesystem::path MakeDraftDir() {
  auto dir = FreshTempDir("tessera_tests_spec_draft");
  WritePlaceholderConfig(dir);
  WritePlaceholderWeights(dir);
  return dir;
}

}  // namespace

TEST(SpecTest, Name) {
  auto strategy = CreateDFlash2Strategy();
  EXPECT_EQ(strategy->Name(), "dflash2");
}

TEST(SpecTest, AttachEmptyPath) {
  auto strategy = CreateDFlash2Strategy();
  auto result = strategy->Attach(StrategyOptions{"", 4});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), StatusCode::InvalidArgument);
}

TEST(SpecTest, AttachBlockSize) {
  auto dir = MakeDraftDir();
  // 0 means "use the checkpoint's configured block size" and is accepted.
  auto default_block =
      CreateDFlash2Strategy()->Attach(StrategyOptions{dir.string(), 0});
  EXPECT_TRUE(default_block.has_value());
  // Out-of-range blocks are rejected.
  auto too_big =
      CreateDFlash2Strategy()->Attach(StrategyOptions{dir.string(), 9});
  ASSERT_FALSE(too_big.has_value());
  EXPECT_EQ(too_big.error(), StatusCode::InvalidArgument);
}

TEST(SpecTest, AttachMissingDir) {
  auto strategy = CreateDFlash2Strategy();
  auto result =
      strategy->Attach(StrategyOptions{"/nonexistent/draft", 4});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), StatusCode::FileNotFound);
}

TEST(SpecTest, AttachMissingConfig) {
  auto dir = FreshTempDir("tessera_tests_spec_draft");
  WritePlaceholderWeights(dir);  // weights without config.json
  auto strategy = CreateDFlash2Strategy();
  auto result = strategy->Attach(StrategyOptions{dir.string(), 4});
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), StatusCode::MalformedFile);
}

TEST(SpecTest, AttachValidIsIdempotent) {
  auto dir = MakeDraftDir();
  auto strategy = CreateDFlash2Strategy();
  auto first = strategy->Attach(StrategyOptions{dir.string(), 4});
  ASSERT_TRUE(first.has_value()) << tessera::ToString(first.error());
  auto second = strategy->Attach(StrategyOptions{dir.string(), 4});
  ASSERT_TRUE(second.has_value()) << tessera::ToString(second.error());
}

namespace {

// A minimal but complete DFlash2 draft config: hidden 5120, 5 layers of 32
// heads over 8 kv heads of dim 128, conv group 16, 2 taps, selector top-4.
constexpr const char* kDraftConfig = R"({
  "hidden_size": 5120,
  "num_hidden_layers": 5,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 17408,
  "vocab_size": 248320,
  "rms_norm_eps": 1e-06,
  "sliding_window": 2048,
  "rope_parameters": {"rope_theta": 10000000.0},
  "layer_types": ["sliding_attention", "sliding_attention",
                   "sliding_attention", "sliding_attention",
                   "sliding_attention"],
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 4,
    "target_layer_ids": [5, 19, 33, 47, 61]
  }
})";

}  // namespace

TEST(SpecConfigTest, ParsesValidConfig) {
  auto config = ParseDFlash2Config(kDraftConfig);
  ASSERT_TRUE(config.has_value()) << tessera::ToString(config.error());
  EXPECT_EQ(config->hidden_size, 5120u);
  EXPECT_EQ(config->num_layers, 5u);
  EXPECT_EQ(config->num_heads, 32u);
  EXPECT_EQ(config->num_kv_heads, 8u);
  EXPECT_EQ(config->head_dim, 128u);
  EXPECT_EQ(config->vocab_size, 248320u);
  EXPECT_EQ(config->sliding_window, 2048u);
  EXPECT_EQ(config->block_size, 8u);
  EXPECT_EQ(config->conv_group_size, 16u);
  EXPECT_EQ(config->conv_kernel_size, 2u);
  EXPECT_EQ(config->mask_token_id, 248070u);
  EXPECT_EQ(config->selector_rank, 256u);
  EXPECT_EQ(config->selector_top_k, 4u);
  EXPECT_EQ(config->target_layer_ids, (std::vector<std::uint32_t>{5, 19, 33, 47, 61}));
  EXPECT_EQ(config->layer_types.size(), 5u);
  EXPECT_DOUBLE_EQ(config->rope_theta, 10000000.0);
  // Sliding attention with no explicit is_causal resolves to causal.
  EXPECT_TRUE(config->attn_causal);
}

TEST(SpecConfigTest, ResolvesAttentionCausality) {
  // An explicit top-level is_causal wins (the Qwen3.8 DFlash2 draft case).
  std::string explicit_false = kDraftConfig;
  explicit_false.insert(explicit_false.find("\"sliding_window\""),
                        "\"is_causal\": false, ");
  auto non_causal = ParseDFlash2Config(explicit_false);
  ASSERT_TRUE(non_causal.has_value()) << tessera::ToString(non_causal.error());
  EXPECT_FALSE(non_causal->attn_causal);

  // dflash_config.causal is the fallback when is_causal is absent.
  std::string with_override = kDraftConfig;
  with_override.replace(with_override.find("\"dflash_config\": {"),
                        std::string("\"dflash_config\": {").size(),
                        "\"dflash_config\": {\"causal\": false,");
  auto override = ParseDFlash2Config(with_override);
  ASSERT_TRUE(override.has_value()) << tessera::ToString(override.error());
  EXPECT_FALSE(override->attn_causal);

  // Full attention with no explicit flag resolves to non-causal.
  std::string full = kDraftConfig;
  for (std::size_t pos = full.find("sliding_attention"); pos != std::string::npos;
       pos = full.find("sliding_attention", pos + 1)) {
    full.replace(pos, std::string("sliding_attention").size(),
                 "full_attention");
  }
  auto full_attn = ParseDFlash2Config(full);
  ASSERT_TRUE(full_attn.has_value()) << tessera::ToString(full_attn.error());
  EXPECT_FALSE(full_attn->attn_causal);
}

TEST(SpecConfigTest, RejectsMalformedConfigs) {
  // Missing a required top-level field.
  auto missing = ParseDFlash2Config(R"({"num_hidden_layers": 5})");
  ASSERT_FALSE(missing.has_value());
  EXPECT_EQ(missing.error(), StatusCode::MalformedFile);

  // Zero hidden size.
  auto zero = ParseDFlash2Config(R"({
    "hidden_size": 0, "num_hidden_layers": 5, "num_attention_heads": 32,
    "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 17408,
    "vocab_size": 248320, "rms_norm_eps": 1e-06, "sliding_window": 2048,
    "rope_parameters": {"rope_theta": 1e7},
    "layer_types": ["a", "a", "a", "a", "a"],
    "dflash_config": {"block_size": 8, "conv_group_size": 16,
      "conv_kernel_size": 2, "mask_token_id": 1, "selector_rank": 256,
      "selector_top_k": 4, "target_layer_ids": [5]}})");
  ASSERT_FALSE(zero.has_value());
  EXPECT_EQ(zero.error(), StatusCode::MalformedFile);

  // conv_group_size does not divide hidden_size.
  std::string not_divisible = kDraftConfig;
  not_divisible.replace(not_divisible.find("\"conv_group_size\": 16"),
                        std::string("\"conv_group_size\": 16").size(),
                        "\"conv_group_size\": 17");
  auto bad_group = ParseDFlash2Config(not_divisible);
  ASSERT_FALSE(bad_group.has_value());
  EXPECT_EQ(bad_group.error(), StatusCode::MalformedFile);

  // layer_types length does not match num_hidden_layers.
  std::string bad_layers = kDraftConfig;
  bad_layers.replace(bad_layers.find("\"num_hidden_layers\": 5"),
                     std::string("\"num_hidden_layers\": 5").size(),
                     "\"num_hidden_layers\": 4");
  auto layers = ParseDFlash2Config(bad_layers);
  ASSERT_FALSE(layers.has_value());
  EXPECT_EQ(layers.error(), StatusCode::MalformedFile);

  // A non-numeric target layer id.
  std::string bad_ids = kDraftConfig;
  bad_ids.replace(bad_ids.find("[5, 19, 33, 47, 61]"),
                  std::string("[5, 19, 33, 47, 61]").size(),
                  "[5, \"x\"]");
  auto ids = ParseDFlash2Config(bad_ids);
  ASSERT_FALSE(ids.has_value());
  EXPECT_EQ(ids.error(), StatusCode::MalformedFile);
}

// The real DFlash2 draft config parses when the directory is provided
// (set TESSERA_TEST_DFLASH2_DIR).
TEST(SpecConfigTest, LoadsRealConfigWhenProvided) {
  const char* dir = std::getenv("TESSERA_TEST_DFLASH2_DIR");
  if (dir == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_DFLASH2_DIR not set";
  }
  auto config = LoadDFlash2Config(dir);
  ASSERT_TRUE(config.has_value()) << tessera::ToString(config.error());
  EXPECT_EQ(config->num_layers, 5u);
  EXPECT_EQ(config->hidden_size, 5120u);
  EXPECT_EQ(config->conv_group_size, 16u);
  EXPECT_EQ(config->conv_kernel_size, 2u);
  EXPECT_EQ(config->selector_rank, 256u);
  EXPECT_EQ(config->selector_top_k, 16u);
  EXPECT_EQ(config->target_layer_ids.size(), 5u);
  EXPECT_EQ(config->layer_types.size(), 5u);
  // The Qwen3.8 DFlash2 draft is non-causal.
  EXPECT_FALSE(config->attn_causal);
}


// The real DFlash2 draft checkpoint loads and binds when the directory is
// provided (set TESSERA_TEST_DFLASH2_DIR).
TEST(SpecWeightsTest, LoadsRealDraftWhenProvided) {
  const char* dir = std::getenv("TESSERA_TEST_DFLASH2_DIR");
  if (dir == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_DFLASH2_DIR not set";
  }
  std::unique_ptr<tessera::Engine> engine;
  tessera::testing::MakeEngineOrSkip(engine);
  auto config = LoadDFlash2Config(dir);
  ASSERT_TRUE(config.has_value()) << tessera::ToString(config.error());
  auto store = tessera::spec::DraftWeightStore::Load(engine->Owner(), dir,
                                                     *config);
  ASSERT_TRUE(store.has_value()) << tessera::ToString(store.error());
  EXPECT_EQ(store->Weights().layers.size(), config->num_layers);
  EXPECT_NE(store->Weights().fc, nullptr);
  EXPECT_NE(store->Weights().hidden_norm, nullptr);
  EXPECT_NE(store->Weights().final_norm, nullptr);
  for (const auto& layer : store->Weights().layers) {
    EXPECT_NE(layer.q_w, nullptr);
    EXPECT_NE(layer.down_w, nullptr);
    EXPECT_NE(layer.hidden_norm, nullptr);
  }
}

// The real DFlash2 drafter loads and runs one draft block on the device
// when the checkpoint directory is provided (set TESSERA_TEST_DFLASH2_DIR).
// A small dummy head keeps the test light; the logits must be finite.
TEST(SpecDrafterTest, RunsRealDraftWhenProvided) {
  const char* dir = std::getenv("TESSERA_TEST_DFLASH2_DIR");
  if (dir == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_DFLASH2_DIR not set";
  }
  std::unique_ptr<tessera::Engine> engine;
  tessera::testing::MakeEngineOrSkip(engine);
  auto config = LoadDFlash2Config(dir);
  ASSERT_TRUE(config.has_value()) << tessera::ToString(config.error());
  auto drafter = tessera::spec::DFlash2Drafter::Create(engine->Owner(), dir,
                                                       *config);
  ASSERT_TRUE(drafter.has_value()) << tessera::ToString(drafter.error());
  const std::size_t rows = config->block_size;
  const std::size_t ctx = 1;
  const std::size_t n = config->target_layer_ids.size();
  const std::size_t hidden = config->hidden_size;
  const std::size_t vocab = 32;
  auto& backend = engine->Owner();
  auto make = [&](std::size_t count, float seed) {
    std::vector<float> data(count);
    for (std::size_t i = 0; i < count; ++i) {
      data[i] = std::sin(static_cast<float>(i) * seed) * 0.01f;
    }
    auto buffer = backend.AllocateBuffer(count * 4, tessera::MemoryKind::Device);
    backend.CopyH2D(**buffer, std::span<const std::byte>(
                                  reinterpret_cast<const std::byte*>(data.data()),
                                  data.size() * 4));
    return std::move(*buffer);
  };
  auto mask = make(rows * hidden, 0.1f);
  auto outw = make(vocab * hidden, 0.3f);
  auto logits = backend.AllocateBuffer(rows * vocab * 4, tessera::MemoryKind::Device);
  ASSERT_TRUE(logits.has_value());
  auto head = backend.LoadKernel("gemm_f32", {});
  ASSERT_TRUE(head.has_value());
  std::vector<std::unique_ptr<tessera::Buffer>> captures;
  std::vector<const tessera::Buffer*> capture_ptrs;
  for (std::size_t i = 0; i < n; ++i) {
    captures.push_back(make(ctx * hidden, 0.2f));
    capture_ptrs.push_back(captures.back().get());
  }
  auto appended =
      drafter->AppendContext(backend, capture_ptrs, /*row_offset=*/0, ctx);
  ASSERT_TRUE(appended.has_value()) << tessera::ToString(appended.error());
  auto status = drafter->Run(backend, *mask, *outw, **head, **logits, rows, ctx,
                             0, vocab);
  ASSERT_TRUE(status.has_value()) << tessera::ToString(status.error());
  backend.Synchronize();
  std::vector<float> got(rows * vocab);
  backend.CopyD2H(**logits, reinterpret_cast<std::byte*>(got.data()),
                  got.size() * 4);
  for (const float value : got) {
    EXPECT_TRUE(std::isfinite(value));
  }
}
