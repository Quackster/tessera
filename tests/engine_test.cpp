#include <gtest/gtest.h>
#include <cstdio>

#include <chrono>
#include <cstdlib>
#include <cstring>

#include <optional>
#include <random>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "core/sampling.hpp"
#include "core/timing.hpp"
#include "test_helpers.hpp"
#include "tessera/architecture.hpp"
#include "tessera/engine.hpp"
#include "tessera/speculative.hpp"
#include "tessera/image.hpp"
#include "tessera/vision.hpp"
#include "tessera/types.hpp"

using tessera::Engine;
using tessera::EngineOptions;
using tessera::GenerateOptions;
using tessera::ModelFormat;
using tessera::TensorEntry;
using tessera::ModelOptions;
using tessera::StatusCode;
using tessera::TransformerConfig;
using tessera::testing::DrawValue;
using tessera::testing::FreshTempDir;
using tessera::testing::GgufBuilder;
using tessera::testing::MakeEngineOrSkip;
using tessera::testing::MakeSafetensorsContainer;
using tessera::testing::MakeValidGguf;
using tessera::testing::QuantizeRows;
using tessera::testing::WriteBytes;
using tessera::testing::WriteHybridFixture;
using tessera::testing::WritePlaceholderConfig;
using tessera::testing::WritePlaceholderWeights;

namespace {

std::filesystem::path WriteGgufFixture(const std::string& name) {
  auto dir = FreshTempDir("tessera_tests_engine");
  auto path = dir / name;
  WriteBytes(path, MakeValidGguf());
  return path;
}

// A 1-layer vanilla transformer GGUF: hidden 256, 4 heads over 2 kv
// groups of dim 64, ffn 512, vocab 32. Projections are Q4_K, vectors
// F32, tensor names follow the llama.cpp convention.
std::filesystem::path WriteTinyModelFixture(const std::string& name) {
  std::mt19937 rng(7);
  struct Spec {
    const char* tensor;
    std::uint32_t type;
    std::vector<std::uint64_t> dims;
    std::size_t rows;  // Q4_K row count (0 for F32)
    std::size_t cols;  // Q4_K row length (0 for F32)
  };
  const std::vector<Spec> specs = {
      {"token_embd.weight", 0, {256, 32}, 0, 0},
      {"blk.0.attn_norm.weight", 0, {256}, 0, 0},
      {"blk.0.attn_q.weight", 12, {256, 256}, 256, 256},
      {"blk.0.attn_k.weight", 12, {256, 128}, 128, 256},
      {"blk.0.attn_v.weight", 12, {256, 128}, 128, 256},
      {"blk.0.attn_output.weight", 12, {256, 256}, 256, 256},
      {"blk.0.ffn_norm.weight", 0, {256}, 0, 0},
      {"blk.0.ffn_gate.weight", 12, {256, 512}, 512, 256},
      {"blk.0.ffn_up.weight", 12, {256, 512}, 512, 256},
      {"blk.0.ffn_down.weight", 12, {512, 256}, 256, 512},
      {"output_norm.weight", 0, {256}, 0, 0},
      {"output.weight", 12, {256, 32}, 32, 256},
  };
  GgufBuilder builder;
  builder.Header(0x46554747, 3, specs.size(), 10);
  builder.KvString("general.name", "tiny-vanilla");
  builder.KvString("general.architecture", "test-vanilla");
  builder.KvU32("test-vanilla.block_count", 1);
  builder.KvU32("test-vanilla.embedding_length", 256);
  builder.KvU32("test-vanilla.feed_forward_length", 512);
  builder.KvU32("test-vanilla.attention.head_count", 4);
  builder.KvU32("test-vanilla.attention.head_count_kv", 2);
  builder.KvU32("test-vanilla.rope.dimension_count", 64);
  builder.KvF32("test-vanilla.rope.freq_base", 10000.0f);
  builder.KvF32("test-vanilla.attention.layer_norm_rms_epsilon", 1e-5f);
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
  auto dir = FreshTempDir("tessera_tests_tiny_model");
  auto path = dir / name;
  WriteBytes(path, builder.bytes);
  return path;
}

}  // namespace

TEST(EngineTest, CreateInitializesBackend) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  EXPECT_TRUE(engine->Owner().Name() == "vulkan" ||
              engine->Owner().Name() == "rocm");
  EXPECT_NE(engine->Owner().DeviceName().size(), 0u);
  EXPECT_EQ(engine->Speculative(), nullptr);
}

TEST(EngineTest, LoadGgufModelParsesManifest) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteGgufFixture("model.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto& loaded = *model;
  EXPECT_EQ(loaded->Format(), ModelFormat::Gguf);
  ASSERT_EQ(loaded->Tensors().size(), 1u);
  EXPECT_EQ(loaded->Tensors()[0].name, "w_a");
  EXPECT_EQ(loaded->Tensors()[0].dtype, tessera::DType::F32);
  EXPECT_EQ(loaded->MaxContextLength(), 1024u);
  EXPECT_EQ(loaded->Name(), "test-model");
  EXPECT_EQ(loaded->Path(), path.string());
}

TEST(EngineTest, LoadGgufModelUploadsWeights) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 1);
  builder.KvString("general.name", "test-model");
  builder.Tensor("w_a", 1, {4}, 0, 0);  // F32, 1x4, offset 0
  builder.PadTo(((builder.bytes.size() + 31) & ~31u));
  builder.PushF32(1.0f);
  builder.PushF32(2.0f);
  builder.PushF32(3.0f);
  builder.PushF32(4.0f);
  auto dir = FreshTempDir("tessera_tests_weights");
  auto path = dir / "weights.gguf";
  WriteBytes(path, builder.bytes);
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto& loaded = *model;
  ASSERT_EQ(loaded->Weights().size(), 1u);
  EXPECT_EQ(loaded->Weights()[0].manifest.name, "w_a");
  const auto* buffer = loaded->FindWeight("w_a");
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->Size(), 16u);
  EXPECT_EQ(loaded->FindWeight("missing"), nullptr);
  // The name index backs both lookups; a hit returns the tensor and a miss
  // returns nullptr.
  const tessera::DeviceTensor* tensor = loaded->FindDeviceTensor("w_a");
  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->manifest.name, "w_a");
  EXPECT_EQ(tensor->device.get(), buffer);
  EXPECT_EQ(loaded->FindDeviceTensor("missing"), nullptr);
  std::vector<std::byte> readback(16);
  auto download = engine->Owner().CopyD2H(*buffer, readback.data(), 16);
  ASSERT_TRUE(download.has_value())
      << tessera::ToString(download.error());
  const auto* values = reinterpret_cast<const float*>(readback.data());
  EXPECT_FLOAT_EQ(values[0], 1.0f);
  EXPECT_FLOAT_EQ(values[1], 2.0f);
  EXPECT_FLOAT_EQ(values[2], 3.0f);
  EXPECT_FLOAT_EQ(values[3], 4.0f);
}




TEST(EngineTest, TinyModelDecodesDeterministically) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("tiny.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto config = (*model)->Config();
  ASSERT_TRUE(config.has_value()) << tessera::ToString(config.error());
  EXPECT_EQ(config->layers, 1u);
  EXPECT_EQ(config->vocab_size, 32u);
  tessera::core::DecodeCache cache;
  auto first =
      tessera::core::DecodeStep(engine->Owner(), **model, cache, 0);
  ASSERT_TRUE(first.has_value()) << tessera::ToString(first.error());
  EXPECT_LT(*first, 32u);
  // A second step runs against the grown cache.
  auto second =
      tessera::core::DecodeStep(engine->Owner(), **model, cache, *first);
  ASSERT_TRUE(second.has_value()) << tessera::ToString(second.error());
  EXPECT_LT(*second, 32u);
  // A fresh cache replays the first step exactly.
  tessera::core::DecodeCache replay;
  auto again =
      tessera::core::DecodeStep(engine->Owner(), **model, replay, 0);
  ASSERT_TRUE(again.has_value()) << tessera::ToString(again.error());
  EXPECT_EQ(*again, *first);
}

TEST(EngineTest, GenerateTinyModelGreedy) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("generate.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto ids = engine->Generate(**model, GenerateOptions{4, 0});
  ASSERT_TRUE(ids.has_value()) << tessera::ToString(ids.error());
  ASSERT_EQ(ids->size(), 4u);
  auto again = engine->Generate(**model, GenerateOptions{4, 0});
  ASSERT_TRUE(again.has_value()) << tessera::ToString(again.error());
  EXPECT_EQ(*ids, *again);
}

// The model exposes the stop token ids its definition declares (the scalar
// `tokenizer.ggml.eos_token_id` and the array `tokenizer.ggml.eos_token_ids`).
TEST(EngineTest, LoadGgufModelReadsStopTokens) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 0, 3);
  builder.KvString("general.name", "stop-model");
  builder.KvU32("tokenizer.ggml.eos_token_id", 7);
  builder.KvArrayU32("tokenizer.ggml.eos_token_ids", {7, 9});
  auto dir = FreshTempDir("tessera_tests_stop_tokens");
  auto path = dir / "stops.gguf";
  WriteBytes(path, builder.bytes);
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  const std::span<const std::uint32_t> stops = (*model)->StopTokens();
  ASSERT_EQ(stops.size(), 2u);
  EXPECT_EQ(stops[0], 7u);
  EXPECT_EQ(stops[1], 9u);
}

// A generation loop ends at a stop token and does not emit it, for both the
// model's declared stops and the caller's extras.
TEST(EngineTest, GenerateStopsAtStopToken) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("stopping.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  GenerateOptions free_opts;
  free_opts.first_token = 0;
  free_opts.max_completion_tokens = 8;
  auto ids = engine->Generate(**model, free_opts);
  ASSERT_TRUE(ids.has_value()) << tessera::ToString(ids.error());
  ASSERT_GE(ids->size(), 3u);
  // Stop on the third generated token: the first two survive, the stop does
  // not appear in the output.
  GenerateOptions stop_opts = free_opts;
  stop_opts.stop_tokens = {(*ids)[2]};
  auto stopped = engine->Generate(**model, stop_opts);
  ASSERT_TRUE(stopped.has_value()) << tessera::ToString(stopped.error());
  ASSERT_EQ(stopped->size(), 2u);
  EXPECT_EQ((*stopped)[0], (*ids)[0]);
  EXPECT_EQ((*stopped)[1], (*ids)[1]);
  // The streaming path reports the end reason: the model's stop token.
  auto streamed = engine->GenerateStreaming(
      **model, stop_opts, [](std::uint32_t) { return true; });
  ASSERT_TRUE(streamed.has_value()) << tessera::ToString(streamed.error());
  EXPECT_EQ(streamed->produced, 2u);
  EXPECT_EQ(streamed->reason, tessera::FinishReason::Stop);
}

// A zero max_completion_tokens fills the remaining context: context minus prompt,
// saturating at zero when the prompt already fills it.
TEST(EngineTest, GenerateZeroMaxCompletionTokensFillsContext) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("generate_fill.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 6});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  // No prompt tokens: the fallback first token leaves 5 of 6 free.
  auto filled = engine->Generate(**model, GenerateOptions{0, 0});
  ASSERT_TRUE(filled.has_value()) << tessera::ToString(filled.error());
  EXPECT_EQ(filled->size(), 5u);
  // An explicit count still passes through unchanged.
  auto four = engine->Generate(**model, GenerateOptions{4, 0});
  ASSERT_TRUE(four.has_value()) << tessera::ToString(four.error());
  EXPECT_EQ(four->size(), 4u);
  // A prompt past the context produces nothing instead of underflowing.
  GenerateOptions full;
  full.prompt_tokens = {0, 1, 2, 3, 4, 5, 6, 7};
  auto capped = engine->Generate(**model, full);
  ASSERT_TRUE(capped.has_value()) << tessera::ToString(capped.error());
  EXPECT_TRUE(capped->empty());
}

// An explicit token count with a prompt past the context is rejected
// before any device work: a 15k-token prompt must fail fast instead
// of losing the device in a giant prefill.
TEST(EngineTest, GenerateRejectsPromptBeyondContext) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("generate_toolong.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 6});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  GenerateOptions options;
  options.max_completion_tokens = 4;
  options.prompt_tokens = {0, 1, 2, 3, 4, 5, 6, 7};
  auto rejected = engine->Generate(**model, options);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error(), StatusCode::InvalidArgument);
  auto streamed = engine->GenerateStreaming(
      **model, options, [](std::uint32_t) { return true; });
  ASSERT_FALSE(streamed.has_value());
  EXPECT_EQ(streamed.error(), StatusCode::InvalidArgument);
}

// Strategies without batched prefill keep the per-token path: the MTP
// tail keeps every position, the empty call probes support, and a real
// chunk is refused.

// A multi-token prompt prefills through the forward-only path (all but the
// last prompt token skip the output head) and still generates.
TEST(EngineTest, GenerateWithPromptTokensPrefills) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("prefill.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  GenerateOptions options;
  options.first_token = 0;
  options.max_completion_tokens = 4;
  options.prompt_tokens = {0, 1, 2};
  auto ids = engine->Generate(**model, options);
  ASSERT_TRUE(ids.has_value()) << tessera::ToString(ids.error());
  EXPECT_EQ(ids->size(), 4u);
  auto again = engine->Generate(**model, options);
  ASSERT_TRUE(again.has_value()) << tessera::ToString(again.error());
  EXPECT_EQ(*ids, *again);
}

// GenerateStreaming emits one callback per produced token, matches the
// blocking Generate sequence, and stops early when asked. The CLI
// streams through this so output tokens reach the console at once.
TEST(EngineTest, GenerateStreamingEmitsIncrementally) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("streaming.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  GenerateOptions options;
  options.first_token = 0;
  options.max_completion_tokens = 4;
  auto blocked = engine->Generate(**model, options);
  ASSERT_TRUE(blocked.has_value()) << tessera::ToString(blocked.error());
  ASSERT_EQ(blocked->size(), 4u);
  std::vector<std::uint32_t> streamed;
  auto count = engine->GenerateStreaming(
      **model, options, [&](std::uint32_t token) {
        streamed.push_back(token);
        return true;
      });
  ASSERT_TRUE(count.has_value()) << tessera::ToString(count.error());
  EXPECT_EQ(count->produced, 4u);
  EXPECT_EQ(count->reason, tessera::FinishReason::Length);
  EXPECT_EQ(streamed, *blocked);
  // Early stop after two tokens reports the produced prefix only.
  std::vector<std::uint32_t> prefix;
  auto partial = engine->GenerateStreaming(
      **model, options, [&](std::uint32_t token) {
        if (prefix.size() >= 2) {
          return false;
        }
        prefix.push_back(token);
        return true;
      });
  ASSERT_TRUE(partial.has_value()) << tessera::ToString(partial.error());
  EXPECT_EQ(partial->produced, 2u);
  EXPECT_EQ(partial->reason, tessera::FinishReason::Aborted);
  ASSERT_EQ(prefix.size(), 2u);
  EXPECT_EQ(prefix[0], (*blocked)[0]);
  EXPECT_EQ(prefix[1], (*blocked)[1]);
}

TEST(EngineTest, RealModelLoadPathWhenProvided) {
  const char* raw = std::getenv("TESSERA_TEST_MODEL");
  if (raw == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_MODEL not set";
  }
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{raw, 1024});
  // The first-class target is a hybrid attention/SSM model: the load
  // succeeds and a greedy step produces a token in range.
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto config = (*model)->Config();
  ASSERT_TRUE(config.has_value()) << tessera::ToString(config.error());
  EXPECT_TRUE(config->hybrid);
  EXPECT_EQ(config->layers, 64u);
  EXPECT_EQ(config->attention.head_dim, 256u);
  EXPECT_EQ(config->vocab_size, 248320u);
  EXPECT_EQ(config->ssm.inner_size, 6144u);
  EXPECT_EQ(config->full_attention_interval, 4u);
  ASSERT_EQ(config->rope_sections.size(), 3u);
  EXPECT_EQ(config->rope_sections[0], 11u);
  EXPECT_EQ(config->rope_sections[1], 11u);
  EXPECT_EQ(config->rope_sections[2], 10u);
  tessera::core::DecodeCache cache;
  auto step = tessera::core::DecodeStep(engine->Owner(), **model, cache, 0);
  ASSERT_TRUE(step.has_value()) << tessera::ToString(step.error());
  EXPECT_LT(*step, config->vocab_size);
}

// The MXFP4 target loads through its module and decodes when its
// directory is provided (set TESSERA_TEST_MXFP4_DIR).
TEST(EngineTest, MxFp4GeneratesWhenProvided) {
  const char* dir = std::getenv("TESSERA_TEST_MXFP4_DIR");
  if (dir == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_MXFP4_DIR not set";
  }
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{dir, 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto config = (*model)->Config();
  ASSERT_TRUE(config.has_value()) << tessera::ToString(config.error());
  EXPECT_TRUE(config->hybrid);
  EXPECT_EQ(config->layers, 64u);
  GenerateOptions options;
  options.max_completion_tokens = 4;
  options.prompt_tokens = {760, 6511, 314, 9338, 369};
  auto generated = engine->Generate(**model, options);
  ASSERT_TRUE(generated.has_value()) << tessera::ToString(generated.error());
  EXPECT_EQ(generated->size(), 4u);
  for (std::size_t i = 0; i < generated->size(); ++i) {
    std::fprintf(stderr, "mxfp4 token %zu: %u\n", i, (*generated)[i]);
  }
}

// Same prompt as MxFp4GeneratesWhenProvided, on the GGUF target, so the
// per-layer trace can be compared (set TESSERA_TEST_MODEL).
TEST(EngineTest, GgufGeneratesWhenProvided) {
  const char* path = std::getenv("TESSERA_TEST_MODEL");
  if (path == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_MODEL not set";
  }
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{path, 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  GenerateOptions options;
  options.max_completion_tokens = 4;
  options.prompt_tokens = {760, 6511, 314, 9338, 369};
  auto generated = engine->Generate(**model, options);
  ASSERT_TRUE(generated.has_value()) << tessera::ToString(generated.error());
  for (std::size_t i = 0; i < generated->size(); ++i) {
    std::fprintf(stderr, "gguf token %zu: %u\n", i, (*generated)[i]);
  }
}

// Print a tensor's dequantized fp32 statistics for cross-target
// comparison (set TESSERA_DUMP_MODEL and TESSERA_DUMP_TENSOR).



TEST(EngineTest, ParsesMoeTextConfigJson) {
  auto arch = tessera::CreateArchitecture("qwen35moe");
  ASSERT_NE(arch, nullptr);
  const std::string json = R"({
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "text_config": {
      "hidden_size": 2048, "num_hidden_layers": 40,
      "num_attention_heads": 16, "num_key_value_heads": 2,
      "head_dim": 256, "vocab_size": 248320, "rms_norm_eps": 1e-6,
      "full_attention_interval": 4, "partial_rotary_factor": 0.25,
      "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
      "linear_num_key_heads": 16, "linear_num_value_heads": 32,
      "linear_value_head_dim": 128,
      "moe_intermediate_size": 512, "num_experts": 256,
      "num_experts_per_tok": 8, "shared_expert_intermediate_size": 512,
      "rope_parameters": {"rope_theta": 10000000.0,
        "mrope_section": [11, 11, 10]}
    }
  })";
  auto config = arch->ParseConfigJson(json);
  ASSERT_TRUE(config.has_value()) << tessera::ToString(config.error());
  EXPECT_TRUE(config->IsMoe());
  EXPECT_EQ(config->num_experts, 256u);
  EXPECT_EQ(config->experts_per_tok, 8u);
  EXPECT_EQ(config->moe_intermediate, 512u);
  EXPECT_EQ(config->shared_expert_intermediate, 512u);
  EXPECT_EQ(config->ffn_dim, 0u);
  EXPECT_TRUE(config->hybrid);
}

TEST(EngineTest, LoadGgufModelMoeConfig) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 22);
  builder.KvString("general.name", "tiny-moe");
  builder.KvString("general.architecture", "qwen35moe");
  builder.KvU32("qwen35moe.block_count", 2);
  builder.KvU32("qwen35moe.embedding_length", 64);
  builder.KvF32("qwen35moe.attention.layer_norm_rms_epsilon", 1e-5f);
  builder.KvU32("qwen35moe.attention.head_count", 4);
  builder.KvU32("qwen35moe.attention.head_count_kv", 2);
  builder.KvU32("qwen35moe.attention.key_length", 16);
  builder.KvU32("qwen35moe.attention.value_length", 16);
  builder.KvU32("qwen35moe.rope.dimension_count", 8);
  builder.KvF32("qwen35moe.rope.freq_base", 10000.0f);
  builder.PushString("qwen35moe.rope.dimension_sections");
  builder.PushU32(9);
  builder.PushU32(4);
  builder.PushU64(4);
  for (std::uint32_t s : {1u, 1u, 1u, 0u}) {
    builder.PushU32(s);
  }
  builder.KvU32("qwen35moe.ssm.conv_kernel", 2);
  builder.KvU32("qwen35moe.ssm.state_size", 4);
  builder.KvU32("qwen35moe.ssm.group_count", 2);
  builder.KvU32("qwen35moe.ssm.time_step_rank", 4);
  builder.KvU32("qwen35moe.ssm.inner_size", 16);
  builder.KvU32("qwen35moe.full_attention_interval", 2);
  builder.KvU32("qwen35moe.expert_count", 4);
  builder.KvU32("qwen35moe.expert_used_count", 2);
  builder.KvU32("qwen35moe.expert_feed_forward_length", 8);
  builder.KvU32("qwen35moe.expert_shared_feed_forward_length", 8);
  builder.Tensor("output.weight", 2, {64, 8}, 0, 0);
  builder.PadTo(((builder.bytes.size() + 31) & ~31u) + 4096);
  auto dir = FreshTempDir("tessera_tests_moe_model");
  auto path = dir / "moe.gguf";
  WriteBytes(path, builder.bytes);
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto config = (*model)->Config();
  ASSERT_TRUE(config.has_value()) << tessera::ToString(config.error());
  EXPECT_TRUE(config->IsMoe());
  EXPECT_EQ(config->num_experts, 4u);
  EXPECT_EQ(config->experts_per_tok, 2u);
  EXPECT_EQ(config->moe_intermediate, 8u);
  EXPECT_EQ(config->shared_expert_intermediate, 8u);
  EXPECT_EQ(config->ffn_dim, 0u);
}

TEST(EngineTest, LoadModelMissingFile) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model =
      engine->LoadModel(ModelOptions{"/nonexistent/model.gguf", 1024});
  ASSERT_FALSE(model.has_value());
  EXPECT_EQ(model.error(), StatusCode::FileNotFound);
}

TEST(EngineTest, LoadModelEmptyPath) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{"", 1024});
  ASSERT_FALSE(model.has_value());
  EXPECT_EQ(model.error(), StatusCode::InvalidArgument);
}

TEST(EngineTest, LoadGgufModelAttentionParams) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 7);
  builder.KvString("general.name", "test-model");
  builder.KvString("general.architecture", "test-arch");
  builder.KvU32("test-arch.attention.head_count", 8);
  builder.KvU32("test-arch.attention.head_count_kv", 2);
  builder.KvU32("test-arch.embedding_length", 256);
  builder.KvU32("test-arch.rope.dimension_count", 16);
  builder.KvF32("test-arch.rope.freq_base", 10000.0f);
  builder.Tensor("w_a", 1, {4}, 0, 0);
  builder.PadTo(((builder.bytes.size() + 31) & ~31u) + 16);
  auto dir = FreshTempDir("tessera_tests_attention");
  auto path = dir / "attn.gguf";
  WriteBytes(path, builder.bytes);
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto params = (*model)->Attention();
  ASSERT_TRUE(params.has_value()) << tessera::ToString(params.error());
  EXPECT_EQ(params->heads, 8u);
  EXPECT_EQ(params->kv_heads, 2u);
  EXPECT_EQ(params->head_dim, 32u);
  EXPECT_EQ(params->rope_dim, 16u);
  EXPECT_DOUBLE_EQ(params->rope_theta, 10000.0);
}




TEST(EngineTest, LoadGgufModelHybridConfig) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteHybridFixture("hybrid.gguf", true);
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto config = (*model)->Config();
  ASSERT_TRUE(config.has_value()) << tessera::ToString(config.error());
  EXPECT_TRUE(config->hybrid);
  EXPECT_EQ(config->layers, 3u);
  EXPECT_EQ(config->attention.head_dim, 64u);
  EXPECT_EQ(config->ssm.conv_kernel, 2u);
  EXPECT_EQ(config->ssm.state_size, 8u);
  EXPECT_EQ(config->ssm.group_count, 2u);
  EXPECT_EQ(config->ssm.time_step_rank, 4u);
  EXPECT_EQ(config->ssm.inner_size, 32u);
  EXPECT_EQ(config->full_attention_interval, 2u);
  ASSERT_EQ(config->rope_sections.size(), 3u);
  EXPECT_EQ(config->rope_sections[0], 1u);
  EXPECT_EQ(config->rope_sections[1], 1u);
  EXPECT_EQ(config->rope_sections[2], 1u);
  EXPECT_FALSE(config->IsFullAttentionLayer(0));
  EXPECT_TRUE(config->IsFullAttentionLayer(1));
  EXPECT_FALSE(config->IsFullAttentionLayer(2));
  TransformerConfig vanilla;
  EXPECT_TRUE(vanilla.IsFullAttentionLayer(5));
}




TEST(EngineTest, LoadModelRejectsMalformedGguf) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto dir = FreshTempDir("tessera_tests_engine");
  auto path = dir / "bad.gguf";
  WriteBytes(path, std::vector<std::byte>(32, std::byte{0x41}));
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_FALSE(model.has_value());
  EXPECT_EQ(model.error(), StatusCode::MalformedFile);
}

TEST(EngineTest, LoadModelRejectsBadDirectory) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto dir = FreshTempDir("tessera_tests_engine");
  auto model = engine->LoadModel(ModelOptions{dir.string(), 1024});
  ASSERT_FALSE(model.has_value());
  EXPECT_EQ(model.error(), StatusCode::MalformedFile);
}

TEST(EngineTest, LoadModelAcceptsMxFp4Directory) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto dir = FreshTempDir("tessera_tests_engine");
  WritePlaceholderConfig(dir);
  WritePlaceholderWeights(dir);
  auto model = engine->LoadModel(ModelOptions{dir.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto& loaded = *model;
  EXPECT_EQ(loaded->Format(), ModelFormat::MxFp4);
  ASSERT_EQ(loaded->Tensors().size(), 1u);
  EXPECT_EQ(loaded->Tensors()[0].name, "w");
  EXPECT_EQ(loaded->Tensors()[0].dtype, tessera::DType::F32);
  ASSERT_EQ(loaded->Weights().size(), 1u);
  EXPECT_EQ(loaded->Weights()[0].device->Size(), 16u);
  const auto* buffer = loaded->FindWeight("w");
  ASSERT_NE(buffer, nullptr);
  std::vector<std::byte> readback(16);
  auto download = engine->Owner().CopyD2H(*buffer, readback.data(), 16);
  ASSERT_TRUE(download.has_value())
      << tessera::ToString(download.error());
  const auto* values = reinterpret_cast<const float*>(readback.data());
  EXPECT_FLOAT_EQ(values[0], 1.0f);
  EXPECT_FLOAT_EQ(values[3], 4.0f);
}




// MTP attaches like DFlash2, with no checkpoint path (it drafts from the
// target's own nextn head). This is the attachment the CLI makes for
// --speculate, on both the run and the serve path.

// --list-gpus path: when a device is present the backend enumerates at
// least that device, and the index matches the one Engine::Create picks.

// Engine::Create selects the GPU by index (default 0). An out-of-range
// index is InvalidArgument, not a crash or a silent fallback.
TEST(EngineTest, RejectsOutOfRangeDeviceIndex) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  EngineOptions options;
  options.device_index = 100000;
  auto bad = Engine::Create(options);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
}

// Prefill chunk plumbing: the request value wins, then the engine
// default, then the automatic default (kDefaultPrefillChunkTokens).
TEST(EngineTest, ResolvePrefillChunkTokens) {
  EXPECT_EQ(tessera::ResolvePrefillChunkTokens(0, 0, 4096),
            tessera::kDefaultPrefillChunkTokens);
  EXPECT_EQ(tessera::ResolvePrefillChunkTokens(512, 0, 4096), 512u);
  EXPECT_EQ(tessera::ResolvePrefillChunkTokens(512, 256, 4096), 256u);
  EXPECT_EQ(tessera::ResolvePrefillChunkTokens(0, 256, 4096), 256u);
  EXPECT_EQ(tessera::ResolvePrefillChunkTokens(0, 0, 0),
            tessera::kDefaultPrefillChunkTokens);
}

// New options default to automatic (0); the engine keeps its
// configured value for generation time.
TEST(EngineTest, PrefillChunkDefaultsToAuto) {
  EXPECT_EQ(EngineOptions{}.prefill_chunk_tokens, 0u);
  EXPECT_EQ(GenerateOptions{}.prefill_chunk_tokens, 0u);
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  EXPECT_EQ(engine->PrefillChunkTokens(), 0u);
}

// Sampling: temperature 0 and top_k 1 both reduce to the argmax, the
// draw is reproducible for a fixed seed, and the repetition penalty
// lowers a repeated token.
TEST(EngineTest, SampleTokenFilters) {
  const std::vector<float> logits = {1.0f, 3.0f, 2.0f};
  tessera::SamplingOptions greedy;
  greedy.temperature = 0.0f;
  std::mt19937_64 rng(1);
  EXPECT_EQ(tessera::core::SampleToken(
                std::span<const float>(logits), greedy, {}, rng),
            1u);
  tessera::SamplingOptions top1;
  top1.temperature = 1.0f;
  top1.top_k = 1;
  std::mt19937_64 rng2(1);
  EXPECT_EQ(tessera::core::SampleToken(
                std::span<const float>(logits), top1, {}, rng2),
            1u);

  const std::vector<float> many = {0.5f, 0.2f, 1.0f, 0.1f};
  tessera::SamplingOptions full;
  full.temperature = 1.0f;
  std::mt19937_64 a(7), b(7);
  EXPECT_EQ(
      tessera::core::SampleToken(std::span<const float>(many), full, {}, a),
      tessera::core::SampleToken(std::span<const float>(many), full, {}, b));

  const std::vector<float> pair = {5.0f, 4.9f};
  tessera::SamplingOptions penalty;
  penalty.temperature = 0.0f;
  penalty.repetition_penalty = 2.0f;
  const std::vector<std::uint32_t> history = {0};
  std::mt19937_64 rng3(1);
  EXPECT_EQ(tessera::core::SampleToken(std::span<const float>(pair), penalty,
                                       history, rng3),
            1u);
}

// Sampling through the engine is reproducible for a fixed seed, and
// invalid parameters are rejected.
TEST(EngineTest, SampleGenerationIsDeterministic) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("sample.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  GenerateOptions options;
  options.max_completion_tokens = 4;
  options.sample = true;
  options.seed = 42;
  auto first = engine->Generate(**model, options);
  ASSERT_TRUE(first.has_value()) << tessera::ToString(first.error());
  EXPECT_EQ(first->size(), 4u);
  auto again = engine->Generate(**model, options);
  ASSERT_TRUE(again.has_value()) << tessera::ToString(again.error());
  EXPECT_EQ(*first, *again);
  options.sampling.top_p = 0.0f;
  auto bad = engine->Generate(**model, options);
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error(), StatusCode::InvalidArgument);
}

// The DFlash2 draft must be output preserving: speculative generation
// equals plain greedy on the real target. Needs TESSERA_TEST_MODEL (the
// target) and TESSERA_TEST_DFLASH2_DIR (the draft).
TEST(EngineTest, DFlash2MatchesGreedyOnModel) {
  const char* target = std::getenv("TESSERA_TEST_MODEL");
  const char* draft = std::getenv("TESSERA_TEST_DFLASH2_DIR");
  if (target == nullptr || draft == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_MODEL / TESSERA_TEST_DFLASH2_DIR not set";
  }
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{target, 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  GenerateOptions options;
  options.max_completion_tokens = 8;
  if (const char* tokens = std::getenv("TESSERA_DFLASH2_TOKENS");
      tokens != nullptr) {
    const int value = std::atoi(tokens);
    if (value > 0) {
      options.max_completion_tokens = static_cast<std::size_t>(value);
    }
  }
  if (const char* kv = std::getenv("TESSERA_DFLASH2_KV"); kv != nullptr) {
    const std::string_view name(kv);
    if (name == "f16") {
      options.kv_type = tessera::KvCacheType::F16;
    } else if (name == "q8") {
      options.kv_type = tessera::KvCacheType::Q8;
    } else if (name == "q4") {
      options.kv_type = tessera::KvCacheType::Q4;
    } else if (name == "fp8") {
      options.kv_type = tessera::KvCacheType::FP8;
    }
  }
  options.prompt_tokens = {760, 6511, 314, 9338, 369};
  auto greedy = engine->Generate(**model, options);
  ASSERT_TRUE(greedy.has_value()) << tessera::ToString(greedy.error());
  auto spec = engine->GenerateDraft(**model, options, draft);
  ASSERT_TRUE(spec.has_value()) << tessera::ToString(spec.error());
  EXPECT_EQ(*spec, *greedy);
}

// MTP speculative decode on the real model: output equals greedy, and the
// log line reports tokens/s for both so the speculative speedup is visible.
TEST(EngineTest, MtpDecodeOnModel) {
  const char* target = std::getenv("TESSERA_TEST_MODEL");
  if (target == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_MODEL not set";
  }
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{target, 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  GenerateOptions options;
  options.max_completion_tokens = 32;
  if (const char* tokens = std::getenv("TESSERA_MTP_TOKENS");
      tokens != nullptr) {
    const int value = std::atoi(tokens);
    if (value > 0) {
      options.max_completion_tokens = static_cast<std::size_t>(value);
    }
  }
  if (const char* block = std::getenv("TESSERA_MTP_BLOCK");
      block != nullptr) {
    options.draft_tokens = static_cast<std::size_t>(std::atoi(block));
  }
  options.prompt_tokens = {760, 6511, 314, 9338, 369};
  auto greedy = engine->Generate(**model, options);
  ASSERT_TRUE(greedy.has_value()) << tessera::ToString(greedy.error());
  auto attached =
      engine->AttachSpeculative(tessera::CreateMtpStrategy());
  ASSERT_TRUE(attached.has_value()) << tessera::ToString(attached.error());
  auto spec = engine->GenerateSpeculative(**model, options);
  ASSERT_TRUE(spec.has_value()) << tessera::ToString(spec.error());
  EXPECT_EQ(*spec, *greedy);
  // The attachment also drives the plain generate path, which is the
  // path the HTTP server runs for every served turn.
  auto served = engine->Generate(**model, options);
  ASSERT_TRUE(served.has_value()) << tessera::ToString(served.error());
  EXPECT_EQ(*served, *greedy);
}

// The vision config parser rejects a non-CLIP GGUF.

// The real mmproj parses when its path is provided
// (set TESSERA_TEST_MMPROJ).

// The real mmproj loads and encodes a synthetic image to finite embeddings
// when its path is provided (set TESSERA_TEST_MMPROJ).

// Host: a binary PPM loads to fp32 RGB, and bilinear resize works.

// Load instrumentation: the phase timer reports name, milliseconds and
// detail through the diagnostics sink exactly once.

// A null diagnostics channel makes the timer a no-op (tests and tools
// that build an engine without a sink).


// The shared transfer summary reports bytes, copies, time and MiB/s; a
// zero duration reports a zero rate instead of dividing by zero.

// A GGUF load reports one line per phase through the engine diagnostics,
// so a slow load is traceable from the log alone.
