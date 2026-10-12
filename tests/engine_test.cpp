#include <gtest/gtest.h>
#include <cstdio>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <optional>
#include <random>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <sys/socket.h>
#include <unistd.h>

#include <arpa/inet.h>
#include <netinet/in.h>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "core/json.hpp"
#include "core/loaders/shard.hpp"
#include "core/sampling.hpp"
#include "tessera/budget.hpp"
#include "core/timing.hpp"
#include "test_helpers.hpp"
#include "tessera/architecture.hpp"
#include "tessera/engine.hpp"
#include "tessera/runner.hpp"
#include "tessera/scheduler.hpp"
#include "tessera/speculative.hpp"
#include "tessera/image.hpp"
#include "tessera/tokenizer.hpp"
#include "tessera/vision.hpp"
#include "tessera/types.hpp"
#include "serve/http.hpp"
#include "serve/openai.hpp"
#include "serve/openai_shared.hpp"

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
using tessera::testing::TestDeviceIndex;
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

// A tiny vanilla transformer GGUF: hidden 256, 4 heads over 2 kv
// groups of dim 64, ffn 512, vocab 32. Projections are Q4_K, vectors
// F32, tensor names follow the llama.cpp convention. `layers` repeats
// the block specs as blk.0 through blk.(layers - 1).
std::filesystem::path WriteTinyModelFixture(const std::string& name,
                                            std::size_t layers = 1) {
  std::mt19937 rng(7);
  struct Spec {
    const char* tensor;
    std::uint32_t type;
    std::vector<std::uint64_t> dims;
    std::size_t rows;  // Q4_K row count (0 for F32)
    std::size_t cols;  // Q4_K row length (0 for F32)
  };
  std::vector<Spec> specs;
  specs.push_back({"token_embd.weight", 0, {256, 32}, 0, 0});
  struct BlockSpec {
    const char* suffix;
    std::uint32_t type;
    std::vector<std::uint64_t> dims;
    std::size_t rows;
    std::size_t cols;
  };
  // One block's tensors; the layer loop below stamps them as blk.l.*.
  const std::vector<BlockSpec> block = {
      {"attn_norm.weight", 0, {256}, 0, 0},
      {"attn_q.weight", 12, {256, 256}, 256, 256},
      {"attn_k.weight", 12, {256, 128}, 128, 256},
      {"attn_v.weight", 12, {256, 128}, 128, 256},
      {"attn_output.weight", 12, {256, 256}, 256, 256},
      {"ffn_norm.weight", 0, {256}, 0, 0},
      {"ffn_gate.weight", 12, {256, 512}, 512, 256},
      {"ffn_up.weight", 12, {256, 512}, 512, 256},
      {"ffn_down.weight", 12, {512, 256}, 256, 512},
  };
  // Own every block tensor name before the specs borrow them, so no
  // push reallocates while a spec points at it.
  std::vector<std::string> block_names;
  block_names.reserve(layers * block.size());
  for (std::size_t l = 0; l < layers; ++l) {
    const std::string base = "blk." + std::to_string(l) + ".";
    for (const auto& part : block) {
      block_names.push_back(base + part.suffix);
    }
  }
  std::size_t next = 0;
  for (std::size_t l = 0; l < layers; ++l) {
    (void)l;
    for (const auto& part : block) {
      specs.push_back({block_names[next].c_str(), part.type, part.dims,
                       part.rows, part.cols});
      ++next;
    }
  }
  specs.push_back({"output_norm.weight", 0, {256}, 0, 0});
  specs.push_back({"output.weight", 12, {256, 32}, 32, 256});
  GgufBuilder builder;
  builder.Header(0x46554747, 3, specs.size(), 10);
  builder.KvString("general.name", "tiny-vanilla");
  builder.KvString("general.architecture", "test-vanilla");
  builder.KvU32("test-vanilla.block_count",
                static_cast<std::uint32_t>(layers));
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

// The chat template sees the thinking options: thinking stays off by
// default, enable_thinking opens it, and a reasoning effort reaches the
// template as the reasoning_effort argument.
TEST(EngineTest, ChatPromptThinkingFlags) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 2);
  builder.KvString("general.name", "test-model");
  builder.KvString("tokenizer.chat_template",
                   "{% if enable_thinking %}T{% else %}F{% endif %}"
                   "{{ reasoning_effort }}");
  builder.Tensor("w_a", 1, {4}, 0, 0);  // F32, 1x4, offset 0
  builder.PadTo(((builder.bytes.size() + 31) & ~31u) + 16);
  auto dir = FreshTempDir("tessera_tests_chat");
  auto path = dir / "chat.gguf";
  WriteBytes(path, builder.bytes);
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto prompt = (*model)->ChatPrompt("hi");
  ASSERT_TRUE(prompt.has_value()) << tessera::ToString(prompt.error());
  EXPECT_EQ(*prompt, "F");
  prompt = (*model)->ChatPrompt("hi", true);
  ASSERT_TRUE(prompt.has_value()) << tessera::ToString(prompt.error());
  EXPECT_EQ(*prompt, "T");
  prompt = (*model)->ChatPrompt("hi", true, "xhigh");
  ASSERT_TRUE(prompt.has_value()) << tessera::ToString(prompt.error());
  EXPECT_EQ(*prompt, "Txhigh");
  prompt = (*model)->ChatPrompt("hi", false, "low");
  ASSERT_TRUE(prompt.has_value()) << tessera::ToString(prompt.error());
  EXPECT_EQ(*prompt, "Flow");
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

// A paged load keeps early layers in RAM and decodes exactly like the
// full upload: a paged name misses before its layer is ensured, resolves
// after, and the generated tokens match the resident baseline.
TEST(EngineTest, PagedModelMatchesFullUpload) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("paged.gguf", 2);
  auto full = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(full.has_value()) << tessera::ToString(full.error());
  auto full_ids = engine->Generate(**full, GenerateOptions{4, 0});
  ASSERT_TRUE(full_ids.has_value()) << tessera::ToString(full_ids.error());
  ModelOptions paged_options{path.string(), 1024};
  paged_options.gpu_layers = 2;
  auto paged = engine->LoadModel(paged_options);
  ASSERT_TRUE(paged.has_value()) << tessera::ToString(paged.error());
  // Two resident layers: the head plus blk.1; blk.0 pages.
  EXPECT_EQ((*paged)->FindDeviceTensor("blk.0.attn_q.weight"), nullptr);
  EXPECT_NE((*paged)->FindDeviceTensor("blk.1.attn_q.weight"), nullptr);
  EXPECT_NE((*paged)->FindDeviceTensor("output.weight"), nullptr);
  EXPECT_LT((*paged)->Weights().size(), (*full)->Weights().size());
  EXPECT_FALSE((*paged)->Paged().empty());
  auto ensured = (*paged)->EnsureLayerResident(0);
  ASSERT_TRUE(ensured.has_value()) << tessera::ToString(ensured.error());
  EXPECT_NE((*paged)->FindDeviceTensor("blk.0.attn_q.weight"), nullptr);
  auto paged_ids = engine->Generate(**paged, GenerateOptions{4, 0});
  ASSERT_TRUE(paged_ids.has_value()) << tessera::ToString(paged_ids.error());
  EXPECT_EQ(*paged_ids, *full_ids);
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

// GenerateBatch interleaves chunked prefills with decodes: a long prompt
// prefills chunk by chunk while a short sequence already decodes, and each
// sequence still equals a separate Generate.
TEST(EngineTest, GenerateBatchInterleavesChunkedPrefill) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("batch_interleave.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  GenerateOptions first;
  first.prompt_tokens = {0, 1, 2, 3, 4};
  first.max_completion_tokens = 3;
  first.prefill_chunk_tokens = 2;
  GenerateOptions second;
  second.prompt_tokens = {5, 6};
  second.max_completion_tokens = 3;
  second.prefill_chunk_tokens = 2;
  // Progress hooks observe the chunk rounds: the long prompt reports three
  // rounds while the short one finishes in one, which only happens when
  // prefill runs chunk by chunk instead of blocking up front. The hooks
  // live on the batch copies so the equivalence Generates below stay clean.
  std::vector<std::pair<std::size_t, std::size_t>> first_progress;
  std::vector<std::pair<std::size_t, std::size_t>> second_progress;
  std::vector<GenerateOptions> requests = {first, second};
  requests[0].prefill_progress = [&](std::size_t done, std::size_t total) {
    first_progress.emplace_back(done, total);
  };
  requests[1].prefill_progress = [&](std::size_t done, std::size_t total) {
    second_progress.emplace_back(done, total);
  };
  auto batched = engine->GenerateBatch(**model, requests);
  ASSERT_TRUE(batched.has_value()) << tessera::ToString(batched.error());
  ASSERT_EQ(batched->size(), 2u);
  auto expect_first = engine->Generate(**model, first);
  ASSERT_TRUE(expect_first.has_value())
      << tessera::ToString(expect_first.error());
  auto expect_second = engine->Generate(**model, second);
  ASSERT_TRUE(expect_second.has_value())
      << tessera::ToString(expect_second.error());
  EXPECT_EQ((*batched)[0], *expect_first);
  EXPECT_EQ((*batched)[1], *expect_second);
  const std::vector<std::pair<std::size_t, std::size_t>> expect_first_progress{
      {2, 5}, {4, 5}, {5, 5}};
  const std::vector<std::pair<std::size_t, std::size_t>> expect_second_progress{
      {2, 2}};
  EXPECT_EQ(first_progress, expect_first_progress);
  EXPECT_EQ(second_progress, expect_second_progress);
}

// The scheduler time-slices one device across sessions: a long turn
// prefills chunk by chunk while two short turns decode beside it, and
// every turn equals a separate Generate with the same prompt.
TEST(EngineTest, SchedulerSharesOneDeviceAcrossSessions) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("scheduler_share.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  std::vector<std::uint32_t> long_prompt(200);
  for (std::size_t i = 0; i < long_prompt.size(); ++i) {
    long_prompt[i] = static_cast<std::uint32_t>(i % 32);
  }
  std::vector<GenerateOptions> requests(3);
  // TEMPORARY bisect (revert): one token plus three completions each,
  // so no KV growth ever happens (4 rows fit the initial capacity).
  requests[0].prompt_tokens = {10};
  requests[0].max_completion_tokens = 3;
  requests[1].prompt_tokens = {1};
  requests[1].max_completion_tokens = 3;
  requests[2].prompt_tokens = {3};
  requests[2].max_completion_tokens = 3;
  std::vector<std::vector<std::uint32_t>> expected;
  for (const auto& request : requests) {
    auto solo = engine->Generate(**model, request);
    ASSERT_TRUE(solo.has_value()) << tessera::ToString(solo.error());
    expected.push_back(*solo);
  }
  tessera::RunnerPool pool;
  pool.Add(std::make_unique<tessera::Runner>(
      std::move(engine), std::move(*model), TestDeviceIndex()));
  std::atomic<int> entered{0};
  std::atomic<std::size_t> max_active{0};
  struct Turn {
    std::vector<std::uint32_t> tokens;
    std::size_t produced = 0;
    bool ok = false;
  };
  std::vector<Turn> turns(3);
  std::vector<std::thread> workers;
  for (int t = 0; t < 3; ++t) {
    workers.emplace_back([&, t] {
      entered.fetch_add(1);
      while (entered.load() < 3) {
        std::this_thread::yield();
      }
      auto outcome = pool.SubmitLeastLoaded(
          requests[t], [&](std::uint32_t token) {
            turns[t].tokens.push_back(token);
            std::size_t active = pool.Active()[0];
            std::size_t seen = max_active.load();
            while (seen < active &&
                   !max_active.compare_exchange_weak(seen, active)) {
            }
            return true;
          });
      turns[t].ok = outcome.has_value();
      if (outcome) {
        turns[t].produced = outcome->produced;
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (int t = 0; t < 3; ++t) {
    EXPECT_TRUE(turns[t].ok) << "session " << t << " failed";
    EXPECT_EQ(turns[t].tokens, expected[t]) << "session " << t << " drifted";
    EXPECT_EQ(turns[t].produced, expected[t].size());
  }
  // All three turns decoded together at some point; a serial device
  // would never admit more than one.
  EXPECT_EQ(max_active.load(), 3u);
}

// Two CLI sessions share one device: both decode together and each
// streams while the other runs, matching a solo Generate per prompt.
TEST(EngineTest, CliParallelSessionsStreamTogether) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("cli_parallel.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  std::vector<GenerateOptions> requests(2);
  requests[0].prompt_tokens = {10};
  requests[0].max_completion_tokens = 8;
  requests[1].prompt_tokens = {3};
  requests[1].max_completion_tokens = 8;
  std::vector<std::vector<std::uint32_t>> expected;
  for (const auto& request : requests) {
    auto solo = engine->Generate(**model, request);
    ASSERT_TRUE(solo.has_value()) << tessera::ToString(solo.error());
    expected.push_back(*solo);
  }
  tessera::RunnerPool pool;
  pool.Add(std::make_unique<tessera::Runner>(
      std::move(engine), std::move(*model), TestDeviceIndex()));
  std::atomic<int> entered{0};
  std::atomic<std::size_t> max_active{0};
  struct Turn {
    std::vector<std::uint32_t> tokens;
    bool ok = false;
  };
  std::vector<Turn> turns(2);
  std::vector<std::thread> workers;
  for (int t = 0; t < 2; ++t) {
    workers.emplace_back([&, t] {
      entered.fetch_add(1);
      while (entered.load() < 2) {
        std::this_thread::yield();
      }
      auto outcome = pool.SubmitLeastLoaded(
          requests[t], [&](std::uint32_t token) {
            turns[t].tokens.push_back(token);
            std::size_t active = pool.Active()[0];
            std::size_t seen = max_active.load();
            while (seen < active &&
                   !max_active.compare_exchange_weak(seen, active)) {
            }
            return true;
          });
      turns[t].ok = outcome.has_value();
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (int t = 0; t < 2; ++t) {
    EXPECT_TRUE(turns[t].ok) << "session " << t << " failed";
    EXPECT_EQ(turns[t].tokens, expected[t]) << "session " << t << " drifted";
  }
  // Both sessions decoded together; a serial device admits one at most.
  EXPECT_EQ(max_active.load(), 2u);
}




// The scheduler queues past its bound: with one slot a long turn
// admits first while a short turn waits, then both read back equal
// to separate Generates.
TEST(EngineTest, SchedulerQueuesBeyondMaxConcurrent) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("scheduler_queue.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  std::vector<std::uint32_t> long_prompt(400);
  for (std::size_t i = 0; i < long_prompt.size(); ++i) {
    long_prompt[i] = static_cast<std::uint32_t>(i % 32);
  }
  GenerateOptions first;
  first.prompt_tokens = long_prompt;
  first.max_completion_tokens = 4;
  first.prefill_chunk_tokens = 2;
  GenerateOptions second;
  second.prompt_tokens = {7};
  second.max_completion_tokens = 4;
  auto expect_first = engine->Generate(**model, first);
  ASSERT_TRUE(expect_first.has_value())
      << tessera::ToString(expect_first.error());
  auto expect_second = engine->Generate(**model, second);
  ASSERT_TRUE(expect_second.has_value())
      << tessera::ToString(expect_second.error());
  std::mutex device;
  tessera::SchedulerOptions bound;
  bound.max_concurrent = 1;
  tessera::Scheduler shared(*engine, **model, device, bound);
  std::vector<std::uint32_t> first_tokens;
  std::vector<std::uint32_t> second_tokens;
  bool first_ok = false;
  bool second_ok = false;
  std::thread head([&] {
    auto outcome = shared.Submit(first, [&](std::uint32_t token) {
      first_tokens.push_back(token);
      return true;
    });
    first_ok = outcome.has_value();
  });
  while (shared.Active() == 0) {
    std::this_thread::yield();
  }
  std::thread tail([&] {
    auto outcome = shared.Submit(second, [&](std::uint32_t token) {
      second_tokens.push_back(token);
      return true;
    });
    second_ok = outcome.has_value();
  });
  bool saw_queued = false;
  for (int i = 0; i < 1000 && !saw_queued; ++i) {
    saw_queued = shared.Pending() > 0;
    if (!saw_queued) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
  head.join();
  tail.join();
  EXPECT_TRUE(first_ok);
  EXPECT_TRUE(second_ok);
  EXPECT_EQ(first_tokens, *expect_first);
  EXPECT_EQ(second_tokens, *expect_second);
  EXPECT_TRUE(saw_queued);
}

// The scheduler rejects bad turns without touching the device: an
// oversize prompt, a thinking budget and bad sampling fail fast, and
// an empty pool has no device to route to.
TEST(EngineTest, SchedulerRejectsBadRequests) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("scheduler_reject.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 8});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  tessera::RunnerPool pool;
  pool.Add(std::make_unique<tessera::Runner>(
      std::move(engine), std::move(*model), TestDeviceIndex()));
  const auto sink = [](std::uint32_t) { return true; };
  GenerateOptions too_long;
  too_long.prompt_tokens = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  too_long.max_completion_tokens = 1;
  auto rejected = pool.SubmitLeastLoaded(too_long, sink);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error(), StatusCode::InvalidArgument);
  GenerateOptions thinking;
  thinking.prompt_tokens = {0};
  thinking.max_completion_tokens = 1;
  thinking.max_thinking_tokens = 1;
  auto unthought = pool.SubmitLeastLoaded(thinking, sink);
  ASSERT_FALSE(unthought.has_value());
  EXPECT_EQ(unthought.error(), StatusCode::UnsupportedFeature);
  GenerateOptions bad_sample;
  bad_sample.prompt_tokens = {0};
  bad_sample.max_completion_tokens = 1;
  bad_sample.sample = true;
  bad_sample.sampling.temperature = -1.0f;
  auto unsampled = pool.SubmitLeastLoaded(bad_sample, sink);
  ASSERT_FALSE(unsampled.has_value());
  EXPECT_EQ(unsampled.error(), StatusCode::InvalidArgument);
  tessera::RunnerPool empty;
  GenerateOptions fine;
  fine.prompt_tokens = {0};
  fine.max_completion_tokens = 1;
  auto noroute = empty.SubmitLeastLoaded(fine, sink);
  ASSERT_FALSE(noroute.has_value());
  EXPECT_EQ(noroute.error(), StatusCode::InvalidArgument);
}

namespace {

// Stand-in for a speculative drafter in serve routing tests: never
// proposes, so the exclusive loop decodes plainly (output equals
// greedy) while the engine counts as strategy-backed for routing.
struct NonProposingStrategy : public tessera::SpeculativeStrategy {
  std::string_view Name() const override { return "non-proposing"; }
  std::expected<void, tessera::StatusCode> Attach(
      const tessera::StrategyOptions&) override {
    return {};
  }
  std::expected<void, tessera::StatusCode> Prepare(
      tessera::Backend&, tessera::Model&) override {
    return {};
  }
  std::size_t DraftBlock() const override { return 4; }
  std::span<const std::size_t> CaptureLayers() const override { return {}; }
  std::span<tessera::Buffer* const> CaptureBuffers() override { return {}; }
  std::expected<void, tessera::StatusCode> AppendPrefill(
      tessera::Backend&, tessera::Model&, tessera::core::DecodeCache&,
      std::span<const std::uint32_t>, std::uint64_t,
      std::span<tessera::Buffer* const>) override {
    return std::unexpected(tessera::StatusCode::UnsupportedFeature);
  }
  std::expected<void, tessera::StatusCode> OnAnchor(
      tessera::Backend&, tessera::Model&, tessera::core::DecodeCache&,
      std::uint32_t token, std::uint64_t position,
      std::span<const float>) override {
    anchors.emplace_back(token, position);
    return {};
  }
  std::expected<std::size_t, tessera::StatusCode> Draft(
      tessera::Backend&, tessera::Model&, tessera::core::DecodeCache&,
      std::span<const float>, std::uint32_t,
      std::span<std::uint32_t>) override {
    return 0;
  }
  std::expected<void, tessera::StatusCode> Commit(
      tessera::Backend&, tessera::Model&, tessera::core::DecodeCache&,
      std::size_t) override {
    return {};
  }
  std::vector<std::pair<std::uint32_t, std::uint64_t>> anchors;
};

}  // namespace

// A strategy-backed engine still shares the scheduler for plain
// turns: the gate is the thinking budget, not the strategy.
TEST(EngineTest, SchedulerAdmitsStrategyBackedTurns) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto strategy = std::make_unique<NonProposingStrategy>();
  ASSERT_TRUE(engine->AttachSpeculative(std::move(strategy)).has_value());
  GenerateOptions plain;
  plain.prompt_tokens = {3};
  EXPECT_TRUE(tessera::CanSchedule(*engine, plain));
  GenerateOptions thinking = plain;
  thinking.max_thinking_tokens = 1;
  EXPECT_FALSE(tessera::CanSchedule(*engine, thinking));
}

// TryAcquire skips a busy device instead of queueing behind it.
TEST(EngineTest, TryAcquireSkipsBusyDevice) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("try_acquire.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  tessera::RunnerPool pool;
  pool.Add(std::make_unique<tessera::Runner>(
      std::move(engine), std::move(*model), TestDeviceIndex()));
  auto held = pool.Acquire();
  ASSERT_TRUE(held.has_value());
  EXPECT_FALSE(pool.TryAcquire().has_value());
  held->Release();
  auto fast = pool.TryAcquire();
  ASSERT_TRUE(fast.has_value());
}

// YieldFor drops the device lock so a waiter acquires mid-turn.
TEST(EngineTest, LeaseYieldReleasesDevice) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("lease_yield.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  tessera::RunnerPool pool;
  pool.Add(std::make_unique<tessera::Runner>(
      std::move(engine), std::move(*model), TestDeviceIndex()));
  auto held = pool.Acquire();
  ASSERT_TRUE(held.has_value());
  std::atomic<bool> entered{false};
  std::thread waiter([&] {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!entered.load() && std::chrono::steady_clock::now() < deadline) {
      if (auto slot = pool.TryAcquire()) {
        entered.store(true);
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  held->YieldFor(std::chrono::milliseconds(50));
  waiter.join();
  EXPECT_TRUE(entered.load());
}

// A long exclusive turn on a strategy-backed engine streams beside a
// second session instead of behind it: the second routes shared and
// both match solo greedy (the pelican-SVG case).
TEST(EngineTest, StrategyBackedServeStreamsConcurrentSessions) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("strategy_share.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  GenerateOptions first_solo_options;
  first_solo_options.prompt_tokens = {10};
  first_solo_options.max_completion_tokens = 48;
  auto first_solo = engine->Generate(**model, first_solo_options);
  ASSERT_TRUE(first_solo.has_value())
      << tessera::ToString(first_solo.error());
  GenerateOptions second_solo_options;
  second_solo_options.prompt_tokens = {3};
  second_solo_options.max_completion_tokens = 8;
  auto second_solo = engine->Generate(**model, second_solo_options);
  ASSERT_TRUE(second_solo.has_value())
      << tessera::ToString(second_solo.error());
  auto strategy = std::make_unique<NonProposingStrategy>();
  NonProposingStrategy* watched = strategy.get();
  ASSERT_TRUE(engine->AttachSpeculative(std::move(strategy)).has_value());
  tessera::RunnerPool pool;
  pool.Add(std::make_unique<tessera::Runner>(
      std::move(engine), std::move(*model), TestDeviceIndex()));
  const auto idle = [] { return false; };
  // First session arrives on an idle device: exclusive fast path.
  tessera::RunnerLease first_lease;
  tessera::RunnerLease* first_slot = nullptr;
  ASSERT_TRUE(tessera::serve::RouteDevice(pool, 0, false, idle, first_lease,
                                          &first_slot));
  ASSERT_NE(first_slot, nullptr);
  GenerateOptions first_options = first_solo_options;
  std::vector<std::uint32_t> first_tokens;
  std::atomic<std::size_t> first_count{0};
  int first_yield = 0;
  std::expected<tessera::GenerateOutcome, tessera::StatusCode> first_outcome =
      std::unexpected(tessera::StatusCode::Ok);
  std::thread first([&] {
    first_outcome = first_slot->engine().GenerateStreaming(
        first_slot->model(), first_options, [&](std::uint32_t token) {
          first_tokens.push_back(token);
          first_count.fetch_add(1);
          tessera::serve::MaybeYieldExclusive(*first_slot, first_yield);
          return true;
        });
  });
  // Wait for the first turn to produce before routing the second, so
  // the second must share instead of racing for the fast path.
  const auto start = std::chrono::steady_clock::now();
  while (first_count.load() == 0 &&
         std::chrono::steady_clock::now() - start < std::chrono::seconds(30)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_GT(first_count.load(), 0u);
  // Second session arrives while the first holds the device: shared.
  tessera::RunnerLease second_lease;
  tessera::RunnerLease* second_slot = nullptr;
  ASSERT_TRUE(tessera::serve::RouteDevice(pool, 0, false, idle, second_lease,
                                          &second_slot));
  EXPECT_EQ(second_slot, nullptr);
  std::atomic<bool> first_done{false};
  std::atomic<std::size_t> max_active{0};
  std::thread watcher([&] {
    while (!first_done.load()) {
      const std::size_t active = pool.Active()[0];
      std::size_t seen = max_active.load();
      while (seen < active &&
             !max_active.compare_exchange_weak(seen, active)) {
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  std::vector<std::uint32_t> second_tokens;
  auto second_outcome = pool.SubmitLeastLoaded(
      second_solo_options, [&](std::uint32_t token) {
        second_tokens.push_back(token);
        return true;
      });
  first.join();
  first_done.store(true);
  watcher.join();
  ASSERT_TRUE(first_outcome.has_value())
      << tessera::ToString(first_outcome.error());
  ASSERT_TRUE(second_outcome.has_value())
      << tessera::ToString(second_outcome.error());
  EXPECT_EQ(first_tokens, *first_solo);
  EXPECT_EQ(second_tokens, *second_solo);
  EXPECT_GT(watched->anchors.size(), 0u);
  // Both turns held the device together; a serial serve never admits
  // more than one.
  EXPECT_EQ(max_active.load(), 2u);
}

// A busy port fails at bind time, before any weight loading starts,
// so a conflicting server surfaces immediately instead of after a
// full model upload.
TEST(EngineTest, BindConflictFailsFast) {
  tessera::log::Diagnostics log;
  const int holder = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(holder, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(0);
  ASSERT_EQ(::bind(holder, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)),
            0);
  ASSERT_EQ(::listen(holder, 1), 0);
  sockaddr_in bound{};
  socklen_t bound_len = sizeof(bound);
  ASSERT_EQ(::getsockname(holder, reinterpret_cast<sockaddr*>(&bound),
                          &bound_len),
            0);
  const auto port = static_cast<std::uint16_t>(ntohs(bound.sin_port));
  auto conflicted = tessera::serve::BindHttpServer("127.0.0.1", port, log);
  EXPECT_FALSE(conflicted.has_value());
  ::close(holder);
  auto free = tessera::serve::BindHttpServer("127.0.0.1", port, log);
  ASSERT_TRUE(free.has_value());
  ::close(free->fd);
}

// Least-loaded routing splits concurrent turns across runners: with
// two runners each turn lands on a different device slot while both
// still match their solo greedy text.
TEST(EngineTest, SubmitLeastLoadedSplitsAcrossRunners) {
  std::unique_ptr<Engine> first_engine;
  MakeEngineOrSkip(first_engine);
  EngineOptions second_options;
  // Same device: proves least-loaded routing splits turns across
  // runners. Live multi-GPU serve stays repo-pending (heavy loads),
  // and a busy device is never borrowed for tests.
  second_options.device_index = TestDeviceIndex();
  auto second_engine = Engine::Create(second_options);
  if (!second_engine) {
    GTEST_SKIP() << "no second engine: "
                 << tessera::ToString(second_engine.error());
  }
  auto fixture_path = WriteTinyModelFixture("route.gguf");
  auto first_model =
      first_engine->LoadModel(ModelOptions{fixture_path.string(), 1024});
  ASSERT_TRUE(first_model.has_value())
      << tessera::ToString(first_model.error());
  auto second_model = (*second_engine)->LoadModel(
      ModelOptions{fixture_path.string(), 1024});
  ASSERT_TRUE(second_model.has_value())
      << tessera::ToString(second_model.error());
  std::vector<std::uint32_t> prompt(100);
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    prompt[i] = static_cast<std::uint32_t>(i % 32);
  }
  GenerateOptions request;
  request.prompt_tokens = prompt;
  request.max_completion_tokens = 4;
  request.prefill_chunk_tokens = 2;
  auto expect_first =
      first_engine->Generate(**first_model, request);
  ASSERT_TRUE(expect_first.has_value())
      << tessera::ToString(expect_first.error());
  auto expect_second =
      (*second_engine)->Generate(**second_model, request);
  ASSERT_TRUE(expect_second.has_value())
      << tessera::ToString(expect_second.error());
  tessera::RunnerPool pool;
  pool.Add(std::make_unique<tessera::Runner>(
      std::move(first_engine), std::move(*first_model),
      TestDeviceIndex()));
  pool.Add(std::make_unique<tessera::Runner>(
      std::move(*second_engine), std::move(*second_model),
      TestDeviceIndex()));
  struct Turn {
    std::vector<std::uint32_t> tokens;
    bool ok = false;
  };
  std::vector<Turn> turns(2);
  std::thread head([&] {
    auto outcome = pool.SubmitLeastLoaded(
        request, [&](std::uint32_t token) {
          turns[0].tokens.push_back(token);
          return true;
        });
    turns[0].ok = outcome.has_value();
  });
  // Wait until the first turn is admitted so the second routes away
  // from it; simultaneous submits would both see idle runners.
  for (int i = 0; i < 1000 && pool.Active() != std::vector<std::size_t>{1, 0};
       ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_EQ(pool.Active(), (std::vector<std::size_t>{1, 0}));
  std::thread tail([&] {
    auto outcome = pool.SubmitLeastLoaded(
        request, [&](std::uint32_t token) {
          turns[1].tokens.push_back(token);
          return true;
        });
    turns[1].ok = outcome.has_value();
  });
  bool saw_split = false;
  for (int i = 0; i < 1000 && !saw_split; ++i) {
    saw_split = pool.Active() == std::vector<std::size_t>{1, 1};
    if (!saw_split) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
  head.join();
  tail.join();
  EXPECT_TRUE(turns[0].ok);
  EXPECT_TRUE(turns[1].ok);
  EXPECT_EQ(turns[0].tokens, *expect_first);
  EXPECT_EQ(turns[1].tokens, *expect_second);
  // Both runners held one turn at once; a first-free-wins router
  // would stack both on runner zero.
  EXPECT_TRUE(saw_split);
}

// The OpenAI completions endpoint drives the same shared scheduler:
// a stateless, non-streaming turn returns exactly the solo greedy
// text for the same prompt.
TEST(EngineTest, OpenAiCompleteMatchesSoloThroughScheduler) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteTinyModelFixture("openai_complete.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  // Single letters plus digits: 32 entries, so every id the tiny
  // model emits decodes back to text.
  std::vector<std::string> vocab;
  for (char c = 'a'; c <= 'z'; ++c) {
    vocab.emplace_back(1, c);
  }
  for (char c = '0'; c <= '5'; ++c) {
    vocab.emplace_back(1, c);
  }
  ASSERT_EQ(vocab.size(), 32u);
  tessera::Tokenizer tokenizer(vocab, std::vector<std::int32_t>(32, 1), {});
  auto ids = tokenizer.Encode("abc");
  ASSERT_TRUE(ids.has_value()) << tessera::ToString(ids.error());
  GenerateOptions solo_options;
  solo_options.prompt_tokens = *ids;
  solo_options.max_completion_tokens = 4;
  auto solo = engine->Generate(**model, solo_options);
  ASSERT_TRUE(solo.has_value()) << tessera::ToString(solo.error());
  auto want = tokenizer.Decode(*solo);
  ASSERT_TRUE(want.has_value()) << tessera::ToString(want.error());
  tessera::RunnerPool pool;
  pool.Add(std::make_unique<tessera::Runner>(
      std::move(engine), std::move(*model), TestDeviceIndex()));
  tessera::serve::SessionStore sessions;
  tessera::core::Json body = tessera::core::Json::Object();
  body.Set("prompt", tessera::core::Json::String("abc"));
  body.Set("max_completion_tokens", tessera::core::Json::Number(4));
  body.Set("stream", tessera::core::Json::Bool(false));
  int fds[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
  {
    tessera::serve::ResponseWriter writer(fds[0]);
    tessera::serve::OpenAiComplete(pool, tokenizer, sessions, writer, body,
                                   /*default_max=*/8);
    writer.Close();
  }
  ASSERT_EQ(shutdown(fds[0], SHUT_WR), 0);
  std::string raw;
  char chunk[1024];
  for (;;) {
    const ssize_t count = read(fds[1], chunk, sizeof(chunk));
    if (count <= 0) {
      break;
    }
    raw.append(chunk, static_cast<std::size_t>(count));
  }
  close(fds[0]);
  close(fds[1]);
  const std::size_t split = raw.find("\r\n\r\n");
  ASSERT_NE(split, std::string::npos);
  EXPECT_EQ(raw.substr(0, 12), "HTTP/1.1 200");
  auto response = tessera::core::Json::Parse(
      std::string_view(raw).substr(split + 4));
  ASSERT_TRUE(response != nullptr);
  const tessera::core::Json* choices = response->Find("choices");
  ASSERT_TRUE(choices != nullptr && choices->isArray());
  ASSERT_EQ(choices->AsArray().size(), 1u);
  const tessera::core::Json* text = choices->AsArray()[0].Find("text");
  ASSERT_TRUE(text != nullptr && text->isString());
  EXPECT_EQ(text->AsString(), *want);
  const tessera::core::Json* reason =
      choices->AsArray()[0].Find("finish_reason");
  ASSERT_TRUE(reason != nullptr && reason->isString());
  EXPECT_EQ(reason->AsString(), "length");
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
  // Prefill reports both windows: the end-to-end serving number and the
  // forwards-only compute number nested inside it.
  EXPECT_GT(count->prefill_ms, 0.0);
  EXPECT_GT(count->prefill_compute_ms, 0.0);
  EXPECT_LE(count->prefill_compute_ms, count->prefill_ms);
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

// The MXFP4 weight map honors the MTP head selection, so the loader can
// skip the nextn tensors when no attached strategy reads them.
TEST(EngineTest, WeightMapSkipsMtpHeadWhenNotWanted) {
  auto arch = tessera::CreateArchitecture("qwen35");
  ASSERT_NE(arch, nullptr);
  tessera::TransformerConfig config;
  config.layers = 64;
  tessera::WeightLoadOptions with_head;
  tessera::WeightLoadOptions without_head;
  without_head.include_mtp_head = false;
  auto mapped = arch->MapWeightName("mtp.fc.weight", config, with_head);
  ASSERT_TRUE(mapped.has_value());
  EXPECT_EQ(*mapped, "blk.64.nextn.eh_proj.weight");
  EXPECT_FALSE(
      arch->MapWeightName("mtp.fc.weight", config, without_head).has_value());
  const char* trunk =
      "model.language_model.layers.3.self_attn.q_proj.weight";
  auto trunk_with = arch->MapWeightName(trunk, config, with_head);
  auto trunk_without = arch->MapWeightName(trunk, config, without_head);
  ASSERT_TRUE(trunk_with.has_value());
  ASSERT_TRUE(trunk_without.has_value());
  EXPECT_EQ(*trunk_with, "blk.3.attn_q.weight");
  EXPECT_EQ(*trunk_without, "blk.3.attn_q.weight");
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
  // A quantized KV cache must take the same MTP path. The MTP head's KV
  // cache shares the cache type; before that was set the first MTP step
  // failed with a device error. The quantized cache is lossy by design,
  // so compare the speculative output with its own greedy baseline.
  options.kv_type = tessera::KvCacheType::Q8;
  auto quantized_greedy = engine->Generate(**model, options);
  ASSERT_TRUE(quantized_greedy.has_value())
      << tessera::ToString(quantized_greedy.error());
  auto quantized_spec = engine->GenerateSpeculative(**model, options);
  ASSERT_TRUE(quantized_spec.has_value())
      << tessera::ToString(quantized_spec.error());
  EXPECT_EQ(*quantized_spec, *quantized_greedy);
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

TEST(EngineTest, GpuLayerResidentSplitsLastLayers) {
  using tessera::core::GpuLayerResident;
  constexpr std::size_t kTrunk = 40;
  // Zero keeps everything: the full-upload default.
  EXPECT_TRUE(GpuLayerResident("blk.0.attn_q.weight", 0, kTrunk));
  EXPECT_TRUE(GpuLayerResident("blk.39.ffn_down.weight", 0, kTrunk));
  // One keeps the head only: no trunk block stays resident.
  EXPECT_TRUE(GpuLayerResident("output.weight", 1, kTrunk));
  EXPECT_TRUE(GpuLayerResident("token_embd.weight", 1, kTrunk));
  EXPECT_TRUE(GpuLayerResident("blk.40.nextn.eh_proj.weight", 1, kTrunk));
  EXPECT_FALSE(GpuLayerResident("blk.39.attn_q.weight", 1, kTrunk));
  EXPECT_FALSE(GpuLayerResident("blk.0.attn_q.weight", 1, kTrunk));
  // Twenty keeps the output plus the last 19 blocks (21 through 39).
  EXPECT_FALSE(GpuLayerResident("blk.20.attn_q.weight", 20, kTrunk));
  EXPECT_TRUE(GpuLayerResident("blk.21.attn_q.weight", 20, kTrunk));
  EXPECT_TRUE(GpuLayerResident("blk.39.ffn_down_exps.weight", 20, kTrunk));
  // Past the layer count keeps everything resident.
  EXPECT_TRUE(GpuLayerResident("blk.0.attn_q.weight", 41, kTrunk));
  EXPECT_TRUE(GpuLayerResident("blk.0.attn_q.weight", 999, kTrunk));
  // An unknown trunk count keeps everything: blocks cannot be ordered.
  EXPECT_TRUE(GpuLayerResident("blk.3.attn_q.weight", 5, 0));
  // Non-block globals stay resident at any setting.
  EXPECT_TRUE(GpuLayerResident("position_embd.weight", 1, kTrunk));
}

TEST(EngineTest, RoutedExpertKindMatchesSuffix) {
  using tessera::core::IsRoutedExpertKind;
  EXPECT_TRUE(IsRoutedExpertKind("ffn_gate_exps.weight"));
  EXPECT_TRUE(IsRoutedExpertKind("ffn_up_exps.weight"));
  EXPECT_TRUE(IsRoutedExpertKind("ffn_down_exps.weight"));
  EXPECT_TRUE(IsRoutedExpertKind("blk.0.ffn_gate_exps.weight"));
  // Dense weights never match: shared experts, router, attention, head.
  EXPECT_FALSE(IsRoutedExpertKind("ffn_gate_shexp.weight"));
  EXPECT_FALSE(IsRoutedExpertKind("ffn_gate_inp.weight"));
  EXPECT_FALSE(IsRoutedExpertKind("attn_q.weight"));
  EXPECT_FALSE(IsRoutedExpertKind("output.weight"));
  EXPECT_FALSE(IsRoutedExpertKind(""));
}

// The MoE dense refinement keeps every non-expert tensor of a paged
// trunk block resident, so only routed experts page per token. Dense
// models, full uploads, and already-resident names never match.
TEST(EngineTest, PagedMoeDenseResidentKeepsMoeDense) {
  using tessera::core::PagedMoeDenseResident;
  constexpr std::size_t kTrunk = 40;
  // A paged MoE block keeps its dense tensors: attention, norms, router
  // and shared experts.
  EXPECT_TRUE(PagedMoeDenseResident("blk.0.attn_q.weight", 20, kTrunk, true));
  EXPECT_TRUE(PagedMoeDenseResident("blk.0.post_attention_norm.weight", 20,
                                    kTrunk, true));
  EXPECT_TRUE(
      PagedMoeDenseResident("blk.0.ffn_gate_inp.weight", 20, kTrunk, true));
  EXPECT_TRUE(
      PagedMoeDenseResident("blk.0.ffn_gate_shexp.weight", 20, kTrunk, true));
  // Routed experts still page.
  EXPECT_FALSE(
      PagedMoeDenseResident("blk.0.ffn_gate_exps.weight", 20, kTrunk, true));
  EXPECT_FALSE(
      PagedMoeDenseResident("blk.0.ffn_down_exps.weight", 20, kTrunk, true));
  // Already-resident names are not double counted: resident blocks, the
  // MTP block, the head and non-block globals.
  EXPECT_FALSE(PagedMoeDenseResident("blk.39.attn_q.weight", 20, kTrunk, true));
  EXPECT_FALSE(PagedMoeDenseResident("blk.40.attn_q.weight", 20, kTrunk, true));
  EXPECT_FALSE(PagedMoeDenseResident("output.weight", 20, kTrunk, true));
  EXPECT_FALSE(PagedMoeDenseResident("position_embd.weight", 20, kTrunk, true));
  // Dense models never match, so their paging is unchanged.
  EXPECT_FALSE(PagedMoeDenseResident("blk.0.attn_q.weight", 20, kTrunk, false));
  // Full uploads and unknown trunk counts never match.
  EXPECT_FALSE(PagedMoeDenseResident("blk.0.attn_q.weight", 0, kTrunk, true));
  EXPECT_FALSE(PagedMoeDenseResident("blk.0.attn_q.weight", 20, 0, true));
}

// KV cache math pins the AppendKv growth: one F32 row per key and value,
// geometric capacity, per-row scales only for quantized types.
TEST(EngineTest, FullKvLayerBytesFollowsAppendKv) {
  tessera::TransformerConfig config;
  config.attention.kv_heads = 2;
  config.attention.head_dim = 64;
  // kv_dim 128, capacity 1024 at context 1024.
  EXPECT_EQ(tessera::core::FullKvLayerBytes(config, tessera::KvCacheType::F32,
                                            1024),
            2u * 1024u * 128u * 4u);
  EXPECT_EQ(tessera::core::FullKvLayerBytes(config, tessera::KvCacheType::Q8,
                                            1024),
            2u * 1024u * 128u + 2u * 1024u * 4u);
  EXPECT_EQ(
      tessera::core::FullKvLayerBytes(config, tessera::KvCacheType::F32, 0),
      0u);
}

// Linear state is context independent; dense configs carry none.
TEST(EngineTest, LinearStateLayerBytesNeedsHybrid) {
  tessera::TransformerConfig dense;
  EXPECT_EQ(tessera::core::LinearStateLayerBytes(dense), 0u);
  tessera::TransformerConfig hybrid;
  hybrid.hybrid = true;
  hybrid.ssm.conv_kernel = 4;
  hybrid.ssm.state_size = 128;
  hybrid.ssm.group_count = 16;
  hybrid.ssm.time_step_rank = 32;
  hybrid.ssm.inner_size = 4096;
  // hist 8192 x 3 x 4 plus state 32 x 128 x 128 x 4.
  EXPECT_EQ(tessera::core::LinearStateLayerBytes(hybrid),
            8192u * 3u * 4u + 32u * 128u * 128u * 4u);
}

// The layer advisor on the 2-layer tiny fixture: total layers, full fit
// at ample memory, paging below it, and no advice without free bytes.
TEST(EngineTest, RecommendGpuLayersRanksResidentSets) {
  using tessera::core::RecommendGpuLayers;
  using tessera::core::ResidentWeightBytes;
  auto path = WriteTinyModelFixture("budget.gguf", 2);
  auto full = RecommendGpuLayers(path.string(), 1024, tessera::KvCacheType::F32,
                                 1ull << 60, "none");
  ASSERT_TRUE(full.has_value()) << tessera::ToString(full.error());
  EXPECT_EQ(full->trunk_layers, 2u);
  EXPECT_EQ(full->total_layers, 3u);
  EXPECT_EQ(full->full_layers, 2u);
  EXPECT_EQ(full->linear_layers, 0u);
  EXPECT_GT(full->full_bytes, 0u);
  EXPECT_GT(full->kv_bytes, 0u);
  EXPECT_TRUE(full->has_best);
  EXPECT_EQ(full->best, 0u);
  EXPECT_EQ(full->best_bytes, full->full_bytes);
  auto budget = tessera::core::MeasureWeights(path.string(), true);
  ASSERT_TRUE(budget.has_value()) << tessera::ToString(budget.error());
  const std::uint64_t all = ResidentWeightBytes(*budget, 0);
  const std::uint64_t two = ResidentWeightBytes(*budget, 2);
  const std::uint64_t one = ResidentWeightBytes(*budget, 1);
  EXPECT_EQ(all, full->full_bytes);
  EXPECT_LT(one, two);
  EXPECT_LE(two, all);
  // Just below the full fit: paging becomes the advice.
  auto paged = RecommendGpuLayers(path.string(), 1024,
                                  tessera::KvCacheType::F32,
                                  full->full_bytes + full->kv_bytes +
                                      (1024ull << 20) - 1,
                                  "none");
  ASSERT_TRUE(paged.has_value()) << tessera::ToString(paged.error());
  EXPECT_TRUE(paged->has_best);
  EXPECT_GE(paged->best, 1u);
  EXPECT_LT(paged->best_bytes, full->full_bytes);
  // Without reported memory there is no recommendation.
  auto unknown = RecommendGpuLayers(path.string(), 1024,
                                    tessera::KvCacheType::F32, 0, "none");
  ASSERT_TRUE(unknown.has_value()) << tessera::ToString(unknown.error());
  EXPECT_FALSE(unknown->has_best);
  EXPECT_TRUE(RecommendGpuLayers("/nonexistent.gguf", 1024,
                                 tessera::KvCacheType::F32, 1ull << 60,
                                 "none")
                  .error() == tessera::StatusCode::FileNotFound);
}

// The fit check judges the requested value: it fits at ample memory, it
// refuses below resident plus cache plus headroom, and larger N costs
// more resident bytes.
TEST(EngineTest, FitGpuLayersJudgesRequestedValue) {
  using tessera::core::FitGpuLayers;
  auto path = WriteTinyModelFixture("fit.gguf", 2);
  auto full = FitGpuLayers(path.string(), 1024, tessera::KvCacheType::F32,
                           1ull << 60, "none", 2);
  ASSERT_TRUE(full.has_value()) << tessera::ToString(full.error());
  EXPECT_EQ(full->total_layers, 3u);
  EXPECT_TRUE(full->fits);
  EXPECT_FALSE(full->budget_is_total);
  EXPECT_GT(full->resident_bytes, 0u);
  EXPECT_GT(full->kv_bytes, 0u);
  auto one = FitGpuLayers(path.string(), 1024, tessera::KvCacheType::F32,
                          1ull << 60, "none", 1);
  ASSERT_TRUE(one.has_value()) << tessera::ToString(one.error());
  EXPECT_TRUE(one->fits);
  EXPECT_LT(one->resident_bytes, full->resident_bytes);
  // One byte below resident plus cache plus the 1024 MiB reserve refuses.
  const std::uint64_t tight =
      one->resident_bytes + one->kv_bytes + (1024ull << 20) - 1;
  auto refused = FitGpuLayers(path.string(), 1024, tessera::KvCacheType::F32,
                              tight, "none", 1);
  ASSERT_TRUE(refused.has_value()) << tessera::ToString(refused.error());
  EXPECT_FALSE(refused->fits);
  EXPECT_TRUE(FitGpuLayers("/nonexistent.gguf", 1024,
                           tessera::KvCacheType::F32, 1ull << 60, "none", 1)
                  .error() == tessera::StatusCode::FileNotFound);
}
