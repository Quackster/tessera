#include <gtest/gtest.h>

#include <cstdlib>
#include <random>
#include <vector>

#include "core/decode.hpp"
#include "test_helpers.hpp"
#include "tessera/engine.hpp"
#include "tessera/speculative.hpp"
#include "tessera/types.hpp"

using tessera::Engine;
using tessera::EngineOptions;
using tessera::ModelFormat;
using tessera::TensorEntry;
using tessera::ModelOptions;
using tessera::StatusCode;
using tessera::TransformerConfig;
using tessera::testing::DrawValue;
using tessera::testing::FreshTempDir;
using tessera::testing::GgufBuilder;
using tessera::testing::MakeSafetensorsContainer;
using tessera::testing::MakeValidGguf;
using tessera::testing::QuantizeRows;
using tessera::testing::WriteBytes;
using tessera::testing::WriteHybridFixture;
using tessera::testing::WritePlaceholderConfig;
using tessera::testing::WritePlaceholderWeights;

namespace {

// A device is required for the engine; skip cleanly without one.
void MakeEngineOrSkip(std::unique_ptr<Engine>& engine) {
  auto created = Engine::Create(EngineOptions{});
  if (!created) {
    GTEST_SKIP()
        << "no device available: " << tessera::ToString(created.error());
  }
  engine = std::move(*created);
}

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

TEST(EngineTest, LoadGgufModelUploadsQuantizedSize) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 1);
  builder.KvString("general.name", "test-model");
  builder.Tensor("w_q", 1, {256}, 12, 0);  // Q4_K, 256 elements
  builder.PadTo(((builder.bytes.size() + 31) & ~31u));
  builder.PadPayload(144);
  auto dir = FreshTempDir("tessera_tests_weights_q4k");
  auto path = dir / "weights_q4k.gguf";
  WriteBytes(path, builder.bytes);
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  ASSERT_EQ((*model)->Weights().size(), 1u);
  EXPECT_EQ((*model)->Weights()[0].device->Size(), 144u);
}

TEST(EngineTest, LoadGgufModelRejectsUnsizedLayout) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 1);
  builder.KvString("general.name", "test-model");
  builder.Tensor("w_q2", 1, {256}, 10, 0);  // Q2_K has no sized layout
  builder.PadTo(((builder.bytes.size() + 31) & ~31u));
  auto dir = FreshTempDir("tessera_tests_weights_q2k");
  auto path = dir / "weights_q2k.gguf";
  WriteBytes(path, builder.bytes);
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_FALSE(model.has_value());
  EXPECT_EQ(model.error(), StatusCode::UnsupportedFeature);
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

TEST(EngineTest, RealModelLoadPathWhenProvided) {
  const char* raw = std::getenv("TESSERA_TEST_GGUF");
  if (raw == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_GGUF not set";
  }
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{raw, 1024});
  // The first-class target is a hybrid attention/SSM model: the load
  // succeeds and decode reports UnsupportedFeature until the
  // recurrent kernels land (see docs/PROGRESS.md).
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
  ASSERT_FALSE(step.has_value());
  EXPECT_EQ(step.error(), StatusCode::UnsupportedFeature);
}

TEST(EngineTest, LoadMxFp4BlobAndScalePair) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  // A 4x32 MXFP4 weight (64 blob bytes) with its 4x1 E8M0 scales.
  std::string json =
      R"({"w.weight":{"dtype":"U8","shape":[4,16],"data_offsets":[0,64]},)"
      R"("w.weight_scale":{"dtype":"U8","shape":[4,1],)"
      R"("data_offsets":[64,68]}})";
  auto container = MakeSafetensorsContainer(json);
  container.insert(container.end(), 68, std::byte{0});
  auto dir = FreshTempDir("tessera_tests_mxfp4_pair");
  WritePlaceholderConfig(dir);
  WriteBytes(dir / "model.safetensors", container);
  auto model = engine->LoadModel(ModelOptions{dir.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto& loaded = *model;
  ASSERT_EQ(loaded->Tensors().size(), 2u);
  const TensorEntry* blob = nullptr;
  const TensorEntry* scale = nullptr;
  for (const auto& entry : loaded->Tensors()) {
    if (entry.name == "w.weight") {
      blob = &entry;
    } else if (entry.name == "w.weight_scale") {
      scale = &entry;
    }
  }
  ASSERT_NE(blob, nullptr);
  ASSERT_NE(scale, nullptr);
  EXPECT_EQ(blob->dtype, tessera::DType::F4E2M1);
  ASSERT_EQ(blob->shape.rank, 2u);
  EXPECT_EQ(blob->shape.dims[0], 4u);
  EXPECT_EQ(blob->shape.dims[1], 32u);
  EXPECT_EQ(scale->dtype, tessera::DType::F8E8M0);
  EXPECT_EQ(loaded->FindWeight("w.weight")->Size(), 64u);
  EXPECT_EQ(loaded->FindWeight("w.weight_scale")->Size(), 4u);
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

TEST(EngineTest, LoadGgufModelAttentionMissingKeys) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteGgufFixture("model.gguf");
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto params = (*model)->Attention();
  ASSERT_FALSE(params.has_value());
  EXPECT_EQ(params.error(), StatusCode::MalformedFile);
}

TEST(EngineTest, LoadGgufModelAttentionBadValues) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 7);
  builder.KvString("general.name", "test-model");
  builder.KvString("general.architecture", "test-arch");
  builder.KvU32("test-arch.attention.head_count", 0);
  builder.KvU32("test-arch.attention.head_count_kv", 2);
  builder.KvU32("test-arch.embedding_length", 256);
  builder.KvU32("test-arch.rope.dimension_count", 16);
  builder.KvF32("test-arch.rope.freq_base", 10000.0f);
  builder.Tensor("w_a", 1, {4}, 0, 0);
  builder.PadTo(((builder.bytes.size() + 31) & ~31u) + 16);
  auto dir = FreshTempDir("tessera_tests_attention_bad");
  auto path = dir / "bad_attn.gguf";
  WriteBytes(path, builder.bytes);
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_FALSE(model.has_value());
  EXPECT_EQ(model.error(), StatusCode::MalformedFile);
}

TEST(EngineTest, LoadGgufModelAttentionExplicitHeadDim) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 9);
  builder.KvString("general.name", "test-model");
  builder.KvString("general.architecture", "test-hybrid");
  builder.KvU32("test-hybrid.attention.head_count", 24);
  builder.KvU32("test-hybrid.attention.head_count_kv", 4);
  builder.KvU32("test-hybrid.embedding_length", 5120);
  builder.KvU32("test-hybrid.attention.key_length", 256);
  builder.KvU32("test-hybrid.attention.value_length", 256);
  builder.KvU32("test-hybrid.rope.dimension_count", 64);
  builder.KvF32("test-hybrid.rope.freq_base", 10000000.0f);
  builder.Tensor("w_a", 1, {4}, 0, 0);
  builder.PadTo(((builder.bytes.size() + 31) & ~31u) + 16);
  auto dir = FreshTempDir("tessera_tests_explicit_headdim");
  auto path = dir / "headdim.gguf";
  WriteBytes(path, builder.bytes);
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto params = (*model)->Attention();
  ASSERT_TRUE(params.has_value()) << tessera::ToString(params.error());
  EXPECT_EQ(params->heads, 24u);
  EXPECT_EQ(params->kv_heads, 4u);
  EXPECT_EQ(params->head_dim, 256u);
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

TEST(EngineTest, LoadGgufModelHybridIncompleteKeys) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteHybridFixture("hybrid_bad.gguf", false);
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_FALSE(model.has_value());
  EXPECT_EQ(model.error(), StatusCode::MalformedFile);
}

TEST(EngineTest, LoadGgufModelHybridMissingSections) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteHybridFixture("hybrid_nosections.gguf", true, false);
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_FALSE(model.has_value());
  EXPECT_EQ(model.error(), StatusCode::MalformedFile);
}

TEST(EngineTest, HybridDecodeReturnsUnsupported) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto path = WriteHybridFixture("hybrid_decode.gguf", true);
  auto model = engine->LoadModel(ModelOptions{path.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  tessera::core::DecodeCache cache;
  auto step = tessera::core::DecodeStep(engine->Owner(), **model, cache, 0);
  ASSERT_FALSE(step.has_value());
  EXPECT_EQ(step.error(), StatusCode::UnsupportedFeature);
}

TEST(EngineTest, LoadModelAttentionMxFp4Unsupported) {

  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto dir = FreshTempDir("tessera_tests_attention_mxfp4");
  WritePlaceholderConfig(dir);
  WritePlaceholderWeights(dir);
  auto model = engine->LoadModel(ModelOptions{dir.string(), 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  auto params = (*model)->Attention();
  ASSERT_FALSE(params.has_value());
  EXPECT_EQ(params.error(), StatusCode::UnsupportedFeature);
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

TEST(EngineTest, AttachSpeculativeNull) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto result = engine->AttachSpeculative(nullptr);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), StatusCode::InvalidArgument);
}

TEST(EngineTest, AttachSpeculativeTwice) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto first = engine->AttachSpeculative(tessera::CreateDFlash2Strategy());
  ASSERT_TRUE(first.has_value()) << tessera::ToString(first.error());
  auto second = engine->AttachSpeculative(tessera::CreateDFlash2Strategy());
  ASSERT_FALSE(second.has_value());
  EXPECT_EQ(second.error(), StatusCode::InvalidArgument);
}

TEST(EngineTest, AttachSpeculativeValid) {
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto draft_dir = FreshTempDir("tessera_tests_engine_draft");
  WritePlaceholderConfig(draft_dir);
  WritePlaceholderWeights(draft_dir);
  auto strategy = tessera::CreateDFlash2Strategy();
  ASSERT_TRUE(strategy->Attach(tessera::StrategyOptions{draft_dir.string(), 4})
                  .has_value());
  auto attached = engine->AttachSpeculative(std::move(strategy));
  ASSERT_TRUE(attached.has_value()) << tessera::ToString(attached.error());
  ASSERT_NE(engine->Speculative(), nullptr);
  EXPECT_EQ(engine->Speculative()->Name(), "dflash2");
}
