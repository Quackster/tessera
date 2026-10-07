#include <gtest/gtest.h>

#include "test_helpers.hpp"
#include "tessera/engine.hpp"
#include "tessera/speculative.hpp"
#include "tessera/types.hpp"

using tessera::Engine;
using tessera::EngineOptions;
using tessera::ModelFormat;
using tessera::ModelOptions;
using tessera::StatusCode;
using tessera::testing::FreshTempDir;
using tessera::testing::GgufBuilder;
using tessera::testing::MakeValidGguf;
using tessera::testing::WriteBytes;
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
  EXPECT_EQ(loaded->Tensors().size(), 0u);  // manifest in milestone 6
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
