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
