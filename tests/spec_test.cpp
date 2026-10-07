#include <gtest/gtest.h>

#include "test_helpers.hpp"
#include "tessera/speculative.hpp"
#include "tessera/types.hpp"

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

TEST(SpecTest, AttachBadBlockSize) {
  auto dir = MakeDraftDir();
  auto strategy = CreateDFlash2Strategy();
  auto too_small = strategy->Attach(StrategyOptions{dir.string(), 0});
  ASSERT_FALSE(too_small.has_value());
  EXPECT_EQ(too_small.error(), StatusCode::InvalidArgument);
  auto too_big = strategy->Attach(StrategyOptions{dir.string(), 9});
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
