#include <gtest/gtest.h>

#include "core/loaders/safetensors.hpp"
#include "test_helpers.hpp"

using tessera::core::InspectMxFp4Directory;
using tessera::core::InspectSafetensorsFile;
using tessera::StatusCode;
using tessera::testing::FreshTempDir;
using tessera::testing::MakeSafetensorsContainer;
using tessera::testing::WriteBytes;
using tessera::testing::WriteString;

namespace {

// The JSON literal used in these tests is 31 bytes long.
constexpr std::uint64_t kPlaceholderJsonLen = 31;

}  // namespace

TEST(SafetensorsTest, AcceptsValidContainer) {
  auto dir = tessera::testing::FreshTempDir("tessera_tests_safetensors");
  auto path = dir / "model.safetensors";
  WriteBytes(path,
             MakeSafetensorsContainer(R"({"model":{"a":{"dtype":"F32"}}})"));
  auto check = InspectSafetensorsFile(path);
  ASSERT_TRUE(check.has_value()) << tessera::ToString(check.error());
  EXPECT_EQ(check->header_len, kPlaceholderJsonLen);
  EXPECT_EQ(check->file_size, 8u + kPlaceholderJsonLen);
}

TEST(SafetensorsTest, RejectsMissingFile) {
  auto dir = tessera::testing::FreshTempDir("tessera_tests_safetensors");
  auto check = InspectSafetensorsFile(dir / "absent.safetensors");
  ASSERT_FALSE(check.has_value());
  EXPECT_EQ(check.error(), StatusCode::FileNotFound);
}

TEST(SafetensorsTest, RejectsFileTooSmall) {
  auto dir = tessera::testing::FreshTempDir("tessera_tests_safetensors");
  auto path = dir / "small.safetensors";
  WriteBytes(path, {std::byte{0}, std::byte{0}, std::byte{'{'}, std::byte{'}'}});
  auto check = InspectSafetensorsFile(path);
  ASSERT_FALSE(check.has_value());
  EXPECT_EQ(check.error(), StatusCode::MalformedFile);
}

TEST(SafetensorsTest, RejectsZeroHeaderLen) {
  auto dir = tessera::testing::FreshTempDir("tessera_tests_safetensors");
  auto path = dir / "zerolen.safetensors";
  WriteBytes(path, MakeSafetensorsContainer("{}", true, 0));
  auto check = InspectSafetensorsFile(path);
  ASSERT_FALSE(check.has_value());
  EXPECT_EQ(check.error(), StatusCode::MalformedFile);
}

TEST(SafetensorsTest, RejectsHeaderLenBeyondEnd) {
  auto dir = tessera::testing::FreshTempDir("tessera_tests_safetensors");
  auto path = dir / "long.safetensors";
  // Claimed header length far exceeds the actual file.
  WriteBytes(path, MakeSafetensorsContainer("{}", true, 4096));
  auto check = InspectSafetensorsFile(path);
  ASSERT_FALSE(check.has_value());
  EXPECT_EQ(check.error(), StatusCode::MalformedFile);
}

TEST(SafetensorsTest, RejectsHeaderNotJson) {
  auto dir = tessera::testing::FreshTempDir("tessera_tests_safetensors");
  auto path = dir / "notjson.safetensors";
  WriteBytes(path, MakeSafetensorsContainer("[\"model\"]"));
  auto check = InspectSafetensorsFile(path);
  ASSERT_FALSE(check.has_value());
  EXPECT_EQ(check.error(), StatusCode::MalformedFile);
}

TEST(SafetensorsTest, AcceptsValidDirectory) {
  auto dir = tessera::testing::FreshTempDir("tessera_tests_safetensors");
  tessera::testing::WritePlaceholderConfig(dir);
  tessera::testing::WritePlaceholderWeights(dir);
  auto layout = InspectMxFp4Directory(dir);
  ASSERT_TRUE(layout.has_value()) << tessera::ToString(layout.error());
  EXPECT_EQ(layout->weights_path, (dir / "model.safetensors").string());
  EXPECT_EQ(layout->header_len, kPlaceholderJsonLen);
}

TEST(SafetensorsTest, RejectsMissingDirectory) {
  auto dir = tessera::testing::FreshTempDir("tessera_tests_safetensors");
  auto layout = InspectMxFp4Directory(dir / "absent");
  ASSERT_FALSE(layout.has_value());
  EXPECT_EQ(layout.error(), StatusCode::FileNotFound);
}

TEST(SafetensorsTest, RejectsDirectoryWithoutConfig) {
  auto dir = tessera::testing::FreshTempDir("tessera_tests_safetensors");
  tessera::testing::WritePlaceholderWeights(dir);
  auto layout = InspectMxFp4Directory(dir);
  ASSERT_FALSE(layout.has_value());
  EXPECT_EQ(layout.error(), StatusCode::MalformedFile);
}

TEST(SafetensorsTest, RejectsDirectoryWithoutWeights) {
  auto dir = tessera::testing::FreshTempDir("tessera_tests_safetensors");
  tessera::testing::WritePlaceholderConfig(dir);
  auto layout = InspectMxFp4Directory(dir);
  ASSERT_FALSE(layout.has_value());
  EXPECT_EQ(layout.error(), StatusCode::MalformedFile);
}

TEST(SafetensorsTest, RejectsDirectoryWithMultipleWeights) {
  auto dir = tessera::testing::FreshTempDir("tessera_tests_safetensors");
  tessera::testing::WritePlaceholderConfig(dir);
  tessera::testing::WritePlaceholderWeights(dir, "a.safetensors");
  tessera::testing::WritePlaceholderWeights(dir, "b.safetensors");
  auto layout = InspectMxFp4Directory(dir);
  ASSERT_FALSE(layout.has_value());
  EXPECT_EQ(layout.error(), StatusCode::MalformedFile);
}

TEST(SafetensorsTest, RejectsDirectoryWithBrokenWeight) {
  auto dir = tessera::testing::FreshTempDir("tessera_tests_safetensors");
  tessera::testing::WritePlaceholderConfig(dir);
  WriteBytes(dir / "model.safetensors",
             {std::byte{0}, std::byte{0}, std::byte{'x'}});
  auto layout = InspectMxFp4Directory(dir);
  ASSERT_FALSE(layout.has_value());
  EXPECT_EQ(layout.error(), StatusCode::MalformedFile);
}
