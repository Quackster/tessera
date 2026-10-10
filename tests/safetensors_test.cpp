#include <gtest/gtest.h>

#include <utility>

#include "core/files.hpp"
#include "core/loaders/safetensors.hpp"
#include "test_helpers.hpp"

using tessera::core::InspectMxFp4Directory;
using tessera::core::InspectSafetensorsFile;
using tessera::core::ParseSafetensorsMap;
using tessera::StatusCode;
using tessera::testing::FreshTempDir;
using tessera::testing::MakeSafetensorsContainer;
using tessera::testing::WriteBytes;
using tessera::testing::WriteString;

namespace {

// The JSON literal used in these tests is 31 bytes long.
constexpr std::uint64_t kPlaceholderJsonLen = 31;
// The placeholder weight map holds one F32 tensor.
constexpr std::uint64_t kPlaceholderMapLen = 55;

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




TEST(SafetensorsTest, AcceptsValidDirectory) {
  auto dir = tessera::testing::FreshTempDir("tessera_tests_safetensors");
  tessera::testing::WritePlaceholderConfig(dir);
  tessera::testing::WritePlaceholderWeights(dir);
  auto layout = InspectMxFp4Directory(dir);
  ASSERT_TRUE(layout.has_value()) << tessera::ToString(layout.error());
  EXPECT_EQ(layout->weights_path, (dir / "model.safetensors").string());
  EXPECT_EQ(layout->header_len, kPlaceholderMapLen);
}

TEST(SafetensorsTest, ParsesTensorMap) {
  const std::string json =
      R"({"__metadata__":{"format":"pt"},)"
      R"("w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},)"
      R"("m":{"dtype":"BF16","shape":[2,2],"data_offsets":[16,24]}})";
  auto bytes = MakeSafetensorsContainer(json);
  bytes.insert(bytes.end(), 24, std::byte{0});
  auto map = ParseSafetensorsMap(std::span<const std::byte>(bytes));
  ASSERT_TRUE(map.has_value()) << tessera::ToString(map.error());
  ASSERT_EQ(map->size(), 2u);
  EXPECT_EQ((*map)[0].name, "w");
  EXPECT_EQ((*map)[0].entry.dtype, tessera::DType::F32);
  EXPECT_EQ((*map)[0].begin, 8u + json.size());
  EXPECT_EQ((*map)[0].end, 8u + json.size() + 16);
  EXPECT_EQ((*map)[1].entry.dtype, tessera::DType::BF16);
  ASSERT_EQ((*map)[1].entry.shape.rank, 2u);
  EXPECT_EQ((*map)[1].entry.shape.dims[1], 2u);
}

TEST(SafetensorsTest, ParsesMxFp4BlobPair) {
  const std::string json =
      R"({"w.weight":{"dtype":"U8","shape":[4,16],"data_offsets":[0,64]},)"
      R"("w.weight_scale":{"dtype":"U8","shape":[4,1],)"
      R"("data_offsets":[64,68]}})";
  auto bytes = MakeSafetensorsContainer(json);
  bytes.insert(bytes.end(), 68, std::byte{0});
  auto map = ParseSafetensorsMap(std::span<const std::byte>(bytes));
  ASSERT_TRUE(map.has_value()) << tessera::ToString(map.error());
  ASSERT_EQ(map->size(), 2u);
  EXPECT_EQ((*map)[0].entry.dtype, tessera::DType::F4E2M1);
  ASSERT_EQ((*map)[0].entry.shape.rank, 2u);
  EXPECT_EQ((*map)[0].entry.shape.dims[0], 4u);
  EXPECT_EQ((*map)[0].entry.shape.dims[1], 32u);
  EXPECT_EQ((*map)[1].entry.dtype, tessera::DType::F8E8M0);
}

TEST(SafetensorsTest, RejectsMapWithUnknownDtype) {
  auto bytes = MakeSafetensorsContainer(
      R"({"w":{"dtype":"F64","shape":[4],"data_offsets":[0,32]}})");
  bytes.insert(bytes.end(), 32, std::byte{0});
  auto map = ParseSafetensorsMap(std::span<const std::byte>(bytes));
  ASSERT_FALSE(map.has_value());
  EXPECT_EQ(map.error(), StatusCode::UnsupportedFeature);
}

TEST(SafetensorsTest, RejectsMapWithBadOffsets) {
  auto bytes = MakeSafetensorsContainer(
      R"({"w":{"dtype":"F32","shape":[4],"data_offsets":[8,16]}})");
  auto map = ParseSafetensorsMap(std::span<const std::byte>(bytes));
  ASSERT_FALSE(map.has_value());
  EXPECT_EQ(map.error(), StatusCode::MalformedFile);
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



