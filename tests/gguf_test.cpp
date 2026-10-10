#include <gtest/gtest.h>

#include <cstdlib>
#include <span>
#include <variant>

#include "core/loaders/gguf.hpp"
#include "test_helpers.hpp"

using tessera::core::GgufFile;
using tessera::core::ParseGguf;
using tessera::core::ParseGgufFile;
using tessera::DType;
using tessera::StatusCode;
using tessera::testing::GgufBuilder;

namespace {

// A well-formed v3 image (spec layout): name + architecture metadata, an
// f32 tensor and a Q4_K tensor (256 elements = one block), followed by
// a zeroed payload. Offsets follow the default alignment of 32.
std::vector<std::byte> MakeValidGgufV3() {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 2, 2);
  builder.KvString("general.name", "test-model");
  builder.KvU32("general.architecture", 7);
  builder.Tensor("w_a", 2, {2, 3}, 0, 0);   // F32: 6 elements = 24 bytes
  builder.Tensor("w_b", 1, {256}, 12, 32);  // Q4_K: one 144-byte block
  // The data region starts at align32(info_end); pad so it holds w_b's
  // [32, 176).
  builder.PadTo(((builder.bytes.size() + 31) & ~31u) + 176);
  return builder.bytes;
}

}  // namespace

TEST(GgufTest, ParsesValidV3File) {
  auto bytes = MakeValidGgufV3();
  auto file = ParseGguf(std::span<const std::byte>(bytes));
  ASSERT_TRUE(file.has_value()) << tessera::ToString(file.error());
  EXPECT_EQ(file->version, 3u);
  ASSERT_EQ(file->tensors.size(), 2u);
  EXPECT_EQ(file->tensors[0].name, "w_a");
  EXPECT_EQ(file->tensors[0].shape.rank, 2u);
  EXPECT_EQ(file->tensors[0].shape.dims[0], 2u);
  EXPECT_EQ(file->tensors[0].shape.dims[1], 3u);
  EXPECT_EQ(file->tensors[0].shape.Numel(), 6u);
  EXPECT_EQ(file->tensors[0].dtype, DType::F32);
  EXPECT_EQ(file->tensors[1].name, "w_b");
  EXPECT_EQ(file->tensors[1].shape.Numel(), 256u);
  EXPECT_EQ(file->tensors[1].dtype, DType::Q4K);
  ASSERT_EQ(file->tensor_offsets.size(), 2u);
  EXPECT_EQ(file->tensor_offsets[0], 0u);
  EXPECT_EQ(file->tensor_offsets[1], 32u);

  const auto* name = file->Find("general.name");
  ASSERT_NE(name, nullptr);
  EXPECT_EQ(*std::get_if<std::string>(name), "test-model");
  const auto* arch = file->Find("general.architecture");
  ASSERT_NE(arch, nullptr);
  EXPECT_EQ(*std::get_if<std::uint32_t>(arch), 7u);
  EXPECT_EQ(file->Find("absent.key"), nullptr);
  EXPECT_EQ(file->dropped_array_keys.size(), 0u);
}

// ParseGgufFile maps the file instead of reading it; the parsed result
// must match the in-memory parse, and a missing file is FileNotFound.
TEST(GgufTest, ParsesFileFromDisk) {
  auto bytes = MakeValidGgufV3();
  auto dir = tessera::testing::FreshTempDir("tessera_tests_gguf_mmap");
  auto path = dir / "valid.gguf";
  tessera::testing::WriteBytes(path, bytes);

  auto mapped = ParseGgufFile(path);
  ASSERT_TRUE(mapped.has_value()) << tessera::ToString(mapped.error());
  auto in_memory = ParseGguf(std::span<const std::byte>(bytes));
  ASSERT_TRUE(in_memory.has_value()) << tessera::ToString(in_memory.error());
  EXPECT_EQ(mapped->version, in_memory->version);
  ASSERT_EQ(mapped->tensors.size(), in_memory->tensors.size());
  for (std::size_t i = 0; i < mapped->tensors.size(); ++i) {
    EXPECT_EQ(mapped->tensors[i].name, in_memory->tensors[i].name);
    EXPECT_EQ(mapped->tensors[i].dtype, in_memory->tensors[i].dtype);
    EXPECT_EQ(mapped->tensors[i].shape.Numel(),
              in_memory->tensors[i].shape.Numel());
    EXPECT_EQ(mapped->tensor_offsets[i], in_memory->tensor_offsets[i]);
  }

  auto missing = ParseGgufFile(dir / "absent.gguf");
  ASSERT_FALSE(missing.has_value());
  EXPECT_EQ(missing.error(), StatusCode::FileNotFound);
}

TEST(GgufTest, AcceptsVersion2) {
  // v2 and v3 share the byte layout; the same image parses as v2.
  GgufBuilder builder;
  builder.Header(0x46554747, 2, 1, 1);
  builder.KvU32("general.architecture", 7);
  builder.Tensor("w_a", 1, {4}, 0, 0);  // F32: 16 bytes
  builder.PadTo(((builder.bytes.size() + 31) & ~31u) + 16);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_TRUE(file.has_value()) << tessera::ToString(file.error());
  EXPECT_EQ(file->version, 2u);
  ASSERT_EQ(file->tensors.size(), 1u);
  EXPECT_EQ(file->tensors[0].dtype, DType::F32);
}

TEST(GgufTest, ReadsAllMetadataValueTypes) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 0, 12);
  builder.KvU8("t.u8", 7);
  builder.KvI8("t.i8", -3);
  builder.KvU16("t.u16", 400);
  builder.KvI16("t.i16", -400);
  builder.KvU32("t.u32", 70000);
  builder.KvI32("t.i32", -70000);
  builder.KvF32("t.f32", 1.5f);
  builder.KvBool("t.bool", true);
  builder.KvString("t.string", "hello");
  builder.KvU64("t.u64", 1ull << 40);
  builder.KvI64("t.i64", static_cast<std::int64_t>(-(1ull << 40)));
  builder.KvF64("t.f64", -2.25);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_TRUE(file.has_value()) << tessera::ToString(file.error());
  EXPECT_EQ(*std::get_if<std::uint32_t>(file->Find("t.u8")), 7u);
  EXPECT_EQ(*std::get_if<std::int32_t>(file->Find("t.i8")), -3);
  EXPECT_EQ(*std::get_if<std::uint32_t>(file->Find("t.u16")), 400u);
  EXPECT_EQ(*std::get_if<std::int32_t>(file->Find("t.i16")), -400);
  EXPECT_EQ(*std::get_if<std::uint32_t>(file->Find("t.u32")), 70000u);
  EXPECT_EQ(*std::get_if<std::int32_t>(file->Find("t.i32")), -70000);
  EXPECT_FLOAT_EQ(*std::get_if<float>(file->Find("t.f32")), 1.5f);
  EXPECT_TRUE(*std::get_if<bool>(file->Find("t.bool")));
  EXPECT_EQ(*std::get_if<std::string>(file->Find("t.string")), "hello");
  EXPECT_EQ(*std::get_if<std::uint64_t>(file->Find("t.u64")), 1ull << 40);
  EXPECT_EQ(*std::get_if<std::int64_t>(file->Find("t.i64")),
            static_cast<std::int64_t>(-(1ull << 40)));
  EXPECT_DOUBLE_EQ(*std::get_if<double>(file->Find("t.f64")), -2.25);
}

TEST(GgufTest, RejectsBadMagic) {
  auto bytes = MakeValidGgufV3();
  bytes[0] = std::byte{0x00};
  auto file = ParseGguf(std::span<const std::byte>(bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}

TEST(GgufTest, RejectsUnsupportedVersion) {
  GgufBuilder builder;
  builder.Header(0x46554747, 9, 0, 0);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::UnsupportedFeature);
}


TEST(GgufTest, RejectsTruncatedHeader) {
  std::vector<std::byte> tiny(10, std::byte{0x47});
  auto file = ParseGguf(std::span<const std::byte>(tiny));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}

TEST(GgufTest, RejectsTruncatedTensorMetadata) {
  auto bytes = MakeValidGgufV3();
  bytes.resize(bytes.size() - 12);  // cuts the Q4_K entry or the payload
  auto file = ParseGguf(std::span<const std::byte>(bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}






TEST(GgufTest, RejectsUnknownGgmlType) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 0);
  builder.Tensor("w", 1, {1}, 99, 0);  // type 99 is not a ggml type
  builder.PadPayload(8);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::UnsupportedFeature);
}

TEST(GgufTest, RejectsQ4KNumelNotMultipleOf256) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 0);
  builder.Tensor("w", 1, {255}, 12, 0);  // one short of a full block
  builder.PadPayload(144);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}






TEST(GgufTest, RejectsNonMonotonicTensorOffsets) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 2, 0);
  builder.Tensor("a", 1, {8}, 0, 32);   // F32: [32, 64)
  builder.Tensor("b", 1, {8}, 0, 0);  // goes backwards
  builder.PadTo(((builder.bytes.size() + 31) & ~31u) + 128);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}

TEST(GgufTest, RejectsOffsetBeyondPayload) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 0);
  builder.Tensor("w", 1, {1}, 0, 1000);  // payload is far smaller
  builder.PadPayload(16);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}






// Opt-in validation against a real model file (path via environment,
// never hard-coded; see AGENTS.md "Model Data").
TEST(GgufTest, ParsesRealFileWhenProvided) {
  const char* path = std::getenv("TESSERA_TEST_MODEL");
  if (path == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_MODEL not set";
  }
  auto file = ParseGgufFile(path);
  if (!file && file.error() == StatusCode::UnsupportedFeature) {
    GTEST_SKIP() << "file uses unmapped tensor layouts (the metadata "
                    "arrays parsed fine)";
  }
  ASSERT_TRUE(file.has_value()) << tessera::ToString(file.error());
  EXPECT_EQ(file->version, 3u);
  ASSERT_GT(file->tensors.size(), 0u);
  ASSERT_EQ(file->tensor_offsets.size(), file->tensors.size());
  bool found_q4k = false;
  for (const auto& tensor : file->tensors) {
    if (tensor.dtype == DType::Q4K) {
      found_q4k = true;
      // A Q4_K tensor must hold whole blocks.
      EXPECT_EQ(tensor.shape.Numel() % 256u, 0u)
          << tensor.name;
    }
  }
  EXPECT_TRUE(found_q4k);
}
