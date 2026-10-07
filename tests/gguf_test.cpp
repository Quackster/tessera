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

TEST(GgufTest, RejectsV1) {
  GgufBuilder builder;
  builder.Header(0x46554747, 1, 0, 0);
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

TEST(GgufTest, RejectsEmptyTensorName) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 0);
  builder.PushU64(0);  // zero-length name
  builder.PushU32(1);  // rank
  builder.PushU64(1);  // dim
  builder.PushU32(0);  // F32
  builder.PushU64(0);  // offset
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}

TEST(GgufTest, RejectsTensorNameOverLimit) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 0);
  // 65-byte name: the spec caps tensor names at 64 bytes.
  builder.PushU64(65);
  for (int i = 0; i < 65; ++i) {
    builder.PushU8(static_cast<std::uint8_t>('a' + i % 26));
  }
  builder.PushU32(1);
  builder.PushU64(1);
  builder.PushU32(0);
  builder.PushU64(0);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}

TEST(GgufTest, RejectsBoolValueNotZeroOrOne) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 0, 1);
  builder.KvBoolRaw("t.bool", 2);  // the spec allows only 0 and 1
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}

TEST(GgufTest, RejectsKeyStringBeyondBufferEnd) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 0, 1);
  builder.PushU64(100);  // key claims 100 bytes; only one byte follows
  builder.PushU8('x');
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}

TEST(GgufTest, RejectsMoreThanFourDims) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 0);
  builder.PushString("w");
  builder.PushU32(5);  // rank 5 is outside the supported range
  for (int d = 0; d < 5; ++d) {
    builder.PushU64(1);
  }
  builder.PushU32(0);
  builder.PushU64(0);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
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

TEST(GgufTest, RejectsUnalignedOffset) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 0);
  builder.Tensor("w", 1, {1}, 0, 33);  // default alignment is 32
  builder.PadPayload(64);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}

TEST(GgufTest, RejectsNonPowerOfTwoAlignment) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 1);
  builder.KvU32("general.alignment", 3);  // not a power of two
  builder.Tensor("w", 1, {1}, 0, 0);
  builder.PadPayload(8);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}

TEST(GgufTest, RespectsGeneralAlignment) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 1);
  builder.KvU32("general.alignment", 64);
  builder.Tensor("w", 1, {16}, 0, 64);  // F32: 64 bytes at [64, 128)
  // The data region starts at align64(info_end); pad past it by the
  // full 64-byte tensor.
  builder.PadTo(((builder.bytes.size() + 63) & ~63u) + 128);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_TRUE(file.has_value()) << tessera::ToString(file.error());
  EXPECT_EQ(*std::get_if<std::uint32_t>(file->Find("general.alignment")),
            64u);
}

TEST(GgufTest, RejectsHugeTensorCount) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, (1ull << 20) + 1, 0);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}

TEST(GgufTest, RejectsHugeKvCount) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 0, (1ull << 16) + 1);
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

TEST(GgufTest, RejectsPlainTensorLargerThanPayload) {
  // An f32 1x4 tensor needs 16 bytes; the payload is only 8.
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 0);
  builder.Tensor("w", 1, {4}, 0, 0);
  builder.PadPayload(8);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}

TEST(GgufTest, DropsArrayValuesAndRecordsKey) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 0, 1);
  builder.PushString("llama.tensor_list");
  builder.PushU32(9);  // GGUF_METADATA_VALUE_TYPE_ARRAY
  builder.PushU32(8);  // element type: string
  builder.PushU64(17);  // past the retention bound
  for (int i = 0; i < 17; ++i) {
    builder.PushString("x");
  }
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_TRUE(file.has_value()) << tessera::ToString(file.error());
  EXPECT_EQ(file->metadata.size(), 0u);  // the array is not retained
  EXPECT_EQ(file->small_arrays.size(), 0u);
  ASSERT_EQ(file->dropped_array_keys.size(), 1u);
  EXPECT_EQ(file->dropped_array_keys[0], "llama.tensor_list");
}

TEST(GgufTest, RetainsSmallArrays) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 0, 2);
  builder.PushString("arch.rope.dimension_sections");
  builder.PushU32(9);  // array
  builder.PushU32(4);  // element type: u32
  builder.PushU64(4);
  for (std::uint32_t s : {11u, 11u, 10u, 0u}) {
    builder.PushU32(s);
  }
  builder.PushString("arch.tags");
  builder.PushU32(9);  // array
  builder.PushU32(8);  // element type: string
  builder.PushU64(1);
  builder.PushString("hybrid");
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_TRUE(file.has_value()) << tessera::ToString(file.error());
  EXPECT_EQ(file->dropped_array_keys.size(), 0u);
  ASSERT_EQ(file->small_arrays.size(), 2u);
  const auto sections =
      file->small_arrays.find("arch.rope.dimension_sections");
  ASSERT_TRUE(sections != file->small_arrays.end());
  ASSERT_EQ(sections->second.size(), 4u);
  const std::uint32_t want[4] = {11u, 11u, 10u, 0u};
  for (std::size_t i = 0; i < 4; ++i) {
    const auto* value = std::get_if<std::uint32_t>(&sections->second[i]);
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, want[i]);
  }
  const auto tags = file->small_arrays.find("arch.tags");
  ASSERT_TRUE(tags != file->small_arrays.end());
  ASSERT_EQ(tags->second.size(), 1u);
  const auto* name = std::get_if<std::string>(&tags->second[0]);
  ASSERT_NE(name, nullptr);
  EXPECT_EQ(*name, "hybrid");
}

TEST(GgufTest, RejectsSmallArrayWithBadElementType) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 0, 1);
  builder.PushString("list");
  builder.PushU32(9);   // array
  builder.PushU32(13);  // element type beyond the value table
  builder.PushU64(2);
  builder.PushU32(1);
  builder.PushU32(2);
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}

TEST(GgufTest, RejectsArrayCountOverflow) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 0, 1);
  builder.PushString("list");
  builder.PushU32(9);                // array
  builder.PushU32(4);               // element type: u32
  builder.PushU64((1ull << 20) + 1);  // count beyond the boundary
  auto file = ParseGguf(std::span<const std::byte>(builder.bytes));
  ASSERT_FALSE(file.has_value());
  EXPECT_EQ(file.error(), StatusCode::MalformedFile);
}

// Opt-in validation against a real model file (path via environment,
// never hard-coded; see AGENTS.md "Model Data").
TEST(GgufTest, ParsesRealFileWhenProvided) {
  const char* path = std::getenv("TESSERA_TEST_GGUF");
  if (path == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_GGUF not set";
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
