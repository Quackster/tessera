#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include "cli_common.hpp"
#include "core/files.hpp"
#include "core/sha256.hpp"
#include "tessera/mtp_fetch.hpp"
#include "test_helpers.hpp"
#include "tessera/types.hpp"

using tessera::DType;
using tessera::StatusCode;
using tessera::TensorEntry;
using tessera::TensorManifest;
using tessera::TensorShape;


TEST(TensorShapeTest, NumelTracksRank) {
  TensorShape scalar;  // rank 0: one element
  EXPECT_EQ(scalar.Numel(), 1u);

  TensorShape vector;
  vector.rank = 1;
  vector.dims[0] = 7;
  EXPECT_EQ(vector.Numel(), 7u);

  TensorShape matrix;
  matrix.rank = 2;
  matrix.dims[0] = 4;
  matrix.dims[1] = 8;
  EXPECT_EQ(matrix.Numel(), 32u);

  TensorShape volume;
  volume.rank = 4;
  volume.dims[0] = 2;
  volume.dims[1] = 3;
  volume.dims[2] = 5;
  volume.dims[3] = 9;
  EXPECT_EQ(volume.Numel(), 270u);
}




TEST(DTypeTest, TensorBytesSizesTensors) {
  EXPECT_EQ(*tessera::TensorBytes(DType::F32, 4), 16u);
  EXPECT_EQ(*tessera::TensorBytes(DType::F16, 4), 8u);
  EXPECT_EQ(*tessera::TensorBytes(DType::I64, 3), 24u);
  EXPECT_EQ(*tessera::TensorBytes(DType::Q40, 32), 17u);
  EXPECT_EQ(*tessera::TensorBytes(DType::Q80, 64), 68u);
  EXPECT_EQ(*tessera::TensorBytes(DType::Q4K, 256), 144u);
  EXPECT_EQ(*tessera::TensorBytes(DType::Q5K, 512), 352u);
  EXPECT_EQ(*tessera::TensorBytes(DType::Q6K, 256), 210u);
  EXPECT_EQ(*tessera::TensorBytes(DType::Q8K, 256), 258u);
  EXPECT_EQ(*tessera::TensorBytes(DType::F4E2M1, 256), 128u);
  EXPECT_EQ(*tessera::TensorBytes(DType::Q3K, 256), 110u);
  EXPECT_EQ(*tessera::TensorBytes(DType::IQ4_NL, 32), 18u);
  EXPECT_EQ(*tessera::TensorBytes(DType::IQ4_XS, 256), 136u);
  EXPECT_EQ(*tessera::TensorBytes(DType::IQ3_S, 256), 110u);
  EXPECT_EQ(*tessera::TensorBytes(DType::IQ3_XXS, 256), 98u);
  EXPECT_EQ(*tessera::TensorBytes(DType::IQ2_S, 256), 82u);
  EXPECT_EQ(*tessera::TensorBytes(DType::Q2_0, 64), 18u);
  EXPECT_EQ(tessera::TensorBytes(DType::Q2_0, 100).error(),
            StatusCode::InvalidArgument);
  // No known layout for the K-quant leftover.
  EXPECT_EQ(tessera::TensorBytes(DType::Q2K, 256).error(),
            StatusCode::UnsupportedFeature);
  // 100 elements are not a whole number of Q4_K blocks; 3 elements
  // are not a whole number of F4E2M1 pairs.
  EXPECT_EQ(tessera::TensorBytes(DType::Q4K, 100).error(),
            StatusCode::InvalidArgument);
  EXPECT_EQ(tessera::TensorBytes(DType::F4E2M1, 3).error(),
            StatusCode::InvalidArgument);
}

TEST(FileReaderTest, ReadsRangeAndWholeFile) {
  using tessera::testing::FreshTempDir;
  using tessera::testing::WriteBytes;
  // A known 100-byte file (byte i is i).
  std::vector<std::byte> expected(100);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<std::byte>(i);
  }
  auto dir = FreshTempDir("tessera_tests_file_reader");
  auto path = dir / "data.bin";
  WriteBytes(path, expected);

  auto reader = tessera::core::FileReader::Open(path);
  ASSERT_TRUE(reader.has_value());
  EXPECT_EQ(reader->size(), expected.size());

  // A mid-file range.
  auto range = reader->ReadRange(10, 20);
  ASSERT_TRUE(range.has_value());
  ASSERT_EQ(range->size(), 20u);
  for (std::size_t i = 0; i < range->size(); ++i) {
    EXPECT_EQ((*range)[i], expected[10 + i]);
  }

  // The whole file.
  auto whole = reader->ReadRange(0, expected.size());
  ASSERT_TRUE(whole.has_value());
  ASSERT_EQ(whole->size(), expected.size());
  for (std::size_t i = 0; i < whole->size(); ++i) {
    EXPECT_EQ((*whole)[i], expected[i]);
  }

  // A zero-length read is empty.
  auto empty = reader->ReadRange(5, 0);
  ASSERT_TRUE(empty.has_value());
  EXPECT_EQ(empty->size(), 0u);
}

TEST(FileReaderTest, RejectsMissingAndNonRegularFiles) {
  using tessera::testing::FreshTempDir;
  auto dir = FreshTempDir("tessera_tests_file_reader_reject");
  auto missing = tessera::core::FileReader::Open(dir / "nope.bin");
  EXPECT_EQ(missing.error(), StatusCode::FileNotFound);
  auto non_regular = tessera::core::FileReader::Open(dir);  // a directory
  EXPECT_EQ(non_regular.error(), StatusCode::MalformedFile);
}

TEST(FileReaderTest, RejectsOutOfBoundsRange) {
  using tessera::testing::FreshTempDir;
  using tessera::testing::WriteBytes;
  std::vector<std::byte> expected(64);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<std::byte>(i);
  }
  auto dir = FreshTempDir("tessera_tests_file_reader_oob");
  auto path = dir / "data.bin";
  WriteBytes(path, expected);
  auto reader = tessera::core::FileReader::Open(path);
  ASSERT_TRUE(reader.has_value());
  // The count overruns the 64-byte file; the short read is rejected.
  auto overrun = reader->ReadRange(60, 10);
  EXPECT_EQ(overrun.error(), StatusCode::MalformedFile);
}

TEST(Sha256Test, KnownVectors) {
  auto hex = [](std::string_view text) {
    return tessera::core::Sha256::Hex(std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(text.data()), text.size()));
  };
  EXPECT_EQ(hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(
      hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(Sha256Test, StreamedFileMatchesInMemory) {
  using tessera::testing::FreshTempDir;
  using tessera::testing::WriteBytes;
  std::string text;
  for (int i = 0; i < 5000; ++i) {
    text += "the quick brown fox jumps over the lazy dog ";
  }
  auto dir = FreshTempDir("tessera_tests_sha256_file");
  auto path = dir / "blob.bin";
  std::vector<std::byte> bytes(text.size());
  std::memcpy(bytes.data(), text.data(), text.size());
  WriteBytes(path, bytes);
  auto from_file = tessera::core::Sha256File(path);
  ASSERT_TRUE(from_file.has_value());
  EXPECT_EQ(*from_file, tessera::core::Sha256::Hex(std::span<const std::byte>(
                            reinterpret_cast<const std::byte*>(text.data()),
                            text.size())));
  auto missing = tessera::core::Sha256File(dir / "absent.bin");
  EXPECT_EQ(missing.error(), StatusCode::FileNotFound);
}

TEST(ParseGpuListTest, AcceptsOneDeviceOrAList) {
  using tessera::cli::ParseGpuList;
  EXPECT_EQ(ParseGpuList("0"), (std::vector<int>{0}));
  EXPECT_EQ(ParseGpuList("1"), (std::vector<int>{1}));
  EXPECT_EQ(ParseGpuList("0,1"), (std::vector<int>{0, 1}));
  EXPECT_EQ(ParseGpuList("2,3,5"), (std::vector<int>{2, 3, 5}));
}

TEST(ParseGpuListTest, SkipsEmptyItems) {
  using tessera::cli::ParseGpuList;
  EXPECT_EQ(ParseGpuList("0,"), (std::vector<int>{0}));
  EXPECT_EQ(ParseGpuList(",1"), (std::vector<int>{1}));
  EXPECT_TRUE(ParseGpuList("").empty());
}

TEST(DeviceRoutingHintTest, SingleDeviceIsSilent) {
  using tessera::cli::DeviceRouteKind;
  using tessera::cli::DeviceRoutingHint;
  const std::vector<int> one{0};
  EXPECT_TRUE(DeviceRoutingHint(one, DeviceRouteKind::kRunSingle, false).empty());
  EXPECT_TRUE(
      DeviceRoutingHint(one, DeviceRouteKind::kServeReplicas, false).empty());
  EXPECT_TRUE(DeviceRoutingHint({}, DeviceRouteKind::kRunSingle, false).empty());
}

TEST(DeviceRoutingHintTest, RunWithoutSplitNamesFirstDeviceAndFlags) {
  using tessera::cli::DeviceRouteKind;
  using tessera::cli::DeviceRoutingHint;
  const std::vector<int> devices{0, 1};
  const std::string hint =
      DeviceRoutingHint(devices, DeviceRouteKind::kRunSingle, false);
  EXPECT_NE(hint.find("cli: "), std::string::npos);
  EXPECT_NE(hint.find("only gpu 0 is used"), std::string::npos);
  EXPECT_NE(hint.find("--pipeline"), std::string::npos);
  EXPECT_NE(hint.find("--tp"), std::string::npos);
  EXPECT_NE(hint.find("--cp"), std::string::npos);
  EXPECT_NE(hint.find("--spec-split"), std::string::npos);
  EXPECT_NE(hint.find("--hybrid"), std::string::npos);
}

TEST(DeviceRoutingHintTest, RunWithSplitIsSilent) {
  using tessera::cli::DeviceRouteKind;
  using tessera::cli::DeviceRoutingHint;
  const std::vector<int> devices{0, 1};
  EXPECT_TRUE(
      DeviceRoutingHint(devices, DeviceRouteKind::kRunSingle, true).empty());
}

TEST(DeviceRoutingHintTest, ServeReportsReplicasAndFlags) {
  using tessera::cli::DeviceRouteKind;
  using tessera::cli::DeviceRoutingHint;
  const std::vector<int> devices{2, 3, 5};
  const std::string hint =
      DeviceRoutingHint(devices, DeviceRouteKind::kServeReplicas, false);
  EXPECT_NE(hint.find("--gpus 2, 3, 5"), std::string::npos);
  EXPECT_NE(hint.find("replica"), std::string::npos);
  EXPECT_NE(hint.find("--pipeline, --tp, --cp, --spec-split or --hybrid"),
            std::string::npos);
}

TEST(MtpFetchTest, DigestsCoverEveryTensorOnce) {
  const auto digests = tessera::MtpTensorDigests();
  EXPECT_EQ(digests.size(), 31u);
  std::set<std::string> names;
  for (const auto& digest : digests) {
    const std::string name(digest.name);
    EXPECT_EQ(name.rfind("mtp.", 0), 0u);
    EXPECT_EQ(digest.sha256.size(), 64u);
    EXPECT_TRUE(names.insert(name).second) << "duplicate " << name;
  }
}

