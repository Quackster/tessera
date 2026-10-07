#include <gtest/gtest.h>

#include <iterator>

#include "tessera/types.hpp"

using tessera::DType;
using tessera::StatusCode;
using tessera::TensorEntry;
using tessera::TensorManifest;
using tessera::TensorShape;

TEST(StatusCodeTest, ToStringCoversEveryCodeUniquely) {
  const StatusCode codes[] = {
      StatusCode::Ok,
      StatusCode::InvalidArgument,
      StatusCode::FileNotFound,
      StatusCode::MalformedFile,
      StatusCode::UnsupportedFeature,
      StatusCode::OutOfMemory,
      StatusCode::DeviceError,
  };
  for (std::size_t i = 0; i < std::size(codes); ++i) {
    EXPECT_NE(tessera::ToString(codes[i]).size(), 0u)
        << "empty name for code " << static_cast<int>(codes[i]);
    for (std::size_t j = i + 1; j < std::size(codes); ++j) {
      EXPECT_NE(tessera::ToString(codes[i]), tessera::ToString(codes[j]))
          << "codes " << static_cast<int>(codes[i]) << " and "
          << static_cast<int>(codes[j]) << " share a name";
    }
  }
}

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

TEST(TensorShapeTest, Equality) {
  TensorShape a;
  a.rank = 2;
  a.dims[0] = 4;
  a.dims[1] = 8;
  TensorShape b = a;
  EXPECT_EQ(a, b);
  b.dims[1] = 9;
  EXPECT_NE(a, b);
  TensorShape c;  // rank 0
  EXPECT_NE(a, c);
}

TEST(DTypeTest, ElementBytesForPlainTypes) {
  EXPECT_EQ(tessera::ElementBytes(DType::F32), 4);
  EXPECT_EQ(tessera::ElementBytes(DType::F16), 2);
  EXPECT_EQ(tessera::ElementBytes(DType::BF16), 2);
  EXPECT_EQ(tessera::ElementBytes(DType::F8E4M3), 1);
  EXPECT_EQ(tessera::ElementBytes(DType::F8E5M2), 1);
  EXPECT_EQ(tessera::ElementBytes(DType::F8E8M0), 1);
  EXPECT_EQ(tessera::ElementBytes(DType::I32), 4);
  EXPECT_EQ(tessera::ElementBytes(DType::I64), 8);
}

TEST(DTypeTest, ElementBytesRejectsBlockTypes) {
  int value = 0;
  EXPECT_THROW(value = tessera::ElementBytes(DType::Q2K),
               std::invalid_argument);
  EXPECT_THROW(value = tessera::ElementBytes(DType::Q40),
               std::invalid_argument);
  EXPECT_THROW(value = tessera::ElementBytes(DType::Q4K),
               std::invalid_argument);
  EXPECT_THROW(value = tessera::ElementBytes(DType::Q8K),
               std::invalid_argument);
  EXPECT_THROW(value = tessera::ElementBytes(DType::F4E2M1),
               std::invalid_argument);
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

TEST(TensorManifestTest, TotalNumelSumsTensors) {
  TensorManifest manifest;
  TensorEntry first;
  first.name = "a";
  first.shape.rank = 2;
  first.shape.dims[0] = 2;
  first.shape.dims[1] = 3;
  TensorEntry second;
  second.name = "b";
  second.shape.rank = 1;
  second.shape.dims[0] = 4;
  manifest.tensors.push_back(first);
  manifest.tensors.push_back(second);
  EXPECT_EQ(manifest.TotalNumel(), 10u);

  TensorManifest empty;
  EXPECT_EQ(empty.TotalNumel(), 0u);
}
