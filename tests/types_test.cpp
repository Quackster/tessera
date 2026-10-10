#include <gtest/gtest.h>

#include <iterator>

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

