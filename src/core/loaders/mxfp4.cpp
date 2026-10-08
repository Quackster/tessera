#include "core/loaders/mxfp4.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/json.hpp"
#include "tessera/architecture.hpp"
#include "tessera/backend.hpp"

namespace tessera::core {

std::string MxFp4ArchitectureName(std::string_view json) {
  auto document = Json::Parse(json);
  if (!document || !document->isObject()) {
    return {};
  }
  const Json* architectures = document->Find("architectures");
  if (architectures != nullptr && architectures->isArray() &&
      !architectures->AsArray().empty() &&
      architectures->AsArray().front().isString()) {
    return architectures->AsArray().front().AsString();
  }
  const Json* model_type = document->Find("model_type");
  if (model_type != nullptr && model_type->isString()) {
    return model_type->AsString();
  }
  return {};
}

namespace {

std::size_t PlainElemBytes(DType dtype) {
  switch (dtype) {
    case DType::F32:
      return 4;
    case DType::BF16:
      return 2;
    default:
      return 0;
  }
}

float ReadPlain(DType dtype, const std::byte* p) {
  if (dtype == DType::F32) {
    float value = 0.0f;
    std::memcpy(&value, p, 4);
    return value;
  }
  std::uint16_t half = 0;
  std::memcpy(&half, p, 2);
  const std::uint32_t bits = static_cast<std::uint32_t>(half) << 16;
  float value = 0.0f;
  std::memcpy(&value, &bits, 4);
  return value;
}

// The value dimension granularity: one element per unit for plain data,
// one packed nibble byte (2 elements) per unit for an MXFP4 blob, and one
// E8M0 scale byte (32 elements) per unit for the scales.
enum class Lane { Plain, Blob, Scale };

std::size_t LaneUnits(Lane lane, std::size_t cols) {
  switch (lane) {
    case Lane::Plain:
      return cols;
    case Lane::Blob:
      return cols / 2;
    case Lane::Scale:
      return cols / 32;
  }
  return cols;
}

std::size_t LaneSource(Lane lane, const std::vector<std::size_t>& src,
                       std::size_t unit) {
  switch (lane) {
    case Lane::Plain:
      return src[unit];
    case Lane::Blob:
      return src[unit * 2] / 2;
    case Lane::Scale:
      return src[unit * 32] / 32;
  }
  return unit;
}

// Permute a row-major buffer of `rows` x cols elements along the outer or
// inner value dimension.
std::expected<std::vector<std::byte>, StatusCode> PermuteValue(
    std::span<const std::byte> in, std::size_t rows, std::size_t cols,
    Lane lane, std::size_t elem_bytes, bool inner,
    const std::vector<std::size_t>& src) {
  const std::size_t per = lane == Lane::Plain ? elem_bytes : 1;
  const std::size_t units = LaneUnits(lane, cols);
  const std::size_t row_bytes = units * per;
  if (units == 0 || in.size() != rows * row_bytes) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::vector<std::byte> out(in.size());
  if (!inner) {
    if (src.size() != rows) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    for (std::size_t r = 0; r < rows; ++r) {
      if (src[r] >= rows) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      std::memcpy(out.data() + r * row_bytes, in.data() + src[r] * row_bytes,
                  row_bytes);
    }
    return out;
  }
  if (src.size() != cols) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  // The value-head reorder moves whole blocks (many consecutive units), so
  // copy maximal runs of consecutive source units. A per-unit memcpy here
  // costs seconds on a 27B checkpoint.
  for (std::size_t r = 0; r < rows; ++r) {
    std::byte* dst = out.data() + r * row_bytes;
    const std::byte* base = in.data() + r * row_bytes;
    std::size_t u = 0;
    while (u < units) {
      const std::size_t s = LaneSource(lane, src, u);
      if (s >= units) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      std::size_t run = 1;
      while (u + run < units) {
        const std::size_t ns = LaneSource(lane, src, u + run);
        if (ns != s + run || ns >= units) {
          break;
        }
        ++run;
      }
      std::memcpy(dst + u * per, base + s * per, run * per);
      u += run;
    }
  }
  return out;
}

// Reverse the dimensions to the internal (ggml ne) convention.
TensorShape ReverseShape(const TensorShape& shape) {
  TensorShape out;
  out.rank = shape.rank;
  for (std::size_t i = 0; i < shape.rank; ++i) {
    out.dims[i] = shape.dims[shape.rank - 1 - i];
  }
  return out;
}

std::expected<std::unique_ptr<Buffer>, StatusCode> Upload(
    Backend& backend, std::span<const std::byte> bytes) {
  auto buffer = backend.AllocateBuffer(bytes.size(), MemoryKind::Device);
  if (!buffer) {
    return std::unexpected(StatusCode::OutOfMemory);
  }
  if (!backend.CopyH2D(**buffer, bytes)) {
    return std::unexpected(StatusCode::DeviceError);
  }
  return std::move(*buffer);
}

}  // namespace

std::expected<std::vector<DeviceTensor>, StatusCode> BuildMxFp4Weights(
    Backend& backend, std::span<const std::byte> file,
    std::span<const SafetensorsTensor> tensors, const Architecture& module,
    const TransformerConfig& config) {
  std::unordered_map<std::string, const SafetensorsTensor*> by_name;
  for (const auto& tensor : tensors) {
    by_name.emplace(tensor.name, &tensor);
  }
  std::vector<DeviceTensor> weights;
  for (const auto& tensor : tensors) {
    if (tensor.entry.dtype == DType::F8E8M0 ||
        tensor.entry.dtype == DType::F8E4M3) {
      continue;  // scales pair with their blob; FP8 MTP is not packed yet
    }
    auto internal = module.MapWeightName(tensor.name, config);
    if (!internal) {
      continue;
    }
    auto conversion = module.ConvertWeight(*internal, config);
    const TensorShape& shape = tensor.entry.shape;
    const std::size_t rank = shape.rank;
    if (rank == 0 || rank > 3) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    const std::size_t rows = shape.dims[0];
    std::size_t cols = 1;
    for (std::size_t d = 1; d < rank; ++d) {
      cols *= shape.dims[d];
    }
    std::span<const std::byte> payload =
        file.subspan(tensor.begin, tensor.end - tensor.begin);

    std::unique_ptr<Buffer> device;
    DType out_dtype = tensor.entry.dtype;
    if (tensor.entry.dtype == DType::F4E2M1) {
      const auto scale = by_name.find(tensor.name + "_scale");
      if (scale == by_name.end()) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      std::span<const std::byte> scale_bytes =
          file.subspan(scale->second->begin,
                       scale->second->end - scale->second->begin);
      // Pack blob||scales in one allocation. Appending to a blob vector
      // reallocates the whole blob and turned a 27B load into minutes; a
      // second copy is skipped for the tensors with no reorder.
      std::vector<std::byte> packed;
      if (conversion) {
        std::vector<std::byte> blob(payload.begin(), payload.end());
        std::vector<std::byte> scales(scale_bytes.begin(), scale_bytes.end());
        auto permuted_blob = PermuteValue(blob, rows, cols, Lane::Blob, 0,
                                          conversion->inner, conversion->src);
        if (!permuted_blob) {
          return std::unexpected(permuted_blob.error());
        }
        auto permuted_scales =
            PermuteValue(scales, rows, cols, Lane::Scale, 0, conversion->inner,
                         conversion->src);
        if (!permuted_scales) {
          return std::unexpected(permuted_scales.error());
        }
        blob = std::move(*permuted_blob);
        scales = std::move(*permuted_scales);
        packed.resize(blob.size() + scales.size());
        std::memcpy(packed.data(), blob.data(), blob.size());
        std::memcpy(packed.data() + blob.size(), scales.data(), scales.size());
      } else {
        packed.resize(payload.size() + scale_bytes.size());
        std::memcpy(packed.data(), payload.data(), payload.size());
        std::memcpy(packed.data() + payload.size(), scale_bytes.data(),
                    scale_bytes.size());
      }
      auto buffer = Upload(backend, packed);
      if (!buffer) {
        return std::unexpected(buffer.error());
      }
      device = std::move(*buffer);
      out_dtype = DType::F4E2M1;
    } else {
      const std::size_t elem_bytes = PlainElemBytes(tensor.entry.dtype);
      if (elem_bytes == 0) {
        return std::unexpected(StatusCode::UnsupportedFeature);
      }
      if (payload.size() != rows * cols * elem_bytes) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      if (*internal == "token_embd.weight") {
        // The embedding stays BF16; GatherEmbedding reads it row-wise.
        auto buffer = Upload(backend, payload);
        if (!buffer) {
          return std::unexpected(buffer.error());
        }
        device = std::move(*buffer);
      } else {
        std::vector<float> values(rows * cols);
        if (!conversion && tensor.entry.dtype == DType::BF16) {
          // Bulk bf16 -> f32: a wider load per element lets the compiler
          // vectorize. The per-element ReadPlain call does not, and the
          // 5 GB lm_head alone then takes half a minute.
          const std::uint16_t* src =
              reinterpret_cast<const std::uint16_t*>(payload.data());
          for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] =
                std::bit_cast<float>(static_cast<std::uint32_t>(src[i]) << 16);
          }
        } else if (!conversion && tensor.entry.dtype == DType::F32) {
          std::memcpy(values.data(), payload.data(), values.size() * 4);
        } else {
          for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] =
                ReadPlain(tensor.entry.dtype, payload.data() + i * elem_bytes);
            if (conversion && conversion->exp_negate) {
              values[i] = -std::exp(values[i]);
            }
            if (conversion && conversion->add_one) {
              values[i] += 1.0f;
            }
          }
        }
        std::span<const std::byte> f32_bytes(
            reinterpret_cast<const std::byte*>(values.data()),
            values.size() * 4);
        std::span<const std::byte> out_bytes = f32_bytes;
        std::vector<std::byte> permuted_bytes;
        if (conversion && !conversion->src.empty()) {
          auto permuted = PermuteValue(f32_bytes, rows, cols, Lane::Plain, 4,
                                       conversion->inner, conversion->src);
          if (!permuted) {
            return std::unexpected(permuted.error());
          }
          permuted_bytes = std::move(*permuted);
          out_bytes = permuted_bytes;
        }
        auto buffer = Upload(backend, out_bytes);
        if (!buffer) {
          return std::unexpected(buffer.error());
        }
        device = std::move(*buffer);
        out_dtype = DType::F32;
      }
    }

    TensorEntry entry;
    entry.name = std::move(*internal);
    entry.shape = ReverseShape(shape);
    entry.dtype = out_dtype;
    weights.push_back(DeviceTensor{std::move(entry), std::move(device)});
  }
  return weights;
}

}  // namespace tessera::core
