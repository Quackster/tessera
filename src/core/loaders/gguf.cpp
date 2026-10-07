#include "core/files.hpp"
#include "core/loaders/gguf.hpp"

#include <cstring>
#include <optional>

namespace tessera::core {

namespace {

// Format constants (GGUF magic and version range) - never configurable.
constexpr std::uint32_t kGgufMagic = 0x46554747u;  // "GGUF"
constexpr std::uint32_t kGgufMinVersion = 2;
constexpr std::uint32_t kGgufMaxVersion = 3;
// Boundary checks: reject absurd counts before trusting the image.
constexpr std::uint64_t kMaxGgufTensorCount = 1ull << 20;
constexpr std::uint64_t kMaxGgufKvCount = 1ull << 16;
// Spec caps: metadata keys and string values at 65535 bytes, tensor
// names at 64 bytes.
constexpr std::uint64_t kMaxGgufStringLen = 65535;
constexpr std::uint64_t kMaxGgufTensorNameLen = 64;
// Arrays above this length stay dropped (bulk data, e.g. vocabularies);
// shorter ones are retained as definition metadata.
constexpr std::uint64_t kMaxRetainedArrayElements = 16;
// Skipped arrays are walked bounds-checked, so large counts are cheap;
// real vocabularies reach 248k entries.
constexpr std::uint64_t kMaxGgufArrayCount = 1ull << 20;
constexpr std::uint32_t kMaxGgufRank = 4;
constexpr std::uint64_t kMaxGgufDim = 1ull << 24;
// Spec default for the general.alignment metadata key.
constexpr std::uint32_t kGgufDefaultAlignment = 32;

// Metadata value types (gguf spec; v2 and v3 share the table).
enum class GgufValueType : std::uint32_t {
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12,
};
constexpr std::uint32_t kMaxGgufValueType =
    static_cast<std::uint32_t>(GgufValueType::Float64);

// Bounds-checked cursor over the GGUF byte stream (little-endian).
struct Cursor {
  std::span<const std::byte> data;
  std::size_t pos = 0;
  bool failed = false;

  bool Need(std::size_t n) {
    if (pos + n > data.size()) {
      failed = true;
      return false;
    }
    return true;
  }

  std::uint8_t U8() {
    std::uint8_t value = 0;
    if (Need(1)) {
      value = static_cast<std::uint8_t>(data[pos]);
      pos += 1;
    }
    return value;
  }

  std::uint16_t U16() {
    std::uint16_t value = 0;
    if (Need(2)) {
      std::memcpy(&value, data.data() + pos, 2);
      pos += 2;
    }
    return value;
  }

  std::uint32_t U32() {
    std::uint32_t value = 0;
    if (Need(4)) {
      std::memcpy(&value, data.data() + pos, 4);
      pos += 4;
    }
    return value;
  }

  std::uint64_t U64() {
    std::uint64_t value = 0;
    if (Need(8)) {
      std::memcpy(&value, data.data() + pos, 8);
      pos += 8;
    }
    return value;
  }

  float F32() {
    float value = 0.0f;
    if (Need(4)) {
      std::memcpy(&value, data.data() + pos, 4);
      pos += 4;
    }
    return value;
  }

  double F64() {
    double value = 0.0;
    if (Need(8)) {
      std::memcpy(&value, data.data() + pos, 8);
      pos += 8;
    }
    return value;
  }

  // The spec allows only 0 and 1; anything else is malformed.
  bool Bool() {
    const std::uint8_t value = U8();
    if (!failed && value != 0 && value != 1) {
      failed = true;
    }
    return value == 1;
  }

  // A GGUF string: u64 byte length + the bytes (no NUL, per the spec).
  std::string String(std::uint64_t min_len, std::uint64_t max_len) {
    const std::uint64_t len = U64();
    if (failed || len < min_len || len > max_len) {
      failed = true;
      return {};
    }
    if (!Need(len)) {
      return {};
    }
    std::string result(reinterpret_cast<const char*>(data.data() + pos), len);
    pos += static_cast<std::size_t>(len);
    return result;
  }

  // Advance past one scalar value of `type` without retaining it.
  // Array is not skippable here; the caller expands it.
  [[nodiscard]] bool SkipScalar(std::uint32_t type) {
    switch (static_cast<GgufValueType>(type)) {
      case GgufValueType::Uint8:
      case GgufValueType::Int8:
      case GgufValueType::Bool: U8(); break;
      case GgufValueType::Uint16:
      case GgufValueType::Int16: U16(); break;
      case GgufValueType::Uint32:
      case GgufValueType::Int32:
      case GgufValueType::Float32: U32(); break;
      case GgufValueType::Uint64:
      case GgufValueType::Int64:
      case GgufValueType::Float64: U64(); break;
      case GgufValueType::String: String(0, kMaxGgufStringLen); break;
      default: return false;
    }
    return !failed;
  }
};

// Read one scalar metadata value (not an array).
std::optional<GgufValue> ReadScalar(Cursor& cursor, std::uint32_t type) {
  switch (static_cast<GgufValueType>(type)) {
    case GgufValueType::Uint8:
      return GgufValue(static_cast<std::uint32_t>(cursor.U8()));
    case GgufValueType::Int8:
      return GgufValue(static_cast<std::int32_t>(
          static_cast<std::int8_t>(cursor.U8())));
    case GgufValueType::Uint16:
      return GgufValue(static_cast<std::uint32_t>(cursor.U16()));
    case GgufValueType::Int16:
      return GgufValue(static_cast<std::int32_t>(
          static_cast<std::int16_t>(cursor.U16())));
    case GgufValueType::Uint32:
      return GgufValue(cursor.U32());
    case GgufValueType::Int32:
      return GgufValue(static_cast<std::int32_t>(cursor.U32()));
    case GgufValueType::Float32:
      return GgufValue(cursor.F32());
    case GgufValueType::Bool:
      return GgufValue(cursor.Bool());
    case GgufValueType::String:
      return GgufValue(cursor.String(0, kMaxGgufStringLen));
    case GgufValueType::Uint64:
      return GgufValue(cursor.U64());
    case GgufValueType::Int64:
      return GgufValue(static_cast<std::int64_t>(cursor.U64()));
    case GgufValueType::Float64:
      return GgufValue(cursor.F64());
    default:
      return std::nullopt;
  }
}

// ggml type id (current spec numbering) -> engine DType.
std::optional<DType> GgmlTypeToDType(std::uint32_t id) {
  switch (id) {
    case 0: return DType::F32;
    case 1: return DType::F16;
    case 2: return DType::Q40;  // Q4_0
    case 8: return DType::Q80;  // Q8_0
    case 10: return DType::Q2K;
    case 11: return DType::Q3K;
    case 12: return DType::Q4K;
    case 13: return DType::Q5K;
    case 14: return DType::Q6K;
    case 15: return DType::Q8K;
    case 20: return DType::IQ4_NL;
    case 21: return DType::IQ3_S;
    case 23: return DType::IQ4_XS;
    case 26: return DType::I32;
    case 27: return DType::I64;
    case 30: return DType::BF16;
    default: return std::nullopt;
  }
}

// The file alignment: general.alignment (spec: a power of two) or the
// spec default. Returns false when a present value is not a power of
// two; other encodings fall back to the default.
bool FileAlignment(const GgufFile& file, std::uint32_t* alignment) {
  *alignment = kGgufDefaultAlignment;
  const auto* value = file.Find("general.alignment");
  if (value == nullptr) {
    return true;
  }
  std::uint64_t raw = 0;
  if (const auto* as_u32 = std::get_if<std::uint32_t>(value)) {
    raw = *as_u32;
  } else if (const auto* as_u64 = std::get_if<std::uint64_t>(value)) {
    raw = *as_u64;
  } else {
    return true;
  }
  *alignment = static_cast<std::uint32_t>(raw);
  return *alignment != 0 && (*alignment & (*alignment - 1)) == 0;
}

}  // namespace

const GgufValue* GgufFile::Find(std::string_view key) const {
  auto it = metadata.find(std::string(key));
  return it == metadata.end() ? nullptr : &it->second;
}

std::expected<GgufFile, StatusCode> ParseGguf(
    std::span<const std::byte> data) {
  constexpr std::size_t kHeaderSize = 24;  // magic + version + counts
  if (data.size() < kHeaderSize) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  Cursor cursor{data};

  if (cursor.U32() != kGgufMagic) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  GgufFile file;
  file.version = cursor.U32();
  if (file.version < kGgufMinVersion || file.version > kGgufMaxVersion) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  const std::uint64_t tensor_count = cursor.U64();
  const std::uint64_t kv_count = cursor.U64();
  if (tensor_count > kMaxGgufTensorCount || kv_count > kMaxGgufKvCount) {
    return std::unexpected(StatusCode::MalformedFile);
  }

  for (std::uint64_t i = 0; i < kv_count; ++i) {
    std::string key = cursor.String(1, kMaxGgufStringLen);
    std::uint32_t type = cursor.U32();
    if (cursor.failed) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    if (type == static_cast<std::uint32_t>(GgufValueType::Array)) {
      // Arrays are retained below the retention bound and skipped
      // past it (recording the drop); either way the walk is bounded.
      const std::uint32_t elem_type = cursor.U32();
      const std::uint64_t count = cursor.U64();
      if (cursor.failed || count > kMaxGgufArrayCount) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      if (count > kMaxRetainedArrayElements) {
        bool ok = true;
        for (std::uint64_t e = 0; e < count && ok; ++e) {
          ok = cursor.SkipScalar(elem_type);
        }
        if (!ok) {
          return std::unexpected(StatusCode::MalformedFile);
        }
        file.dropped_array_keys.push_back(std::move(key));
      } else {
        std::vector<GgufValue> kept;
        kept.reserve(static_cast<std::size_t>(count));
        bool ok = true;
        for (std::uint64_t e = 0; e < count && ok; ++e) {
          auto value = ReadScalar(cursor, elem_type);
          if (!value || cursor.failed) {
            ok = false;
          } else {
            kept.push_back(std::move(*value));
          }
        }
        if (!ok) {
          return std::unexpected(StatusCode::MalformedFile);
        }
        file.small_arrays[std::move(key)] = std::move(kept);
      }
    } else if (type <= kMaxGgufValueType) {
      if (auto value = ReadScalar(cursor, type)) {
        if (cursor.failed) {
          return std::unexpected(StatusCode::MalformedFile);
        }
        file.metadata[std::move(key)] = std::move(*value);
      } else {
        return std::unexpected(StatusCode::UnsupportedFeature);
      }
    } else {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
  }

  std::uint32_t alignment = kGgufDefaultAlignment;
  if (!FileAlignment(file, &alignment)) {
    return std::unexpected(StatusCode::MalformedFile);
  }

  struct TensorRecord {
    std::string name;
    TensorShape shape;
    DType dtype = DType::F32;
    std::uint64_t offset = 0;
  };
  std::vector<TensorRecord> records;
  records.reserve(static_cast<std::size_t>(tensor_count));
  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    TensorRecord record;
    record.name = cursor.String(1, kMaxGgufTensorNameLen);
    if (cursor.failed || record.name.empty()) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    const std::uint32_t rank = cursor.U32();
    if (cursor.failed || rank > kMaxGgufRank) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    record.shape.rank = rank;
    bool dims_ok = true;
    for (std::uint32_t d = 0; d < rank; ++d) {
      const std::uint64_t dim = cursor.U64();
      if (dim > kMaxGgufDim) {
        dims_ok = false;
      }
      record.shape.dims[d] = static_cast<std::size_t>(dim);
    }
    if (cursor.failed || !dims_ok) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    const std::uint32_t ggml_type = cursor.U32();
    auto dtype = GgmlTypeToDType(ggml_type);
    if (cursor.failed) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    if (!dtype) {
      return std::unexpected(StatusCode::UnsupportedFeature);
    }
    record.dtype = *dtype;
    record.offset = cursor.U64();
    if (cursor.failed) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    records.push_back(std::move(record));
  }

  // The tensor data starts at the aligned position after the last
  // tensor info; offsets are relative to that start.
  const std::uint64_t info_end = cursor.pos;
  const std::uint64_t padding =
      (static_cast<std::uint64_t>(alignment) -
       info_end % static_cast<std::uint64_t>(alignment)) %
      static_cast<std::uint64_t>(alignment);
  const std::uint64_t data_start =
      info_end + padding <= data.size() ? info_end + padding : data.size();
  const std::uint64_t data_size = data.size() - data_start;

  std::uint64_t prev_offset = 0;
  for (const auto& record : records) {
    if (record.offset % alignment != 0) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    if (record.offset < prev_offset || record.offset > data_size) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    prev_offset = record.offset;
    // Exact size validation through the canonical sizer. Layouts
    // without a known size keep the historical size 0 and are
    // rejected at upload time instead.
    std::uint64_t tensor_bytes = 0;
    auto sized = TensorBytes(record.dtype, record.shape.Numel());
    if (!sized) {
      if (sized.error() != StatusCode::UnsupportedFeature) {
        return std::unexpected(StatusCode::MalformedFile);
      }
    } else {
      tensor_bytes = static_cast<std::uint64_t>(*sized);
    }
    if (tensor_bytes > data_size - record.offset) {
      return std::unexpected(StatusCode::MalformedFile);
    }
  }

  file.tensor_data_start = data_start;
  file.tensors.reserve(records.size());
  file.tensor_offsets.reserve(records.size());
  for (auto& record : records) {
    file.tensor_offsets.push_back(record.offset);
    file.tensors.push_back(
        TensorEntry{std::move(record.name), record.shape, record.dtype});
  }
  return file;
}

std::expected<GgufFile, StatusCode> ParseGgufFile(
    const std::filesystem::path& path) {
  auto data = ReadFile(path);
  if (!data) {
    return std::unexpected(data.error());
  }
  return ParseGguf(std::span<const std::byte>(*data));
}

}  // namespace tessera::core
