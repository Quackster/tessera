#include "core/files.hpp"
#include "core/loaders/safetensors.hpp"

#include <algorithm>
#include <cstring>

namespace tessera::core {

namespace {

// Boundary checks for the container (never configurable).
constexpr std::size_t kSafetensorsMinSize = 10;  // 8-byte prefix + '{}'
constexpr std::uint64_t kMaxSafetensorsHeaderLen = 1ull << 30;
// Fixed header schema needs no nesting beyond entry objects.
constexpr int kMaxJsonDepth = 8;

// Bounds-checked cursor over the JSON header bytes.
struct JsonCursor {
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

  char Peek() {
    if (!Need(1)) {
      return 0;
    }
    return static_cast<char>(data[pos]);
  }

  void SkipWs() {
    while (!failed && (Peek() == ' ' || Peek() == '\t' || Peek() == '\n' ||
                       Peek() == '\r')) {
      ++pos;
    }
  }

  bool Expect(char c) {
    SkipWs();
    if (failed || Peek() != c) {
      failed = true;
      return false;
    }
    ++pos;
    return true;
  }
};

// One \uXXXX unit; combines surrogate pairs, rejects lone units.
bool ParseHex4(JsonCursor& cursor, std::uint32_t* unit) {
  *unit = 0;
  for (int i = 0; i < 4; ++i) {
    if (!cursor.Need(1)) {
      return false;
    }
    const char c = static_cast<char>(cursor.data[cursor.pos++]);
    *unit <<= 4;
    if (c >= '0' && c <= '9') {
      *unit |= static_cast<std::uint32_t>(c - '0');
    } else if (c >= 'a' && c <= 'f') {
      *unit |= static_cast<std::uint32_t>(c - 'a' + 10);
    } else if (c >= 'A' && c <= 'F') {
      *unit |= static_cast<std::uint32_t>(c - 'A' + 10);
    } else {
      return false;
    }
  }
  return true;
}

void AppendUtf8(std::string& out, std::uint32_t point) {
  if (point < 0x80) {
    out.push_back(static_cast<char>(point));
  } else if (point < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (point >> 6)));
    out.push_back(static_cast<char>(0x80 | (point & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xE0 | (point >> 12)));
    out.push_back(static_cast<char>(0x80 | ((point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (point & 0x3F)));
  }
}

bool ParseJsonString(JsonCursor& cursor, std::string* out) {
  out->clear();
  if (!cursor.Expect('"')) {
    return false;
  }
  while (!cursor.failed) {
    if (!cursor.Need(1)) {
      return false;
    }
    const char c = static_cast<char>(cursor.data[cursor.pos++]);
    if (c == '"') {
      return true;
    }
    if (c == '\\') {
      if (!cursor.Need(1)) {
        return false;
      }
      const char e = static_cast<char>(cursor.data[cursor.pos++]);
      switch (e) {
        case '"': out->push_back('"'); break;
        case '\\': out->push_back('\\'); break;
        case '/': out->push_back('/'); break;
        case 'b': out->push_back('\b'); break;
        case 'f': out->push_back('\f'); break;
        case 'n': out->push_back('\n'); break;
        case 'r': out->push_back('\r'); break;
        case 't': out->push_back('\t'); break;
        case 'u': {
          std::uint32_t unit = 0;
          if (!ParseHex4(cursor, &unit)) {
            return false;
          }
          if (unit >= 0xD800 && unit <= 0xDBFF) {
            if (!cursor.Need(2) ||
                static_cast<char>(cursor.data[cursor.pos]) != '\\' ||
                static_cast<char>(cursor.data[cursor.pos + 1]) != 'u') {
              return false;
            }
            cursor.pos += 2;
            std::uint32_t low = 0;
            if (!ParseHex4(cursor, &low) || low < 0xDC00 ||
                low > 0xDFFF) {
              return false;
            }
            unit = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00);
          } else if (unit >= 0xDC00 && unit <= 0xDFFF) {
            return false;
          }
          if (unit > 0x10FFFF) {
            return false;
          }
          if (unit < 0x10000) {
            AppendUtf8(*out, unit);
          } else {
            // Non-BMP as raw UTF-8 bytes (4-byte form).
            out->push_back(static_cast<char>(0xF0 | (unit >> 18)));
            out->push_back(static_cast<char>(0x80 | ((unit >> 12) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | ((unit >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (unit & 0x3F)));
          }
          break;
        }
        default: return false;
      }
    } else if (c < 0x20) {
      return false;  // raw control characters are not valid JSON
    } else {
      out->push_back(c);
    }
  }
  return false;
}

bool ParseJsonU64(JsonCursor& cursor, std::uint64_t* out) {
  cursor.SkipWs();
  *out = 0;
  bool any = false;
  while (!cursor.failed) {
    if (!cursor.Need(1)) {
      break;
    }
    const char c = static_cast<char>(cursor.data[cursor.pos]);
    if (c < '0' || c > '9') {
      break;
    }
    if (*out > (0xFFFFFFFFFFFFFFFFull - 9) / 10) {
      return false;
    }
    *out = *out * 10 + static_cast<std::uint64_t>(c - '0');
    ++cursor.pos;
    any = true;
  }
  return any && !cursor.failed;
}

// Skip one generic value (only used for __metadata__); bounded depth.
bool SkipJsonValue(JsonCursor& cursor, int depth) {
  if (depth > kMaxJsonDepth) {
    return false;
  }
  cursor.SkipWs();
  if (cursor.failed) {
    return false;
  }
  const char c = cursor.Peek();
  if (c == '{') {
    ++cursor.pos;
    cursor.SkipWs();
    if (cursor.Peek() == '}') {
      ++cursor.pos;
      return true;
    }
    while (true) {
      std::string ignored;
      if (!ParseJsonString(cursor, &ignored) || !cursor.Expect(':') ||
          !SkipJsonValue(cursor, depth + 1)) {
        return false;
      }
      cursor.SkipWs();
      if (cursor.Peek() == ',') {
        ++cursor.pos;
        continue;
      }
      return cursor.Expect('}');
    }
  }
  if (c == '[') {
    ++cursor.pos;
    cursor.SkipWs();
    if (cursor.Peek() == ']') {
      ++cursor.pos;
      return true;
    }
    while (true) {
      if (!SkipJsonValue(cursor, depth + 1)) {
        return false;
      }
      cursor.SkipWs();
      if (cursor.Peek() == ',') {
        ++cursor.pos;
        continue;
      }
      return cursor.Expect(']');
    }
  }
  if (c == '"') {
    std::string ignored;
    return ParseJsonString(cursor, &ignored);
  }
  if (c == 't') {
    const char* want = "true";
    for (int i = 0; i < 4; ++i) {
      if (!cursor.Need(1) ||
          static_cast<char>(cursor.data[cursor.pos++]) != want[i]) {
        return false;
      }
    }
    return true;
  }
  if (c == 'f') {
    const char* want = "false";
    for (int i = 0; i < 5; ++i) {
      if (!cursor.Need(1) ||
          static_cast<char>(cursor.data[cursor.pos++]) != want[i]) {
        return false;
      }
    }
    return true;
  }
  if (c == 'n') {
    const char* want = "null";
    for (int i = 0; i < 4; ++i) {
      if (!cursor.Need(1) ||
          static_cast<char>(cursor.data[cursor.pos++]) != want[i]) {
        return false;
      }
    }
    return true;
  }
  std::uint64_t ignored = 0;
  if (c == '-' || (c >= '0' && c <= '9')) {
    if (c == '-') {
      ++cursor.pos;
    }
    if (!ParseJsonU64(cursor, &ignored)) {
      return false;
    }
    // No floats or exponents in skipped metadata; a fraction mark
    // ends the skip.
    cursor.SkipWs();
    if (!cursor.failed && (cursor.Peek() == '.' || cursor.Peek() == 'e' ||
                           cursor.Peek() == 'E')) {
      return false;
    }
    return true;
  }
  return false;
}

struct MapEntry {
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
  bool has_dtype = false;
  bool has_shape = false;
  bool has_offsets = false;
};

bool ParseEntryObject(JsonCursor& cursor, MapEntry* entry) {
  if (!cursor.Expect('{')) {
    return false;
  }
  cursor.SkipWs();
  if (cursor.Peek() == '}') {
    ++cursor.pos;
    return false;  // empty entries carry no tensor
  }
  while (true) {
    std::string key;
    if (!ParseJsonString(cursor, &key) || !cursor.Expect(':')) {
      return false;
    }
    if (key == "dtype") {
      if (entry->has_dtype || !ParseJsonString(cursor, &entry->dtype)) {
        return false;
      }
      entry->has_dtype = true;
    } else if (key == "shape") {
      if (entry->has_shape || !cursor.Expect('[')) {
        return false;
      }
      entry->has_shape = true;
      cursor.SkipWs();
      if (cursor.Peek() == ']') {
        ++cursor.pos;
      } else {
        while (true) {
          std::uint64_t dim = 0;
          if (!ParseJsonU64(cursor, &dim)) {
            return false;
          }
          entry->shape.push_back(dim);
          cursor.SkipWs();
          if (cursor.Peek() == ',') {
            ++cursor.pos;
            continue;
          }
          if (!cursor.Expect(']')) {
            return false;
          }
          break;
        }
      }
    } else if (key == "data_offsets") {
      if (entry->has_offsets || !cursor.Expect('[')) {
        return false;
      }
      if (!ParseJsonU64(cursor, &entry->begin) || !cursor.Expect(',') ||
          !ParseJsonU64(cursor, &entry->end)) {
        return false;
      }
      if (!cursor.Expect(']')) {
        return false;
      }
      entry->has_offsets = true;
    } else {
      return false;  // the schema has no other fields
    }
    cursor.SkipWs();
    if (cursor.Peek() == ',') {
      ++cursor.pos;
      continue;
    }
    if (!cursor.Expect('}')) {
      return false;
    }
    return entry->has_dtype && entry->has_shape && entry->has_offsets;
  }
}

std::optional<DType> SafetensorsDType(std::string_view name) {
  if (name == "F32") return DType::F32;
  if (name == "F16") return DType::F16;
  if (name == "BF16") return DType::BF16;
  if (name == "F8_E4M3") return DType::F8E4M3;
  if (name == "F8_E5M2") return DType::F8E5M2;
  if (name == "I32") return DType::I32;
  if (name == "I64") return DType::I64;
  return std::nullopt;
}

bool EndsWith(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() &&
         text.substr(text.size() - suffix.size()) == suffix;
}

}  // namespace

std::expected<SafetensorsFileCheck, StatusCode> InspectSafetensorsFile(
    const std::filesystem::path& path) {
  auto data = ReadFile(path);
  if (!data) {
    return std::unexpected(data.error());
  }
  if (data->size() < kSafetensorsMinSize) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::uint64_t header_len = 0;
  std::memcpy(&header_len, data->data(), 8);  // little-endian on all targets
  if (header_len == 0 || header_len > kMaxSafetensorsHeaderLen) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (data->size() < 8 + static_cast<std::size_t>(header_len)) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (data->at(8) != static_cast<std::byte>('{')) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return SafetensorsFileCheck{data->size(), header_len};
}

std::expected<MxFp4Layout, StatusCode> InspectMxFp4Directory(
    const std::filesystem::path& dir) {
  std::error_code ec;
  if (!std::filesystem::exists(dir, ec) || ec) {
    return std::unexpected(StatusCode::FileNotFound);
  }
  if (!std::filesystem::is_directory(dir, ec) || ec) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (!std::filesystem::exists(dir / "config.json", ec) || ec) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  // Deterministic choice: lexicographically first *.safetensors.
  std::vector<std::filesystem::path> weights;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (!entry.is_regular_file(ec) || ec) continue;
    if (entry.path().extension() == ".safetensors") {
      weights.push_back(entry.path());
    }
  }
  if (ec) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (weights.empty()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  if (weights.size() > 1) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::sort(weights.begin(), weights.end());
  auto check = InspectSafetensorsFile(weights.front());
  if (!check) {
    return std::unexpected(check.error());
  }
  return MxFp4Layout{
      weights.front().string(), check->header_len, check->file_size};
}

std::expected<std::vector<SafetensorsTensor>, StatusCode> ParseSafetensorsMap(
    std::span<const std::byte> data) {
  if (data.size() < kSafetensorsMinSize) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::uint64_t header_len = 0;
  std::memcpy(&header_len, data.data(), 8);
  if (header_len == 0 || header_len > kMaxSafetensorsHeaderLen ||
      data.size() < 8 + static_cast<std::size_t>(header_len) ||
      data[8] != static_cast<std::byte>('{')) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  JsonCursor cursor{data.subspan(8, static_cast<std::size_t>(header_len))};
  if (!cursor.Expect('{')) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  struct RawEntry {
    std::string name;
    MapEntry fields;
  };
  std::vector<RawEntry> raw;
  cursor.SkipWs();
  if (cursor.Peek() != '}') {
    while (true) {
      std::string key;
      if (!ParseJsonString(cursor, &key) || !cursor.Expect(':')) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      if (key == "__metadata__") {
        if (!SkipJsonValue(cursor, 0)) {
          return std::unexpected(StatusCode::MalformedFile);
        }
      } else {
        MapEntry entry;
        if (!ParseEntryObject(cursor, &entry)) {
          return std::unexpected(StatusCode::MalformedFile);
        }
        for (const auto& seen : raw) {
          if (seen.name == key) {
            return std::unexpected(StatusCode::MalformedFile);
          }
        }
        raw.push_back(RawEntry{key, std::move(entry)});
      }
      cursor.SkipWs();
      if (cursor.Peek() == ',') {
        ++cursor.pos;
        continue;
      }
      break;
    }
  }
  if (!cursor.Expect('}') || cursor.failed) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  cursor.SkipWs();
  if (cursor.pos != cursor.data.size()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  const std::uint64_t data_size =
      static_cast<std::uint64_t>(data.size()) - 8 - header_len;
  const auto find = [&raw](std::string_view name) -> const MapEntry* {
    for (const auto& seen : raw) {
      if (seen.name == name) {
        return &seen.fields;
      }
    }
    return nullptr;
  };
  std::vector<SafetensorsTensor> tensors;
  tensors.reserve(raw.size());
  for (const auto& item : raw) {
    const MapEntry& fields = item.fields;
    if (fields.begin > fields.end ||
        fields.end > data_size ||
        fields.shape.size() > TensorShape::kMaxRank) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    TensorShape shape;
    shape.rank = fields.shape.size();
    std::uint64_t numel = 1;
    for (std::size_t d = 0; d < shape.rank; ++d) {
      shape.dims[d] = static_cast<std::size_t>(fields.shape[d]);
      if (fields.shape[d] != 0 &&
          numel > data_size / fields.shape[d]) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      numel *= fields.shape[d];
    }
    DType dtype = DType::F32;
    if (fields.dtype == "U8") {
      // MXFP4 layout: blobs pair with a scale entry, U8 scales are
      // E8M0; anything else U8 is uninterpretable.
      if (EndsWith(item.name, ".weight_scale")) {
        dtype = DType::F8E8M0;
      } else {
        // One blob byte covers 2 elements, one scale byte covers 32:
        // blob[d1] pairs with scale[d1 / 16] on the same row count.
        constexpr std::uint64_t kU64HalfMax = 0x7FFFFFFFFFFFFFFFull;
        constexpr std::uint64_t kU64Max16 = 0x0FFFFFFFFFFFFFFFull;
        const MapEntry* scale = find(item.name + "_scale");
        if (scale == nullptr || scale->dtype != "U8" ||
            fields.shape.size() != 2 || scale->shape.size() != 2 ||
            fields.shape[0] != scale->shape[0] ||
            fields.shape[1] > kU64HalfMax ||
            scale->shape[1] > kU64Max16 ||
            fields.shape[1] * 2 != scale->shape[1] * 32) {
          return std::unexpected(StatusCode::MalformedFile);
        }
        const std::uint64_t wide = fields.shape[1] * 2;
        if (shape.dims[0] != 0 && wide > kU64HalfMax * 2 / shape.dims[0]) {
          return std::unexpected(StatusCode::MalformedFile);
        }
        dtype = DType::F4E2M1;
        shape.dims[1] = static_cast<std::size_t>(wide);
        numel = static_cast<std::uint64_t>(shape.dims[0]) * wide;
      }
    } else {
      auto mapped = SafetensorsDType(fields.dtype);
      if (!mapped) {
        return std::unexpected(StatusCode::UnsupportedFeature);
      }
      dtype = *mapped;
    }
    auto sized = TensorBytes(dtype, static_cast<std::size_t>(numel));
    if (!sized || *sized != fields.end - fields.begin) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    TensorEntry entry;
    entry.name = item.name;
    entry.shape = shape;
    entry.dtype = dtype;
    tensors.push_back(
        SafetensorsTensor{entry.name, entry, 8 + header_len + fields.begin,
                          8 + header_len + fields.end});
  }
  return tensors;
}

}  // namespace tessera::core
