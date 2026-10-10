#pragma once

// Shared helpers for the tessera test suite (single test binary; see
// AGENTS.md "Building and Testing").

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/numerics/quant.hpp"
#include "tessera/engine.hpp"

namespace tessera::testing {

// GPU index the test run targets (default 0). Reads TESSERA_TEST_GPU so
// a run can pick a free device when another process holds the default
// one. Every device-test helper uses this, so the suite targets one GPU.
inline int TestDeviceIndex() {
  if (const char* gpu = std::getenv("TESSERA_TEST_GPU")) {
    return std::atoi(gpu);
  }
  return 0;
}

// Create an engine for a device test; skips cleanly without a device.
inline void MakeEngineOrSkip(std::unique_ptr<Engine>& engine) {
  EngineOptions options;
  options.device_index = TestDeviceIndex();
  auto created = Engine::Create(options);
  if (!created) {
    GTEST_SKIP() << "no device available: "
                 << tessera::ToString(created.error());
  }
  engine = std::move(*created);
}

// Defined below; forward-declared so fixtures can use them.
inline float DrawValue(std::mt19937& rng);
inline std::vector<std::byte> QuantizeRows(const std::vector<float>& values,
                                           std::size_t n, std::size_t k);

// Builds a GGUF image in memory (little-endian, v2/v3 spec layout:
// u64 string lengths without NUL, 13-type value table, variable dims,
// u32 ggml type).
struct GgufBuilder {
  std::vector<std::byte> bytes;

  void PushU8(std::uint8_t value) {
    bytes.push_back(static_cast<std::byte>(value));
  }

  void PushU16(std::uint16_t value) {
    for (int i = 0; i < 2; ++i) {
      bytes.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xFF));
    }
  }

  void PushU32(std::uint32_t value) {
    for (int i = 0; i < 4; ++i) {
      bytes.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xFF));
    }
  }

  void PushU64(std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
      bytes.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xFF));
    }
  }

  void PushF32(float value) {
    std::uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    PushU32(bits);
  }

  void PushF64(double value) {
    std::uint64_t bits;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    PushU64(bits);
  }

  // A GGUF string: u64 byte length + the bytes (no NUL, per the spec).
  void PushString(const char* text) {
    const auto len = std::strlen(text);
    PushU64(static_cast<std::uint64_t>(len));
    for (std::size_t i = 0; i < len; ++i) {
      bytes.push_back(static_cast<std::byte>(text[i]));
    }
  }

  void Header(std::uint32_t magic, std::uint32_t version,
              std::uint64_t tensor_count, std::uint64_t kv_count) {
    PushU32(magic);
    PushU32(version);
    PushU64(tensor_count);
    PushU64(kv_count);
  }

  // Metadata key + value; the types follow the spec value table.
  void KvU8(const char* key, std::uint8_t value) {
    PushString(key);
    PushU32(0);
    PushU8(value);
  }

  void KvI8(const char* key, std::int8_t value) {
    PushString(key);
    PushU32(1);
    PushU8(static_cast<std::uint8_t>(value));
  }

  void KvU16(const char* key, std::uint16_t value) {
    PushString(key);
    PushU32(2);
    PushU16(value);
  }

  void KvI16(const char* key, std::int16_t value) {
    PushString(key);
    PushU32(3);
    PushU16(static_cast<std::uint16_t>(value));
  }

  void KvU32(const char* key, std::uint32_t value) {
    PushString(key);
    PushU32(4);
    PushU32(value);
  }

  void KvI32(const char* key, std::int32_t value) {
    PushString(key);
    PushU32(5);
    PushU32(static_cast<std::uint32_t>(value));
  }

  void KvF32(const char* key, float value) {
    PushString(key);
    PushU32(6);
    PushF32(value);
  }

  void KvBool(const char* key, bool value) {
    PushString(key);
    PushU32(7);
    PushU8(value ? 1 : 0);
  }

  // A bool byte the spec rejects (only 0 and 1 are valid).
  void KvBoolRaw(const char* key, std::uint8_t raw) {
    PushString(key);
    PushU32(7);
    PushU8(raw);
  }

  void KvString(const char* key, const char* value) {
    PushString(key);
    PushU32(8);
    PushString(value);
  }

  void KvU64(const char* key, std::uint64_t value) {
    PushString(key);
    PushU32(10);
    PushU64(value);
  }

  void KvI64(const char* key, std::int64_t value) {
    PushString(key);
    PushU32(11);
    PushU64(static_cast<std::uint64_t>(value));
  }

  void KvF64(const char* key, double value) {
    PushString(key);
    PushU32(12);
    PushF64(value);
  }

  // Metadata array of u32 values (type 9: element type, count, values).
  void KvArrayU32(const char* key,
                  std::initializer_list<std::uint32_t> values) {
    PushString(key);
    PushU32(9);
    PushU32(4);  // element type uint32
    PushU64(values.size());
    for (std::uint32_t value : values) {
      PushU32(value);
    }
  }

  // Metadata string array (GGUF array of strings, element type 8).
  void KvArrayString(const char* key,
                     std::initializer_list<const char*> values) {
    PushString(key);
    PushU32(9);
    PushU32(8);  // element type string
    PushU64(values.size());
    for (const char* value : values) {
      PushString(value);
    }
  }

  // Metadata int32 array (element type 5; GGUF has no byte-wide int).
  void KvArrayI32(const char* key,
                  std::initializer_list<std::int32_t> values) {
    PushString(key);
    PushU32(9);
    PushU32(5);  // element type int32
    PushU64(values.size());
    for (std::int32_t value : values) {
      PushU32(static_cast<std::uint32_t>(value));
    }
  }

  // Tensor entry: name, rank, rank dims (u64), ggml type, offset.
  void Tensor(const char* name, std::uint32_t rank,
              std::initializer_list<std::uint64_t> dims, std::uint32_t ggml_type,
              std::uint64_t offset) {
    PushString(name);
    PushU32(rank);
    auto dim_it = dims.begin();
    for (std::uint32_t d = 0; d < rank; ++d) {
      PushU64(dim_it != dims.end() ? *dim_it++ : 1);
    }
    PushU32(ggml_type);
    PushU64(offset);
  }

  // Append zero bytes as the (fake) tensor payload region.
  void PadPayload(std::size_t count) {
    bytes.insert(bytes.end(), count, std::byte{0});
  }

  // Grow the image with zero bytes to an absolute size.
  void PadTo(std::size_t size) {
    if (bytes.size() < size) {
      bytes.insert(bytes.end(), size - bytes.size(), std::byte{0});
    }
  }
};

// Build a minimal valid GGUF v3 file: one string kv, one f32 tensor.
inline std::vector<std::byte> MakeValidGguf() {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1, 1);
  builder.KvString("general.name", "test-model");
  builder.Tensor("w_a", 1, {4}, 0, 0);  // F32, 1x4
  // The payload region starts at align32(info_end); pad past it.
  builder.PadTo(((builder.bytes.size() + 31) & ~31u) + 16);
  return builder.bytes;
}

// Write raw bytes to a path, creating/truncating it.
inline void WriteBytes(const std::filesystem::path& path,
                       const std::vector<std::byte>& data) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!data.empty()) {
    stream.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size()));
  }
}

inline void WriteString(const std::filesystem::path& path,
                        const std::string& text) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(text.data(), static_cast<std::streamsize>(text.size()));
}

// A stable (wall-clock independent) temp directory for test fixtures.
inline std::filesystem::path FreshTempDir(const std::string& name) {
  auto dir = std::filesystem::temp_directory_path() / name;
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir, ec);
  return dir;
}

// A tiny hybrid-shaped GGUF (interval 2, one MTP block, vocab 8).
// Skipping the interval key leaves the ssm set partial for tests.
inline std::filesystem::path WriteHybridFixture(const std::string& name,
                                                bool complete,
                                                bool sections = true) {
  GgufBuilder builder;
  builder.Header(0x46554747, 3, 1,
                 complete ? (sections ? 20 : 19) : (sections ? 19 : 18));
  builder.KvString("general.name", "tiny-hybrid");
  builder.KvString("general.architecture", "qwen35");
  builder.KvU32("qwen35.block_count", 4);
  builder.KvU32("qwen35.embedding_length", 256);
  builder.KvU32("qwen35.feed_forward_length", 512);
  builder.KvF32("qwen35.attention.layer_norm_rms_epsilon", 1e-5f);
  builder.KvU32("qwen35.attention.head_count", 4);
  builder.KvU32("qwen35.attention.head_count_kv", 2);
  builder.KvU32("qwen35.attention.key_length", 64);
  builder.KvU32("qwen35.attention.value_length", 64);
  builder.KvU32("qwen35.rope.dimension_count", 32);
  builder.KvF32("qwen35.rope.freq_base", 10000.0f);
  if (sections) {
    builder.PushString("qwen35.rope.dimension_sections");
    builder.PushU32(9);
    builder.PushU32(4);
    builder.PushU64(4);
    for (std::uint32_t s : {1u, 1u, 1u, 0u}) {
      builder.PushU32(s);
    }
  }
  builder.KvU32("qwen35.ssm.conv_kernel", 2);
  builder.KvU32("qwen35.ssm.state_size", 8);
  builder.KvU32("qwen35.ssm.group_count", 2);
  builder.KvU32("qwen35.ssm.time_step_rank", 4);
  builder.KvU32("qwen35.ssm.inner_size", 32);
  if (complete) {
    builder.KvU32("qwen35.full_attention_interval", 2);
  }
  builder.KvU32("qwen35.nextn_predict_layers", 1);
  builder.Tensor("output.weight", 2, {256, 8}, 0, 0);
  builder.PadTo(((builder.bytes.size() + 31) & ~31u) + 8192);
  auto dir = FreshTempDir("tessera_tests_hybrid_model");
  auto path = dir / name;
  WriteBytes(path, builder.bytes);
  return path;
}

// A valid safetensors container: 8-byte LE header length + JSON bytes.
inline std::vector<std::byte> MakeSafetensorsContainer(
    const std::string& json, bool override_len = false,
    std::uint64_t len_override = 0) {
  std::vector<std::byte> out;
  const std::uint64_t len =
      override_len ? len_override : static_cast<std::uint64_t>(json.size());
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::byte>((len >> (8 * i)) & 0xFF));
  }
  for (char c : json) {
    out.push_back(static_cast<std::byte>(c));
  }
  return out;
}

// Per backend tolerance (AGENTS.md: assert per backend tolerance, never
// a single hard-coded epsilon). Both backends run fp32 sequential
// accumulation; the table is the assertion point per device.
struct GemmTolerance {
  float abs = 0.0f;
  float rel = 0.0f;
};
inline GemmTolerance ToleranceFor(std::string_view backend) {
  if (backend == "vulkan") {
    return {1.0e-4f, 1.0e-5f};
  }
  if (backend == "rocm") {
    return {1.0e-4f, 1.0e-5f};
  }
  return {1.0e-3f, 1.0e-4f};
}

// Deterministic draw in [-1, 1] (fixed seeds, no wall clock).
inline float DrawValue(std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  return dist(rng);
}

// The 6-bit scale/min pair of sub-block j (test oracle mirroring the
// core packing).
inline void TestGetScaleMin(std::size_t j, const std::byte* scales,
                            std::uint8_t* scale, std::uint8_t* min) {
  const auto byte = [scales](std::size_t i) {
    return static_cast<std::uint8_t>(scales[i]);
  };
  if (j < 4) {
    *scale = byte(j) & 63;
    *min = byte(j + 4) & 63;
  } else {
    *scale = (byte(j + 4) & 15) | ((byte(j - 4) >> 6) << 4);
    *min = (byte(j + 4) >> 4) | ((byte(j) >> 6) << 4);
  }
}

// Quantize n rows of k values (k % 256 == 0) into Q4_K blocks.
inline std::vector<std::byte> QuantizeRows(const std::vector<float>& values,
                                           std::size_t n, std::size_t k) {
  std::vector<std::byte> out(n * (k / tessera::kQ4KBlockElements) *
                             core::kQ4KBlockBytes);
  std::vector<float> block(tessera::kQ4KBlockElements);
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t b = 0; b < k / tessera::kQ4KBlockElements; ++b) {
      for (std::size_t i = 0; i < tessera::kQ4KBlockElements; ++i) {
        block[i] = values[r * k + b * tessera::kQ4KBlockElements + i];
      }
      core::QuantizeQ4K(
          std::span<const float>(block),
          out.data() + (r * (k / tessera::kQ4KBlockElements) + b) *
              core::kQ4KBlockBytes);
    }
  }
  return out;
}

// The fp16 header bytes of a block, read little-endian.
inline std::uint16_t BlockU16(const std::byte* block, std::size_t offset) {
  return static_cast<std::uint16_t>(
      static_cast<std::uint16_t>(
          static_cast<std::uint8_t>(block[offset])) |
      (static_cast<std::uint16_t>(
          static_cast<std::uint8_t>(block[offset + 1])) << 8));
}

// A config.json placeholder good enough for layout checks.
inline void WritePlaceholderConfig(const std::filesystem::path& dir) {
  WriteString(dir / "config.json", R"({"architectures":["TesseraTest"]})");
}

inline void WritePlaceholderWeights(const std::filesystem::path& dir,
                                    const std::string& file_name =
                                        "model.safetensors") {
  auto container = MakeSafetensorsContainer(
      R"({"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})");
  // Payload: four little-endian floats (1, 2, 3, 4).
  const std::uint32_t words[4] = {0x3F800000, 0x40000000, 0x40400000,
                                  0x40800000};
  for (auto bits : words) {
    for (int i = 0; i < 4; ++i) {
      container.push_back(static_cast<std::byte>((bits >> (8 * i)) & 0xFF));
    }
  }
  WriteBytes(dir / file_name, container);
}

}  // namespace tessera::testing
