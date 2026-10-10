#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>

namespace tessera::backends::vulkan {

// SPIR-V opcodes and decorations used to recover a kernel's per-binding
// access. glslang emits NonWritable for a `readonly buffer` block and
// NonReadable for a `writeonly buffer` block; a plain block gets neither and
// counts as read+write.
inline constexpr std::uint32_t kSpirvMagicWord = 0x07230203u;
inline constexpr std::uint32_t kSpirvOpDecorate = 71;
inline constexpr std::uint32_t kSpirvDecorationBinding = 33;
inline constexpr std::uint32_t kSpirvDecorationNonWritable = 24;
inline constexpr std::uint32_t kSpirvDecorationNonReadable = 25;

// Decode a SPIR-V module's buffer-block access into read_mask/write_mask
// (bit i set for binding i). Leaves both zero when the module declares no
// such binding, so a caller can fall back to treating every binding as
// read+write. Tolerates truncated or non-SPIR-V input by returning zero.
inline void ParseSpirvBindingAccess(std::span<const std::byte> code,
                                    std::uint32_t* read_mask,
                                    std::uint32_t* write_mask) {
  *read_mask = 0;
  *write_mask = 0;
  if (code.size() < 20 || code.size() % 4 != 0) {
    return;
  }
  const std::uint32_t* words =
      reinterpret_cast<const std::uint32_t*>(code.data());
  const std::size_t count = code.size() / 4;
  if (words[0] != kSpirvMagicWord) {
    return;
  }
  struct Access {
    int binding = -1;
    bool non_writable = false;
    bool non_readable = false;
  };
  std::unordered_map<std::uint32_t, Access> access;
  std::size_t i = 5;  // skip the five-word module header
  while (i < count) {
    const std::uint32_t inst = words[i];
    const std::uint32_t op = inst & 0xffffu;
    const std::uint32_t wc = inst >> 16;
    if (wc == 0 || i + wc > count) {
      break;
    }
    if (op == kSpirvOpDecorate && wc >= 3) {
      const std::uint32_t id = words[i + 1];
      const std::uint32_t decoration = words[i + 2];
      Access& entry = access[id];
      if (decoration == kSpirvDecorationBinding && wc >= 4) {
        entry.binding = static_cast<int>(words[i + 3]);
      } else if (decoration == kSpirvDecorationNonWritable) {
        entry.non_writable = true;
      } else if (decoration == kSpirvDecorationNonReadable) {
        entry.non_readable = true;
      }
    }
    i += wc;
  }
  std::uint32_t read_bits = 0;
  std::uint32_t write_bits = 0;
  for (const auto& [id, entry] : access) {
    if (entry.binding < 0 || entry.binding >= 32) {
      continue;
    }
    const std::uint32_t bit = 1u << entry.binding;
    // NonReadable means write-only, so its absence means the shader reads the
    // buffer; NonWritable means read-only, so its absence means it writes.
    if (!entry.non_readable) {
      read_bits |= bit;
    }
    if (!entry.non_writable) {
      write_bits |= bit;
    }
  }
  *read_mask = read_bits;
  *write_mask = write_bits;
}

}  // namespace tessera::backends::vulkan
