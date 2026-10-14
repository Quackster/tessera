#include "core/sha256.hpp"

#include <array>
#include <cstring>
#include <fstream>

namespace tessera::core {

namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr std::array<std::uint32_t, 8> kInitialState = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

inline std::uint32_t RotateRight(std::uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32 - bits));
}

}  // namespace

Sha256::Sha256() : state_(kInitialState) {}

void Sha256::Transform(const std::uint8_t* block) {
  std::uint32_t w[64];
  for (std::size_t i = 0; i < 16; ++i) {
    w[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
           (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
           (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
           static_cast<std::uint32_t>(block[i * 4 + 3]);
  }
  for (std::size_t i = 16; i < 64; ++i) {
    const std::uint32_t s0 = RotateRight(w[i - 15], 7) ^
                             RotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const std::uint32_t s1 = RotateRight(w[i - 2], 17) ^
                             RotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
  std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
  for (std::size_t i = 0; i < 64; ++i) {
    const std::uint32_t s1 =
        RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
    const std::uint32_t ch = (e & f) ^ (~e & g);
    const std::uint32_t temp1 = h + s1 + ch + kRoundConstants[i] + w[i];
    const std::uint32_t s0 =
        RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
    const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temp2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

void Sha256::Update(std::span<const std::byte> data) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(data.data());
  std::size_t remaining = data.size();
  bit_count_ += static_cast<std::uint64_t>(data.size()) * 8;
  while (remaining > 0) {
    const std::size_t take = std::min(remaining, buffer_.size() - buffer_len_);
    std::memcpy(buffer_.data() + buffer_len_, bytes, take);
    buffer_len_ += take;
    bytes += take;
    remaining -= take;
    if (buffer_len_ == buffer_.size()) {
      Transform(buffer_.data());
      buffer_len_ = 0;
    }
  }
}

std::string Sha256::HexDigest() {
  const std::uint64_t bit_length = bit_count_;
  const std::uint8_t marker = 0x80;
  Update(std::span<const std::byte>(reinterpret_cast<const std::byte*>(&marker),
                                    1));
  const std::uint8_t zero = 0x00;
  while (buffer_len_ != 56) {
    Update(std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(&zero), 1));
  }
  std::uint8_t length_be[8];
  for (std::size_t i = 0; i < 8; ++i) {
    length_be[i] =
        static_cast<std::uint8_t>(bit_length >> ((7 - i) * 8));
  }
  // The length bytes are the final block's tail; append without counting
  // them again in the message length.
  std::memcpy(buffer_.data() + 56, length_be, 8);
  Transform(buffer_.data());
  buffer_len_ = 0;
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (const std::uint32_t word : state_) {
    for (int shift = 28; shift >= 0; shift -= 4) {
      out.push_back(kHex[(word >> shift) & 0xF]);
    }
  }
  return out;
}

std::string Sha256::Hex(std::span<const std::byte> data) {
  Sha256 hasher;
  hasher.Update(data);
  return hasher.HexDigest();
}

std::expected<std::string, StatusCode> Sha256File(
    const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return std::unexpected(StatusCode::FileNotFound);
  }
  Sha256 hasher;
  std::array<std::byte, 1 << 20> chunk{};
  while (file) {
    file.read(reinterpret_cast<char*>(chunk.data()),
              static_cast<std::streamsize>(chunk.size()));
    const std::streamsize read = file.gcount();
    if (read > 0) {
      hasher.Update(std::span<const std::byte>(chunk.data(),
                                               static_cast<std::size_t>(read)));
    }
  }
  return hasher.HexDigest();
}

}  // namespace tessera::core
