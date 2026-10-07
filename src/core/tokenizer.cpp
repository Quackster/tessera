#include "tessera/tokenizer.hpp"

#include <array>
#include <cctype>
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tessera {

namespace {

// Append codepoint `cp` to `out` as UTF-8.
void Utf8Append(std::uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Append the codepoints of `s` (UTF-8) to `out`; invalid bytes are
// passed through as their own value.
void Utf8Decode(std::string_view s, std::vector<std::uint32_t>& out) {
  std::size_t i = 0;
  while (i < s.size()) {
    const auto b0 = static_cast<std::uint8_t>(s[i]);
    std::uint32_t cp = b0;
    std::size_t len = 1;
    if (b0 >= 0xF0) {
      len = 4;
      cp = b0 & 0x07;
    } else if (b0 >= 0xE0) {
      len = 3;
      cp = b0 & 0x0F;
    } else if (b0 >= 0xC0) {
      len = 2;
      cp = b0 & 0x1F;
    }
    if (i + len > s.size()) {
      len = 1;
      cp = b0;
    } else {
      for (std::size_t k = 1; k < len; ++k) {
        cp = (cp << 6) | (static_cast<std::uint8_t>(s[i + k]) & 0x3F);
      }
    }
    out.push_back(cp);
    i += len;
  }
}

// GPT-2 byte <-> unicode tables: 256 byte pieces and the reverse map.
struct ByteUnicode {
  std::array<std::string, 256> piece;
  std::unordered_map<std::uint32_t, std::uint8_t> to_byte;

  ByteUnicode() {
    std::vector<std::uint32_t> bs;
    for (std::uint32_t b = 0x21; b <= 0x7E; ++b) {
      bs.push_back(b);
    }
    for (std::uint32_t b = 0xA1; b <= 0xAC; ++b) {
      bs.push_back(b);
    }
    for (std::uint32_t b = 0xAE; b <= 0xFF; ++b) {
      bs.push_back(b);
    }
    std::vector<std::uint32_t> cs = bs;
    std::uint32_t n = 0;
    for (std::uint32_t b = 0; b < 256; ++b) {
      bool present = false;
      for (std::uint32_t x : bs) {
        if (x == b) {
          present = true;
          break;
        }
      }
      if (!present) {
        bs.push_back(b);
        cs.push_back(256 + n);
        ++n;
      }
    }
    for (std::size_t i = 0; i < bs.size(); ++i) {
      Utf8Append(cs[i], piece[bs[i]]);
      to_byte[cs[i]] = static_cast<std::uint8_t>(bs[i]);
    }
  }
};

const ByteUnicode& ByteTable() {
  static const ByteUnicode table;
  return table;
}

bool IsSpaceByte(std::uint8_t b) {
  return b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D || b == 0x0B ||
         b == 0x0C;
}
bool IsNewlineByte(std::uint8_t b) { return b == 0x0A || b == 0x0D; }
bool IsDigitByte(std::uint8_t b) { return b >= '0' && b <= '9'; }
bool IsAsciiLetter(std::uint8_t b) {
  return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z');
}
// Bytes >= 0x80 are treated as letters (UTF-8 lead/continuation bytes of
// letters); this matches the reference for Latin and CJK text.
bool IsLetterByte(std::uint8_t b) { return b >= 0x80 || IsAsciiLetter(b); }

// Length (bytes) of the Qwen pre-tokenization match starting at `i`.
std::size_t PreTokenLength(std::string_view s, std::size_t i) {
  const auto at = [&s](std::size_t k) {
    return static_cast<std::uint8_t>(s[k]);
  };
  const std::size_t n = s.size();
  // 1. Case-insensitive contractions: 's 't 're 've 'm 'll 'd.
  if (at(i) == '\'') {
    static const char* kSuffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (const char* suffix : kSuffixes) {
      std::size_t k = 1;
      const std::size_t len = std::char_traits<char>::length(suffix);
      while (k <= len && i + k < n) {
        const char lower = static_cast<char>(
            std::tolower(static_cast<unsigned char>(at(i + k))));
        if (lower != suffix[k - 1]) {
          break;
        }
        ++k;
      }
      if (k == len + 1) {
        return 1 + len;
      }
    }
  }
  // 2. Optional non-letter/number (not newline) char, then a letter run.
  if (IsLetterByte(at(i))) {
    std::size_t j = i;
    while (j < n && IsLetterByte(at(j))) {
      ++j;
    }
    return j - i;
  }
  if (!IsNewlineByte(at(i)) && !IsDigitByte(at(i)) && i + 1 < n &&
      IsLetterByte(at(i + 1))) {
    std::size_t j = i + 1;
    while (j < n && IsLetterByte(at(j))) {
      ++j;
    }
    return j - i;
  }
  // 3. A single digit.
  if (IsDigitByte(at(i))) {
    return 1;
  }
  // 4. Optional space, a punctuation run, then any trailing newlines.
  {
    std::size_t j = (at(i) == ' ') ? i + 1 : i;
    std::size_t start = j;
    while (j < n && !IsSpaceByte(at(j)) && !IsLetterByte(at(j)) &&
           !IsDigitByte(at(j))) {
      ++j;
    }
    if (j > start) {
      while (j < n && IsNewlineByte(at(j))) {
        ++j;
      }
      return j - i;
    }
  }
  // 5. Whitespace ending in a newline.
  if (IsSpaceByte(at(i))) {
    std::size_t j = i;
    while (j < n && IsSpaceByte(at(j)) && !IsNewlineByte(at(j))) {
      ++j;
    }
    if (j < n && IsNewlineByte(at(j))) {
      while (j < n && IsNewlineByte(at(j))) {
        ++j;
      }
      return j - i;
    }
    // 6. \s+(?!\S): trailing whitespace matches fully; otherwise
    // backtrack one space so the last space seeds the next
    // "space + word" token (the reference behavior).
    std::size_t k = i;
    while (k < n && IsSpaceByte(at(k))) {
      ++k;
    }
    const std::size_t run = k - i;
    if (k >= n) {
      return run;
    }
    if (run >= 2) {
      return run - 1;
    }
    return 1;  // 7. \s+
  }
  return 1;
}

// Byte-pair-encode one pre-token (raw bytes) into byte-level symbols.
std::vector<std::string> BpeEncode(
    const std::vector<std::string>& symbols_in,
    const std::unordered_map<std::string, std::int32_t>& ranks) {
  std::vector<std::string> symbols = symbols_in;
  while (symbols.size() > 1) {
    std::int32_t best_rank = 0;
    std::size_t best = symbols.size();
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      const std::string key = symbols[i] + '\x01' + symbols[i + 1];
      auto it = ranks.find(key);
      if (it != ranks.end() && (best == symbols.size() || it->second < best_rank)) {
        best_rank = it->second;
        best = i;
      }
    }
    if (best == symbols.size()) {
      break;
    }
    symbols[best] += symbols[best + 1];
    symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1));
  }
  return symbols;
}

}  // namespace

Tokenizer::Tokenizer(std::vector<std::string> vocab,
                     std::vector<std::int32_t> token_types,
                     std::vector<std::string> merges)
    : vocab_(std::move(vocab)), token_types_(std::move(token_types)) {
  token_to_id_.reserve(vocab_.size() * 2);
  for (std::size_t id = 0; id < vocab_.size(); ++id) {
    token_to_id_.emplace(vocab_[id], static_cast<std::uint32_t>(id));
  }
  for (std::size_t id = 0; id < vocab_.size() && id < token_types_.size();
       ++id) {
    // GGUF token types: 3 = control, 4 = user-defined; both are special.
    if (token_types_[id] == 3 || token_types_[id] == 4) {
      special_.emplace_back(vocab_[id], static_cast<std::uint32_t>(id));
    }
  }
  std::sort(special_.begin(), special_.end(),
            [](const auto& a, const auto& b) {
              return a.first.size() > b.first.size();
            });
  std::int32_t rank = 0;
  for (const std::string& merge : merges) {
    const std::size_t space = merge.find(' ');
    if (space == std::string::npos || space == 0 ||
        space + 1 >= merge.size()) {
      ++rank;
      continue;
    }
    merge_rank_.emplace(
        merge.substr(0, space) + '\x01' + merge.substr(space + 1), rank);
    ++rank;
  }
}

std::size_t Tokenizer::VocabSize() const { return vocab_.size(); }

std::expected<std::vector<std::uint32_t>, StatusCode> Tokenizer::Encode(
    std::string_view text) const {
  const ByteUnicode& table = ByteTable();
  std::vector<std::uint32_t> ids;
  std::size_t i = 0;
  while (i < text.size()) {
    // Match a special token literally (longest first) before BPE.
    bool matched = false;
    for (const auto& [token, id] : special_) {
      if (text.size() - i >= token.size() &&
          text.compare(i, token.size(), token) == 0) {
        ids.push_back(id);
        i += token.size();
        matched = true;
        break;
      }
    }
    if (matched) {
      continue;
    }
    const std::size_t len = PreTokenLength(text, i);
    std::vector<std::string> symbols;
    symbols.reserve(len);
    for (std::size_t k = 0; k < len; ++k) {
      symbols.push_back(
          table.piece[static_cast<std::uint8_t>(text[i + k])]);
    }
    const std::vector<std::string> merged = BpeEncode(symbols, merge_rank_);
    for (const std::string& piece : merged) {
      auto it = token_to_id_.find(piece);
      if (it != token_to_id_.end()) {
        ids.push_back(it->second);
      } else {
        // Fall back to the individual bytes (always present for Qwen).
        for (char c : piece) {
          auto single = token_to_id_.find(std::string(1, c));
          if (single != token_to_id_.end()) {
            ids.push_back(single->second);
          }
        }
      }
    }
    i += len;
  }
  return ids;
}

std::expected<std::string, StatusCode> Tokenizer::Decode(
    std::span<const std::uint32_t> ids) const {
  const ByteUnicode& table = ByteTable();
  std::string out;
  for (std::uint32_t id : ids) {
    if (id >= vocab_.size()) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    std::vector<std::uint32_t> cps;
    Utf8Decode(vocab_[id], cps);
    for (std::uint32_t cp : cps) {
      auto it = table.to_byte.find(cp);
      if (it != table.to_byte.end()) {
        out.push_back(static_cast<char>(it->second));
      }
    }
  }
  return out;
}

}  // namespace tessera
