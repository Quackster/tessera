#include "core/json.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <string>

namespace tessera::core {

namespace {

constexpr int kMaxDepth = 64;
constexpr std::size_t kMaxValues = 1 << 20;

struct Parser {
  std::string_view text;
  std::size_t pos = 0;
  std::size_t count = 0;

  void SkipWs() {
    while (pos < text.size() &&
           (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' ||
            text[pos] == '\r')) {
      ++pos;
    }
  }

  bool Fail() {
    pos = text.size() + 1;
    return false;
  }

  std::unique_ptr<Json> Value(int depth) {
    if (depth > kMaxDepth || ++count > kMaxValues) {
      Fail();
      return nullptr;
    }
    SkipWs();
    if (pos >= text.size()) {
      Fail();
      return nullptr;
    }
    const char c = text[pos];
    if (c == '{') {
      return Object(depth);
    }
    if (c == '[') {
      return Array(depth);
    }
    if (c == '"') {
      auto s = String();
      if (pos > text.size()) {
        return nullptr;
      }
      return std::make_unique<Json>(Json::String(std::move(s)));
    }
    if (c == 't' || c == 'f') {
      return Literal();
    }
    if (c == 'n') {
      if (text.substr(pos, 4) != "null") {
        Fail();
        return nullptr;
      }
      pos += 4;
      return std::make_unique<Json>(Json());
    }
    return Number();
  }

  std::unique_ptr<Json> Literal() {
    if (text.substr(pos, 4) == "true") {
      pos += 4;
      return std::make_unique<Json>(Json::Bool(true));
    }
    if (text.substr(pos, 5) == "false") {
      pos += 5;
      return std::make_unique<Json>(Json::Bool(false));
    }
    Fail();
    return nullptr;
  }

  std::unique_ptr<Json> Number() {
    const std::size_t start = pos;
    if (pos < text.size() && (text[pos] == '-' || text[pos] == '+')) {
      ++pos;
    }
    bool any = false;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
      ++pos;
      any = true;
    }
    if (pos < text.size() && text[pos] == '.') {
      ++pos;
      while (pos < text.size() &&
             std::isdigit(static_cast<unsigned char>(text[pos]))) {
        ++pos;
        any = true;
      }
    }
    if (!any) {
      Fail();
      return nullptr;
    }
    if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
      ++pos;
      if (pos < text.size() && (text[pos] == '-' || text[pos] == '+')) {
        ++pos;
      }
      while (pos < text.size() &&
             std::isdigit(static_cast<unsigned char>(text[pos]))) {
        ++pos;
      }
    }
    const double value = std::strtod(std::string(text.substr(start, pos - start)).c_str(),
                                     nullptr);
    auto out = std::make_unique<Json>(Json::Number(value));
    return out;
  }

  std::string String() {
    std::string out;
    if (pos >= text.size() || text[pos] != '"') {
      Fail();
      return out;
    }
    ++pos;
    while (pos < text.size()) {
      const char c = text[pos++];
      if (c == '"') {
        return out;
      }
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (pos >= text.size()) {
        break;
      }
      const char esc = text[pos++];
      switch (esc) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          if (pos + 4 > text.size()) {
            Fail();
            return out;
          }
          unsigned cp = 0;
          for (int i = 0; i < 4; ++i) {
            const char h = text[pos++];
            cp <<= 4;
            if (h >= '0' && h <= '9') {
              cp |= static_cast<unsigned>(h - '0');
            } else if (h >= 'a' && h <= 'f') {
              cp |= static_cast<unsigned>(h - 'a' + 10);
            } else if (h >= 'A' && h <= 'F') {
              cp |= static_cast<unsigned>(h - 'A' + 10);
            } else {
              Fail();
              return out;
            }
          }
          if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
          } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
          } else {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
          }
          break;
        }
        default:
          Fail();
          return out;
      }
    }
    Fail();
    return out;
  }

  std::unique_ptr<Json> Array(int depth) {
    ++pos;  // '['
    Json out = Json::Array();
    SkipWs();
    if (pos < text.size() && text[pos] == ']') {
      ++pos;
      return std::make_unique<Json>(std::move(out));
    }
    while (true) {
      auto value = Value(depth + 1);
      if (value == nullptr) {
        return nullptr;
      }
      out.Push(std::move(*value));
      SkipWs();
      if (pos >= text.size()) {
        Fail();
        return nullptr;
      }
      if (text[pos] == ',') {
        ++pos;
        continue;
      }
      if (text[pos] == ']') {
        ++pos;
        return std::make_unique<Json>(std::move(out));
      }
      Fail();
      return nullptr;
    }
  }

  std::unique_ptr<Json> Object(int depth) {
    ++pos;  // '{'
    Json out = Json::Object();
    SkipWs();
    if (pos < text.size() && text[pos] == '}') {
      ++pos;
      return std::make_unique<Json>(std::move(out));
    }
    while (true) {
      SkipWs();
      std::string key = String();
      if (pos > text.size()) {
        return nullptr;
      }
      SkipWs();
      if (pos >= text.size() || text[pos] != ':') {
        Fail();
        return nullptr;
      }
      ++pos;
      auto value = Value(depth + 1);
      if (value == nullptr) {
        return nullptr;
      }
      out.Set(std::move(key), std::move(*value));
      SkipWs();
      if (pos >= text.size()) {
        Fail();
        return nullptr;
      }
      if (text[pos] == ',') {
        ++pos;
        continue;
      }
      if (text[pos] == '}') {
        ++pos;
        return std::make_unique<Json>(std::move(out));
      }
      Fail();
      return nullptr;
    }
  }
};

void AppendEscaped(const std::string& s, std::string& out) {
  out.push_back('"');
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
}

void DumpInto(const Json& value, std::string& out) {
  switch (value.type()) {
    case Json::Type::Null: out += "null"; break;
    case Json::Type::Bool: out += value.AsBool() ? "true" : "false"; break;
    case Json::Type::Number: {
      const double n = value.AsNumber();
      char buf[32];
      if (n == std::floor(n) && std::fabs(n) < 1e15) {
        std::snprintf(buf, sizeof(buf), "%lld",
                      static_cast<long long>(n));
      } else {
        std::snprintf(buf, sizeof(buf), "%.10g", n);
      }
      out += buf;
      break;
    }
    case Json::Type::String: AppendEscaped(value.AsString(), out); break;
    case Json::Type::Array: {
      out.push_back('[');
      bool first = true;
      for (const Json& item : value.AsArray()) {
        if (!first) {
          out.push_back(',');
        }
        first = false;
        DumpInto(item, out);
      }
      out.push_back(']');
      break;
    }
    case Json::Type::Object: {
      out.push_back('{');
      bool first = true;
      for (const auto& [key, item] : value.AsObject()) {
        if (!first) {
          out.push_back(',');
        }
        first = false;
        AppendEscaped(key, out);
        out.push_back(':');
        DumpInto(item, out);
      }
      out.push_back('}');
      break;
    }
  }
}

}  // namespace

Json Json::Bool(bool value) {
  Json out;
  out.type_ = Type::Bool;
  out.boolean_ = value;
  return out;
}

Json Json::Number(double value) {
  Json out;
  out.type_ = Type::Number;
  out.number_ = value;
  return out;
}

Json Json::String(std::string value) {
  Json out;
  out.type_ = Type::String;
  out.string_ = std::move(value);
  return out;
}

Json Json::Array() {
  Json out;
  out.type_ = Type::Array;
  return out;
}

Json Json::Object() {
  Json out;
  out.type_ = Type::Object;
  return out;
}

const Json* Json::Find(std::string_view key) const {
  if (type_ != Type::Object) {
    return nullptr;
  }
  auto it = object_.find(std::string(key));
  return it == object_.end() ? nullptr : &it->second;
}

void Json::Set(std::string key, Json value) {
  object_[std::move(key)] = std::move(value);
}

void Json::Push(Json value) { array_.push_back(std::move(value)); }

std::string Json::Dump() const {
  std::string out;
  DumpInto(*this, out);
  return out;
}

std::unique_ptr<Json> Json::Parse(std::string_view text) {
  Parser parser{text};
  parser.SkipWs();
  auto value = parser.Value(0);
  if (value == nullptr) {
    return nullptr;
  }
  parser.SkipWs();
  if (parser.pos != text.size()) {
    return nullptr;
  }
  return value;
}

}  // namespace tessera::core
