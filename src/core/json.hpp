#pragma once

#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tessera/types.hpp"

// Minimal JSON DOM for the serving layer (parse and serialize). Bounded:
// nesting depth and value sizes are capped so a malformed body cannot
// exhaust memory. Not a general JSON library.

namespace tessera::core {

class Json {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Json() = default;
  static Json Bool(bool value);
  static Json Number(double value);
  static Json String(std::string value);
  static Json Array();
  static Json Object();

  [[nodiscard]] Type type() const { return type_; }
  [[nodiscard]] bool isNull() const { return type_ == Type::Null; }
  [[nodiscard]] bool isString() const { return type_ == Type::String; }
  [[nodiscard]] bool isObject() const { return type_ == Type::Object; }
  [[nodiscard]] bool isArray() const { return type_ == Type::Array; }

  [[nodiscard]] double AsNumber() const { return number_; }
  [[nodiscard]] bool AsBool() const { return boolean_; }
  [[nodiscard]] const std::string& AsString() const { return string_; }
  [[nodiscard]] const std::vector<Json>& AsArray() const { return array_; }
  [[nodiscard]] const std::map<std::string, Json>& AsObject() const {
    return object_;
  }

  // Object lookup; nullptr when absent or not an object.
  [[nodiscard]] const Json* Find(std::string_view key) const;
  void Set(std::string key, Json value);
  void Push(Json value);

  // Serialize compactly (no whitespace).
  [[nodiscard]] std::string Dump() const;

  // Parse `text`; nullptr (in `ok`) when malformed.
  [[nodiscard]] static std::unique_ptr<Json> Parse(std::string_view text);

 private:
  Type type_ = Type::Null;
  bool boolean_ = false;
  double number_ = 0.0;
  std::string string_;
  std::vector<Json> array_;
  std::map<std::string, Json> object_;
};

// Required-field accessors for model config parsing. Each returns
// MalformedFile when the key is absent or the value has the wrong type;
// JsonU64 also rejects negative, fractional and huge numbers, JsonReal
// rejects non-positive numbers. Shared by every config parser so the
// validation rules stay in one place.
[[nodiscard]] std::expected<std::uint64_t, StatusCode> JsonU64(
    const Json& object, std::string_view key);
[[nodiscard]] std::expected<double, StatusCode> JsonReal(const Json& object,
                                                         std::string_view key);
[[nodiscard]] std::expected<bool, StatusCode> JsonBool(const Json& object,
                                                       std::string_view key);
[[nodiscard]] std::expected<std::string_view, StatusCode> JsonString(
    const Json& object, std::string_view key);

// The text-model sub-object of a multimodal config (the "text_config"
// key), or the root itself when the config is text-only.
[[nodiscard]] const Json& JsonTextConfig(const Json& root);

}  // namespace tessera::core
