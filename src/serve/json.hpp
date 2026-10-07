#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// Minimal JSON DOM for the serving layer (parse and serialize). Bounded:
// nesting depth and value sizes are capped so a malformed body cannot
// exhaust memory. Not a general JSON library.

namespace tessera::serve {

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

}  // namespace tessera::serve
