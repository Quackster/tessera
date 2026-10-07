#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// A compact Jinja2 subset engine for chat templates. It supports the
// constructs the model chat templates use: {{ expr }}, {% if/elif/else/
// endif %}, {% for x in y %}, {% set %}, {% macro %}, namespace(),
// filters (trim, tojson, items, length, string, safe, default), tests
// (is string/iterable/mapping/undefined/defined/none/true/false),
// slicing, loop.* variables, string concat (~, +), comparisons and
// boolean operators. Whitespace control ({%- -%}) is honored.
//
// Values: undefined, none, bool, int, double, string, list, map
// (ordered), and callables (builtins).

namespace tessera::serve::jinja {

class Value;

struct Callable;

class Value {
 public:
  enum class Type { Undefined, None, Bool, Int, Double, Str, List, Map,
                    Callable };

  Value() = default;
  static Value None();
  static Value Bool(bool value);
  static Value Int(std::int64_t value);
  static Value Double(double value);
  static Value Str(std::string value);
  static Value List(std::vector<Value> value);
  static Value Map(std::vector<std::pair<std::string, Value>> value);
  static Value Undefined(std::string name);
  static Value Fn(std::shared_ptr<Callable> fn);

  [[nodiscard]] Type type() const { return type_; }
  [[nodiscard]] bool isUndefined() const { return type_ == Type::Undefined; }
  [[nodiscard]] bool isList() const { return type_ == Type::List; }
  [[nodiscard]] bool isMap() const { return type_ == Type::Map; }
  [[nodiscard]] bool isStr() const { return type_ == Type::Str; }
  [[nodiscard]] bool isNumber() const {
    return type_ == Type::Int || type_ == Type::Double;
  }

  [[nodiscard]] bool asBool() const { return bool_; }
  [[nodiscard]] std::int64_t asInt() const;
  [[nodiscard]] double asDouble() const;
  [[nodiscard]] const std::string& asStr() const { return str_; }
  [[nodiscard]] const std::string& undefinedName() const { return str_; }
  [[nodiscard]] std::vector<Value>& list() { return list_; }
  [[nodiscard]] const std::vector<Value>& list() const { return list_; }
  [[nodiscard]] std::vector<std::pair<std::string, Value>>& map() {
    return map_;
  }
  [[nodiscard]] const std::vector<std::pair<std::string, Value>>& map() const {
    return map_;
  }
  [[nodiscard]] const std::shared_ptr<Callable>& callable() const {
    return fn_;
  }

  // Truthiness (undefined/none/false/0/empty string/empty list/empty map
  // are false).
  [[nodiscard]] bool truthy() const;
  // String form (Jinja str()).
  [[nodiscard]] std::string str() const;

  [[nodiscard]] const Value* find(std::string_view key) const;
  void set(std::string key, Value value);

 private:
  Type type_ = Type::Undefined;
  bool bool_ = false;
  std::int64_t int_ = 0;
  double double_ = 0.0;
  std::string str_;
  std::vector<Value> list_;
  std::vector<std::pair<std::string, Value>> map_;
  std::shared_ptr<Callable> fn_;
};

using Kwarg = std::pair<std::string, Value>;

// A builtin function (raise_exception, namespace, range, ...).
struct Callable {
  std::function<Value(const std::vector<Value>&, const std::vector<Kwarg>&)>
      call;
};

// Result of rendering a template.
struct RenderResult {
  bool ok = false;
  std::string text;
  std::string error;
};

// Render `template_text` with the given top-level context. On a template
// error (parse or raise_exception), ok is false and error explains.
[[nodiscard]] RenderResult Render(std::string_view template_text,
                                  const std::vector<Kwarg>& context);

}  // namespace tessera::serve::jinja
