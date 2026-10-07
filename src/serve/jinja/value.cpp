#include "serve/jinja/jinja.hpp"

#include <cmath>
#include <cstdio>

namespace tessera::serve::jinja {

Value Value::None() {
  Value v;
  v.type_ = Type::None;
  return v;
}
Value Value::Bool(bool value) {
  Value v;
  v.type_ = Type::Bool;
  v.bool_ = value;
  return v;
}
Value Value::Int(std::int64_t value) {
  Value v;
  v.type_ = Type::Int;
  v.int_ = value;
  return v;
}
Value Value::Double(double value) {
  Value v;
  v.type_ = Type::Double;
  v.double_ = value;
  return v;
}
Value Value::Str(std::string value) {
  Value v;
  v.type_ = Type::Str;
  v.str_ = std::move(value);
  return v;
}
Value Value::List(std::vector<Value> value) {
  Value v;
  v.type_ = Type::List;
  v.list_ = std::move(value);
  return v;
}
Value Value::Map(std::vector<std::pair<std::string, Value>> value) {
  Value v;
  v.type_ = Type::Map;
  v.map_ = std::move(value);
  return v;
}
Value Value::Undefined(std::string name) {
  Value v;
  v.type_ = Type::Undefined;
  v.str_ = std::move(name);
  return v;
}
Value Value::Fn(std::shared_ptr<Callable> fn) {
  Value v;
  v.type_ = Type::Callable;
  v.fn_ = std::move(fn);
  return v;
}

std::int64_t Value::asInt() const {
  if (type_ == Type::Int) {
    return int_;
  }
  if (type_ == Type::Double) {
    return static_cast<std::int64_t>(double_);
  }
  if (type_ == Type::Bool) {
    return bool_ ? 1 : 0;
  }
  return 0;
}

double Value::asDouble() const {
  if (type_ == Type::Double) {
    return double_;
  }
  if (type_ == Type::Int) {
    return static_cast<double>(int_);
  }
  if (type_ == Type::Bool) {
    return bool_ ? 1.0 : 0.0;
  }
  return 0.0;
}

bool Value::truthy() const {
  switch (type_) {
    case Type::Undefined:
    case Type::None: return false;
    case Type::Bool: return bool_;
    case Type::Int: return int_ != 0;
    case Type::Double: return double_ != 0.0;
    case Type::Str: return !str_.empty();
    case Type::List: return !list_.empty();
    case Type::Map: return !map_.empty();
    case Type::Callable: return true;
  }
  return false;
}

std::string Value::str() const {
  switch (type_) {
    case Type::Undefined: return "";
    case Type::None: return "None";
    case Type::Bool: return bool_ ? "True" : "False";
    case Type::Int: return std::to_string(int_);
    case Type::Double: {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%g", double_);
      return buf;
    }
    case Type::Str: return str_;
    case Type::Callable: return "<function>";
    case Type::List: {
      std::string out = "[";
      for (std::size_t i = 0; i < list_.size(); ++i) {
        if (i != 0) {
          out += ", ";
        }
        out += list_[i].isStr() ? std::string("'") + list_[i].str() + "'"
                                : list_[i].str();
      }
      out += "]";
      return out;
    }
    case Type::Map: {
      std::string out = "{";
      for (std::size_t i = 0; i < map_.size(); ++i) {
        if (i != 0) {
          out += ", ";
        }
        out += "'" + map_[i].first + "': " + map_[i].second.str();
      }
      out += "}";
      return out;
    }
  }
  return "";
}

const Value* Value::find(std::string_view key) const {
  if (type_ == Type::Map) {
    for (const auto& [k, v] : map_) {
      if (k == key) {
        return &v;
      }
    }
  }
  return nullptr;
}

void Value::set(std::string key, Value value) {
  for (auto& [k, v] : map_) {
    if (k == key) {
      v = std::move(value);
      return;
    }
  }
  map_.emplace_back(std::move(key), std::move(value));
}

}  // namespace tessera::serve::jinja
