#include "serve/jinja/ast.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace tessera::serve::jinja {

namespace {

// Thrown by raise_exception to abort rendering with a message.
struct Abort {
  std::string message;
};

std::string JsonEscape(std::string_view s) {
  std::string out = "\"";
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out.push_back(c);
    }
  }
  out.push_back('"');
  return out;
}

std::string ToJson(const Value& value) {
  switch (value.type()) {
    case Value::Type::Undefined:
    case Value::Type::None: return "null";
    case Value::Type::Bool: return value.asBool() ? "true" : "false";
    case Value::Type::Int: return std::to_string(value.asInt());
    case Value::Type::Double: {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%g", value.asDouble());
      return buf;
    }
    case Value::Type::Str: return JsonEscape(value.asStr());
    case Value::Type::Callable: return "null";
    case Value::Type::List: {
      std::string out = "[";
      const auto& items = value.list();
      for (std::size_t i = 0; i < items.size(); ++i) {
        if (i != 0) {
          out += ",";
        }
        out += ToJson(items[i]);
      }
      out += "]";
      return out;
    }
    case Value::Type::Map: {
      std::string out = "{";
      bool first = true;
      for (const auto& [k, v] : value.map()) {
        if (!first) {
          out += ",";
        }
        first = false;
        out += JsonEscape(k) + ":" + ToJson(v);
      }
      out += "}";
      return out;
    }
  }
  return "null";
}

struct Env {
  std::vector<std::unordered_map<std::string, Value>> scopes;
  std::unordered_map<std::string, const Node*> macros;
  std::string output;

  Env() { scopes.emplace_back(); }

  Value* Find(std::string_view name) {
    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
      auto found = it->find(std::string(name));
      if (found != it->end()) {
        return &found->second;
      }
    }
    return nullptr;
  }
  void Set(const std::string& name, Value value) {
    scopes.back()[name] = std::move(value);
  }
  void Push() { scopes.emplace_back(); }
  void Pop() { scopes.pop_back(); }
};

Value EvalExpr(Env& env, const Expr& expr);
void EvalNodes(Env& env, const std::vector<NodePtr>& nodes);

bool IsIterable(const Value& v) { return v.isList() || v.isMap(); }

std::string TrimAscii(std::string s) {
  std::size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  std::size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

Value ApplyFilter(Env& env, const std::string& name, const Value& input,
                  const std::vector<Value>& args) {
  (void)env;
  if (name == "trim") return Value::Str(TrimAscii(input.str()));
  if (name == "tojson") return Value::Str(ToJson(input));
  if (name == "safe" || name == "string") return Value::Str(input.str());
  if (name == "length" || name == "count") {
    if (input.isList()) return Value::Int((std::int64_t)input.list().size());
    if (input.isMap()) return Value::Int((std::int64_t)input.map().size());
    if (input.isStr()) return Value::Int((std::int64_t)input.asStr().size());
    return Value::Int(0);
  }
  if (name == "items") {
    std::vector<Value> pairs;
    if (input.isMap()) {
      for (const auto& [k, v] : input.map()) {
        std::vector<Value> pair{Value::Str(k), v};
        pairs.push_back(Value::List(std::move(pair)));
      }
    }
    return Value::List(std::move(pairs));
  }
  if (name == "default" || name == "d") {
    return input.isUndefined() && !args.empty() ? args[0] : input;
  }
  if (name == "first") {
    return input.isList() && !input.list().empty() ? input.list().front()
                                                   : Value::Undefined("first");
  }
  if (name == "last") {
    return input.isList() && !input.list().empty() ? input.list().back()
                                                   : Value::Undefined("last");
  }
  if (name == "join") {
    const std::string sep = args.empty() ? "" : args[0].str();
    std::string out;
    if (input.isList()) {
      for (std::size_t i = 0; i < input.list().size(); ++i) {
        if (i != 0) out += sep;
        out += input.list()[i].str();
      }
    }
    return Value::Str(out);
  }
  if (name == "lower" || name == "upper") {
    std::string s = input.str();
    for (char& c : s) {
      c = static_cast<char>(name == "lower"
                                ? std::tolower(static_cast<unsigned char>(c))
                                : std::toupper(static_cast<unsigned char>(c)));
    }
    return Value::Str(s);
  }
  return input;
}

bool ApplyTest(const std::string& name, const Value& value) {
  if (name == "string") return value.isStr();
  if (name == "number" || name == "integer" || name == "float") {
    return value.isNumber();
  }
  if (name == "mapping") return value.isMap();
  if (name == "iterable" || name == "sequence") return IsIterable(value);
  if (name == "defined") return !value.isUndefined();
  if (name == "undefined") return value.isUndefined();
  if (name == "none" || name == "null") {
    return value.type() == Value::Type::None;
  }
  if (name == "true") return value.type() == Value::Type::Bool && value.asBool();
  if (name == "false") {
    return value.type() == Value::Type::Bool && !value.asBool();
  }
  if (name == "boolean") return value.type() == Value::Type::Bool;
  if (name == "callable") return value.type() == Value::Type::Callable;
  return false;
}

Value EvalCall(Env& env, const Expr& expr);

std::vector<Value> EvalArgs(Env& env, const Expr& expr, std::size_t first) {
  std::vector<Value> args;
  for (std::size_t k = first; k < expr.a.size(); ++k) {
    args.push_back(EvalExpr(env, *expr.a[k]));
  }
  return args;
}

Value InvokeMacro(Env& env, const Node& macro,
                  const std::vector<Value>& args) {
  env.Push();
  for (std::size_t k = 0; k < macro.params.size(); ++k) {
    if (k < args.size()) {
      env.Set(macro.params[k], args[k]);
    } else if (k < macro.param_defaults.size() &&
               macro.param_defaults[k] != nullptr) {
      env.Set(macro.params[k], EvalExpr(env, *macro.param_defaults[k]));
    } else {
      env.Set(macro.params[k], Value::Undefined(macro.params[k]));
    }
  }
  const std::string saved = env.output;
  env.output.clear();
  EvalNodes(env, macro.body);
  std::string result = std::move(env.output);
  env.output = saved;
  env.Pop();
  return Value::Str(std::move(result));
}

Value EvalCallWithName(Env& env, const std::string& name,
                       const std::vector<Value>& args) {
  if (name == "raise_exception") {
    throw Abort{args.empty() ? std::string() : args[0].str()};
  }
  if (name == "namespace") {
    return Value::Map({});
  }
  if (name == "range") {
    std::vector<Value> items;
    std::int64_t start = 0;
    std::int64_t stop = args.empty() ? 0 : args[0].asInt();
    std::int64_t step = 1;
    if (args.size() >= 2) {
      start = args[0].asInt();
      stop = args[1].asInt();
    }
    if (args.size() >= 3) step = args[2].asInt();
    if (step > 0) {
      for (std::int64_t k = start; k < stop; k += step) {
        items.push_back(Value::Int(k));
      }
    } else if (step < 0) {
      for (std::int64_t k = start; k > stop; k += step) {
        items.push_back(Value::Int(k));
      }
    }
    return Value::List(std::move(items));
  }
  auto it = env.macros.find(name);
  if (it != env.macros.end()) {
    return InvokeMacro(env, *it->second, args);
  }
  return Value::Undefined(name);
}

Value EvalCall(Env& env, const Expr& expr) {
  // Namespace(...) with kwargs: build a map from kwargs.
  const Expr& callee = *expr.a[0];
  std::vector<Value> args = EvalArgs(env, expr, 1);
  if (callee.kind == Expr::K::Variable && callee.text == "namespace") {
    std::vector<std::pair<std::string, Value>> entries;
    for (const auto& [key, value] : expr.kw) {
      entries.emplace_back(key, EvalExpr(env, *value));
    }
    return Value::Map(std::move(entries));
  }
  if (callee.kind == Expr::K::Variable) {
    Value* found = env.Find(callee.text);
    if (found != nullptr && found->type() == Value::Type::Callable) {
      return found->callable()->call(args, {});
    }
    return EvalCallWithName(env, callee.text, args);
  }
  if (callee.kind == Expr::K::Attr) {
    Value target = EvalExpr(env, *callee.a[0]);
    if (target.type() == Value::Type::Callable) {
      return target.callable()->call(args, {});
    }
  }
  return Value::Undefined("call");
}

Value EvalExpr(Env& env, const Expr& expr) {
  switch (expr.kind) {
    case Expr::K::Literal:
      return expr.lit;
    case Expr::K::Variable: {
      Value* found = env.Find(expr.text);
      return found != nullptr ? *found : Value::Undefined(expr.text);
    }
    case Expr::K::Attr: {
      Value base = EvalExpr(env, *expr.a[0]);
      if (base.isStr() &&
          (expr.text == "startswith" || expr.text == "endswith")) {
        const std::string captured = base.asStr();
        const bool starts = expr.text == "startswith";
        auto callable = std::make_shared<Callable>();
        callable->call = [captured, starts](
                             const std::vector<Value>& args,
                             const std::vector<Kwarg>&) {
          if (args.empty()) return Value::Bool(false);
          const std::string needle = args[0].str();
          const bool result =
              starts ? captured.compare(0, needle.size(), needle) == 0
                     : (needle.size() <= captured.size() &&
                        captured.compare(captured.size() - needle.size(),
                                         needle.size(), needle) == 0);
          return Value::Bool(result);
        };
        return Value::Fn(callable);
      }
      const Value* found = base.find(expr.text);
      return found != nullptr ? *found : Value::Undefined(expr.text);
    }
    case Expr::K::Item: {
      Value base = EvalExpr(env, *expr.a[0]);
      Value index = EvalExpr(env, *expr.a[1]);
      if (base.isList()) {
        std::int64_t idx = index.asInt();
        if (idx < 0) idx += (std::int64_t)base.list().size();
        if (idx >= 0 && idx < (std::int64_t)base.list().size()) {
          return base.list()[(std::size_t)idx];
        }
        return Value::Undefined("item");
      }
      if (base.isMap()) {
        const Value* found = base.find(index.str());
        return found != nullptr ? *found : Value::Undefined("item");
      }
      return Value::Undefined("item");
    }
    case Expr::K::Slice: {
      Value base = EvalExpr(env, *expr.a[0]);
      if (!base.isList()) return base;
      const auto& items = base.list();
      const std::int64_t n = (std::int64_t)items.size();
      std::int64_t step = (expr.a.size() > 3 && expr.a[3] != nullptr)
                              ? EvalExpr(env, *expr.a[3]).asInt()
                              : 1;
      if (step == 0) step = 1;
      std::int64_t start = (expr.a.size() > 1 && expr.a[1] != nullptr)
                               ? EvalExpr(env, *expr.a[1]).asInt()
                               : (step > 0 ? 0 : n - 1);
      std::int64_t stop = (expr.a.size() > 2 && expr.a[2] != nullptr)
                              ? EvalExpr(env, *expr.a[2]).asInt()
                              : (step > 0 ? n : -1);
      if (start < 0) start += n;
      if (stop < 0 && expr.a.size() > 2 && expr.a[2] != nullptr) stop += n;
      std::vector<Value> out;
      if (step > 0) {
        for (std::int64_t k = start; k < stop && k < n; k += step) {
          if (k >= 0) out.push_back(items[(std::size_t)k]);
        }
      } else {
        for (std::int64_t k = start; k > stop && k >= 0; k += step) {
          if (k < n) out.push_back(items[(std::size_t)k]);
        }
      }
      return Value::List(std::move(out));
    }
    case Expr::K::Call:
      return EvalCall(env, expr);
    case Expr::K::Filter: {
      Value input = EvalExpr(env, *expr.a[0]);
      std::vector<Value> args = EvalArgs(env, expr, 1);
      return ApplyFilter(env, expr.text, input, args);
    }
    case Expr::K::Test: {
      Value input = EvalExpr(env, *expr.a[0]);
      bool result = ApplyTest(expr.text, input);
      if (expr.lit.type() == Value::Type::Bool && expr.lit.asBool()) {
        result = !result;
      }
      return Value::Bool(result);
    }
    case Expr::K::Not:
      return Value::Bool(!EvalExpr(env, *expr.a[0]).truthy());
    case Expr::K::Neg: {
      Value v = EvalExpr(env, *expr.a[0]);
      return v.type() == Value::Type::Int ? Value::Int(-v.asInt())
                                          : Value::Double(-v.asDouble());
    }
    case Expr::K::Concat:
      return Value::Str(EvalExpr(env, *expr.a[0]).str() +
                        EvalExpr(env, *expr.a[1]).str());
    case Expr::K::Ternary:
      return EvalExpr(env, *expr.a[2]).truthy() ? EvalExpr(env, *expr.a[1])
                                                : EvalExpr(env, *expr.a[0]);
    case Expr::K::ListLit: {
      std::vector<Value> items;
      for (const auto& item : expr.a) items.push_back(EvalExpr(env, *item));
      return Value::List(std::move(items));
    }
    case Expr::K::MapLit: {
      std::vector<std::pair<std::string, Value>> entries;
      for (const auto& [key, value] : expr.kw) {
        entries.emplace_back(key, EvalExpr(env, *value));
      }
      return Value::Map(std::move(entries));
    }
    case Expr::K::Binary: {
      const std::string& op = expr.text;
      if (op == "and") {
        return Value::Bool(EvalExpr(env, *expr.a[0]).truthy() &&
                           EvalExpr(env, *expr.a[1]).truthy());
      }
      if (op == "or") {
        return Value::Bool(EvalExpr(env, *expr.a[0]).truthy() ||
                           EvalExpr(env, *expr.a[1]).truthy());
      }
      Value lhs = EvalExpr(env, *expr.a[0]);
      Value rhs = EvalExpr(env, *expr.a[1]);
      if (op == "==") {
        if (lhs.isNumber() && rhs.isNumber()) {
          return Value::Bool(lhs.asDouble() == rhs.asDouble());
        }
        return Value::Bool(lhs.type() == rhs.type() && lhs.str() == rhs.str());
      }
      if (op == "!=") {
        if (lhs.isNumber() && rhs.isNumber()) {
          return Value::Bool(lhs.asDouble() != rhs.asDouble());
        }
        return Value::Bool(!(lhs.type() == rhs.type() && lhs.str() == rhs.str()));
      }
      if (op == "in" || op == "not in") {
        bool found = false;
        if (rhs.isList()) {
          for (const Value& v : rhs.list()) {
            if ((v.isNumber() && lhs.isNumber() && v.asDouble() == lhs.asDouble()) ||
                v.str() == lhs.str()) {
              found = true;
              break;
            }
          }
        } else if (rhs.isMap()) {
          found = rhs.find(lhs.str()) != nullptr;
        } else if (rhs.isStr()) {
          found = rhs.asStr().find(lhs.str()) != std::string::npos;
        }
        return Value::Bool(op == "in" ? found : !found);
      }
      if (lhs.isNumber() && rhs.isNumber() &&
          (lhs.type() == Value::Type::Double ||
           rhs.type() == Value::Type::Double)) {
        const double a = lhs.asDouble();
        const double b = rhs.asDouble();
        if (op == "<") return Value::Bool(a < b);
        if (op == ">") return Value::Bool(a > b);
        if (op == "<=") return Value::Bool(a <= b);
        if (op == ">=") return Value::Bool(a >= b);
        if (op == "+") return Value::Double(a + b);
        if (op == "-") return Value::Double(a - b);
        if (op == "*") return Value::Double(a * b);
        if (op == "/") return Value::Double(b != 0 ? a / b : 0.0);
      }
      const std::int64_t a = lhs.asInt();
      const std::int64_t b = rhs.asInt();
      if (op == "<") return Value::Bool(a < b);
      if (op == ">") return Value::Bool(a > b);
      if (op == "<=") return Value::Bool(a <= b);
      if (op == ">=") return Value::Bool(a >= b);
      if (op == "+") {
        if (lhs.isStr() || rhs.isStr()) {
          return Value::Str(lhs.str() + rhs.str());
        }
        return Value::Int(a + b);
      }
      if (op == "-") return Value::Int(a - b);
      if (op == "*") return Value::Int(a * b);
      if (op == "/") return Value::Int(b != 0 ? a / b : 0);
      if (op == "//") return Value::Int(b != 0 ? a / b : 0);
      if (op == "%") return Value::Int(b != 0 ? a % b : 0);
      return Value::Undefined("op");
    }
  }
  return Value::Undefined("expr");
}

std::vector<std::string> SplitVars(const std::string& text) {
  std::vector<std::string> vars;
  std::size_t start = 0;
  while (start <= text.size()) {
    std::size_t comma = text.find(',', start);
    const std::string name = TrimAscii(
        comma == std::string::npos ? text.substr(start)
                                   : text.substr(start, comma - start));
    if (!name.empty()) vars.push_back(name);
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return vars;
}

Value MakeLoop(const std::vector<Value>& items, std::size_t index) {
  std::vector<std::pair<std::string, Value>> entries;
  entries.emplace_back("index0", Value::Int((std::int64_t)index));
  entries.emplace_back("index", Value::Int((std::int64_t)index + 1));
  entries.emplace_back("first", Value::Bool(index == 0));
  entries.emplace_back("last", Value::Bool(index + 1 == items.size()));
  entries.emplace_back("length", Value::Int((std::int64_t)items.size()));
  entries.emplace_back("revindex0",
                       Value::Int((std::int64_t)(items.size() - index - 1)));
  entries.emplace_back("revindex",
                       Value::Int((std::int64_t)(items.size() - index)));
  entries.emplace_back("previtem",
                       index == 0 ? Value::Undefined("previtem")
                                  : items[index - 1]);
  entries.emplace_back("nextitem",
                       index + 1 >= items.size()
                           ? Value::Undefined("nextitem")
                           : items[index + 1]);
  return Value::Map(std::move(entries));
}

void EvalNode(Env& env, const Node& node) {
  switch (node.kind) {
    case Node::K::Text:
      env.output += node.text;
      return;
    case Node::K::Output:
      env.output += EvalExpr(env, *node.expr).str();
      return;
    case Node::K::Set:
      env.Set(node.text, EvalExpr(env, *node.expr));
      return;
    case Node::K::SetAttr: {
      Value* target = env.Find(node.text);
      if (target != nullptr) {
        target->set(node.attr, EvalExpr(env, *node.expr));
      }
      return;
    }
    case Node::K::If: {
      for (const auto& [cond, body] : node.branches) {
        if (EvalExpr(env, *cond).truthy()) {
          EvalNodes(env, body);
          return;
        }
      }
      EvalNodes(env, node.else_body);
      return;
    }
    case Node::K::Macro:
      env.macros[node.text] = &node;
      return;
    case Node::K::For: {
      Value iterable = EvalExpr(env, *node.iter);
      std::vector<Value> items;
      if (iterable.isList()) {
        items = iterable.list();
      } else if (iterable.isMap()) {
        for (const auto& [k, v] : iterable.map()) {
          (void)v;
          items.push_back(Value::Str(k));
        }
      }
      const std::vector<std::string> vars = SplitVars(node.text);
      for (std::size_t idx = 0; idx < items.size(); ++idx) {
        env.Push();
        if (vars.size() == 1) {
          env.Set(vars[0], items[idx]);
        } else if (items[idx].isList() && items[idx].list().size() == vars.size()) {
          for (std::size_t v = 0; v < vars.size(); ++v) {
            env.Set(vars[v], items[idx].list()[v]);
          }
        }
        env.Set("loop", MakeLoop(items, idx));
        EvalNodes(env, node.body);
        env.Pop();
      }
      return;
    }
  }
}

void EvalNodes(Env& env, const std::vector<NodePtr>& nodes) {
  for (const NodePtr& node : nodes) {
    EvalNode(env, *node);
  }
}

}  // namespace

RenderResult Render(std::string_view template_text,
                    const std::vector<Kwarg>& context) {
  RenderResult result;
  ParseResult parsed = Parse(template_text);
  if (!parsed.ok) {
    result.error = parsed.error;
    return result;
  }
  Env env;
  for (const auto& [name, value] : context) {
    env.Set(name, value);
  }
  try {
    EvalNodes(env, parsed.program.nodes);
  } catch (const Abort& abort) {
    result.error = abort.message.empty() ? "template aborted" : abort.message;
    return result;
  }
  result.ok = true;
  result.text = std::move(env.output);
  return result;
}

}  // namespace tessera::serve::jinja
