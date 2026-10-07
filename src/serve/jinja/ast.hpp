#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "serve/jinja/jinja.hpp"

// Internal AST for the Jinja subset (shared by the parser and the
// evaluator, not part of the public API).

namespace tessera::serve::jinja {

struct Expr;
using ExprPtr = std::unique_ptr<Expr>;

struct Expr {
  enum class K {
    Literal,   // lit
    Variable,  // text = name
    Attr,      // a[0] object, text = attribute
    Item,      // a[0] object, a[1] index
    Slice,     // a[0] object, a[1] start, a[2] stop, a[3] step (nullable)
    Call,      // a[0] callee, a[1..] positional, kw = keyword args
    Filter,    // a[0] value, text = filter, a[1..] args
    Test,      // a[0] value, text = test, a[1..] args
    Not,       // a[0]
    Neg,       // a[0]
    Binary,    // text = op, a[0]/a[1]
    Concat,    // a[0] ~ a[1]
    Ternary,   // a[0] if a[1] else a[2]
    ListLit,   // a = items
    MapLit,    // kw = entries
  };
  K kind = K::Literal;
  Value lit;
  std::string text;
  std::vector<ExprPtr> a;
  std::vector<std::pair<std::string, ExprPtr>> kw;
};

struct Node;
using NodePtr = std::unique_ptr<Node>;

struct Node {
  enum class K { Text, Output, If, For, Set, SetAttr, Macro };
  K kind = K::Text;
  std::string text;   // Text; Set/SetAttr target name; For variable
  std::string attr;   // SetAttr attribute
  ExprPtr expr;       // Output; Set/SetAttr value; If condition (branch 0)
  ExprPtr iter;       // For iterable
  std::vector<NodePtr> body;
  // If branches: (condition, body); branch 0 uses expr for compatibility.
  std::vector<std::pair<ExprPtr, std::vector<NodePtr>>> branches;
  std::vector<NodePtr> else_body;
  std::vector<std::string> params;        // Macro parameters
  std::vector<ExprPtr> param_defaults;    // Parallel to params (nullable)
};

struct Program {
  std::vector<NodePtr> nodes;
};

// Parse `text` into a program; ok=false with error on failure.
struct ParseResult {
  bool ok = false;
  Program program;
  std::string error;
};
[[nodiscard]] ParseResult Parse(std::string_view text);

}  // namespace tessera::serve::jinja
