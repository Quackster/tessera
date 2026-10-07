#include "serve/jinja/ast.hpp"

#include <cctype>
#include <cstdlib>
#include <string>
#include <string_view>

namespace tessera::serve::jinja {

namespace {

bool IsIdentStart(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}
bool IsIdentChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// --- Expression parser ---------------------------------------------------

struct ExprParser {
  std::string_view s;
  std::size_t i = 0;
  bool failed = false;
  std::string error;

  void SkipWs() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' ||
                            s[i] == '\r')) {
      ++i;
    }
  }
  bool Eof() {
    SkipWs();
    return i >= s.size();
  }
  bool Peek(std::string_view tok) {
    SkipWs();
    return s.substr(i, tok.size()) == tok;
  }
  bool Take(std::string_view tok) {
    if (Peek(tok)) {
      i += tok.size();
      return true;
    }
    return false;
  }
  void Fail(std::string message) {
    if (!failed) {
      failed = true;
      error = std::move(message);
    }
  }

  ExprPtr Make() { return std::make_unique<Expr>(); }

  std::string ParseIdent() {
    SkipWs();
    std::size_t start = i;
    if (i < s.size() && IsIdentStart(s[i])) {
      ++i;
      while (i < s.size() && IsIdentChar(s[i])) {
        ++i;
      }
    }
    return std::string(s.substr(start, i - start));
  }

  ExprPtr ParseString() {
    SkipWs();
    const char quote = s[i++];
    std::string out;
    while (i < s.size() && s[i] != quote) {
      if (s[i] == '\\' && i + 1 < s.size()) {
        ++i;
        char e = s[i++];
        switch (e) {
          case 'n': out.push_back('\n'); break;
          case 't': out.push_back('\t'); break;
          case 'r': out.push_back('\r'); break;
          default: out.push_back(e); break;
        }
      } else {
        out.push_back(s[i++]);
      }
    }
    if (i >= s.size()) {
      Fail("unterminated string");
      return nullptr;
    }
    ++i;  // closing quote
    auto e = Make();
    e->kind = Expr::K::Literal;
    e->lit = Value::Str(std::move(out));
    return e;
  }

  ExprPtr ParseNumber() {
    SkipWs();
    std::size_t start = i;
    while (i < s.size() &&
           (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.')) {
      ++i;
    }
    const std::string num(s.substr(start, i - start));
    auto e = Make();
    e->kind = Expr::K::Literal;
    if (num.find('.') != std::string::npos) {
      e->lit = Value::Double(std::strtod(num.c_str(), nullptr));
    } else {
      e->lit = Value::Int(std::strtoll(num.c_str(), nullptr, 10));
    }
    return e;
  }

  ExprPtr ParsePrimary() {
    SkipWs();
    if (i >= s.size()) {
      Fail("unexpected end of expression");
      return nullptr;
    }
    const char c = s[i];
    if (c == '(') {
      ++i;
      auto first = ParseExpr();
      SkipWs();
      if (Take(",")) {
        auto e = Make();
        e->kind = Expr::K::ListLit;
        e->a.push_back(std::move(first));
        while (true) {
          SkipWs();
          if (Take(")")) {
            break;
          }
          auto item = ParseExpr();
          if (item == nullptr) {
            return nullptr;
          }
          e->a.push_back(std::move(item));
          if (Take(",")) {
            continue;
          }
          if (Take(")")) {
            break;
          }
          Fail("expected ',' or ')'");
          return nullptr;
        }
        return e;
      }
      if (!Take(")")) {
        Fail("expected ')'");
      }
      return first;
    }
    if (c == '[') {
      ++i;
      auto e = Make();
      e->kind = Expr::K::ListLit;
      SkipWs();
      if (!Take("]")) {
        while (true) {
          auto item = ParseExpr();
          if (item == nullptr) {
            return nullptr;
          }
          e->a.push_back(std::move(item));
          if (Take(",")) {
            continue;
          }
          if (Take("]")) {
            break;
          }
          Fail("expected ',' or ']'");
          return nullptr;
        }
      }
      return e;
    }
    if (c == '{') {
      ++i;
      auto e = Make();
      e->kind = Expr::K::MapLit;
      SkipWs();
      if (!Take("}")) {
        while (true) {
          auto key = ParseExpr();
          if (!Take(":")) {
            Fail("expected ':' in map");
            return nullptr;
          }
          auto value = ParseExpr();
          if (key == nullptr || value == nullptr) {
            return nullptr;
          }
          std::string key_str =
              key->kind == Expr::K::Literal ? key->lit.str() : "";
          e->kw.emplace_back(std::move(key_str), std::move(value));
          if (Take(",")) {
            continue;
          }
          if (Take("}")) {
            break;
          }
          Fail("expected ',' or '}'");
          return nullptr;
        }
      }
      return e;
    }
    if (c == '"' || c == '\'') {
      return ParseString();
    }
    if (std::isdigit(static_cast<unsigned char>(c))) {
      return ParseNumber();
    }
    const std::string ident = ParseIdent();
    if (ident.empty()) {
      Fail("unexpected character in expression");
      return nullptr;
    }
    auto e = Make();
    if (ident == "true" || ident == "True") {
      e->kind = Expr::K::Literal;
      e->lit = Value::Bool(true);
    } else if (ident == "false" || ident == "False") {
      e->kind = Expr::K::Literal;
      e->lit = Value::Bool(false);
    } else if (ident == "none" || ident == "None") {
      e->kind = Expr::K::Literal;
      e->lit = Value::None();
    } else {
      e->kind = Expr::K::Variable;
      e->text = ident;
    }
    return e;
  }

  ExprPtr ParseArgs(ExprPtr callee) {
    // callee is a Call node; parse (a, b, key=val).
    SkipWs();
    if (!Take("(")) {
      return callee;
    }
    SkipWs();
    if (Take(")")) {
      return callee;
    }
    while (true) {
      const std::size_t save = i;
      const std::string name = ParseIdent();
      SkipWs();
      if (!name.empty() && i < s.size() && s[i] == '=' &&
          (i + 1 >= s.size() || s[i + 1] != '=')) {
        ++i;
        auto value = ParseExpr();
        if (value == nullptr) {
          return nullptr;
        }
        callee->kw.emplace_back(name, std::move(value));
      } else {
        i = save;
        auto value = ParseExpr();
        if (value == nullptr) {
          return nullptr;
        }
        callee->a.push_back(std::move(value));
      }
      if (Take(",")) {
        continue;
      }
      if (Take(")")) {
        break;
      }
      Fail("expected ',' or ')'");
      return nullptr;
    }
    return callee;
  }

  // Attr/index/call/slice (postfix, no filter/test).
  ExprPtr ParsePostfix() {
    auto e = ParsePrimary();
    if (e == nullptr) {
      return nullptr;
    }
    while (true) {
      if (Peek(".") && !Peek("..")) {
        // Avoid consuming a number's decimal point (numbers consumed already).
        if (i < s.size() && s[i] == '.' &&
            i + 1 < s.size() &&
            std::isdigit(static_cast<unsigned char>(s[i + 1]))) {
          break;
        }
        ++i;
        auto node = Make();
        node->kind = Expr::K::Attr;
        node->text = ParseIdent();
        node->a.push_back(std::move(e));
        e = std::move(node);
        continue;
      }
      if (Peek("[")) {
        ++i;
        auto node = Make();
        node->kind = Expr::K::Item;
        node->a.push_back(std::move(e));
        if (Take(":")) {
          node->kind = Expr::K::Slice;
          node->a.push_back(nullptr);  // start
          if (!Peek(":") && !Peek("]")) {
            node->a.push_back(ParseExpr());
          } else {
            node->a.push_back(nullptr);
          }
          if (Take(":")) {
            if (!Peek("]")) {
              node->a.push_back(ParseExpr());
            } else {
              node->a.push_back(nullptr);
            }
          } else {
            node->a.push_back(nullptr);
          }
          if (!Take("]")) {
            Fail("expected ']'");
            return nullptr;
          }
          e = std::move(node);
          continue;
        }
        auto index = ParseExpr();
        if (!Take("]")) {
          Fail("expected ']'");
          return nullptr;
        }
        node->a.push_back(std::move(index));
        e = std::move(node);
        continue;
      }
      if (Peek("(")) {
        auto node = Make();
        node->kind = Expr::K::Call;
        node->a.push_back(std::move(e));
        e = ParseArgs(std::move(node));
        if (e == nullptr) {
          return nullptr;
        }
        continue;
      }
      break;
    }
    return e;
  }

  // Filters (|name(args)...) and tests (is name(args)? ) — postfix, high
  // precedence.
  ExprPtr ParseFilterTest() {
    auto e = ParsePostfix();
    if (e == nullptr) {
      return nullptr;
    }
    while (true) {
      SkipWs();
      if (s.substr(i, 2) == "is" &&
          (i + 2 >= s.size() || !IsIdentChar(s[i + 2]))) {
        i += 2;
        const bool negated = Take("not");
        auto node = Make();
        node->kind = Expr::K::Test;
        node->text = ParseIdent();
        node->lit = Value::Bool(negated);
        node->a.push_back(std::move(e));
        // Optional test args: is defined(...)
        if (Peek("(")) {
          auto call = Make();
          call->kind = Expr::K::Call;
          call->a.push_back(std::move(node));
          auto parsed = ParseArgs(std::move(call));
          if (parsed == nullptr) {
            return nullptr;
          }
          e = std::move(parsed);
          continue;
        }
        e = std::move(node);
        continue;
      }
      if (i < s.size() && s[i] == '|' &&
          !(i + 1 < s.size() && s[i + 1] == '|')) {
        ++i;
        auto node = Make();
        node->kind = Expr::K::Filter;
        node->text = ParseIdent();
        node->a.push_back(std::move(e));
        SkipWs();
        if (Take("(")) {
          SkipWs();
          if (!Take(")")) {
            while (true) {
              auto arg = ParseExpr();
              if (arg == nullptr) {
                return nullptr;
              }
              node->a.push_back(std::move(arg));
              if (Take(",")) {
                continue;
              }
              if (Take(")")) {
                break;
              }
              Fail("expected ',' or ')'");
              return nullptr;
            }
          }
        }
        e = std::move(node);
        continue;
      }
      break;
    }
    return e;
  }

  ExprPtr ParseUnary() {
    SkipWs();
    if (Take("-")) {
      auto node = Make();
      node->kind = Expr::K::Neg;
      node->a.push_back(ParseUnary());
      return node;
    }
    if (Take("+")) {
      return ParseUnary();
    }
    return ParseFilterTest();
  }

  ExprPtr ParsePow() { return ParseUnary(); }

  ExprPtr ParseMul() {
    auto e = ParsePow();
    if (e == nullptr) {
      return nullptr;
    }
    while (true) {
      SkipWs();
      std::string op;
      if (Peek("//")) {
        op = "//";
      } else if (i < s.size() && (s[i] == '*' || s[i] == '/' || s[i] == '%')) {
        op = std::string(1, s[i]);
      } else {
        break;
      }
      i += op.size();
      auto rhs = ParsePow();
      if (rhs == nullptr) {
        return nullptr;
      }
      auto node = Make();
      node->kind = Expr::K::Binary;
      node->text = op;
      node->a.push_back(std::move(e));
      node->a.push_back(std::move(rhs));
      e = std::move(node);
    }
    return e;
  }

  ExprPtr ParseAdd() {
    auto e = ParseMul();
    if (e == nullptr) {
      return nullptr;
    }
    while (true) {
      SkipWs();
      if (Peek("~")) {
        ++i;
        auto rhs = ParseMul();
        if (rhs == nullptr) {
          return nullptr;
        }
        auto node = Make();
        node->kind = Expr::K::Concat;
        node->a.push_back(std::move(e));
        node->a.push_back(std::move(rhs));
        e = std::move(node);
        continue;
      }
      if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
        std::string op(1, s[i]);
        ++i;
        auto rhs = ParseMul();
        if (rhs == nullptr) {
          return nullptr;
        }
        auto node = Make();
        node->kind = Expr::K::Binary;
        node->text = op;
        node->a.push_back(std::move(e));
        node->a.push_back(std::move(rhs));
        e = std::move(node);
        continue;
      }
      break;
    }
    return e;
  }

  ExprPtr ParseCompare() {
    auto e = ParseAdd();
    if (e == nullptr) {
      return nullptr;
    }
    while (true) {
      SkipWs();
      std::string op;
      if (Take("==")) {
        op = "==";
      } else if (Take("!=")) {
        op = "!=";
      } else if (Take("<=")) {
        op = "<=";
      } else if (Take(">=")) {
        op = ">=";
      } else if (i < s.size() && (s[i] == '<' || s[i] == '>')) {
        op = std::string(1, s[i++]);
      } else if (s.substr(i, 2) == "in" &&
                 (i + 2 >= s.size() || !IsIdentChar(s[i + 2]))) {
        op = "in";
        i += 2;
      } else if (s.substr(i, 6) == "not in" &&
                 (i + 6 >= s.size() || !IsIdentChar(s[i + 6]))) {
        op = "not in";
        i += 6;
      } else {
        break;
      }
      auto rhs = ParseAdd();
      if (rhs == nullptr) {
        return nullptr;
      }
      auto node = Make();
      node->kind = Expr::K::Binary;
      node->text = op;
      node->a.push_back(std::move(e));
      node->a.push_back(std::move(rhs));
      e = std::move(node);
    }
    return e;
  }

  ExprPtr ParseNot() {
    SkipWs();
    if (s.substr(i, 3) == "not" &&
        (i + 3 >= s.size() || !IsIdentChar(s[i + 3]))) {
      i += 3;
      auto node = Make();
      node->kind = Expr::K::Not;
      node->a.push_back(ParseNot());
      return node;
    }
    return ParseCompare();
  }

  ExprPtr ParseAnd() {
    auto e = ParseNot();
    if (e == nullptr) {
      return nullptr;
    }
    while (true) {
      SkipWs();
      if (s.substr(i, 3) == "and" &&
          (i + 3 >= s.size() || !IsIdentChar(s[i + 3]))) {
        i += 3;
        auto rhs = ParseNot();
        if (rhs == nullptr) {
          return nullptr;
        }
        auto node = Make();
        node->kind = Expr::K::Binary;
        node->text = "and";
        node->a.push_back(std::move(e));
        node->a.push_back(std::move(rhs));
        e = std::move(node);
        continue;
      }
      break;
    }
    return e;
  }

  ExprPtr ParseOr() {
    auto e = ParseAnd();
    if (e == nullptr) {
      return nullptr;
    }
    while (true) {
      SkipWs();
      if (s.substr(i, 2) == "or" &&
          (i + 2 >= s.size() || !IsIdentChar(s[i + 2]))) {
        i += 2;
        auto rhs = ParseAnd();
        if (rhs == nullptr) {
          return nullptr;
        }
        auto node = Make();
        node->kind = Expr::K::Binary;
        node->text = "or";
        node->a.push_back(std::move(e));
        node->a.push_back(std::move(rhs));
        e = std::move(node);
        continue;
      }
      break;
    }
    return e;
  }

  ExprPtr ParseExpr() {
    auto e = ParseOr();
    if (e == nullptr) {
      return nullptr;
    }
    SkipWs();
    // Ternary: A if B else C
    if (s.substr(i, 2) == "if" &&
        (i + 2 >= s.size() || !IsIdentChar(s[i + 2]))) {
      i += 2;
      auto cond = ParseOr();
      ExprPtr else_e;
      SkipWs();
      if (s.substr(i, 4) == "else" &&
          (i + 4 >= s.size() || !IsIdentChar(s[i + 4]))) {
        i += 4;
        else_e = ParseExpr();
      }
      auto node = Make();
      node->kind = Expr::K::Ternary;
      node->a.push_back(std::move(else_e));
      node->a.push_back(std::move(e));
      node->a.push_back(std::move(cond));
      return node;
    }
    return e;
  }
};

ExprPtr ParseExpression(std::string_view text, std::string* error) {
  ExprParser parser{text, 0, false, {}};
  auto e = parser.ParseExpr();
  parser.SkipWs();
  if (e != nullptr && parser.i < text.size() && !parser.failed) {
    parser.Fail("trailing characters in expression");
  }
  if (parser.failed) {
    if (error != nullptr) {
      *error = parser.error;
    }
    return nullptr;
  }
  return e;
}

// --- Template lexer ------------------------------------------------------

struct Raw {
  enum class Kind { Text, Output, Stmt };
  Kind kind = Kind::Text;
  std::string content;
  bool trim_left = false;
  bool trim_right = false;
};

std::vector<Raw> LexTemplate(std::string_view text, std::string* error) {
  std::vector<Raw> out;
  std::size_t i = 0;
  std::string pending_text;
  while (i < text.size()) {
    if (text[i] == '{' && i + 1 < text.size() &&
        (text[i + 1] == '{' || text[i + 1] == '%' || text[i + 1] == '#')) {
      const char kind = text[i + 1];
      bool trim_left = false;
      std::size_t j = i + 2;
      if (j < text.size() && text[j] == '-') {
        trim_left = true;
        ++j;
      }
      const std::string close = kind == '{' ? "}}" : (kind == '%' ? "%}" : "#}");
      const std::size_t end = text.find(close, j);
      if (end == std::string_view::npos) {
        if (error != nullptr) {
          *error = "unterminated tag";
        }
        return {};
      }
      bool trim_right = false;
      std::size_t content_end = end;
      if (content_end > j && text[content_end - 1] == '-') {
        trim_right = true;
        --content_end;
      }
      if (kind == '#') {
        if (trim_left) {
          while (!pending_text.empty() &&
                 std::isspace(
                     static_cast<unsigned char>(pending_text.back()))) {
            pending_text.pop_back();
          }
        }
        i = end + 2;
        if (trim_right) {
          while (i < text.size() &&
                 std::isspace(static_cast<unsigned char>(text[i]))) {
            ++i;
          }
        }
        continue;  // comment
      }
      if (trim_left) {
        while (!pending_text.empty() &&
               std::isspace(static_cast<unsigned char>(pending_text.back()))) {
          pending_text.pop_back();
        }
      }
      if (!pending_text.empty()) {
        Raw raw;
        raw.kind = Raw::Kind::Text;
        raw.content = std::move(pending_text);
        pending_text.clear();
        out.push_back(std::move(raw));
      }
      Raw raw;
      raw.kind = kind == '{' ? Raw::Kind::Output : Raw::Kind::Stmt;
      raw.content = std::string(text.substr(j, content_end - j));
      raw.trim_right = trim_right;
      out.push_back(std::move(raw));
      i = end + 2;
      if (trim_right) {
        while (i < text.size() &&
               std::isspace(static_cast<unsigned char>(text[i]))) {
          ++i;
        }
      }
      continue;
    }
    pending_text.push_back(text[i++]);
  }
  if (!pending_text.empty()) {
    Raw raw;
    raw.kind = Raw::Kind::Text;
    raw.content = std::move(pending_text);
    out.push_back(std::move(raw));
  }
  return out;
}

// --- Statement parser ----------------------------------------------------

struct Parser {
  std::vector<Raw> raws;
  std::size_t i = 0;
  bool failed = false;
  std::string error;

  void Fail(std::string message) {
    if (!failed) {
      failed = true;
      error = std::move(message);
    }
  }

  // Parse nodes until one of `stops` (a `{% end... %}` / `{% else %}` /
  // `{% elif ... %}`) which is left unconsumed.
  std::vector<NodePtr> ParseBody(const std::vector<std::string>& stops,
                                 std::string* stop_name,
                                 std::string* stop_arg) {
    std::vector<NodePtr> nodes;
    while (i < raws.size()) {
      Raw& raw = raws[i];
      if (raw.kind == Raw::Kind::Text) {
        auto node = std::make_unique<Node>();
        node->kind = Node::K::Text;
        node->text = raw.content;
        nodes.push_back(std::move(node));
        ++i;
        continue;
      }
      if (raw.kind == Raw::Kind::Output) {
        std::string err;
        auto expr = ParseExpression(raw.content, &err);
        if (expr == nullptr) {
          Fail("output: " + err);
          return nodes;
        }
        auto node = std::make_unique<Node>();
        node->kind = Node::K::Output;
        node->expr = std::move(expr);
        nodes.push_back(std::move(node));
        ++i;
        continue;
      }
      // Statement.
      const std::string kw = FirstWord(raw.content);
      for (const std::string& stop : stops) {
        if (kw == stop) {
          if (stop_name != nullptr) {
            *stop_name = kw;
          }
          if (stop_arg != nullptr) {
            *stop_arg = Rest(raw.content);
          }
          return nodes;
        }
      }
      if (kw == "if") {
        nodes.push_back(ParseIf());
        continue;
      }
      if (kw == "for") {
        nodes.push_back(ParseFor());
        continue;
      }
      if (kw == "set") {
        nodes.push_back(ParseSet());
        continue;
      }
      if (kw == "macro") {
        nodes.push_back(ParseMacro());
        continue;
      }
      Fail("unexpected statement: " + kw);
      return nodes;
    }
    return nodes;
  }

  static std::string FirstWord(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() &&
           std::isspace(static_cast<unsigned char>(s[start]))) {
      ++start;
    }
    std::size_t end = start;
    while (end < s.size() && IsIdentChar(s[end])) {
      ++end;
    }
    return s.substr(start, end - start);
  }
  static std::string Rest(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() &&
           std::isspace(static_cast<unsigned char>(s[start]))) {
      ++start;
    }
    while (start < s.size() && IsIdentChar(s[start])) {
      ++start;
    }
    return s.substr(start);
  }

  NodePtr ParseIf() {
    auto node = std::make_unique<Node>();
    node->kind = Node::K::If;
    std::string cond_text = Rest(raws[i].content);
    ++i;
    while (true) {
      std::string err;
      auto cond = ParseExpression(cond_text, &err);
      if (cond == nullptr) {
        Fail("if: " + err);
        return node;
      }
      std::string stop;
      std::string stop_arg;
      auto body = ParseBody({"elif", "else", "endif"}, &stop, &stop_arg);
      node->branches.emplace_back(std::move(cond), std::move(body));
      if (stop == "elif") {
        cond_text = stop_arg;
        ++i;
        continue;
      }
      if (stop == "else") {
        ++i;
        node->else_body = ParseBody({"endif"}, nullptr, nullptr);
      }
      // consume endif
      if (i < raws.size() && raws[i].kind == Raw::Kind::Stmt &&
          FirstWord(raws[i].content) == "endif") {
        ++i;
      } else {
        Fail("expected endif");
      }
      return node;
    }
  }

  NodePtr ParseFor() {
    auto node = std::make_unique<Node>();
    node->kind = Node::K::For;
    std::string text = Rest(raws[i].content);
    ++i;
    std::size_t in_pos = text.find(" in ");
    if (in_pos == std::string::npos) {
      Fail("for: expected 'in'");
      return node;
    }
    node->text = Trim(text.substr(0, in_pos));
    std::string err;
    auto iter = ParseExpression(text.substr(in_pos + 4), &err);
    if (iter == nullptr) {
      Fail("for: " + err);
      return node;
    }
    node->iter = std::move(iter);
    std::string stop;
    node->body = ParseBody({"endfor"}, &stop, nullptr);
    if (i < raws.size() && FirstWord(raws[i].content) == "endfor") {
      ++i;
    } else {
      Fail("expected endfor");
    }
    return node;
  }

  NodePtr ParseSet() {
    auto node = std::make_unique<Node>();
    node->kind = Node::K::Set;
    std::string text = Rest(raws[i].content);
    ++i;
    std::size_t eq = text.find('=');
    if (eq == std::string::npos) {
      Fail("set: expected '='");
      return node;
    }
    const std::string target = Trim(text.substr(0, eq));
    const std::size_t dot = target.find('.');
    if (dot != std::string::npos) {
      node->kind = Node::K::SetAttr;
      node->text = Trim(target.substr(0, dot));
      node->attr = Trim(target.substr(dot + 1));
    } else {
      node->text = target;
    }
    std::string err;
    auto value = ParseExpression(text.substr(eq + 1), &err);
    if (value == nullptr) {
      Fail("set: " + err);
      return node;
    }
    node->expr = std::move(value);
    return node;
  }

  NodePtr ParseMacro() {
    auto node = std::make_unique<Node>();
    node->kind = Node::K::Macro;
    std::string text = Rest(raws[i].content);
    ++i;
    const std::size_t paren = text.find('(');
    if (paren == std::string::npos) {
      Fail("macro: expected '('");
      return node;
    }
    node->text = Trim(text.substr(0, paren));
    const std::size_t close = text.find(')', paren);
    const std::string params = text.substr(paren + 1, close - paren - 1);
    std::size_t start = 0;
    while (start < params.size()) {
      std::size_t comma = params.find(',', start);
      const std::string name =
          Trim(comma == std::string::npos ? params.substr(start)
                                          : params.substr(start, comma - start));
      const std::size_t eq = name.find('=');
      if (eq != std::string::npos) {
        node->params.push_back(Trim(name.substr(0, eq)));
        std::string err;
        auto def = ParseExpression(Trim(name.substr(eq + 1)), &err);
        if (def == nullptr) {
          Fail("macro default: " + err);
          return node;
        }
        node->param_defaults.push_back(std::move(def));
      } else {
        if (!name.empty()) {
          node->params.push_back(name);
        }
        node->param_defaults.push_back(nullptr);
      }
      if (comma == std::string::npos) {
        break;
      }
      start = comma + 1;
    }
    std::string stop;
    node->body = ParseBody({"endmacro"}, &stop, nullptr);
    if (i < raws.size() && FirstWord(raws[i].content) == "endmacro") {
      ++i;
    } else {
      Fail("expected endmacro");
    }
    return node;
  }

  static std::string Trim(std::string s) {
    std::size_t start = 0;
    while (start < s.size() &&
           std::isspace(static_cast<unsigned char>(s[start]))) {
      ++start;
    }
    std::size_t end = s.size();
    while (end > start &&
           std::isspace(static_cast<unsigned char>(s[end - 1]))) {
      --end;
    }
    return s.substr(start, end - start);
  }

  Program Parse() {
    Program program;
    program.nodes = ParseBody({}, nullptr, nullptr);
    return program;
  }
};

}  // namespace

ParseResult Parse(std::string_view text) {
  ParseResult result;
  std::string error;
  std::vector<Raw> raws = LexTemplate(text, &error);
  if (!error.empty()) {
    result.error = error;
    return result;
  }
  Parser parser{std::move(raws), 0, false, {}};
  result.program = parser.Parse();
  if (parser.failed) {
    result.error = parser.error;
    return result;
  }
  result.ok = true;
  return result;
}

}  // namespace tessera::serve::jinja
