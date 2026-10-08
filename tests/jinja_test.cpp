#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>

#include "test_helpers.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"

#include "serve/jinja/ast.hpp"
#include "serve/jinja/jinja.hpp"

using tessera::serve::jinja::Kwarg;
using tessera::serve::jinja::Render;
using tessera::serve::jinja::Value;
using tessera::Engine;
using tessera::ModelOptions;
using tessera::testing::MakeEngineOrSkip;

namespace {

std::string RenderOk(const std::string& tmpl,
                     const std::vector<Kwarg>& context) {
  auto result = Render(tmpl, context);
  EXPECT_TRUE(result.ok) << result.error;
  return result.text;
}

}  // namespace

TEST(JinjaTest, TextAndOutput) {
  Value name = Value::Str("world");
  EXPECT_EQ(RenderOk("Hello {{ name }}!", {{"name", name}}), "Hello world!");
}

TEST(JinjaTest, IfElse) {
  std::vector<Kwarg> context = {{"x", Value::Int(2)}};
  EXPECT_EQ(RenderOk("{% if x > 1 %}big{% else %}small{% endif %}", context),
            "big");
  context = {{"x", Value::Int(0)}};
  EXPECT_EQ(RenderOk("{% if x > 1 %}big{% else %}small{% endif %}", context),
            "small");
}

TEST(JinjaTest, ForLoopWithLoopVars) {
  std::vector<Value> items = {Value::Str("a"), Value::Str("b")};
  std::vector<Kwarg> context = {{"items", Value::List(items)}};
  EXPECT_EQ(RenderOk("{% for x in items %}{{ loop.index }}:{{ x }}{% if not "
                     "loop.last %},{% endif %}{% endfor %}",
                     context),
            "1:a,2:b");
}

TEST(JinjaTest, Filters) {
  std::vector<Kwarg> context = {{"s", Value::Str("  hi  ")}};
  EXPECT_EQ(RenderOk("{{ s|trim }}", context), "hi");
  std::vector<Kwarg> map = {{"m", Value::Map({{"a", Value::Int(1)}})}};
  EXPECT_EQ(RenderOk("{{ m|items|length }}", map), "1");
}

TEST(JinjaTest, TestsStringIterable) {
  std::vector<Kwarg> context = {{"s", Value::Str("x")},
                                {"l", Value::List({Value::Int(1)})}};
  EXPECT_EQ(RenderOk("{% if s is string %}s{% endif %}{% if l is iterable "
                     "and l is not mapping %}l{% endif %}",
                     context),
            "sl");
}

TEST(JinjaTest, MacroAndNamespace) {
  const std::string tmpl =
      "{% macro wrap(x, suffix='!') %}[{{ x }}{{ suffix }}]{% endmacro %}"
      "{% set ns = namespace(n=0) %}"
      "{% for i in items %}{{ wrap(i) }}{% set ns.n = ns.n + 1 %}{% endfor %}"
      "{{ ns.n }}";
  std::vector<Value> items = {Value::Str("a"), Value::Str("b")};
  std::vector<Kwarg> context = {{"items", Value::List(items)}};
  EXPECT_EQ(RenderOk(tmpl, context), "[a!][b!]2");
}

TEST(JinjaTest, SliceReverse) {
  std::vector<Value> items = {Value::Int(1), Value::Int(2), Value::Int(3)};
  std::vector<Kwarg> context = {{"items", Value::List(items)}};
  EXPECT_EQ(RenderOk("{% for x in items[::-1] %}{{ x }}{% endfor %}", context),
            "321");
}

TEST(JinjaTest, RaiseExceptionFails) {
  auto result = Render("{{ raise_exception('boom') }}", {});
  EXPECT_FALSE(result.ok);
  EXPECT_NE(result.error.find("boom"), std::string::npos);
}

// Renders the model's own chat template and compares to the reference
// (HF Jinja2) output; path via TESSERA_TEST_MODEL (never hard-coded).
TEST(JinjaTest, RealChatTemplateMatchesReference) {
  const char* path = std::getenv("TESSERA_TEST_MODEL");
  if (path == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_MODEL not set";
  }
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{path, 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  const std::string tmpl = std::string((*model)->ChatTemplate());
  ASSERT_FALSE(tmpl.empty());
  auto user_message = Value::Map({{"role", Value::Str("user")},
                                  {"content", Value::Str("Hello")}});
  auto messages = Value::List({user_message});
  const std::string system_line =
      "Reasoning effort is set to xhigh. Please think carefully through the "
      "task, validate key assumptions, consider plausible alternatives, and "
      "prioritize correctness, consistency, and clarity in the final answer.";
  const std::string want_default =
      "<|im_start|>system\n" + system_line +
      "<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n";
  const std::string want_no_think =
      "<|im_start|>user\nHello<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  auto with_think = Render(
      tmpl, {{"messages", messages},
             {"add_generation_prompt", Value::Bool(true)},
             {"tools", Value::None()}});
  ASSERT_TRUE(with_think.ok) << with_think.error;
  EXPECT_EQ(with_think.text, want_default);
  auto without_think = Render(
      tmpl, {{"messages", messages},
             {"add_generation_prompt", Value::Bool(true)},
             {"tools", Value::None()},
             {"enable_thinking", Value::Bool(false)}});
  ASSERT_TRUE(without_think.ok) << without_think.error;
  EXPECT_EQ(without_think.text, want_no_think);
}
