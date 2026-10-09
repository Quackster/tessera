#include "serve/render.hpp"

#include <string>
#include <utility>
#include <vector>

#include "serve/jinja/jinja.hpp"
#include "serve/tools/tool_call.hpp"
#include "tessera/model.hpp"

namespace tessera::serve {

jinja::Value ToJinja(const core::Json& json) {
  switch (json.type()) {
    case core::Json::Type::Null:
      return jinja::Value::None();
    case core::Json::Type::Bool:
      return jinja::Value::Bool(json.AsBool());
    case core::Json::Type::Number:
      return (json.AsNumber() == static_cast<double>(
                                      static_cast<std::int64_t>(
                                          json.AsNumber())))
                 ? jinja::Value::Int(
                       static_cast<std::int64_t>(json.AsNumber()))
                 : jinja::Value::Double(json.AsNumber());
    case core::Json::Type::String:
      return jinja::Value::Str(json.AsString());
    case core::Json::Type::Array: {
      std::vector<jinja::Value> items;
      for (const core::Json& item : json.AsArray()) {
        items.push_back(ToJinja(item));
      }
      return jinja::Value::List(std::move(items));
    }
    case core::Json::Type::Object: {
      std::vector<std::pair<std::string, jinja::Value>> entries;
      for (const auto& [key, value] : json.AsObject()) {
        entries.emplace_back(key, ToJinja(value));
      }
      return jinja::Value::Map(std::move(entries));
    }
  }
  return jinja::Value::None();
}

std::string RenderPrompt(const Model& model, const core::Json& body,
                         std::string* error) {
  const std::string tmpl(model.ChatTemplate());
  if (tmpl.empty()) {
    *error = "model has no chat template";
    return {};
  }
  core::Json messages = core::Json::Array();
  if (!NormalizeToolHistory(body, &messages, error)) {
    return {};
  }
  std::vector<jinja::Value> message_items;
  if (const core::Json* system = body.Find("system");
      system != nullptr && system->isString()) {
    message_items.push_back(jinja::Value::Map(
        {{"role", jinja::Value::Str("system")},
         {"content", jinja::Value::Str(system->AsString())}}));
  }
  for (const core::Json& message : messages.AsArray()) {
    message_items.push_back(ToJinja(message));
  }
  const core::Json* tools = EffectiveTools(body, error);
  if (!error->empty()) {
    return {};
  }
  std::vector<jinja::Kwarg> context;
  context.emplace_back("messages",
                       jinja::Value::List(std::move(message_items)));
  context.emplace_back("add_generation_prompt", jinja::Value::Bool(true));
  context.emplace_back("tools", tools == nullptr ? jinja::Value::None()
                                                 : ToJinja(*tools));
  if (const core::Json* thinking = body.Find("enable_thinking");
      thinking != nullptr &&
      thinking->type() == core::Json::Type::Bool) {
    context.emplace_back("enable_thinking",
                         jinja::Value::Bool(thinking->AsBool()));
  }
  auto rendered = jinja::Render(tmpl, context);
  if (!rendered.ok) {
    *error = rendered.error;
    return {};
  }
  return rendered.text;
}

}  // namespace tessera::serve
