#include <string>
#include <vector>

#include "serve/jinja/jinja.hpp"
#include "tessera/model.hpp"

namespace tessera {

std::expected<std::string, StatusCode> Model::ChatPrompt(
    std::string_view user_text, bool enable_thinking) const {
  const std::string tmpl(ChatTemplate());
  if (tmpl.empty()) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  std::vector<serve::jinja::Value> messages;
  messages.push_back(serve::jinja::Value::Map(
      {{"role", serve::jinja::Value::Str("user")},
       {"content", serve::jinja::Value::Str(std::string(user_text))}}));
  std::vector<serve::jinja::Kwarg> context;
  context.emplace_back("messages",
                       serve::jinja::Value::List(std::move(messages)));
  context.emplace_back("add_generation_prompt", serve::jinja::Value::Bool(true));
  context.emplace_back("tools", serve::jinja::Value::None());
  context.emplace_back("enable_thinking",
                       serve::jinja::Value::Bool(enable_thinking));
  auto rendered = serve::jinja::Render(tmpl, context);
  if (!rendered.ok) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return rendered.text;
}

}  // namespace tessera
