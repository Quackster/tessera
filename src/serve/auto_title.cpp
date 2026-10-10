#include "serve/auto_title.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "serve/render.hpp"
#include "serve/session.hpp"
#include "serve/tools/tool_call.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/tokenizer.hpp"

namespace tessera::serve {

namespace {

// Title turn length: a greedy short answer, not a second conversation.
constexpr std::size_t kAutoTitleTokens = 32;
// First-message characters fed to the title prompt: bounds the prefill.
constexpr std::size_t kAutoTitleSourceChars = 512;

}  // namespace

core::Json TitlePromptBody(const std::string& user_text) {
  core::Json message = core::Json::Object();
  message.Set("role", core::Json::String("user"));
  message.Set("content", core::Json::String(
                             "Give this conversation a short title of at "
                             "most five words. Reply with only the title, "
                             "no quotes.\n\nFirst message:\n" +
                             user_text.substr(0, kAutoTitleSourceChars)));
  core::Json messages = core::Json::Array();
  messages.Push(std::move(message));
  core::Json body = core::Json::Object();
  body.Set("messages", std::move(messages));
  body.Set("enable_thinking", core::Json::Bool(false));
  return body;
}

void MaybeAutoTitle(Engine& engine, Model& model, const Tokenizer& tokenizer,
                    const std::shared_ptr<Session>& session, bool enabled) {
  const SessionView view = session->View();
  if (view.title != kDefaultTitle) {
    return;
  }
  std::string first;
  for (const SessionMessage& message : view.messages) {
    if (message.role == "user" && !message.content.empty()) {
      first = message.content;
      break;
    }
  }
  if (first.empty()) {
    return;
  }
  if (!enabled) {
    session->SetTitle(TitleFromText(first));
    return;
  }
  std::string error;
  const std::string prompt =
      RenderPrompt(model, TitlePromptBody(first), &error);
  if (!error.empty()) {
    session->SetTitle(TitleFromText(first));
    return;
  }
  auto ids = tokenizer.Encode(prompt);
  if (!ids) {
    session->SetTitle(TitleFromText(first));
    return;
  }
  GenerateOptions options;
  options.max_completion_tokens = kAutoTitleTokens;
  options.prompt_tokens = *ids;
  std::vector<std::uint32_t> produced;
  auto streamed = engine.GenerateStreaming(
      model, options, [&](std::uint32_t token) {
        produced.push_back(token);
        return true;
      });
  if (!streamed || produced.empty()) {
    session->SetTitle(TitleFromText(first));
    return;
  }
  auto text = tokenizer.Decode(produced);
  std::string before;
  (void)ParseToolCalls(text ? *text : std::string(), &before);
  if (before.empty()) {
    session->SetTitle(TitleFromText(first));
    return;
  }
  session->SetTitle(TitleFromText(before));
}

}  // namespace tessera::serve
