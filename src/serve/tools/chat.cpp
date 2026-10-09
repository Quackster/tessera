#include "serve/tools/chat.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "serve/render.hpp"
#include "serve/respond.hpp"
#include "serve/tools/tool_call.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/tokenizer.hpp"

namespace tessera::serve {

using core::Json;

namespace {

constexpr char kChatId[] = "chatcmpl-0";

// OpenAI tool_calls array for the parsed model calls.
core::Json ToolCallsJson(const std::vector<ToolCall>& calls) {
  core::Json items = core::Json::Array();
  for (const ToolCall& call : calls) {
    core::Json function = core::Json::Object();
    function.Set("name", core::Json::String(call.name));
    function.Set("arguments", core::Json::String(call.arguments));
    core::Json entry = core::Json::Object();
    entry.Set("id", core::Json::String(call.id));
    entry.Set("type", core::Json::String("function"));
    entry.Set("function", std::move(function));
    items.Push(std::move(entry));
  }
  return items;
}

core::Json StreamChunk(const std::string& model_name, core::Json delta,
                       const std::string& finish) {
  core::Json choice = core::Json::Object();
  choice.Set("index", core::Json::Number(0));
  choice.Set("delta", std::move(delta));
  if (finish.empty()) {
    choice.Set("finish_reason", core::Json());
  } else {
    choice.Set("finish_reason", core::Json::String(finish));
  }
  core::Json choices = core::Json::Array();
  choices.Push(std::move(choice));
  core::Json chunk = core::Json::Object();
  chunk.Set("id", core::Json::String(kChatId));
  chunk.Set("object", core::Json::String("chat.completion.chunk"));
  chunk.Set("model", core::Json::String(model_name));
  chunk.Set("choices", std::move(choices));
  return chunk;
}

void ChatBuffered(Engine& engine, Model& model, const Tokenizer& tokenizer,
                  ResponseWriter& writer, const std::string& model_name,
                  const GenerateOptions& options, std::size_t prompt_size) {
  std::string text;
  auto streamed = engine.GenerateStreaming(
      model, options, [&](std::uint32_t token) {
        if (writer.IsPeerGone()) {
          return false;
        }
        auto piece = tokenizer.Decode(std::span<const std::uint32_t>(&token, 1));
        text += piece ? *piece : std::string();
        return true;
      });
  if (!streamed) {
    SendGenerationError(writer, streamed.error());
    return;
  }
  std::string before;
  std::string reasoning;
  const std::vector<ToolCall> calls =
      ParseToolCalls(text, &before, &reasoning);
  const bool has_calls = !calls.empty();
  core::Json message = core::Json::Object();
  message.Set("role", core::Json::String("assistant"));
  if (!reasoning.empty()) {
    message.Set("reasoning_content", core::Json::String(reasoning));
  }
  message.Set("content",
              before.empty() ? core::Json() : core::Json::String(before));
  if (has_calls) {
    message.Set("tool_calls", ToolCallsJson(calls));
  }
  core::Json choice = core::Json::Object();
  choice.Set("index", core::Json::Number(0));
  choice.Set("message", std::move(message));
  choice.Set("finish_reason",
             core::Json::String(has_calls ? "tool_calls" : "length"));
  core::Json choices = core::Json::Array();
  choices.Push(std::move(choice));
  core::Json response = core::Json::Object();
  response.Set("id", core::Json::String(kChatId));
  response.Set("object", core::Json::String("chat.completion"));
  response.Set("model", core::Json::String(model_name));
  response.Set("choices", std::move(choices));
  response.Set("usage", UsageJson(prompt_size, *streamed));
  SendJson(writer, 200, response);
}

// The turn is generated first, then replayed as SSE: the client cannot
// tell content apart from a tool call until the block is complete.
void ChatStreamed(Engine& engine, Model& model, const Tokenizer& tokenizer,
                  ResponseWriter& writer, const std::string& model_name,
                  const GenerateOptions& options) {
  std::string text;
  auto streamed = engine.GenerateStreaming(
      model, options, [&](std::uint32_t token) {
        if (writer.IsPeerGone()) {
          return false;
        }
        auto piece = tokenizer.Decode(std::span<const std::uint32_t>(&token, 1));
        text += piece ? *piece : std::string();
        return true;
      });
  if (!streamed) {
    core::Json chunk = core::Json::Object();
    chunk.Set("error", core::Json::String(
                           streamed.error() == StatusCode::InvalidArgument
                               ? "prompt exceeds the model context window"
                               : "generation failed"));
    WriteSse(writer, "", chunk, false);
    return;
  }
  (void)writer.SendHeaders(200, "text/event-stream", true);
  core::Json role_delta = core::Json::Object();
  role_delta.Set("role", core::Json::String("assistant"));
  WriteSse(writer, "", StreamChunk(model_name, std::move(role_delta), ""), false);
  std::string before;
  std::string reasoning;
  const std::vector<ToolCall> calls =
      ParseToolCalls(text, &before, &reasoning);
  if (!reasoning.empty()) {
    core::Json delta = core::Json::Object();
    delta.Set("reasoning_content", core::Json::String(reasoning));
    WriteSse(writer, "", StreamChunk(model_name, std::move(delta), ""), false);
  }
  if (!before.empty()) {
    core::Json delta = core::Json::Object();
    delta.Set("content", core::Json::String(before));
    WriteSse(writer, "", StreamChunk(model_name, std::move(delta), ""), false);
  }
  for (std::size_t i = 0; i < calls.size(); ++i) {
    core::Json function = core::Json::Object();
    function.Set("name", core::Json::String(calls[i].name));
    function.Set("arguments", core::Json::String(calls[i].arguments));
    core::Json entry = core::Json::Object();
    entry.Set("index", core::Json::Number(static_cast<double>(i)));
    entry.Set("id", core::Json::String(calls[i].id));
    entry.Set("type", core::Json::String("function"));
    entry.Set("function", std::move(function));
    core::Json items = core::Json::Array();
    items.Push(std::move(entry));
    core::Json delta = core::Json::Object();
    delta.Set("tool_calls", std::move(items));
    WriteSse(writer, "", StreamChunk(model_name, std::move(delta), ""), false);
  }
  core::Json end = core::Json::Object();
  WriteSse(writer, "",
           StreamChunk(model_name, std::move(end),
                       calls.empty() ? "length" : "tool_calls"),
           false);
  (void)writer.Write("data: [DONE]\n\n");
}

}  // namespace

bool WantsTools(const Json& body) {
  std::string error;
  return EffectiveTools(body, &error) != nullptr;
}

void ChatWithTools(Engine& engine, Model& model, const Tokenizer& tokenizer,
                   ResponseWriter& writer, const Json& body,
                   std::size_t default_max) {
  std::string error;
  const std::string prompt = RenderPrompt(model, body, &error);
  if (!error.empty()) {
    SendError(writer, 422, error);
    return;
  }
  auto ids = tokenizer.Encode(prompt);
  if (!ids) {
    SendError(writer, 500, "tokenization failed");
    return;
  }
  GenerateOptions options;
  options.max_tokens = MaxTokensFrom(body, default_max);
  options.prompt_tokens = *ids;
  if (RejectOversizePrompt(writer, ids->size(), options.max_tokens,
                           model.MaxContextLength())) {
    return;
  }
  const std::string model_name(model.Name());
  if (WantsStream(body)) {
    ChatStreamed(engine, model, tokenizer, writer, model_name, options);
  } else {
    ChatBuffered(engine, model, tokenizer, writer, model_name, options,
                 ids->size());
  }
}

}  // namespace tessera::serve
