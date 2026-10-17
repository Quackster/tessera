#include "serve/tools/chat.hpp"

#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "serve/render.hpp"
#include "serve/respond.hpp"
#include "serve/session.hpp"
#include "serve/tools/tool_call.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/tokenizer.hpp"

namespace tessera::serve {

using core::Json;

namespace {

constexpr char kChatId[] = "chatcmpl-0";

// One OpenAI tool_calls entry; streaming entries carry the array index
// (non-streaming message entries do not).
core::Json ToolCallJson(const ToolCall& call, bool with_index,
                        std::size_t index) {
  core::Json function = core::Json::Object();
  function.Set("name", core::Json::String(call.name));
  function.Set("arguments", core::Json::String(call.arguments));
  core::Json entry = core::Json::Object();
  if (with_index) {
    entry.Set("index", core::Json::Number(static_cast<double>(index)));
  }
  entry.Set("id", core::Json::String(call.id));
  entry.Set("type", core::Json::String("function"));
  entry.Set("function", std::move(function));
  return entry;
}

// OpenAI tool_calls array for the parsed model calls.
core::Json ToolCallsJson(const std::vector<ToolCall>& calls) {
  core::Json items = core::Json::Array();
  for (const ToolCall& call : calls) {
    items.Push(ToolCallJson(call, /*with_index=*/false, 0));
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

// Fill the session turn from the split reply: `content` and `reasoning`
// are the concatenated streamer deltas, `calls` the parsed tool calls.
void FillTurn(SessionMessage* out_turn, std::string_view content,
              std::string_view reasoning, const std::vector<ToolCall>& calls,
              bool stopped, std::size_t prompt_tokens,
              std::size_t completion_tokens, long long elapsed_ms) {
  if (out_turn == nullptr) {
    return;
  }
  out_turn->role = "assistant";
  out_turn->content = std::string(content);
  out_turn->reasoning = std::string(reasoning);
  out_turn->tool_calls_json =
      calls.empty() ? std::string() : ToolCallsJson(calls).Dump();
  out_turn->stopped = stopped;
  out_turn->stats.prompt_tokens = prompt_tokens;
  out_turn->stats.completion_tokens = completion_tokens;
  out_turn->stats.elapsed_ms = elapsed_ms;
}

bool ChatBuffered(const ToolGenerate& generate, const Tokenizer& tokenizer,
                  ResponseWriter& writer, const std::string& model_name,
                  const GenerateOptions& options, bool think_expected,
                  SessionMessage* out_turn) {
  std::string text;
  const auto gen_started = std::chrono::steady_clock::now();
  auto streamed = generate(options, [&](std::uint32_t token) {
    if (writer.IsPeerGone()) {
      return false;
    }
    auto piece = tokenizer.Decode(std::span<const std::uint32_t>(&token, 1));
    text += piece ? *piece : std::string();
    return true;
  });
  const long long elapsed_ms =
      MillisBetween(gen_started, std::chrono::steady_clock::now());
  if (!streamed) {
    SendGenerationError(writer, streamed.error());
    return false;
  }
  ReplyStreamer streamer(think_expected);
  ReplyStreamer::Deltas deltas = streamer.Push(text);
  deltas.Append(streamer.Finish());
  FillTurn(out_turn, deltas.content, deltas.reasoning, deltas.calls,
           writer.IsPeerGone(), options.prompt_tokens.size(),
           streamed->produced, elapsed_ms);
  core::Json message = core::Json::Object();
  message.Set("role", core::Json::String("assistant"));
  if (!deltas.reasoning.empty()) {
    message.Set("reasoning_content", core::Json::String(deltas.reasoning));
  }
  message.Set("content",
              deltas.content.empty() ? core::Json()
                                     : core::Json::String(deltas.content));
  if (!deltas.calls.empty()) {
    message.Set("tool_calls", ToolCallsJson(deltas.calls));
  }
  core::Json choice = core::Json::Object();
  choice.Set("index", core::Json::Number(0));
  choice.Set("message", std::move(message));
  choice.Set(
      "finish_reason",
      core::Json::String(deltas.calls.empty()
                             ? FinishReasonName(streamed->reason)
                             : "tool_calls"));
  core::Json choices = core::Json::Array();
  choices.Push(std::move(choice));
  core::Json response = core::Json::Object();
  response.Set("id", core::Json::String(kChatId));
  response.Set("object", core::Json::String("chat.completion"));
  response.Set("model", core::Json::String(model_name));
  response.Set("choices", std::move(choices));
  response.Set("usage",
               UsageJson(options.prompt_tokens.size(), streamed->produced));
  SendJson(writer, 200, response);
  return true;
}

// The turn streams live: reasoning, content and each tool call go out
// as the splitter resolves them, so the client sees the first deltas
// as soon as decoding starts (the buffered variant replays the same
// deltas in one shot).
bool ChatStreamed(const ToolGenerate& generate, const Tokenizer& tokenizer,
                  ResponseWriter& writer, const std::string& model_name,
                  const GenerateOptions& options, bool think_expected,
                  SessionMessage* out_turn) {
  (void)writer.SendHeaders(200, "text/event-stream", true);
  core::Json role_delta = core::Json::Object();
  role_delta.Set("role", core::Json::String("assistant"));
  WriteSse(writer, "", StreamChunk(model_name, std::move(role_delta), ""),
           false);
  ReplyStreamer streamer(think_expected);
  ReplyStreamer::Deltas total;
  const auto emit = [&](const ReplyStreamer::Deltas& part) {
    if (!part.reasoning.empty()) {
      core::Json delta = core::Json::Object();
      delta.Set("reasoning_content", core::Json::String(part.reasoning));
      WriteSse(writer, "", StreamChunk(model_name, std::move(delta), ""),
               false);
      total.reasoning += part.reasoning;
    }
    if (!part.content.empty()) {
      core::Json delta = core::Json::Object();
      delta.Set("content", core::Json::String(part.content));
      WriteSse(writer, "", StreamChunk(model_name, std::move(delta), ""),
               false);
      total.content += part.content;
    }
    for (const ToolCall& call : part.calls) {
      core::Json items = core::Json::Array();
      items.Push(ToolCallJson(call, /*with_index=*/true, total.calls.size()));
      core::Json delta = core::Json::Object();
      delta.Set("tool_calls", std::move(items));
      WriteSse(writer, "", StreamChunk(model_name, std::move(delta), ""),
               false);
      total.calls.push_back(call);
    }
  };
  const auto gen_started = std::chrono::steady_clock::now();
  auto streamed = generate(options, [&](std::uint32_t token) {
    if (writer.IsPeerGone()) {
      return false;
    }
    auto piece = tokenizer.Decode(std::span<const std::uint32_t>(&token, 1));
    emit(streamer.Push(piece ? *piece : std::string()));
    return true;
  });
  emit(streamer.Finish());
  const long long elapsed_ms =
      MillisBetween(gen_started, std::chrono::steady_clock::now());
  if (!streamed) {
    if (!writer.IsPeerGone()) {
      core::Json chunk = core::Json::Object();
      chunk.Set("error",
                core::Json::String(streamed.error() == StatusCode::InvalidArgument
                                       ? "prompt exceeds the model context window"
                                       : "generation failed"));
      WriteSse(writer, "", chunk, false);
    }
    return false;
  }
  FillTurn(out_turn, total.content, total.reasoning, total.calls,
           writer.IsPeerGone(), options.prompt_tokens.size(),
           streamed->produced, elapsed_ms);
  core::Json end = core::Json::Object();
  WriteSse(writer, "",
           StreamChunk(model_name, std::move(end),
                       total.calls.empty() ? FinishReasonName(streamed->reason)
                                           : "tool_calls"),
           false);
  (void)writer.Write("data: [DONE]\n\n");
  return true;
}

}  // namespace

bool WantsTools(const Json& body) {
  std::string error;
  return EffectiveTools(body, &error) != nullptr;
}

bool ChatWithTools(const ToolGenerate& generate, Model& model,
                   const Tokenizer& tokenizer, ResponseWriter& writer,
                   const Json& body, std::size_t default_max,
                   SessionMessage* out_turn, KvCacheType kv_type,
                   const TemplateDefaults& defaults,
                   const TurnSampling& sampling) {
  std::string error;
  const std::string prompt = RenderPrompt(model, body, &error, defaults);
  if (!error.empty()) {
    SendError(writer, 422, error);
    return false;
  }
  auto ids = tokenizer.Encode(prompt);
  if (!ids) {
    SendError(writer, 500, "tokenization failed");
    return false;
  }
  GenerateOptions options;
  options.max_completion_tokens = MaxCompletionTokensFrom(body, default_max);
  options.max_thinking_tokens = MaxThinkingTokensFrom(body);
  options.prompt_tokens = *ids;
  options.kv_type = kv_type;
  if (auto applied = ApplyTurnSampling(body, sampling, &options); !applied) {
    SendError(writer, 422, applied.error());
    return false;
  }
  if (RejectOversizePrompt(writer, ids->size(), model.MaxContextLength())) {
    return false;
  }
  const std::string model_name(model.Name());
  // The template opens the thinking block for thinking turns, so the
  // splitter streams the leading text as reasoning; the same check as
  // the plain chat path (see ThinkExpected).
  const bool think_expected =
      ThinkExpected(prompt, RequestEnableThinking(body, defaults));
  if (WantsStream(body)) {
    return ChatStreamed(generate, tokenizer, writer, model_name, options,
                        think_expected, out_turn);
  }
  return ChatBuffered(generate, tokenizer, writer, model_name, options,
                      think_expected, out_turn);
}

}  // namespace tessera::serve
