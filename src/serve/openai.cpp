#include "serve/openai.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "serve/render.hpp"
#include "serve/respond.hpp"
#include "serve/session_chat.hpp"
#include "serve/tools/chat.hpp"
#include "serve/tools/tool_call.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/tokenizer.hpp"

namespace tessera::serve {

namespace {

using core::Json;

// First user text in a history, for naming a new session.
std::string FirstUserText(const SessionView& view) {
  for (const SessionMessage& message : view.messages) {
    if (message.role == "user" && !message.content.empty()) {
      return message.content;
    }
  }
  return {};
}

// Title a fresh session from its first user turn.
void MaybeNameSession(const std::shared_ptr<Session>& session,
                      const SessionView& before) {
  if (!before.messages.empty() || before.title != kDefaultTitle) {
    return;
  }
  const std::string text = FirstUserText(session->View());
  if (!text.empty()) {
    session->SetTitle(TitleFromText(text));
  }
}

// Assistant turn stored from generated `text`: thinking and tool markup
// stay out of the history, so a resent OpenAI history still matches.
SessionMessage AssistantTurn(std::string_view text, bool stopped,
                             TurnStats stats) {
  std::string before;
  std::string reasoning;
  (void)ParseToolCalls(text, &before, &reasoning);
  return SessionMessage{"assistant", before, reasoning,
                        /*tool_calls_json=*/{}, /*tool_call_id=*/{}, stopped,
                        stats, /*created_ms=*/0};
}

// Reconcile an incoming OpenAI history with the session: 422 on a
// malformed history, otherwise the merged view for rendering.
bool BeginChatTurn(const std::shared_ptr<Session>& session,
                   ResponseWriter& writer, const Json& body,
                   SessionView* view) {
  const Json* messages = body.Find("messages");
  if (messages == nullptr) {
    SendError(writer, 422, "messages must be an array");
    return false;
  }
  std::vector<SessionMessage> incoming;
  std::string error;
  if (!SessionMessagesFromJson(*messages, &incoming, &error)) {
    SendError(writer, 422, error);
    return false;
  }
  const SessionView before = session->View();
  session->SyncHistory(std::move(incoming));
  MaybeNameSession(session, before);
  *view = session->View();
  return true;
}

// Prompt body for a session turn: the full stored history plus the
// request-level fields (system prompt, tools, thinking flag, budgets
// and streaming flag).
Json SessionPromptBody(const Json& body, const SessionView& view) {
  Json prompt_body = Json::Object();
  prompt_body.Set("messages", SessionHistoryJson(view));
  for (const char* key : {"system", "tools", "tool_choice",
                          "enable_thinking", "reasoning_effort", "stream",
                          "max_completion_tokens", "max_thinking_tokens",
                          "chat_template_kwargs"}) {
    if (const Json* value = body.Find(key)) {
      prompt_body.Set(key, *value);
    }
  }
  return prompt_body;
}

// Take the session turn (404 unknown, 409 already generating). The
// guard ends it on every exit path once set.
bool BeginSession(SessionStore& sessions, const std::string& session_id,
                  ResponseWriter& writer, std::shared_ptr<Session>* session,
                  SessionTurn* turn) {
  *session = sessions.Get(session_id);
  if (*session == nullptr) {
    SendError(writer, 404, "unknown session");
    return false;
  }
  if (!(*session)->TryBegin()) {
    SendError(writer, 409, "session is already generating");
    return false;
  }
  turn->session = *session;
  return true;
}

}  // namespace

void OpenAiComplete(Engine& engine, Model& model, const Tokenizer& tokenizer,
                    SessionStore& sessions, std::mutex& generation,
                    ResponseWriter& writer, const Json& body,
                    std::size_t default_max,
                    const CalibrationConfig& calibration) {
  const std::string session_id = SessionIdFrom(body);
  const Json* prompt = body.Find("prompt");
  if (prompt == nullptr || !prompt->isString()) {
    SendError(writer, 422, "prompt must be a string");
    return;
  }
  std::shared_ptr<Session> session;
  SessionTurn turn;
  if (!session_id.empty() &&
      !BeginSession(sessions, session_id, writer, &session, &turn)) {
    return;
  }
  auto slot = WaitForGpu(generation, writer);
  if (!slot) {
    return;
  }
  std::string prompt_text = prompt->AsString();
  if (session != nullptr) {
    const SessionView before = session->View();
    session->Append(SessionMessage{"user", prompt_text, /*reasoning=*/{},
                                   /*tool_calls_json=*/{}, /*tool_call_id=*/{},
                                   false, /*stats=*/{}, /*created_ms=*/0});
    MaybeNameSession(session, before);
    std::string error;
    prompt_text = RenderPrompt(model, SessionPromptBody(body, session->View()),
                               &error);
    if (!error.empty()) {
      SendError(writer, 422, error);
      return;
    }
  }
  auto ids = tokenizer.Encode(prompt_text);
  if (!ids) {
    SendError(writer, 500, "tokenization failed");
    return;
  }
  const std::size_t max_completion_tokens =
      MaxCompletionTokensFrom(body, default_max);
  const bool stream = WantsStream(body);
  GenerateOptions options;
  options.max_completion_tokens = max_completion_tokens;
  options.max_thinking_tokens = MaxThinkingTokensFrom(body);
  options.prompt_tokens = *ids;
  ApplyToGenerateOptions(calibration, options);
  if (RejectOversizePrompt(writer, ids->size(), model.MaxContextLength())) {
    return;
  }
  TurnStats stats;
  stats.prompt_tokens = ids->size();
  if (!stream) {
    // GenerateStreaming with a liveness hook instead of Generate:
    // identical tokens, but a closed window aborts the turn instead
    // of decoding into the void.
    std::vector<std::uint32_t> produced;
    const auto gen_started = std::chrono::steady_clock::now();
    auto streamed = engine.GenerateStreaming(
        model, options, [&](std::uint32_t token) {
          if (writer.IsPeerGone()) {
            return false;
          }
          produced.push_back(token);
          return true;
        });
    stats.elapsed_ms = MillisBetween(gen_started,
                                     std::chrono::steady_clock::now());
    if (!streamed) {
      // A gone peer fails here only on a real error (abort returns a
      // count); anything else is reported when someone listens.
      if (!writer.IsPeerGone()) {
        SendGenerationError(writer, streamed.error());
      }
      return;
    }
    stats.completion_tokens = produced.size();
    auto text = tokenizer.Decode(produced);
    if (!text) {
      SendError(writer, 500, "detokenization failed");
      return;
    }
    if (session != nullptr) {
      session->Append(AssistantTurn(*text, writer.IsPeerGone(), stats));
    }
    Json choice = Json::Object();
    choice.Set("text", Json::String(*text));
    choice.Set("index", Json::Number(0));
    choice.Set("finish_reason", Json::String(FinishReasonName(streamed->reason)));
    Json choices = Json::Array();
    choices.Push(std::move(choice));
    Json response = Json::Object();
    response.Set("id", Json::String("cmpl-0"));
    response.Set("object", Json::String("text_completion"));
    response.Set("model", Json::String(std::string(model.Name())));
    response.Set("choices", std::move(choices));
    response.Set("usage", UsageJson(ids->size(), produced.size()));
    SendJson(writer, 200, response);
    return;
  }
  (void)writer.SendHeaders(200, "text/event-stream", true);
  std::size_t count = 0;
  std::string text;
  const auto gen_started = std::chrono::steady_clock::now();
  auto streamed = engine.GenerateStreaming(
      model, options, [&](std::uint32_t token) {
        if (writer.IsPeerGone()) {
          return false;
        }
        auto piece = tokenizer.Decode(
            std::span<const std::uint32_t>(&token, 1));
        if (piece) {
          text += *piece;
        }
        Json choice = Json::Object();
        choice.Set("text", Json::String(piece ? *piece : std::string()));
        choice.Set("index", Json::Number(0));
        choice.Set("finish_reason", Json());
        Json choices = Json::Array();
        choices.Push(std::move(choice));
        Json chunk = Json::Object();
        chunk.Set("id", Json::String("cmpl-0"));
        chunk.Set("object", Json::String("text_completion"));
        chunk.Set("choices", std::move(choices));
        WriteSse(writer, "", chunk, false);
        ++count;
        return true;
      });
  stats.elapsed_ms = MillisBetween(gen_started,
                                   std::chrono::steady_clock::now());
  stats.completion_tokens = count;
  if (session != nullptr) {
    session->Append(AssistantTurn(text, writer.IsPeerGone(), stats));
  }
  if (!streamed) {
    Json chunk = Json::Object();
    chunk.Set("error", Json::String(
                           streamed.error() == StatusCode::InvalidArgument
                               ? "prompt exceeds the model context window"
                               : "generation failed"));
    WriteSse(writer, "", chunk, false);
  }
  Json done = Json::Object();
  done.Set("choices", Json());
  Json choices = Json::Array();
  Json choice = Json::Object();
  choice.Set("text", Json::String(""));
  choice.Set("index", Json::Number(0));
  choice.Set("finish_reason",
             Json::String(streamed ? FinishReasonName(streamed->reason)
                                   : "length"));
  choices.Push(std::move(choice));
  done.Set("choices", std::move(choices));
  WriteSse(writer, "", done, false);
  (void)writer.Write("data: [DONE]\n\n");
  (void)count;
}

void OpenAiChat(Engine& engine, Model& model, const Tokenizer& tokenizer,
                SessionStore& sessions, std::mutex& generation,
                ResponseWriter& writer, const Json& body,
                std::size_t default_max, bool anthropic,
                const CalibrationConfig& calibration) {
  const std::string session_id = SessionIdFrom(body);
  std::shared_ptr<Session> session;
  SessionTurn turn;
  Json prompt_body = body;
  if (!session_id.empty()) {
    if (!BeginSession(sessions, session_id, writer, &session, &turn)) {
      return;
    }
  }
  auto slot = WaitForGpu(generation, writer);
  if (!slot) {
    return;
  }
  if (session != nullptr) {
    SessionView view{"", "", {}, false, false};
    if (!BeginChatTurn(session, writer, body, &view)) {
      return;
    }
    prompt_body = SessionPromptBody(body, view);
  }
  if (!anthropic && WantsTools(prompt_body)) {
    SessionMessage assistant;
    const bool ok =
        ChatWithTools(engine, model, tokenizer, writer, prompt_body,
                      default_max, session != nullptr ? &assistant : nullptr);
    if (session != nullptr && ok) {
      session->Append(std::move(assistant));
    }
    return;
  }
  std::string error;
  const std::string prompt = RenderPrompt(model, prompt_body, &error);
  if (!error.empty()) {
    SendError(writer, 422, error);
    return;
  }
  auto ids = tokenizer.Encode(prompt);
  if (!ids) {
    SendError(writer, 500, "tokenization failed");
    return;
  }
  const std::size_t max_completion_tokens =
      MaxCompletionTokensFrom(prompt_body, default_max);
  const bool stream = WantsStream(prompt_body);
  GenerateOptions options;
  options.max_completion_tokens = max_completion_tokens;
  options.max_thinking_tokens = MaxThinkingTokensFrom(prompt_body);
  options.prompt_tokens = *ids;
  ApplyToGenerateOptions(calibration, options);
  if (RejectOversizePrompt(writer, ids->size(), model.MaxContextLength())) {
    return;
  }
  const std::string model_name(model.Name());
  TurnStats stats;
  stats.prompt_tokens = ids->size();
  // Thinking is on unless the request turns it off. The chat template
  // then opens the <think> block in the prompt, so the model streams
  // reasoning with no opening tag in its output; ThinkExpected tells
  // the streamer to treat that leading text as reasoning.
  const Json* thinking_flag = prompt_body.Find("enable_thinking");
  const bool enable_thinking =
      thinking_flag == nullptr ||
      thinking_flag->type() != Json::Type::Bool || thinking_flag->AsBool();
  if (anthropic) {
    if (!stream) {
      // GenerateStreaming with a liveness hook instead of Generate:
      // identical tokens, but a closed window aborts the turn instead
      // of decoding into the void.
      std::vector<std::uint32_t> produced;
      const auto gen_started = std::chrono::steady_clock::now();
      auto streamed = engine.GenerateStreaming(
          model, options, [&](std::uint32_t token) {
            if (writer.IsPeerGone()) {
              return false;
            }
            produced.push_back(token);
            return true;
          });
      stats.elapsed_ms = MillisBetween(gen_started,
                                       std::chrono::steady_clock::now());
      if (!streamed) {
        // A gone peer fails here only on a real error (abort returns a
        // count); anything else is reported when someone listens.
        if (!writer.IsPeerGone()) {
          SendGenerationError(writer, streamed.error());
        }
        return;
      }
      stats.completion_tokens = produced.size();
      auto text = tokenizer.Decode(produced);
      if (session != nullptr) {
        session->Append(AssistantTurn(text ? *text : std::string(),
                                      writer.IsPeerGone(), stats));
      }
      Json content = Json::Array();
      Json block = Json::Object();
      block.Set("type", Json::String("text"));
      block.Set("text", Json::String(text ? *text : std::string()));
      content.Push(std::move(block));
      Json usage = Json::Object();
      usage.Set("input_tokens", Json::Number(static_cast<double>(ids->size())));
      usage.Set("output_tokens",
                Json::Number(static_cast<double>(produced.size())));
      Json response = Json::Object();
      response.Set("id", Json::String("msg_0"));
      response.Set("type", Json::String("message"));
      response.Set("role", Json::String("assistant"));
      response.Set("model", Json::String(model_name));
      response.Set("content", std::move(content));
      response.Set("stop_reason",
                   Json::String(streamed->reason == FinishReason::Length
                                    ? "max_tokens"
                                    : "end_turn"));
      response.Set("usage", std::move(usage));
      SendJson(writer, 200, response);
      return;
    }
    (void)writer.SendHeaders(200, "text/event-stream", true);
    Json start = Json::Object();
    start.Set("type", Json::String("message_start"));
    Json message = Json::Object();
    message.Set("id", Json::String("msg_0"));
    message.Set("role", Json::String("assistant"));
    message.Set("content", Json::Array());
    start.Set("message", std::move(message));
    WriteSse(writer, "message_start", start, true);
    Json block_start = Json::Object();
    block_start.Set("type", Json::String("content_block_start"));
    block_start.Set("index", Json::Number(0));
    Json block = Json::Object();
    block.Set("type", Json::String("text"));
    block.Set("text", Json::String(""));
    block_start.Set("content_block", std::move(block));
    WriteSse(writer, "content_block_start", block_start, true);
    std::string text;
    std::size_t count = 0;
    const auto gen_started = std::chrono::steady_clock::now();
    auto streamed = engine.GenerateStreaming(model, options, [&](std::uint32_t token) {
      if (writer.IsPeerGone()) {
        return false;
      }
      auto piece =
          tokenizer.Decode(std::span<const std::uint32_t>(&token, 1));
      if (piece) {
        text += *piece;
      }
      ++count;
      Json delta = Json::Object();
      delta.Set("type", Json::String("content_block_delta"));
      delta.Set("index", Json::Number(0));
      Json text_delta = Json::Object();
      text_delta.Set("type", Json::String("text_delta"));
      text_delta.Set("text", Json::String(piece ? *piece : std::string()));
      delta.Set("delta", std::move(text_delta));
      WriteSse(writer, "content_block_delta", delta, true);
      return true;
    });
    stats.elapsed_ms = MillisBetween(gen_started,
                                     std::chrono::steady_clock::now());
    stats.completion_tokens = count;
    if (session != nullptr) {
      session->Append(AssistantTurn(text, writer.IsPeerGone(), stats));
    }
    Json stop = Json::Object();
    stop.Set("type", Json::String("content_block_stop"));
    stop.Set("index", Json::Number(0));
    WriteSse(writer, "content_block_stop", stop, true);
    Json message_delta = Json::Object();
    message_delta.Set("type", Json::String("message_delta"));
    Json md = Json::Object();
    md.Set("stop_reason",
           Json::String(streamed && streamed->reason == FinishReason::Length
                            ? "max_tokens"
                            : "end_turn"));
    message_delta.Set("delta", std::move(md));
    WriteSse(writer, "message_delta", message_delta, true);
    Json message_stop = Json::Object();
    message_stop.Set("type", Json::String("message_stop"));
    WriteSse(writer, "message_stop", message_stop, true);
    return;
  }
  // OpenAI chat.
  if (!stream) {
    // GenerateStreaming with a liveness hook instead of Generate:
    // identical tokens, but a closed window aborts the turn instead
    // of decoding into the void.
    std::vector<std::uint32_t> produced;
    const auto chat_started = std::chrono::steady_clock::now();
    auto streamed = engine.GenerateStreaming(
        model, options, [&](std::uint32_t token) {
          if (writer.IsPeerGone()) {
            return false;
          }
          produced.push_back(token);
          return true;
        });
    stats.elapsed_ms = MillisBetween(chat_started,
                                     std::chrono::steady_clock::now());
    if (!streamed) {
      // A gone peer fails here only on a real error (abort returns a
      // count); anything else is reported when someone listens.
      if (!writer.IsPeerGone()) {
        SendGenerationError(writer, streamed.error());
      }
      return;
    }
    stats.completion_tokens = produced.size();
    auto text = tokenizer.Decode(produced);
    if (!text) {
      SendError(writer, 500, "detokenization failed");
      return;
    }
    // Split the model's thinking out of the reply: the template opened
    // the <think> block, so `SplitThink` (via the streamer) recovers the
    // reasoning before the `</think>` closer. Without this the reasoning
    // is returned as `content` and the real answer never surfaces.
    ThinkStreamer streamer(ThinkExpected(prompt, enable_thinking));
    const ThinkStreamer::Deltas part = streamer.Push(*text);
    const ThinkStreamer::Deltas tail = streamer.Finish();
    const std::string reasoning_text = part.reasoning + tail.reasoning;
    const std::string content_text = part.content + tail.content;
    if (session != nullptr) {
      session->Append(AssistantTurn(*text, writer.IsPeerGone(), stats));
    }
    Json message = Json::Object();
    message.Set("role", Json::String("assistant"));
    if (!reasoning_text.empty()) {
      message.Set("reasoning_content", Json::String(reasoning_text));
    }
    message.Set("content", Json::String(content_text));
    Json choice = Json::Object();
    choice.Set("index", Json::Number(0));
    choice.Set("message", std::move(message));
    choice.Set("finish_reason", Json::String(FinishReasonName(streamed->reason)));
    Json choices = Json::Array();
    choices.Push(std::move(choice));
    Json response = Json::Object();
    response.Set("id", Json::String("chatcmpl-0"));
    response.Set("object", Json::String("chat.completion"));
    response.Set("model", Json::String(model_name));
    response.Set("choices", std::move(choices));
    response.Set("usage", UsageJson(ids->size(), produced.size()));
    SendJson(writer, 200, response);
    return;
  }
  (void)writer.SendHeaders(200, "text/event-stream", true);
  ThinkStreamer streamer(ThinkExpected(prompt, enable_thinking));
  const auto stream_piece = [&](const ThinkStreamer::Deltas& part) {
    if (part.reasoning.empty() && part.content.empty()) {
      return;
    }
    Json delta = Json::Object();
    if (!part.reasoning.empty()) {
      delta.Set("reasoning_content", Json::String(part.reasoning));
    }
    if (!part.content.empty()) {
      delta.Set("content", Json::String(part.content));
    }
    Json choice = Json::Object();
    choice.Set("index", Json::Number(0));
    choice.Set("delta", std::move(delta));
    choice.Set("finish_reason", Json());
    Json choices = Json::Array();
    choices.Push(std::move(choice));
    Json chunk = Json::Object();
    chunk.Set("id", Json::String("chatcmpl-0"));
    chunk.Set("object", Json::String("chat.completion.chunk"));
    chunk.Set("model", Json::String(model_name));
    chunk.Set("choices", std::move(choices));
    WriteSse(writer, "", chunk, false);
  };
  std::string text;
  std::size_t count = 0;
  const auto chat_started = std::chrono::steady_clock::now();
  auto streamed = engine.GenerateStreaming(model, options, [&](std::uint32_t token) {
    if (writer.IsPeerGone()) {
      return false;
    }
    auto piece = tokenizer.Decode(std::span<const std::uint32_t>(&token, 1));
    if (piece) {
      text += *piece;
    }
    ++count;
    stream_piece(streamer.Push(piece ? *piece : std::string()));
    return true;
  });
  stream_piece(streamer.Finish());
  stats.elapsed_ms = MillisBetween(chat_started,
                                   std::chrono::steady_clock::now());
  stats.completion_tokens = count;
  if (session != nullptr) {
    session->Append(AssistantTurn(text, writer.IsPeerGone(), stats));
  }
  // Final chunk carries the end reason; the deltas above leave it null
  // (OpenAI convention) so a client that reads finish_reason sees "stop"
  // for a completed turn and "length" only on a real truncation.
  Json end_delta = Json::Object();
  Json end_choice = Json::Object();
  end_choice.Set("index", Json::Number(0));
  end_choice.Set("delta", std::move(end_delta));
  end_choice.Set("finish_reason",
                 Json::String(streamed ? FinishReasonName(streamed->reason)
                                       : "length"));
  Json end_choices = Json::Array();
  end_choices.Push(std::move(end_choice));
  Json end_chunk = Json::Object();
  end_chunk.Set("id", Json::String("chatcmpl-0"));
  end_chunk.Set("object", Json::String("chat.completion.chunk"));
  end_chunk.Set("model", Json::String(model_name));
  end_chunk.Set("choices", std::move(end_choices));
  WriteSse(writer, "", end_chunk, false);
  (void)writer.Write("data: [DONE]\n\n");
}

}  // namespace tessera::serve
