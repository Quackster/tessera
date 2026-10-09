#include "serve/session_chat.hpp"

#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "core/json.hpp"
#include "serve/render.hpp"
#include "serve/respond.hpp"
#include "serve/tools/tool_call.hpp"

namespace tessera::serve {

namespace {

using core::Json;

// EndsWith for the think-mode prompt check.
bool EndsWith(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() &&
         text.substr(text.size() - suffix.size()) == suffix;
}

Json MessageJson(const SessionMessage& message) {
  Json item = Json::Object();
  item.Set("role", Json::String(message.role));
  item.Set("content", Json::String(message.content));
  item.Set("reasoning_content", Json::String(message.reasoning));
  item.Set("stopped", Json::Bool(message.stopped));
  return item;
}

Json SessionJson(const SessionView& view) {
  Json items = Json::Array();
  for (const SessionMessage& message : view.messages) {
    items.Push(MessageJson(message));
  }
  Json response = Json::Object();
  response.Set("id", Json::String(view.id));
  response.Set("title", Json::String(view.title));
  response.Set("busy", Json::Bool(view.busy));
  response.Set("paused", Json::Bool(view.paused));
  response.Set("messages", std::move(items));
  return response;
}

Json SessionInfoJson(const SessionInfo& info) {
  Json item = Json::Object();
  item.Set("id", Json::String(info.id));
  item.Set("title", Json::String(info.title));
  item.Set("message_count",
           Json::Number(static_cast<double>(info.message_count)));
  item.Set("busy", Json::Bool(info.busy));
  return item;
}

Json SseDelta(Json delta, const std::string& finish) {
  Json choice = Json::Object();
  choice.Set("index", Json::Number(0));
  choice.Set("delta", std::move(delta));
  if (finish.empty()) {
    choice.Set("finish_reason", Json());
  } else {
    choice.Set("finish_reason", Json::String(finish));
  }
  Json choices = Json::Array();
  choices.Push(std::move(choice));
  Json chunk = Json::Object();
  chunk.Set("id", Json::String("chatcmpl-0"));
  chunk.Set("object", Json::String("chat.completion.chunk"));
  chunk.Set("choices", std::move(choices));
  return chunk;
}

// The UI streams by default; an explicit "stream": false buffers.
bool WantsSessionStream(const Json& body) {
  const Json* value = body.Find("stream");
  if (value != nullptr && value->type() == Json::Type::Bool) {
    return value->AsBool();
  }
  return true;
}

bool BodyFlag(const Json& body, std::string_view key, bool fallback) {
  const Json* value = body.Find(std::string(key));
  if (value != nullptr && value->type() == Json::Type::Bool) {
    return value->AsBool();
  }
  return fallback;
}

// Releases the session turn on every exit path.
struct TurnGuard {
  std::shared_ptr<Session> session;
  ~TurnGuard() {
    if (session != nullptr) {
      session->End();
    }
  }
};

}  // namespace

SessionHandler::SessionHandler(Engine& engine, Model& model,
                               const Tokenizer& tokenizer,
                               SessionStore& sessions, std::mutex& generation,
                               std::size_t default_max_tokens)
    : engine_(engine),
      model_(model),
      tokenizer_(tokenizer),
      sessions_(sessions),
      generation_(generation),
      default_max_tokens_(default_max_tokens) {}

void SessionHandler::HandleList(ResponseWriter& writer) const {
  Json items = Json::Array();
  for (const SessionInfo& info : sessions_.List()) {
    items.Push(SessionInfoJson(info));
  }
  Json response = Json::Object();
  response.Set("sessions", std::move(items));
  SendJson(writer, 200, response);
}

void SessionHandler::HandleCreate(ResponseWriter& writer,
                                  const Json& body) const {
  std::string title;
  if (const Json* value = body.Find("title");
      value != nullptr && value->isString()) {
    title = value->AsString();
  }
  auto session = sessions_.Create(title);
  SendJson(writer, 201, SessionJson(session->View()));
}

void SessionHandler::HandleGet(ResponseWriter& writer,
                               std::string_view id) const {
  auto view = sessions_.View(std::string(id));
  if (!view) {
    SendError(writer, 404, "unknown session");
    return;
  }
  SendJson(writer, 200, SessionJson(*view));
}

void SessionHandler::HandleDelete(ResponseWriter& writer,
                                  std::string_view id) const {
  if (!sessions_.Remove(std::string(id))) {
    SendError(writer, 404, "unknown session");
    return;
  }
  SendJson(writer, 200, Json::Object());
}

void SessionHandler::HandleRename(ResponseWriter& writer, std::string_view id,
                                  const Json& body) const {
  auto session = sessions_.Get(std::string(id));
  if (session == nullptr) {
    SendError(writer, 404, "unknown session");
    return;
  }
  const Json* value = body.Find("title");
  if (value == nullptr || !value->isString() || value->AsString().empty()) {
    SendError(writer, 422, "title must be a non-empty string");
    return;
  }
  session->SetTitle(value->AsString());
  auto view = sessions_.View(std::string(id));
  SendJson(writer, 200, SessionJson(*view));
}

// Shared turn core for chat and retry: render the history, generate
// with live think/content deltas, persist the assistant message. The
// stop flag aborts the decode loop (prefill has no token hook and runs
// to completion); pause blocks it between tokens.
void SessionHandler::RunTurn(const std::shared_ptr<Session>& session,
                            ResponseWriter& writer, std::size_t max_tokens,
                            bool enable_thinking, bool stream) const {
  auto view = sessions_.View(session->Id());
  Json messages = Json::Array();
  for (const SessionMessage& message : view->messages) {
    Json item = Json::Object();
    item.Set("role", Json::String(message.role));
    item.Set("content", Json::String(message.content));
    messages.Push(std::move(item));
  }
  Json prompt_body = Json::Object();
  prompt_body.Set("messages", std::move(messages));
  prompt_body.Set("enable_thinking", Json::Bool(enable_thinking));
  std::string error;
  const std::string prompt = RenderPrompt(model_, prompt_body, &error);
  if (!error.empty()) {
    SendError(writer, 422, error);
    return;
  }
  auto ids = tokenizer_.Encode(prompt);
  if (!ids) {
    SendError(writer, 500, "tokenization failed");
    return;
  }
  if (RejectOversizePrompt(writer, ids->size(), max_tokens,
                           model_.MaxContextLength())) {
    return;
  }
  GenerateOptions options;
  options.max_tokens = max_tokens;
  options.prompt_tokens = *ids;
  ThinkStreamer streamer(EndsWith(prompt, "<think>"));
  std::string reasoning_text;
  std::string content_text;
  bool saw_stop = false;
  // One decode step: stop aborts, pause blocks, deltas stream live
  // (buffered mode only accumulates into the totals below).
  auto on_token = [&](std::uint32_t token, bool live) {
    // A closed window or cancelled request aborts first: pausing
    // below would block the device slot on a client that never
    // resumes. Aborts persist as stopped turns.
    if (writer.IsPeerGone()) {
      saw_stop = true;
      return false;
    }
    if (!session->PollControl()) {
      saw_stop = true;
      return false;
    }
    auto piece =
        tokenizer_.Decode(std::span<const std::uint32_t>(&token, 1));
    const ThinkStreamer::Deltas part =
        streamer.Push(piece ? *piece : std::string());
    reasoning_text += part.reasoning;
    content_text += part.content;
    if (live) {
      if (!part.reasoning.empty()) {
        Json delta = Json::Object();
        delta.Set("reasoning_content", Json::String(part.reasoning));
        WriteSse(writer, "", SseDelta(std::move(delta), ""), false);
      }
      if (!part.content.empty()) {
        Json delta = Json::Object();
        delta.Set("content", Json::String(part.content));
        WriteSse(writer, "", SseDelta(std::move(delta), ""), false);
      }
    }
    return true;
  };
  if (stream) {
    (void)writer.SendHeaders(200, "text/event-stream", true);
    Json role = Json::Object();
    role.Set("role", Json::String("assistant"));
    WriteSse(writer, "", SseDelta(std::move(role), ""), false);
    auto streamed = engine_.GenerateStreaming(
        model_, options,
        [&](std::uint32_t token) { return on_token(token, true); });
    if (!streamed) {
      Json chunk = Json::Object();
      chunk.Set("error", Json::String("generation failed"));
      WriteSse(writer, "", chunk, false);
      return;
    }
    const ThinkStreamer::Deltas tail = streamer.Finish();
    reasoning_text += tail.reasoning;
    content_text += tail.content;
    if (!tail.reasoning.empty()) {
      Json delta = Json::Object();
      delta.Set("reasoning_content", Json::String(tail.reasoning));
      WriteSse(writer, "", SseDelta(std::move(delta), ""), false);
    }
    if (!tail.content.empty()) {
      Json delta = Json::Object();
      delta.Set("content", Json::String(tail.content));
      WriteSse(writer, "", SseDelta(std::move(delta), ""), false);
    }
    Json end = Json::Object();
    WriteSse(writer, "",
             SseDelta(std::move(end), saw_stop ? "stop" : "length"), false);
    (void)writer.Write("data: [DONE]\n\n");
  } else {
    auto streamed = engine_.GenerateStreaming(
        model_, options,
        [&](std::uint32_t token) { return on_token(token, false); });
    if (!streamed) {
      SendGenerationError(writer, streamed.error());
      return;
    }
    const std::size_t produced = *streamed;
    const ThinkStreamer::Deltas tail = streamer.Finish();
    reasoning_text += tail.reasoning;
    content_text += tail.content;
    session->Append(SessionMessage{"assistant", content_text, reasoning_text,
                                   saw_stop});
    Json message = MessageJson(SessionMessage{"assistant", content_text,
                                              reasoning_text, saw_stop});
    Json response = Json::Object();
    response.Set("message", std::move(message));
    response.Set("usage", UsageJson(ids->size(), produced));
    SendJson(writer, 200, response);
    return;
  }
  session->Append(SessionMessage{"assistant", content_text, reasoning_text,
                                 saw_stop});
}

// Acquire the session turn and the GPU slot: 409 when the session is
// busy, 503 when another generation runs. The guard releases the turn.
bool SessionHandler::BeginTurn(const std::shared_ptr<Session>& session,
                               ResponseWriter& writer,
                               std::unique_lock<std::mutex>& gpu,
                               TurnGuard& guard) const {
  if (!session->TryBegin()) {
    SendError(writer, 409, "session is already generating");
    return false;
  }
  gpu = std::unique_lock<std::mutex>(generation_, std::try_to_lock);
  if (!gpu.owns_lock()) {
    session->End();
    SendError(writer, 503, "another generation is already running");
    return false;
  }
  guard.session = session;
  return true;
}

void SessionHandler::HandleChat(ResponseWriter& writer, std::string_view id,
                                const Json& body) const {
  auto session = sessions_.Get(std::string(id));
  if (session == nullptr) {
    SendError(writer, 404, "unknown session");
    return;
  }
  const Json* message = body.Find("message");
  if (message == nullptr || !message->isString() ||
      message->AsString().empty()) {
    SendError(writer, 422, "message must be a non-empty string");
    return;
  }
  const bool first = session->View().messages.empty();
  session->Append(
      SessionMessage{"user", message->AsString(), /*reasoning=*/{}, false});
  if (first) {
    auto view = session->View();
    if (view.title == kDefaultTitle) {
      session->SetTitle(TitleFromText(message->AsString()));
    }
  }
  TurnGuard guard;
  std::unique_lock<std::mutex> gpu;
  if (!BeginTurn(session, writer, gpu, guard)) {
    return;
  }
  RunTurn(session, writer, MaxTokensFrom(body, default_max_tokens_),
          BodyFlag(body, "enable_thinking", true),
          WantsSessionStream(body));
}

void SessionHandler::HandleRetry(ResponseWriter& writer, std::string_view id,
                                 const Json& body) const {
  auto session = sessions_.Get(std::string(id));
  if (session == nullptr) {
    SendError(writer, 404, "unknown session");
    return;
  }
  session->PopTrailingAssistant();
  if (session->View().messages.empty()) {
    SendError(writer, 422, "nothing to retry");
    return;
  }
  TurnGuard guard;
  std::unique_lock<std::mutex> gpu;
  if (!BeginTurn(session, writer, gpu, guard)) {
    return;
  }
  RunTurn(session, writer, MaxTokensFrom(body, default_max_tokens_),
          BodyFlag(body, "enable_thinking", true),
          WantsSessionStream(body));
}

void SessionHandler::HandleStop(ResponseWriter& writer,
                                std::string_view id) const {
  auto session = sessions_.Get(std::string(id));
  if (session == nullptr) {
    SendError(writer, 404, "unknown session");
    return;
  }
  session->RequestStop();
  auto view = sessions_.View(std::string(id));
  SendJson(writer, 200, SessionJson(*view));
}

void SessionHandler::HandlePause(ResponseWriter& writer,
                                 std::string_view id) const {
  auto session = sessions_.Get(std::string(id));
  if (session == nullptr) {
    SendError(writer, 404, "unknown session");
    return;
  }
  session->SetPaused(true);
  auto view = sessions_.View(std::string(id));
  SendJson(writer, 200, SessionJson(*view));
}

void SessionHandler::HandleResume(ResponseWriter& writer,
                                  std::string_view id) const {
  auto session = sessions_.Get(std::string(id));
  if (session == nullptr) {
    SendError(writer, 404, "unknown session");
    return;
  }
  session->SetPaused(false);
  auto view = sessions_.View(std::string(id));
  SendJson(writer, 200, SessionJson(*view));
}

}  // namespace tessera::serve
