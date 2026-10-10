#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "serve/session.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/tokenizer.hpp"

// Web UI session turns over the chat model. One instance is shared by
// all connection threads; generations queue in arrival order on the
// single device (a second turn waits instead of failing). Sessions
// carry no tools: turns are plain user/assistant text with the model
// thinking split into `reasoning_content`.

namespace tessera::serve {

// History as an OpenAI-form messages array (tool turns round-trip),
// the single renderer input for session prompts.
[[nodiscard]] core::Json SessionHistoryJson(const SessionView& view);

// Parse an OpenAI-form messages array into stored history. Content
// must be a string (null reads as empty); anything else is a 422.
// Never fails on tool shapes: they store verbatim and NormalizeToolHistory
// validates them at render time.
[[nodiscard]] bool SessionMessagesFromJson(
    const core::Json& messages, std::vector<SessionMessage>* out,
    std::string* error);

// Read-only session JSON for GET responses (message tool turns included).
[[nodiscard]] core::Json SessionJson(const SessionView& view);
// Small list entry JSON for the sidebar and GET /slots.
[[nodiscard]] core::Json SessionInfoJson(const SessionInfo& info);

class SessionHandler {
 public:
  SessionHandler(Engine& engine, Model& model, const Tokenizer& tokenizer,
                 SessionStore& sessions, std::mutex& generation,
                 std::size_t default_max_completion_tokens, bool auto_title);

  // Session CRUD over JSON bodies; 404 for an unknown id.
  void HandleList(ResponseWriter& writer) const;
  void HandleCreate(ResponseWriter& writer, const core::Json& body) const;
  void HandleGet(ResponseWriter& writer, std::string_view id) const;
  void HandleDelete(ResponseWriter& writer, std::string_view id) const;
  void HandleRename(ResponseWriter& writer, std::string_view id,
                     const core::Json& body) const;
  // Append a user message and generate the turn (SSE when
  // body["stream"] is true, default). 409 when the session is busy;
  // another session's generation queues on the device.
  void HandleChat(ResponseWriter& writer, std::string_view id,
                   const core::Json& body) const;
  // Regenerate after dropping trailing assistant messages.
  void HandleRetry(ResponseWriter& writer, std::string_view id,
                   const core::Json& body) const;
  // Turn controls; 404 for an unknown id, otherwise the live state.
  void HandleStop(ResponseWriter& writer, std::string_view id) const;
  void HandlePause(ResponseWriter& writer, std::string_view id) const;
  void HandleResume(ResponseWriter& writer, std::string_view id) const;

  private:
  // Shared turn core for chat and retry (see the .cpp for the flow).
  void RunTurn(const std::shared_ptr<Session>& session,
               ResponseWriter& writer, const core::Json& request,
               std::size_t max_completion_tokens,
               std::size_t max_thinking_tokens, bool enable_thinking,
               bool stream, std::string_view title) const;
  // 409 when the session is busy; otherwise take the turn and queue
  // for the device behind running generations. The guard releases it.
  bool BeginTurn(const std::shared_ptr<Session>& session,
                 ResponseWriter& writer, std::unique_lock<std::mutex>& gpu,
                 SessionTurn& guard) const;

  Engine& engine_;
  Model& model_;
  const Tokenizer& tokenizer_;
  SessionStore& sessions_;
  std::mutex& generation_;
  std::size_t default_max_completion_tokens_;
  bool auto_title_;
};

}  // namespace tessera::serve
