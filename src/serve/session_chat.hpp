#pragma once

#include <cstddef>
#include <mutex>
#include <string>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "serve/session.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/tokenizer.hpp"

// Web UI session turns over the chat model. One instance is shared by
// all connection threads; generation is serialized through
// `generation` (a second turn while one runs answers 503). Sessions
// carry no tools: turns are plain user/assistant text with the model
// thinking split into `reasoning_content`.

namespace tessera::serve {

class SessionHandler {
 public:
  SessionHandler(Engine& engine, Model& model, const Tokenizer& tokenizer,
                 SessionStore& sessions, std::mutex& generation,
                 std::size_t default_max_tokens);

  // Session CRUD over JSON bodies; 404 for an unknown id.
  void HandleList(ResponseWriter& writer) const;
  void HandleCreate(ResponseWriter& writer, const core::Json& body) const;
  void HandleGet(ResponseWriter& writer, std::string_view id) const;
  void HandleDelete(ResponseWriter& writer, std::string_view id) const;
  void HandleRename(ResponseWriter& writer, std::string_view id,
                     const core::Json& body) const;
  // Append a user message and generate the turn (SSE when
  // body["stream"] is true, default). 409 when the session is busy,
  // 503 when another generation runs.
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
  // Releases the session turn on every exit path.
  struct TurnGuard {
    std::shared_ptr<Session> session;
    ~TurnGuard() {
      if (session != nullptr) {
        session->End();
      }
    }
  };
  // Shared turn core for chat and retry (see the .cpp for the flow).
  void RunTurn(const std::shared_ptr<Session>& session,
               ResponseWriter& writer, std::size_t max_tokens,
               bool enable_thinking, bool stream) const;
  // 409 when the session is busy, 503 when another generation runs.
  bool BeginTurn(const std::shared_ptr<Session>& session,
                ResponseWriter& writer, std::unique_lock<std::mutex>& gpu,
                TurnGuard& guard) const;

  Engine& engine_;
  Model& model_;
  const Tokenizer& tokenizer_;
  SessionStore& sessions_;
  std::mutex& generation_;
  std::size_t default_max_tokens_;
};

}  // namespace tessera::serve
