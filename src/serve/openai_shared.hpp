#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "serve/session.hpp"

// Shared session plumbing for the OpenAI endpoints, implemented once
// in openai.cpp and used by both the completions and the chat
// translations (see openai_chat.cpp).

namespace tessera::serve {

// Assistant turn stored from generated `text`: thinking and tool markup
// stay out of the history, so a resent OpenAI history still matches.
[[nodiscard]] SessionMessage AssistantTurn(std::string_view text, bool stopped,
                                           TurnStats stats);

// Reconcile an incoming OpenAI history with the session: 422 on a
// malformed history, otherwise the merged view for rendering.
[[nodiscard]] bool BeginChatTurn(const std::shared_ptr<Session>& session,
                                 ResponseWriter& writer, const core::Json& body,
                                 SessionView* view);

// Prompt body for a session turn: the full stored history plus the
// request-level fields (system prompt, tools, thinking flag, budgets
// and streaming flag).
[[nodiscard]] core::Json SessionPromptBody(const core::Json& body,
                                           const SessionView& view);

// Take the session turn (404 unknown, 409 already generating). The
// guard ends it on every exit path once set.
bool BeginSession(SessionStore& sessions, const std::string& session_id,
                  ResponseWriter& writer, std::shared_ptr<Session>* session,
                  SessionTurn* turn);

}  // namespace tessera::serve
