#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "serve/session.hpp"
#include "tessera/runner.hpp"

// Shared session plumbing for the OpenAI endpoints, implemented once
// in openai.cpp and used by both the completions and the chat
// translations (see openai_chat.cpp) and the web session handler
// (see session_chat.cpp).

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

// True when no runner holds a lease or an admitted scheduler turn.
// The serve layer takes the exclusive speculative fast path only
// then; otherwise the turn rides the shared scheduler so a long
// generation never starves other sessions.
[[nodiscard]] bool PoolIdle(const RunnerPool& pool);

// Take the device for one turn (session turn already held, or no
// session for stateless requests). Queued exclusive for thinking and
// tools, fast exclusive for an idle strategy-backed device, shared
// (null lease) otherwise. False means the peer went away: return.
bool RouteDevice(RunnerPool& pool, std::size_t max_thinking_tokens,
                 bool wants_tools,
                 const std::function<bool()>& cancelled, RunnerLease& lease,
                 RunnerLease** slot);

// Yield the device every few exclusive tokens so scheduler turns
// progress beside a long exclusive generation. Call from the
// on_token hook with a per-turn counter starting at zero; shared
// turns skip it (null lease).
void MaybeYieldExclusive(RunnerLease& lease, int& since_yield);

}  // namespace tessera::serve
