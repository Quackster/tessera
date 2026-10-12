#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "serve/session.hpp"
#include "tessera/log.hpp"
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

// Take the device for one turn (session turn already held, or no
// session for stateless requests). Queued exclusive for thinking and
// tools, shared (null lease) otherwise: plain turns always share the
// scheduler so a second session time-slices beside the first instead
// of starving behind an exclusive generation. False means the peer
// went away: return.
bool RouteDevice(RunnerPool& pool, std::size_t max_thinking_tokens,
                 bool wants_tools,
                 const std::function<bool()>& cancelled, RunnerLease& lease,
                 RunnerLease** slot);

// Yield the device every few exclusive (thinking, tools) tokens so
// scheduler turns progress beside a long exclusive generation. Call from the
// on_token hook with a per-turn counter starting at zero; shared
// turns skip it (null lease).
void MaybeYieldExclusive(RunnerLease& lease, int& since_yield);

// One live speed report: the per-window decode rate plus the total
// produced so far, for the session SSE stream.
struct SpeedSample {
  double tokens_per_second = 0.0;
  std::size_t completion_tokens = 0;
};

// Per-turn streaming speedometer. Call Begin at generation start and
// NoteToken per emitted token; it logs one console line and returns
// a sample at most every interval (sliding window, so the rate stays
// live instead of converging). Zero interval disables both. The
// clock is injectable so tests stay off wall-clock time.
class TurnSpeedometer {
 public:
  using Clock = std::function<std::chrono::steady_clock::time_point()>;
  TurnSpeedometer(std::chrono::seconds interval, std::string label,
                  log::Diagnostics& log, Clock now = {});
  void Begin();
  // Count one token; empty until the interval elapses. Callers that
  // want console lines only ignore the result.
  std::optional<SpeedSample> NoteToken();

 private:
  std::chrono::milliseconds interval_;
  std::string label_;
  log::Diagnostics* log_;
  Clock now_;
  std::chrono::steady_clock::time_point window_started_{};
  std::size_t window_tokens_ = 0;
  std::size_t total_tokens_ = 0;
  bool begun_ = false;
};

}  // namespace tessera::serve
