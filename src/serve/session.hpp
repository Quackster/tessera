#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// In-memory chat sessions for the web UI. One instance is shared by
// all connection threads; every method is thread-safe. A session
// carries its message history plus the live generation controls
// (busy/paused/stopped), so the UI can retry, pause and stop turns.

namespace tessera::serve {

// Title of a session with no messages yet; the first user message
// names it unless the caller set a title.
inline constexpr char kDefaultTitle[] = "New chat";

// Wall-clock stats for one generated assistant turn, measured around
// GenerateStreaming (queue wait excluded). All zero when the turn did
// not record stats.
struct TurnStats {
  std::size_t prompt_tokens = 0;
  std::size_t completion_tokens = 0;
  long long elapsed_ms = 0;
  // Completion tokens per second over the generation window; 0 when
  // the turn recorded no time or no tokens.
  [[nodiscard]] double TokensPerSecond() const;
};

// One stored chat turn message. Tool turns keep their OpenAI form:
// an assistant message carries the `tool_calls` array dump, a tool
// message its `tool_call_id`, so the history re-renders exactly.
struct SessionMessage {
  std::string role;      // "user", "assistant", "system" or "tool"
  std::string content;
  std::string reasoning;  // assistant thinking (may be empty)
  std::string tool_calls_json;  // assistant tool_calls array dump, else empty
  std::string tool_call_id;     // tool message id, else empty
  bool stopped = false;   // generation was stopped mid-turn
  TurnStats stats;        // generation speed, assistant turns only
  std::uint64_t created_ms = 0;  // unix millis, stamped on store
};

// True when two messages render the same prompt row: the response-side
// metadata (reasoning, stopped, stats, created_ms) is ignored, so a
// resent OpenAI history still matches the stored one.
[[nodiscard]] bool SamePromptRow(const SessionMessage& a,
                                 const SessionMessage& b);

// Read-only session copy for JSON responses.
struct SessionView {
  std::string id;
  std::string title;
  std::vector<SessionMessage> messages;
  bool busy = false;
  bool paused = false;
};

// Small session list entry for the sidebar.
struct SessionInfo {
  std::string id;
  std::string title;
  std::size_t message_count = 0;
  bool busy = false;
};

// One chat session. All state is guarded by control_; use the methods.
class Session {
 public:
  explicit Session(std::string id, std::string title);

  [[nodiscard]] std::string Id() const;
  // Snapshot of the history and flags for JSON responses.
  [[nodiscard]] SessionView View() const;
  [[nodiscard]] SessionInfo Info() const;
  // Append a history message (user before, assistant after a turn).
  void Append(SessionMessage message);
  // Reconcile with an incoming OpenAI history. A common prefix keeps
  // the stored turns and appends the incoming remainder; a divergence
  // after that prefix drops the stored tail first (branch switch). No
  // common prefix means the client sent only new turns, so they append
  // and nothing is lost. A full-history resend and a delta-only client
  // both converge without duplication.
  void SyncHistory(std::vector<SessionMessage> incoming);
  // Drop trailing assistant messages (retry); returns the count.
  std::size_t PopTrailingAssistant();
  void SetTitle(const std::string& title);
  // Start a turn: false when already busy. Resets stopped/paused.
  bool TryBegin();
  void End();
  // Returns false when stopped (abort the turn). Blocks while paused.
  bool PollControl();
  void RequestStop();
  void SetPaused(bool paused);

 private:
  const std::string id_;
  std::string title_;
  std::vector<SessionMessage> messages_;
  mutable std::mutex control_;
  std::condition_variable resume_cv_;
  bool paused_ = false;
  bool stopped_ = false;
  bool busy_ = false;
};

// Session map, insertion ordered. Thread-safe.
class SessionStore {
 public:
  // Create a session; empty title keeps the default until the first
  // user message names it.
  [[nodiscard]] std::shared_ptr<Session> Create(const std::string& title);
  [[nodiscard]] std::vector<SessionInfo> List() const;
  [[nodiscard]] std::shared_ptr<Session> Get(const std::string& id) const;
  [[nodiscard]] std::optional<SessionView> View(const std::string& id) const;
  bool Remove(const std::string& id);

 private:
  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<Session>> sessions_;
  std::size_t next_ = 1;
};

// Releases a session turn; empty when no turn was taken. The single
// canonical guard (the web handler and the OpenAI endpoints share it);
// the destructor ends the turn on every exit path.
struct SessionTurn {
  std::shared_ptr<Session> session;
  ~SessionTurn();
};

// Default title from the first user text: trimmed first line,
// at most kTitleChars characters.
[[nodiscard]] std::string TitleFromText(std::string_view text);

}  // namespace tessera::serve
