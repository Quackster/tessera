#pragma once

#include <condition_variable>
#include <cstddef>
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

// One stored chat turn message.
struct SessionMessage {
  std::string role;      // "user" or "assistant"
  std::string content;
  std::string reasoning;  // assistant thinking (may be empty)
  bool stopped = false;   // generation was stopped mid-turn
};

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

// Default title from the first user text: trimmed first line,
// at most kTitleChars characters.
[[nodiscard]] std::string TitleFromText(std::string_view text);

}  // namespace tessera::serve
