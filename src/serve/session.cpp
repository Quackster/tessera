#include "serve/session.hpp"

#include <string>
#include <string_view>
#include <utility>

namespace tessera::serve {

namespace {

// Title length: a sidebar row, not a sentence.
constexpr std::size_t kTitleChars = 48;

std::string Trim(std::string_view text) {
  const std::size_t begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string_view::npos) {
    return {};
  }
  return std::string(
      text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1));
}

}  // namespace

Session::Session(std::string id, std::string title)
    : id_(std::move(id)), title_(std::move(title)) {}

std::string Session::Id() const {
  return id_;
}

SessionView Session::View() const {
  std::lock_guard<std::mutex> lock(control_);
  return SessionView{id_, title_, messages_, busy_, paused_};
}

SessionInfo Session::Info() const {
  std::lock_guard<std::mutex> lock(control_);
  return SessionInfo{id_, title_, messages_.size(), busy_};
}

void Session::Append(SessionMessage message) {
  std::lock_guard<std::mutex> lock(control_);
  messages_.push_back(std::move(message));
}

std::size_t Session::PopTrailingAssistant() {
  std::lock_guard<std::mutex> lock(control_);
  std::size_t removed = 0;
  while (!messages_.empty() && messages_.back().role == "assistant") {
    messages_.pop_back();
    ++removed;
  }
  return removed;
}

void Session::SetTitle(const std::string& title) {
  std::lock_guard<std::mutex> lock(control_);
  title_ = title;
}

bool Session::TryBegin() {
  std::lock_guard<std::mutex> lock(control_);
  if (busy_) {
    return false;
  }
  busy_ = true;
  paused_ = false;
  stopped_ = false;
  return true;
}

void Session::End() {
  std::lock_guard<std::mutex> lock(control_);
  busy_ = false;
  paused_ = false;
  stopped_ = false;
  resume_cv_.notify_all();
}

bool Session::PollControl() {
  std::unique_lock<std::mutex> lock(control_);
  resume_cv_.wait(lock, [this] { return !paused_ || stopped_; });
  return !stopped_;
}

void Session::RequestStop() {
  std::lock_guard<std::mutex> lock(control_);
  stopped_ = true;
  paused_ = false;
  resume_cv_.notify_all();
}

void Session::SetPaused(bool paused) {
  std::lock_guard<std::mutex> lock(control_);
  if (!busy_) {
    return;
  }
  paused_ = paused;
  resume_cv_.notify_all();
}

std::shared_ptr<Session> SessionStore::Create(const std::string& title) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto session = std::make_shared<Session>("s" + std::to_string(next_++),
                                           title.empty() ? kDefaultTitle
                                                         : title);
  sessions_.push_back(session);
  return session;
}

std::vector<SessionInfo> SessionStore::List() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<SessionInfo> infos;
  infos.reserve(sessions_.size());
  for (const auto& session : sessions_) {
    infos.push_back(session->Info());
  }
  return infos;
}

std::shared_ptr<Session> SessionStore::Get(const std::string& id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& session : sessions_) {
    if (session->Id() == id) {
      return session;
    }
  }
  return nullptr;
}

std::optional<SessionView> SessionStore::View(const std::string& id) const {
  auto session = Get(id);
  if (session == nullptr) {
    return std::nullopt;
  }
  return session->View();
}

bool SessionStore::Remove(const std::string& id) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = sessions_.begin(); it != sessions_.end(); ++it) {
    if ((*it)->Id() == id) {
      sessions_.erase(it);
      return true;
    }
  }
  return false;
}

std::string TitleFromText(std::string_view text) {
  std::string first(Trim(text.substr(0, text.find('\n'))));
  if (first.size() > kTitleChars) {
    first.resize(kTitleChars);
  }
  return first.empty() ? "New chat" : first;
}

}  // namespace tessera::serve
