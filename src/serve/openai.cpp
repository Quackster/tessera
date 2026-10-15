#include "serve/openai.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "serve/openai_shared.hpp"
#include "serve/render.hpp"
#include "serve/respond.hpp"
#include "serve/session_chat.hpp"
#include "serve/tools/tool_call.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/scheduler.hpp"
#include "tessera/tokenizer.hpp"

namespace tessera::serve {

namespace {

using core::Json;

// Exclusive tokens between device yields: frequent enough that a
// queued scheduler turn starts within a few tokens, rare enough
// that solo turns keep full speed (about 1 ms per 8 tokens).
constexpr int kYieldEveryTokens = 8;
constexpr int kYieldQuantumMs = 1;

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

}  // namespace

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

void OpenAiComplete(RunnerPool& pool, const Tokenizer& tokenizer,
                    SessionStore& sessions, ResponseWriter& writer,
                    const Json& body, std::size_t default_max,
                    std::size_t stats_interval_seconds) {
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
  if (pool.empty()) {
    return;
  }
  // Plain turns always share the scheduler so concurrent sessions
  // time-slice instead of queueing behind one exclusive generation;
  // thinking budgets stay queued exclusive (see RouteDevice). The
  // runners hold the same model, so render and metadata read it
  // without holding a device.
  const std::size_t max_thinking_tokens = MaxThinkingTokensFrom(body);
  RunnerLease lease;
  RunnerLease* slot = nullptr;
  if (!RouteDevice(pool, max_thinking_tokens, /*wants_tools=*/false,
                   [&writer] { return writer.IsPeerGone(); }, lease, &slot)) {
    return;
  }
  const bool shared = slot == nullptr;
  Model& model = shared ? pool.At(0).model() : lease.model();
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
  options.max_thinking_tokens = max_thinking_tokens;
  options.prompt_tokens = *ids;
  // The pool carries the launcher's KV storage type (--kv-q8 and
  // friends); without it every serve turn would decode F32 and an
  // unlimited turn would never fit the device.
  options.kv_type = shared ? pool.At(0).kv_type()
                           : lease.runner().kv_type();
  if (!shared) {
    ApplyToGenerateOptions(lease.runner().calibration(), options);
  }
  if (RejectOversizePrompt(writer, ids->size(), model.MaxContextLength())) {
    return;
  }
  // Shared turns enqueue on the least-loaded device and stream back
  // on this thread; exclusive (thinking, tools) turns hold the lease's
  // device while yielding every few tokens (see MaybeYieldExclusive).
  const auto drive = [&](const std::function<bool(std::uint32_t)>& next) {
    if (shared) {
      return pool.SubmitLeastLoaded(
          options, next, [&writer] { return writer.IsPeerGone(); });
    }
    return lease.engine().GenerateStreaming(lease.model(), options, next);
  };
  TurnStats stats;
  stats.prompt_tokens = ids->size();
  // Console tok/s for this turn, every stats interval (zero off).
  // The completions stream shape stays pure: no speed deltas on wire.
  const std::string speed_label =
      session != nullptr ? "session " + session->Id() : "completions";
  log::Diagnostics& speed_log =
      shared ? pool.At(0).engine().Diagnostics() : lease.engine().Diagnostics();
  TurnSpeedometer speedometer(std::chrono::seconds(stats_interval_seconds),
                              speed_label, speed_log);
  speedometer.Begin();
  // Exclusive (thinking, tools) turns yield the device every few
  // tokens so a queued scheduler turn streams beside them instead of
  // behind them.
  int yield_count = 0;
  if (!stream) {
    // GenerateStreaming with a liveness hook instead of Generate:
    // identical tokens, but a closed window aborts the turn instead
    // of decoding into the void.
    std::vector<std::uint32_t> produced;
    const auto gen_started = std::chrono::steady_clock::now();
    auto streamed = drive([&](std::uint32_t token) {
      if (writer.IsPeerGone()) {
        return false;
      }
      produced.push_back(token);
      if (!shared) {
        MaybeYieldExclusive(lease, yield_count);
      }
      (void)speedometer.NoteToken();
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
  auto streamed = drive([&](std::uint32_t token) {
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
    if (!shared) {
      MaybeYieldExclusive(lease, yield_count);
    }
    (void)speedometer.NoteToken();
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

bool RouteDevice(RunnerPool& pool, std::size_t max_thinking_tokens,
                 bool wants_tools,
                 const std::function<bool()>& cancelled, RunnerLease& lease,
                 RunnerLease** slot) {
  *slot = nullptr;
  if (pool.empty()) {
    return false;
  }
  // Thinking budgets and tool calls stay queued exclusive: the
  // scheduler drives neither, so a contended turn waits its turn.
  // Every other turn shares the scheduler (as plain greedy, which
  // equals the speculative output): a solo exclusive fast path would
  // hold the whole device while a second session starves behind its
  // yields, so the serve layer never takes it. A layer-split runner has
  // no scheduler and its Submit serializes, so tool turns share it too.
  GenerateOptions probe;
  probe.max_thinking_tokens = max_thinking_tokens;
  const bool split = pool.At(0).is_split();
  if ((wants_tools && !split) || !pool.At(0).CanSchedule(probe)) {
    auto held = pool.Acquire(cancelled);
    if (!held) {
      return false;
    }
    lease = std::move(*held);
    *slot = &lease;
    return true;
  }
  return true;
}

void MaybeYieldExclusive(RunnerLease& lease, int& since_yield) {
  if (++since_yield < kYieldEveryTokens) {
    return;
  }
  since_yield = 0;
  lease.YieldFor(std::chrono::milliseconds(kYieldQuantumMs));
}

TurnSpeedometer::TurnSpeedometer(std::chrono::seconds interval,
                                 std::string label, log::Diagnostics& log,
                                 Clock now)
    : interval_(interval), label_(std::move(label)), log_(&log) {
  if (now) {
    now_ = std::move(now);
  } else {
    now_ = [] { return std::chrono::steady_clock::now(); };
  }
}

void TurnSpeedometer::Begin() {
  begun_ = true;
  window_started_ = now_();
  window_tokens_ = 0;
  total_tokens_ = 0;
}

std::optional<SpeedSample> TurnSpeedometer::NoteToken() {
  if (!begun_ || interval_.count() <= 0) {
    return std::nullopt;
  }
  ++window_tokens_;
  ++total_tokens_;
  const auto now = now_();
  if (now - window_started_ < interval_) {
    return std::nullopt;
  }
  const double seconds =
      std::chrono::duration<double>(now - window_started_).count();
  const double tps = seconds > 0 ? window_tokens_ / seconds : 0.0;
  char rate[32];
  std::snprintf(rate, sizeof(rate), "%.1f", tps);
  log_->Info("serve", label_ + ": " + std::to_string(total_tokens_) +
                         " completion token(s) at " + rate + " tok/s");
  const SpeedSample sample{tps, total_tokens_};
  window_started_ = now;
  window_tokens_ = 0;
  return sample;
}


}  // namespace tessera::serve
