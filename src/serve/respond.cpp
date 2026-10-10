#include "serve/respond.hpp"

#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#include "tessera/serve.hpp"
#include "tessera/types.hpp"

namespace tessera::serve {

namespace {

// Poll interval while queued for the device: short enough to start
// promptly, long enough to stay out of the way.
constexpr int kQueuePollMs = 10;

}  // namespace

void SendJson(ResponseWriter& writer, int status, const core::Json& body) {
  (void)writer.SendHeaders(status, "application/json");
  (void)writer.Write(body.Dump());
}

void SendError(ResponseWriter& writer, int status,
               const std::string& message) {
  core::Json obj = core::Json::Object();
  obj.Set("error", core::Json::String(message));
  SendJson(writer, status, obj);
}

core::Json UsageJson(std::size_t prompt, std::size_t completion) {
  core::Json usage = core::Json::Object();
  usage.Set("prompt_tokens",
            core::Json::Number(static_cast<double>(prompt)));
  usage.Set("completion_tokens",
            core::Json::Number(static_cast<double>(completion)));
  usage.Set("total_tokens",
            core::Json::Number(static_cast<double>(prompt + completion)));
  return usage;
}

std::size_t MaxCompletionTokensFrom(const core::Json& body,
                                     std::size_t fallback) {
  const core::Json* value = body.Find("max_completion_tokens");
  if (value != nullptr && value->type() == core::Json::Type::Number &&
      value->AsNumber() > 0) {
    return static_cast<std::size_t>(value->AsNumber());
  }
  return fallback;
}

std::size_t MaxThinkingTokensFrom(const core::Json& body) {
  const core::Json* value = body.Find("max_thinking_tokens");
  if (value != nullptr && value->type() == core::Json::Type::Number &&
      value->AsNumber() > 0) {
    return static_cast<std::size_t>(value->AsNumber());
  }
  return 0;
}

void WarnRetiredMaxTokens(log::Diagnostics& diagnostics,
                          const core::Json& body) {
  if (body.Find("max_tokens") != nullptr &&
      body.Find("max_completion_tokens") == nullptr) {
    diagnostics.Warn("serve", "request uses retired max_tokens, which is "
                              "ignored; use max_completion_tokens (0 fills "
                              "the remaining context)");
  }
}

std::string SessionIdFrom(const core::Json& body) {
  const core::Json* value = body.Find("session_id");
  if (value != nullptr && value->isString()) {
    return value->AsString();
  }
  return {};
}

std::optional<std::unique_lock<std::mutex>> WaitForGpu(
    std::mutex& generation, const ResponseWriter& writer) {
  for (;;) {
    std::unique_lock<std::mutex> slot(generation, std::try_to_lock);
    if (slot.owns_lock()) {
      return slot;
    }
    // A peer that disconnects while queued stops waiting instead of
    // holding its place behind a generation it never reads.
    if (writer.IsPeerGone()) {
      return std::nullopt;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kQueuePollMs));
  }
}

long long MillisBetween(std::chrono::steady_clock::time_point start,
                        std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

bool WantsStream(const core::Json& body) {
  const core::Json* value = body.Find("stream");
  return value != nullptr && value->type() == core::Json::Type::Bool &&
         value->AsBool();
}

void SendGenerationError(ResponseWriter& writer, StatusCode code) {
  if (code == StatusCode::InvalidArgument) {
    SendError(writer, 400,
              "prompt exceeds the model context window; shorten the "
              "prompt or raise --context");
    return;
  }
  SendError(writer, 500, "generation failed");
}

bool RejectOversizePrompt(ResponseWriter& writer, std::size_t prompt,
                           std::size_t context) {
  // A prompt past the context fails before any device work, whatever
  // the completion budget (an unlimited request must not decode empty).
  if (prompt > context) {
    SendGenerationError(writer, StatusCode::InvalidArgument);
    return true;
  }
  return false;
}

void WriteSse(ResponseWriter& writer, const std::string& event,
              const core::Json& data, bool with_event) {
  std::string chunk;
  if (with_event) {
    chunk += "event: " + event + "\n";
  }
  chunk += "data: " + data.Dump() + "\n\n";
  (void)writer.Write(chunk);
}

std::string InjectWebDefaults(std::string_view page) {
  constexpr std::string_view kMaxCompletionTokens =
      "@TESSERA_DEFAULT_MAX_COMPLETION_TOKENS@";
  const std::string value = std::to_string(kDefaultMaxCompletionTokens);
  std::string out(page);
  std::size_t pos = 0;
  while ((pos = out.find(kMaxCompletionTokens, pos)) != std::string::npos) {
    out.replace(pos, kMaxCompletionTokens.size(), value);
    pos += value.size();
  }
  return out;
}

}  // namespace tessera::serve
