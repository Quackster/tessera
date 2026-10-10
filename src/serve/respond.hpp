#pragma once

#include <chrono>
#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "tessera/engine.hpp"
#include "tessera/log.hpp"
#include "tessera/types.hpp"

// Shared response helpers for the HTTP endpoints. One canonical
// implementation lives here and both server.cpp and tools/chat.cpp use
// it; do not reimplement these in the endpoints.
namespace tessera::serve {

// Send `body` as a buffered application/json response.
void SendJson(ResponseWriter& writer, int status, const core::Json& body);
// Send `{"error": message}` with `status`.
void SendError(ResponseWriter& writer, int status, const std::string& message);
// OpenAI usage block for `prompt` prompt tokens and `completion` produced.
[[nodiscard]] core::Json UsageJson(std::size_t prompt,
                                   std::size_t completion);
// Positive `max_completion_tokens` from the body, else the positive
// `max_tokens` alias, else `fallback`. The current name wins.
[[nodiscard]] std::size_t MaxCompletionTokensFrom(const core::Json& body,
                                                 std::size_t fallback);
// Positive `max_thinking_tokens` from the body, else 0 (unlimited).
[[nodiscard]] std::size_t MaxThinkingTokensFrom(const core::Json& body);
// Session id from the body (`session_id` string); empty when the
// request is stateless. OpenAI clients ignore the unknown field, so
// passing it is compatible with other servers.
[[nodiscard]] std::string SessionIdFrom(const core::Json& body);
// Queue for the single device behind running generations. Blocks in
// arrival order instead of failing; nullopt when the peer goes away
// while waiting (the waiter gives up its place). The single canonical
// device queue for every endpoint.
[[nodiscard]] std::optional<std::unique_lock<std::mutex>> WaitForGpu(
    std::mutex& generation, const ResponseWriter& writer);
// Wall milliseconds between two steady-clock readings, for turn stats.
[[nodiscard]] long long MillisBetween(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end);
// True when the body asks for server-sent events.
[[nodiscard]] bool WantsStream(const core::Json& body);
// Send the failure of a generation call: InvalidArgument (for example
// a prompt past the context) is a 400 with a short prompt, anything
// else is a 500.
void SendGenerationError(ResponseWriter& writer, StatusCode code);
// 400 when the prompt is past `context`; false otherwise. Call before
// streaming headers so an oversize prompt fails with a status instead
// of a mid-stream error.
[[nodiscard]] bool RejectOversizePrompt(ResponseWriter& writer,
                                        std::size_t prompt,
                                        std::size_t context);
// OpenAI `finish_reason` for a generation outcome: "stop" when the model
// ended the turn (a declared stop token) or the caller cancelled, and
// "length" when the completion budget ran out.
[[nodiscard]] const char* FinishReasonName(FinishReason reason);
// Append one SSE frame (`event: ...` only when `with_event`).
void WriteSse(ResponseWriter& writer, const std::string& event,
              const core::Json& data, bool with_event);
// Fill the web UI placeholders from the static defaults: currently
// the max-completion-tokens input default, so the page and the serving
// defaults change in one place
// (tessera::kDefaultMaxCompletionTokens).
[[nodiscard]] std::string InjectWebDefaults(std::string_view page);

}  // namespace tessera::serve
