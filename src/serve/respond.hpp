#pragma once

#include <cstddef>
#include <string>

#include "core/json.hpp"
#include "serve/http.hpp"
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
// Positive `max_tokens` from the body, else `fallback`.
[[nodiscard]] std::size_t MaxTokensFrom(const core::Json& body,
                                        std::size_t fallback);
// True when the body asks for server-sent events.
[[nodiscard]] bool WantsStream(const core::Json& body);
// Send the failure of a generation call: InvalidArgument (for example
// a prompt past the context) is a 400 with a short prompt, anything
// else is a 500.
void SendGenerationError(ResponseWriter& writer, StatusCode code);
// 400 when an explicit `max_tokens` meets a prompt past `context`;
// false otherwise. Call before streaming headers so an oversize
// prompt fails with a status instead of a mid-stream error.
[[nodiscard]] bool RejectOversizePrompt(ResponseWriter& writer,
                                        std::size_t prompt,
                                        std::size_t max_tokens,
                                        std::size_t context);
// Append one SSE frame (`event: ...` only when `with_event`).
void WriteSse(ResponseWriter& writer, const std::string& event,
              const core::Json& data, bool with_event);

}  // namespace tessera::serve
