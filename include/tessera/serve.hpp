#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/types.hpp"

namespace tessera {

// Zero default completion length: a request that omits
// max_completion_tokens fills the remaining context (see
// Model::EffectiveMaxTokens) instead of stopping after a fixed budget.
//
// Usage:
//   ServeOptions options;
//   options.default_max_completion_tokens = kDefaultMaxCompletionTokens;
constexpr std::size_t kDefaultMaxCompletionTokens = 0;

// Options for Serve.
struct ServeOptions {
  // Interface to bind (dotted IPv4 literal).
  std::string host = "127.0.0.1";
  std::uint16_t port = 8080;
  // Default completion length when a request omits max_completion_tokens
  // (0 fills the remaining context).
  std::size_t default_max_completion_tokens = kDefaultMaxCompletionTokens;
  // Accepted API keys. Empty means no authentication. /health and
  // /metrics stay public.
  std::vector<std::string> api_keys;
  // CORS allowlist. Empty disables CORS. "*" allows any origin.
  std::vector<std::string> allow_origins;
};

// Serve a blocking HTTP API for `model` on the engine: OpenAI-style
// POST /v1/completions, POST /v1/chat/completions (with OpenAI-form
// tool calls) and POST /v1/messages, plus GET /health, GET /metrics,
// GET /v1/models, session management (/api/sessions and the
// /v1/sessions aliases, GET /slots) and a web chat UI at `/`.
// Generations queue in arrival order on the single device, so parallel
// chats wait their turn; a peer that disconnects while queued gives up
// its place. A request carrying `session_id` continues that stored
// session (404 unknown, 409 already generating); without it the
// request is stateless. Responses keep the standard OpenAI shape
// either way. Returns when the server socket fails or closes. The
// listening endpoint and every socket failure (with the failing call,
// the endpoint and the errno text) go to the engine diagnostics
// channel.
//
// Usage:
//   Serve(*engine, *model, {.port = 8080});
[[nodiscard]] std::expected<void, StatusCode> Serve(Engine& engine,
                                                    Model& model,
                                                    const ServeOptions& options);

}  // namespace tessera
