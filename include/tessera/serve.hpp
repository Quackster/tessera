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

// Default completion length (32k tokens) when a request omits
// max_tokens and the caller leaves the default below. Bounding the
// default keeps an unspecified request from decoding to the end of a
// large context window.
//
// Usage:
//   ServeOptions options;
//   options.default_max_tokens = kDefaultMaxTokens;
constexpr std::size_t kDefaultMaxTokens = 32u * 1024u;

// Options for Serve.
struct ServeOptions {
  // Interface to bind (dotted IPv4 literal).
  std::string host = "127.0.0.1";
  std::uint16_t port = 8080;
  // Default completion length when a request omits max_tokens.
  std::size_t default_max_tokens = kDefaultMaxTokens;
  // Accepted API keys. Empty means no authentication. /health and
  // /metrics stay public.
  std::vector<std::string> api_keys;
  // CORS allowlist. Empty disables CORS. "*" allows any origin.
  std::vector<std::string> allow_origins;
};

// Serve a blocking HTTP API for `model` on the engine: OpenAI-style
// POST /v1/completions and GET /health are implemented today. POST
// /v1/chat/completions and POST /v1/messages need the chat-template
// renderer and return 501 until it lands. Requests are handled one at a
// time. Returns when the server socket fails or closes. The listening
// endpoint and every socket failure (with the failing call, the
// endpoint and the errno text) go to the engine diagnostics channel.
//
// Usage:
//   Serve(*engine, *model, {.port = 8080});
[[nodiscard]] std::expected<void, StatusCode> Serve(Engine& engine,
                                                    Model& model,
                                                    const ServeOptions& options);

}  // namespace tessera
