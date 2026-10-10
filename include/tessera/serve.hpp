#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
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
  // Ask the model for a chat title after the first prompt of a new
  // session; off keeps the first user line as the title.
  bool auto_title = true;
};

// Reports the loader's current step (for example "loading weights" or
// "compiling kernels") so /health shows progress instead of silence.
using ServeProgress = std::function<void(std::string_view)>;

// Produces the model for the deferred Serve overload. Runs on a
// background thread after the HTTP socket is bound, so the endpoints
// answer while the model loads and its kernels warm. Call `report` with
// the current step; return the loaded model, or an error that Serve
// records and reports through /health. A model that is not warmed is
// still accepted, but the first request then pays the kernel compile.
//
// Usage:
//   ModelLoader loader = [&](const ServeProgress& report) {
//     report("loading weights");
//     return engine.LoadModel(ModelOptions{path});
//   };
using ModelLoader = std::function<std::expected<std::unique_ptr<Model>,
                                                StatusCode>(const ServeProgress&)>;

// Serve a blocking HTTP API on the engine: OpenAI-style POST
// /v1/completions, POST /v1/chat/completions (with OpenAI-form tool
// calls) and POST /v1/messages, plus GET /health, GET /metrics, GET
// /v1/models, session management (/api/sessions and the /v1/sessions
// aliases, GET /slots) and a web chat UI at `/`. Generations queue in
// arrival order on the single device, so parallel chats wait their
// turn; a peer that disconnects while queued gives up its place. A
// request carrying `session_id` continues that stored session (404
// unknown, 409 already generating); without it the request is
// stateless. Responses keep the standard OpenAI shape either way.
// Returns when the server socket fails or closes. The listening
// endpoint and every socket failure (with the failing call, the
// endpoint and the errno text) go to the engine diagnostics channel.
//
// The deferred overload binds the socket first, then runs `loader` on a
// background thread, so a client can connect while the model loads.
// Until the loader returns, GET /health answers 503 with a JSON body
// `{"status":"loading"|"failed","detail":...}` and every other endpoint
// answers 503; when it returns, /health answers 200 and serving is
// enabled. A load failure keeps the server up so /health can report it.
//
// Usage:
//   Serve(engine, options, loader);   // deferred
//   Serve(*engine, *model, options);  // model already loaded
[[nodiscard]] std::expected<void, StatusCode> Serve(Engine& engine,
                                                    const ServeOptions& options,
                                                    ModelLoader loader);

// Serve an already-loaded `model`, ready immediately. `model` must
// outlive the call.
[[nodiscard]] std::expected<void, StatusCode> Serve(Engine& engine,
                                                    Model& model,
                                                    const ServeOptions& options);

}  // namespace tessera
