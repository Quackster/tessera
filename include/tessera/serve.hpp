#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tessera/calibrate.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/runner.hpp"
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

// Default live speed cadence: streamed turns report tok/s to the
// session SSE stream and the server log every this many seconds.
// Zero disables both; --stats-interval overrides it per serve run.
constexpr std::size_t kDefaultStatsIntervalSeconds = 5;

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
  // Seconds between live tok/s reports on streamed turns, to the
  // session SSE stream and the server log (0 disables both).
  std::size_t stats_interval_seconds = kDefaultStatsIntervalSeconds;
};

// Reports the loader's current step (for example "loading weights" or
// "compiling kernels") so /health shows progress instead of silence.
using ServeProgress = std::function<void(std::string_view)>;

// Produces the runner pool for the deferred Serve overload: one runner
// per device. Runs on a background thread after the socket is bound, so
// /health answers while the weights load. Return an error that Serve
// records and reports through /health.
//
// Usage:
//   PoolLoader loader = [&](const ServeProgress& report) {
//     report("loading weights");
//     return BuildPool(devices);
//   };
using PoolLoader = std::function<std::expected<std::unique_ptr<RunnerPool>,
                                               StatusCode>(const ServeProgress&)>;

// Serve a data-parallel pool of runners (one per device). Takes
// ownership of the pool. Generations route to an idle device and queue
// in arrival order when every device is busy. The OpenAI and session
// endpoints are unchanged; only the device routing differs from the
// single-device overload.
//
// Usage:
//   auto pool = BuildPool({0, 1});
//   if (!pool) return 1;
//   Serve(std::move(*pool), options);
[[nodiscard]] std::expected<void, StatusCode> Serve(
    std::unique_ptr<RunnerPool> pool, const ServeOptions& options);

// Deferred overload: binds the socket first, then runs `loader` on a
// background thread to build the pool, so a client can connect while the
// weights load. Until the loader returns, /health answers 503 with a JSON
// body `{"status":"loading"|"failed","detail":...}`. A load failure
// keeps the server up so /health can report it.
[[nodiscard]] std::expected<void, StatusCode> Serve(const ServeOptions& options,
                                                    PoolLoader loader);

}  // namespace tessera
