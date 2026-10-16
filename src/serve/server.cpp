#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "serve/openai.hpp"
#include "serve/readiness.hpp"
#include "serve/render.hpp"
#include "serve/respond.hpp"
#include "serve/session.hpp"
#include "serve/session_chat.hpp"
#include "tessera/serve.hpp"
#include "web_assets.hpp"

namespace tessera {

namespace {

using core::Json;
using serve::HttpRequest;
using serve::ReasoningEffortsFromTemplate;
using serve::RenderPrompt;
using serve::ResponseWriter;
using serve::SendError;
using serve::SendJson;
using serve::TemplateDefaults;

std::atomic<unsigned long long> g_requests{0};

bool Bridge(const HttpRequest& request, const std::string& path) {
  return request.path == path || request.path == path + "/";
}

// The template defaults the serve options select for requests that omit
// the thinking fields.
TemplateDefaults DefaultsFrom(const ServeOptions& options) {
  return TemplateDefaults{options.default_enable_thinking,
                          options.default_reasoning_effort};
}

// The 503 body a not-ready server returns: status "loading" while the
// loader works, "failed" when it returned an error, plus the step.
void SendNotReady(ResponseWriter& writer,
                  const serve::Readiness::Status& status) {
  const bool failed = status.phase == serve::Readiness::Phase::kFailed;
  Json body = Json::Object();
  body.Set("error",
           Json::String(failed ? "model failed to load" : "model is loading"));
  body.Set("status", Json::String(failed ? "failed" : "loading"));
  body.Set("detail", Json::String(status.detail));
  SendJson(writer, 503, body);
}

// Shared serving core. `readiness` is already seeded (ready for a
// pre-loaded pool; loading otherwise). A non-null `loader` runs on a
// background thread while the socket serves, publishing the pool
// through `readiness`.
std::expected<void, StatusCode> ServeInternal(
    std::unique_ptr<RunnerPool> ready_pool, const ServeOptions& options,
    serve::Readiness& readiness, PoolLoader loader) {
  serve::SessionStore sessions;
  std::unique_ptr<RunnerPool> pool = std::move(ready_pool);
  // Diagnostics sink for the deferred case before the pool exists; the
  // ready case logs through the first runner's engine.
  log::Diagnostics local_log;
  const auto handler = [&pool, &options, &sessions, &readiness](
                           const HttpRequest& request, ResponseWriter& writer) {
    ++g_requests;
    // CORS.
    const std::string origin = request.Header("origin");
    bool cors_ok = false;
    if (!origin.empty() && !options.allow_origins.empty()) {
      for (const std::string& allowed : options.allow_origins) {
        if (allowed == "*" || allowed == origin) {
          cors_ok = true;
          break;
        }
      }
      if (cors_ok) {
        writer.SetHeader("Access-Control-Allow-Origin", origin);
        writer.SetHeader("Vary", "Origin");
      }
    }
    if (request.method == "OPTIONS") {
      writer.SetHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      writer.SetHeader("Access-Control-Allow-Headers",
                       "Authorization, Content-Type, X-Api-Key, x-api-key, "
                       "anthropic-version");
      (void)writer.SendHeaders(204, "text/plain");
      return;
    }
    const std::string path = request.path;
    const bool is_public = Bridge(request, "/health") ||
                           Bridge(request, "/v1/health") ||
                           Bridge(request, "/metrics") ||
                           (request.method == "GET" &&
                            (path == "/" || path == "/index.html" ||
                             path == "/app.js" || path == "/style.css"));
    // API key check.
    if (!options.api_keys.empty() && !is_public) {
      std::string key = request.Header("authorization");
      if (key.empty()) {
        key = request.Header("x-api-key");
      }
      const std::string prefix = "Bearer ";
      if (key.rfind(prefix, 0) == 0) {
        key = key.substr(prefix.size());
      }
      bool valid = false;
      for (const std::string& allowed : options.api_keys) {
        if (allowed == key) {
          valid = true;
          break;
        }
      }
      if (!valid) {
        SendError(writer, 401, "invalid API key");
        return;
      }
    }
    // Health reports readiness so a client can wait instead of failing
    // to connect while a slow model load runs.
    if (request.method == "GET" && Bridge(request, "/health")) {
      const serve::Readiness::Status status = readiness.Get();
      if (status.phase == serve::Readiness::Phase::kReady) {
        SendJson(writer, 200, Json::String("ok"));
      } else {
        SendNotReady(writer, status);
      }
      return;
    }
    if (request.method == "GET" && Bridge(request, "/metrics")) {
      (void)writer.SendHeaders(200, "text/plain; version=0.0.4");
      (void)writer.Write(
          "# HELP tessera_requests_total Requests served.\n"
          "# TYPE tessera_requests_total counter\n"
          "tessera_requests_total " +
          std::to_string(g_requests.load()) + "\n");
      return;
    }
    // Web UI (no build step: single files embedded at configure time).
    if (request.method == "GET" && path == "/app.js") {
      (void)writer.SendHeaders(200, "text/javascript");
      (void)writer.Write(std::string_view(serve::web::kWebAppJs));
      return;
    }
    if (request.method == "GET" && path == "/style.css") {
      (void)writer.SendHeaders(200, "text/css");
      (void)writer.Write(std::string_view(serve::web::kWebStyleCss));
      return;
    }
    if (request.method == "GET" && (path == "/" || path == "/index.html")) {
      (void)writer.SendHeaders(200, "text/html");
      (void)writer.Write(serve::InjectWebDefaults(
          std::string_view(serve::web::kWebIndexHtml)));
      return;
    }
    // Every endpoint below needs the model. Wait until it is loaded and
    // warm, and say why when it is not.
    const serve::Readiness::Status status = readiness.Get();
    if (status.phase != serve::Readiness::Phase::kReady) {
      SendNotReady(writer, status);
      return;
    }
    Model& model = *status.model;
    const Tokenizer* tokenizer = status.tokenizer;
    if (request.method == "GET" &&
        (Bridge(request, "/v1/models") || Bridge(request, "/models"))) {
      Json entry = Json::Object();
      entry.Set("id", Json::String(std::string(model.Name())));
      entry.Set("object", Json::String("model"));
      entry.Set("created", Json::Number(0));
      entry.Set("owned_by", Json::String("tessera"));
      Json data = Json::Array();
      data.Push(std::move(entry));
      Json response = Json::Object();
      response.Set("object", Json::String("list"));
      response.Set("data", std::move(data));
      SendJson(writer, 200, response);
      return;
    }
    if (request.method == "GET" && Bridge(request, "/props")) {
      Json props = Json::Object();
      props.Set("model_path", Json::String(std::string(model.Name())));
      props.Set("n_ctx",
                Json::Number(static_cast<double>(model.MaxContextLength())));
      props.Set("total_slots", Json::Number(static_cast<double>(pool->size())));
      // The reasoning levels the chat template offers, in template order;
      // absent when the template never reads reasoning_effort. The web UI
      // builds its difficulty selector from this list.
      const std::vector<std::string> efforts =
          ReasoningEffortsFromTemplate(model.ChatTemplate());
      if (!efforts.empty()) {
        Json levels = Json::Array();
        for (const std::string& effort : efforts) {
          levels.Push(Json::String(effort));
        }
        props.Set("reasoning_efforts", std::move(levels));
      }
      SendJson(writer, 200, props);
      return;
    }
    if (tokenizer == nullptr) {
      SendError(writer, 422, "this model has no tokenizer");
      return;
    }
    if (pool == nullptr) {
      SendNotReady(writer, readiness.Get());
      return;
    }
    // Generations route to an idle device; when every device is busy a
    // request waits instead of failing, so parallel windows and parallel
    // API clients each get their turn.
    serve::SessionHandler session_handler(
        *pool, *tokenizer, sessions, options.default_max_completion_tokens,
        options.auto_title, options.stats_interval_seconds,
        DefaultsFrom(options));
    if (path == "/api/sessions") {
      if (request.method == "GET") {
        session_handler.HandleList(writer);
        return;
      }
      if (request.method == "POST") {
        auto body = Json::Parse(request.body);
        if (body == nullptr) {
          SendError(writer, 400, "invalid JSON body");
          return;
        }
        session_handler.HandleCreate(writer, *body);
        return;
      }
      SendError(writer, 405, "method not allowed");
      return;
    }
    if (path.rfind("/api/sessions/", 0) == 0) {
      const std::string rest = path.substr(sizeof("/api/sessions/") - 1);
      const std::size_t slash = rest.find('/');
      const std::string id =
          slash == std::string::npos ? rest : rest.substr(0, slash);
      const std::string action =
          slash == std::string::npos ? std::string() : rest.substr(slash + 1);
      if (id.empty()) {
        SendError(writer, 404, "unknown session");
        return;
      }
      if (action.empty()) {
        if (request.method == "GET") {
          session_handler.HandleGet(writer, id);
          return;
        }
        if (request.method == "DELETE") {
          session_handler.HandleDelete(writer, id);
          return;
        }
        if (request.method == "PUT") {
          auto body = Json::Parse(request.body);
          if (body == nullptr) {
            SendError(writer, 400, "invalid JSON body");
            return;
          }
          session_handler.HandleRename(writer, id, *body);
          return;
        }
        SendError(writer, 405, "method not allowed");
        return;
      }
      if (request.method != "POST") {
        SendError(writer, 405, "method not allowed");
        return;
      }
      if (action == "chat" || action == "retry") {
        auto body = Json::Parse(request.body);
        if (body == nullptr) {
          SendError(writer, 400, "invalid JSON body");
          return;
        }
        if (action == "chat") {
          session_handler.HandleChat(writer, id, *body);
        } else {
          session_handler.HandleRetry(writer, id, *body);
        }
        return;
      }
      if (action == "stop") {
        session_handler.HandleStop(writer, id);
        return;
      }
      if (action == "pause") {
        session_handler.HandlePause(writer, id);
        return;
      }
      if (action == "resume") {
        session_handler.HandleResume(writer, id);
        return;
      }
      SendError(writer, 404, "unknown session action");
      return;
    }
    // OpenAI-style session management: the same store as /api/sessions.
    // A created id continues a chat through `session_id` on the
    // OpenAI endpoints below.
    if (Bridge(request, "/v1/sessions")) {
      if (request.method == "GET") {
        session_handler.HandleList(writer);
        return;
      }
      if (request.method == "POST") {
        auto body = Json::Parse(request.body);
        if (body == nullptr) {
          SendError(writer, 400, "invalid JSON body");
          return;
        }
        session_handler.HandleCreate(writer, *body);
        return;
      }
      SendError(writer, 405, "method not allowed");
      return;
    }
    if (request.method == "POST" && Bridge(request, "/tokenize")) {
      auto body = Json::Parse(request.body);
      const Json* content = body == nullptr ? nullptr : body->Find("content");
      if (content == nullptr || !content->isString()) {
        SendError(writer, 422, "content must be a string");
        return;
      }
      auto ids = tokenizer->Encode(content->AsString());
      if (!ids) {
        SendError(writer, 500, "tokenization failed");
        return;
      }
      Json tokens = Json::Array();
      for (std::uint32_t id : *ids) {
        tokens.Push(Json::Number(static_cast<double>(id)));
      }
      Json response = Json::Object();
      response.Set("tokens", std::move(tokens));
      SendJson(writer, 200, response);
      return;
    }
    if (request.method == "POST" && Bridge(request, "/detokenize")) {
      auto body = Json::Parse(request.body);
      const Json* tokens = body == nullptr ? nullptr : body->Find("tokens");
      if (tokens == nullptr || !tokens->isArray()) {
        SendError(writer, 422, "tokens must be an array");
        return;
      }
      std::vector<std::uint32_t> ids;
      for (const Json& token : tokens->AsArray()) {
        ids.push_back(static_cast<std::uint32_t>(token.AsNumber()));
      }
      auto text = tokenizer->Decode(ids);
      if (!text) {
        SendError(writer, 500, "detokenization failed");
        return;
      }
      Json response = Json::Object();
      response.Set("content", Json::String(*text));
      SendJson(writer, 200, response);
      return;
    }
    if (request.method == "POST" &&
        (Bridge(request, "/v1/completions") || Bridge(request, "/completion") ||
         Bridge(request, "/completions"))) {
      auto body = Json::Parse(request.body);
      if (body == nullptr) {
        SendError(writer, 400, "invalid JSON body");
        return;
      }
      serve::OpenAiComplete(*pool, *tokenizer, sessions, writer, *body,
                            options.default_max_completion_tokens,
                            options.stats_interval_seconds,
                            DefaultsFrom(options));
      return;
    }
    if (request.method == "POST" && Bridge(request, "/v1/chat/completions")) {
      auto body = Json::Parse(request.body);
      if (body == nullptr) {
        SendError(writer, 400, "invalid JSON body");
        return;
      }
      serve::OpenAiChat(*pool, *tokenizer, sessions, writer, *body,
                        options.default_max_completion_tokens, false,
                        options.stats_interval_seconds, DefaultsFrom(options));
      return;
    }
    if (request.method == "POST" &&
        Bridge(request, "/v1/messages/count_tokens")) {
      auto body = Json::Parse(request.body);
      if (body == nullptr) {
        SendError(writer, 400, "invalid JSON body");
        return;
      }
      std::string error;
      const std::string prompt =
          RenderPrompt(model, *body, &error, DefaultsFrom(options));
      if (!error.empty()) {
        SendError(writer, 422, error);
        return;
      }
      auto ids = tokenizer->Encode(prompt);
      Json response = Json::Object();
      response.Set("input_tokens",
                   Json::Number(static_cast<double>(ids ? ids->size() : 0)));
      SendJson(writer, 200, response);
      return;
    }
    if (request.method == "POST" &&
        (Bridge(request, "/v1/messages") || Bridge(request, "/messages"))) {
      auto body = Json::Parse(request.body);
      if (body == nullptr) {
        SendError(writer, 400, "invalid JSON body");
        return;
      }
      serve::OpenAiChat(*pool, *tokenizer, sessions, writer, *body,
                        options.default_max_completion_tokens, true,
                        options.stats_interval_seconds, DefaultsFrom(options));
      return;
    }
    // Live session states: one entry per chat with its id, title,
    // message count and busy flag, so parallel windows can watch
    // each other instead of polling blindly.
    if (request.method == "GET" && Bridge(request, "/slots")) {
      Json slots = Json::Array();
      for (const serve::SessionInfo& info : sessions.List()) {
        slots.Push(serve::SessionInfoJson(info));
      }
      SendJson(writer, 200, slots);
      return;
    }
    // Registered but not implemented (capability missing).
    SendError(writer, 501, "endpoint not implemented");
  };
  std::atomic<bool> stop{false};
  std::thread loader_thread;
  log::Diagnostics& serve_log =
      pool ? pool->At(0).engine().Diagnostics() : local_log;
  // Bind before loading: a busy port fails here instead of after a
  // multi-minute weight upload.
  auto bound = serve::BindHttpServer(options.host, options.port, serve_log);
  if (!bound) {
    return std::unexpected(bound.error());
  }
  if (loader) {
    // Bind and serve first, then load: a client reaches /health during
    // a slow load instead of finding the port closed.
    loader_thread = std::thread(
        [&pool, &readiness, &local_log, loader = std::move(loader)] {
          const ServeProgress report = [&readiness](std::string_view detail) {
            readiness.Loading(std::string(detail));
          };
          auto loaded = loader(report);
          if (!loaded) {
            const std::string detail = std::string(ToString(loaded.error()));
            readiness.Failed(detail);
            local_log.Error(
                "serve",
                "model load failed (" + detail +
                    "); /health reports the failure and no request is served");
            return;
          }
          pool = std::move(*loaded);
          readiness.Ready(&pool->At(0).model(), pool->At(0).tokenizer());
          local_log.Info("serve", "model ready");
        });
  }
  auto served =
      serve::RunHttpServerOn(std::move(*bound), handler, &stop, serve_log);
  if (loader_thread.joinable()) {
    loader_thread.join();
  }
  return served;
}

}  // namespace

std::expected<void, StatusCode> Serve(std::unique_ptr<RunnerPool> pool,
                                      const ServeOptions& options) {
  if (pool == nullptr || pool->empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  serve::Readiness readiness;
  readiness.Ready(&pool->At(0).model(), pool->At(0).tokenizer());
  return ServeInternal(std::move(pool), options, readiness, nullptr);
}

std::expected<void, StatusCode> Serve(const ServeOptions& options,
                                      PoolLoader loader) {
  serve::Readiness readiness;
  readiness.Loading("waiting for the model");
  return ServeInternal(nullptr, options, readiness, std::move(loader));
}

}  // namespace tessera
