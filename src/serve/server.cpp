#include "tessera/serve.hpp"

#include <atomic>
#include <cctype>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "serve/http.hpp"
#include "serve/render.hpp"
#include "serve/respond.hpp"
#include "serve/session.hpp"
#include "serve/session_chat.hpp"
#include "serve/tools/chat.hpp"
#include "core/json.hpp"
#include "web_assets.hpp"

namespace tessera {

namespace {

using serve::HttpRequest;
using core::Json;
using serve::ResponseWriter;
using serve::ChatWithTools;
using serve::MaxTokensFrom;
using serve::RenderPrompt;
using serve::SendError;
using serve::SendJson;
using serve::UsageJson;
using serve::WantsStream;
using serve::WantsTools;
using serve::WriteSse;

std::atomic<unsigned long long> g_requests{0};

std::string Lower(std::string_view s) {
  std::string out;
  for (char c : s) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

// Shared completion core for the raw-prompt endpoint.
void Complete(Engine& engine, Model& model, const Tokenizer& tokenizer,
              ResponseWriter& writer, const std::string& prompt_text,
              std::size_t max_tokens, bool stream) {
  auto ids = tokenizer.Encode(prompt_text);
  if (!ids) {
    SendError(writer, 500, "tokenization failed");
    return;
  }
  GenerateOptions options;
  options.max_tokens = max_tokens;
  options.prompt_tokens = *ids;
  if (RejectOversizePrompt(writer, ids->size(), max_tokens,
                           model.MaxContextLength())) {
    return;
  }
  if (!stream) {
    // GenerateStreaming with a liveness hook instead of Generate:
    // identical tokens, but a closed window aborts the turn instead
    // of decoding into the void.
    std::vector<std::uint32_t> produced;
    auto streamed = engine.GenerateStreaming(
        model, options, [&](std::uint32_t token) {
          if (writer.IsPeerGone()) {
            return false;
          }
          produced.push_back(token);
          return true;
        });
    if (!streamed) {
      // A gone peer fails here only on a real error (abort returns a
      // count); anything else is reported when someone listens.
      if (!writer.IsPeerGone()) {
        SendGenerationError(writer, streamed.error());
      }
      return;
    }
    auto text = tokenizer.Decode(produced);
    if (!text) {
      SendError(writer, 500, "detokenization failed");
      return;
    }
    Json choice = Json::Object();
    choice.Set("text", Json::String(*text));
    choice.Set("index", Json::Number(0));
    choice.Set("finish_reason", Json::String("length"));
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
  auto streamed = engine.GenerateStreaming(
      model, options, [&](std::uint32_t token) {
        if (writer.IsPeerGone()) {
          return false;
        }
        auto piece = tokenizer.Decode(
            std::span<const std::uint32_t>(&token, 1));
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
        return true;
      });
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
  choice.Set("finish_reason", Json::String("length"));
  choices.Push(std::move(choice));
  done.Set("choices", std::move(choices));
  WriteSse(writer, "", done, false);
  (void)writer.Write("data: [DONE]\n\n");
  (void)count;
}

void Chat(Engine& engine, Model& model, const Tokenizer& tokenizer,
          ResponseWriter& writer, const Json& body, std::size_t default_max,
          bool anthropic) {
  if (!anthropic && WantsTools(body)) {
    ChatWithTools(engine, model, tokenizer, writer, body, default_max);
    return;
  }
  std::string error;
  const std::string prompt = RenderPrompt(model, body, &error);
  if (!error.empty()) {
    SendError(writer, 422, error);
    return;
  }
  auto ids = tokenizer.Encode(prompt);
  if (!ids) {
    SendError(writer, 500, "tokenization failed");
    return;
  }
  const std::size_t max_tokens = MaxTokensFrom(body, default_max);
  const bool stream = WantsStream(body);
  GenerateOptions options;
  options.max_tokens = max_tokens;
  options.prompt_tokens = *ids;
  if (RejectOversizePrompt(writer, ids->size(), max_tokens,
                           model.MaxContextLength())) {
    return;
  }
  const std::string model_name(model.Name());
  if (anthropic) {
    if (!stream) {
      // GenerateStreaming with a liveness hook instead of Generate:
      // identical tokens, but a closed window aborts the turn instead
      // of decoding into the void.
      std::vector<std::uint32_t> produced;
      auto streamed = engine.GenerateStreaming(
          model, options, [&](std::uint32_t token) {
            if (writer.IsPeerGone()) {
              return false;
            }
            produced.push_back(token);
            return true;
          });
      if (!streamed) {
        // A gone peer fails here only on a real error (abort returns a
        // count); anything else is reported when someone listens.
        if (!writer.IsPeerGone()) {
          SendGenerationError(writer, streamed.error());
        }
        return;
      }
      auto text = tokenizer.Decode(produced);
      Json content = Json::Array();
      Json block = Json::Object();
      block.Set("type", Json::String("text"));
      block.Set("text", Json::String(text ? *text : std::string()));
      content.Push(std::move(block));
      Json usage = Json::Object();
      usage.Set("input_tokens", Json::Number(static_cast<double>(ids->size())));
      usage.Set("output_tokens",
                Json::Number(static_cast<double>(produced.size())));
      Json response = Json::Object();
      response.Set("id", Json::String("msg_0"));
      response.Set("type", Json::String("message"));
      response.Set("role", Json::String("assistant"));
      response.Set("model", Json::String(model_name));
      response.Set("content", std::move(content));
      response.Set("stop_reason", Json::String("max_tokens"));
      response.Set("usage", std::move(usage));
      SendJson(writer, 200, response);
      return;
    }
    (void)writer.SendHeaders(200, "text/event-stream", true);
    Json start = Json::Object();
    start.Set("type", Json::String("message_start"));
    Json message = Json::Object();
    message.Set("id", Json::String("msg_0"));
    message.Set("role", Json::String("assistant"));
    message.Set("content", Json::Array());
    start.Set("message", std::move(message));
    WriteSse(writer, "message_start", start, true);
    Json block_start = Json::Object();
    block_start.Set("type", Json::String("content_block_start"));
    block_start.Set("index", Json::Number(0));
    Json block = Json::Object();
    block.Set("type", Json::String("text"));
    block.Set("text", Json::String(""));
    block_start.Set("content_block", std::move(block));
    WriteSse(writer, "content_block_start", block_start, true);
    (void)engine.GenerateStreaming(model, options, [&](std::uint32_t token) {
      if (writer.IsPeerGone()) {
        return false;
      }
      auto piece =
          tokenizer.Decode(std::span<const std::uint32_t>(&token, 1));
      Json delta = Json::Object();
      delta.Set("type", Json::String("content_block_delta"));
      delta.Set("index", Json::Number(0));
      Json text_delta = Json::Object();
      text_delta.Set("type", Json::String("text_delta"));
      text_delta.Set("text", Json::String(piece ? *piece : std::string()));
      delta.Set("delta", std::move(text_delta));
      WriteSse(writer, "content_block_delta", delta, true);
      return true;
    });
    Json stop = Json::Object();
    stop.Set("type", Json::String("content_block_stop"));
    stop.Set("index", Json::Number(0));
    WriteSse(writer, "content_block_stop", stop, true);
    Json message_delta = Json::Object();
    message_delta.Set("type", Json::String("message_delta"));
    Json md = Json::Object();
    md.Set("stop_reason", Json::String("end_turn"));
    message_delta.Set("delta", std::move(md));
    WriteSse(writer, "message_delta", message_delta, true);
    Json message_stop = Json::Object();
    message_stop.Set("type", Json::String("message_stop"));
    WriteSse(writer, "message_stop", message_stop, true);
    return;
  }
  // OpenAI chat.
  if (!stream) {
    // GenerateStreaming with a liveness hook instead of Generate:
    // identical tokens, but a closed window aborts the turn instead
    // of decoding into the void.
    std::vector<std::uint32_t> produced;
    auto streamed = engine.GenerateStreaming(
        model, options, [&](std::uint32_t token) {
          if (writer.IsPeerGone()) {
            return false;
          }
          produced.push_back(token);
          return true;
        });
    if (!streamed) {
      // A gone peer fails here only on a real error (abort returns a
      // count); anything else is reported when someone listens.
      if (!writer.IsPeerGone()) {
        SendGenerationError(writer, streamed.error());
      }
      return;
    }
    auto text = tokenizer.Decode(produced);
    Json message = Json::Object();
    message.Set("role", Json::String("assistant"));
    message.Set("content", Json::String(text ? *text : std::string()));
    Json choice = Json::Object();
    choice.Set("index", Json::Number(0));
    choice.Set("message", std::move(message));
    choice.Set("finish_reason", Json::String("length"));
    Json choices = Json::Array();
    choices.Push(std::move(choice));
    Json response = Json::Object();
    response.Set("id", Json::String("chatcmpl-0"));
    response.Set("object", Json::String("chat.completion"));
    response.Set("model", Json::String(model_name));
    response.Set("choices", std::move(choices));
    response.Set("usage", UsageJson(ids->size(), produced.size()));
    SendJson(writer, 200, response);
    return;
  }
  (void)writer.SendHeaders(200, "text/event-stream", true);
  (void)engine.GenerateStreaming(model, options, [&](std::uint32_t token) {
    if (writer.IsPeerGone()) {
      return false;
    }
    auto piece = tokenizer.Decode(std::span<const std::uint32_t>(&token, 1));
    Json delta = Json::Object();
    delta.Set("content", Json::String(piece ? *piece : std::string()));
    Json choice = Json::Object();
    choice.Set("index", Json::Number(0));
    choice.Set("delta", std::move(delta));
    choice.Set("finish_reason", Json());
    Json choices = Json::Array();
    choices.Push(std::move(choice));
    Json chunk = Json::Object();
    chunk.Set("id", Json::String("chatcmpl-0"));
    chunk.Set("object", Json::String("chat.completion.chunk"));
    chunk.Set("model", Json::String(model_name));
    chunk.Set("choices", std::move(choices));
    WriteSse(writer, "", chunk, false);
    return true;
  });
  (void)writer.Write("data: [DONE]\n\n");
}

bool Bridge(const HttpRequest& request, const std::string& path) {
  return request.path == path || request.path == path + "/";
}

}  // namespace

std::expected<void, StatusCode> Serve(Engine& engine, Model& model,
                                      const ServeOptions& options) {
  const Tokenizer* tokenizer = model.GetTokenizer();
  serve::SessionStore sessions;
  // One generation at a time on the device; connection threads share
  // it (a second turn answers 503).
  std::mutex generation;
  const auto handler = [&engine, &model, tokenizer, &options, &sessions,
                        &generation](const HttpRequest& request,
                                     ResponseWriter& writer) {
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
    if (request.method == "GET" && Bridge(request, "/health")) {
      SendJson(writer, 200, Json::String("ok"));
      return;
    }
    if (request.method == "GET" && Bridge(request, "/metrics")) {
      Json unused = Json::Object();
      (void)unused;
      (void)writer.SendHeaders(200, "text/plain; version=0.0.4");
      (void)writer.Write("# HELP tessera_requests_total Requests served.\n"
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
      (void)writer.Write(
          serve::InjectWebDefaults(std::string_view(serve::web::kWebIndexHtml)));
      return;
    }
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
      props.Set("n_ctx", Json::Number(
                             static_cast<double>(model.MaxContextLength())));
      props.Set("total_slots", Json::Number(1));
      SendJson(writer, 200, props);
      return;
    }
    if (tokenizer == nullptr) {
      SendError(writer, 422, "this model has no tokenizer");
      return;
    }
    // One generation at a time; a second turn answers 503 instead of
    // queueing behind minutes of prefill on the device.
    auto gpu_slot = [&](ResponseWriter& target)
        -> std::optional<std::unique_lock<std::mutex>> {
      std::unique_lock<std::mutex> slot(generation, std::try_to_lock);
      if (!slot.owns_lock()) {
        SendError(target, 503, "another generation is already running");
        return std::nullopt;
      }
      return slot;
    };
    serve::SessionHandler session_handler(engine, model, *tokenizer, sessions,
                                          generation,
                                          options.default_max_tokens);
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
        (Bridge(request, "/v1/completions") ||
         Bridge(request, "/completion") ||
         Bridge(request, "/completions"))) {
      auto body = Json::Parse(request.body);
      const Json* prompt = body == nullptr ? nullptr : body->Find("prompt");
      if (prompt == nullptr || !prompt->isString()) {
        SendError(writer, 422, "prompt must be a string");
        return;
      }
      auto slot = gpu_slot(writer);
      if (!slot) {
        return;
      }
      Complete(engine, model, *tokenizer, writer, prompt->AsString(),
               MaxTokensFrom(*body, options.default_max_tokens),
               WantsStream(*body));
      return;
    }
    if (request.method == "POST" &&
        Bridge(request, "/v1/chat/completions")) {
      auto body = Json::Parse(request.body);
      if (body == nullptr) {
        SendError(writer, 400, "invalid JSON body");
        return;
      }
      auto slot = gpu_slot(writer);
      if (!slot) {
        return;
      }
      Chat(engine, model, *tokenizer, writer, *body,
           options.default_max_tokens, false);
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
      const std::string prompt = RenderPrompt(model, *body, &error);
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
      auto slot = gpu_slot(writer);
      if (!slot) {
        return;
      }
      Chat(engine, model, *tokenizer, writer, *body,
           options.default_max_tokens, true);
      return;
    }
    if (request.method == "GET" && Bridge(request, "/slots")) {
      SendJson(writer, 200, Json::Array());
      return;
    }
    // Registered but not implemented (capability missing).
    SendError(writer, 501, "endpoint not implemented");
  };
  std::atomic<bool> stop{false};
  return serve::RunHttpServer(options.host, options.port, handler, &stop,
                              engine.Diagnostics());
}

}  // namespace tessera
