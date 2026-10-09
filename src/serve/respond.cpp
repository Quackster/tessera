#include "serve/respond.hpp"

#include <string>

#include "tessera/serve.hpp"
#include "tessera/types.hpp"

namespace tessera::serve {

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

std::size_t MaxTokensFrom(const core::Json& body, std::size_t fallback) {
  const core::Json* value = body.Find("max_tokens");
  if (value != nullptr && value->type() == core::Json::Type::Number &&
      value->AsNumber() > 0) {
    return static_cast<std::size_t>(value->AsNumber());
  }
  return fallback;
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
                           std::size_t max_tokens, std::size_t context) {
  if (max_tokens > 0 && prompt > context) {
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
  constexpr std::string_view kMaxTokens = "@TESSERA_DEFAULT_MAX_TOKENS@";
  const std::string value = std::to_string(kDefaultMaxTokens);
  std::string out(page);
  std::size_t pos = 0;
  while ((pos = out.find(kMaxTokens, pos)) != std::string::npos) {
    out.replace(pos, kMaxTokens.size(), value);
    pos += value.size();
  }
  return out;
}

}  // namespace tessera::serve
