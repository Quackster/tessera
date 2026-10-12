#include "serve/openai.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "serve/openai_shared.hpp"
#include "serve/render.hpp"
#include "serve/respond.hpp"
#include "serve/tools/chat.hpp"
#include "serve/tools/tool_call.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/runner.hpp"
#include "tessera/tokenizer.hpp"

namespace tessera::serve {

namespace {

using core::Json;

}  // namespace

void OpenAiChat(RunnerPool& pool, const Tokenizer& tokenizer,
                SessionStore& sessions, ResponseWriter& writer,
                const Json& body, std::size_t default_max, bool anthropic) {
  const std::string session_id = SessionIdFrom(body);
  std::shared_ptr<Session> session;
  SessionTurn turn;
  Json prompt_body = body;
  if (!session_id.empty()) {
    if (!BeginSession(sessions, session_id, writer, &session, &turn)) {
      return;
    }
  }
  if (pool.empty()) {
    return;
  }
  if (session != nullptr) {
    SessionView view{"", "", {}, false, false};
    if (!BeginChatTurn(session, writer, body, &view)) {
      return;
    }
    prompt_body = SessionPromptBody(body, view);
  }
  // Plain turns always share the scheduler so concurrent sessions
  // time-slice; thinking budgets and tool calls stay queued exclusive
  // (see RouteDevice).
  const bool wants_tools = !anthropic && WantsTools(prompt_body);
  const std::size_t max_thinking_tokens =
      MaxThinkingTokensFrom(prompt_body);
  RunnerLease lease;
  RunnerLease* slot = nullptr;
  if (!RouteDevice(pool, max_thinking_tokens, wants_tools,
                   [&writer] { return writer.IsPeerGone(); }, lease, &slot)) {
    return;
  }
  const bool shared = slot == nullptr;
  Model& model = shared ? pool.At(0).model() : lease.model();
  if (wants_tools) {
    SessionMessage assistant;
    const bool ok = ChatWithTools(lease.engine(), lease.model(), tokenizer,
                                  writer, prompt_body, default_max,
                                  session != nullptr ? &assistant : nullptr);
    if (session != nullptr && ok) {
      session->Append(std::move(assistant));
    }
    return;
  }
  std::string error;
  const std::string prompt = RenderPrompt(model, prompt_body, &error);
  if (!error.empty()) {
    SendError(writer, 422, error);
    return;
  }
  auto ids = tokenizer.Encode(prompt);
  if (!ids) {
    SendError(writer, 500, "tokenization failed");
    return;
  }
  const std::size_t max_completion_tokens =
      MaxCompletionTokensFrom(prompt_body, default_max);
  const bool stream = WantsStream(prompt_body);
  GenerateOptions options;
  options.max_completion_tokens = max_completion_tokens;
  options.max_thinking_tokens = max_thinking_tokens;
  options.prompt_tokens = *ids;
  if (!shared) {
    ApplyToGenerateOptions(lease.runner().calibration(), options);
  }
  if (RejectOversizePrompt(writer, ids->size(), model.MaxContextLength())) {
    return;
  }
  // Shared turns enqueue on the least-loaded device and stream back
  // on this thread; exclusive (thinking) turns hold the lease's device
  // while yielding every few tokens (see MaybeYieldExclusive).
  const auto drive = [&](const std::function<bool(std::uint32_t)>& next) {
    if (shared) {
      return pool.SubmitLeastLoaded(
          options, next, [&writer] { return writer.IsPeerGone(); });
    }
    return lease.engine().GenerateStreaming(lease.model(), options, next);
  };
  const std::string model_name(model.Name());
  TurnStats stats;
  stats.prompt_tokens = ids->size();
  // Exclusive (thinking) turns yield the device every few tokens so
  // a queued scheduler turn streams beside them instead of behind
  // them.
  int yield_count = 0;
  // Thinking is on unless the request turns it off. The chat template
  // then opens the <think> block in the prompt, so the model streams
  // reasoning with no opening tag in its output; ThinkExpected tells
  // the streamer to treat that leading text as reasoning.
  const Json* thinking_flag = prompt_body.Find("enable_thinking");
  const bool enable_thinking =
      thinking_flag == nullptr ||
      thinking_flag->type() != Json::Type::Bool || thinking_flag->AsBool();
  if (anthropic) {
    if (!stream) {
      // GenerateStreaming with a liveness hook instead of Generate:
      // identical tokens, but a closed window aborts the turn instead
      // of decoding into the void.
      std::vector<std::uint32_t> produced;
      const auto gen_started = std::chrono::steady_clock::now();
      auto streamed = drive([&](std::uint32_t token) {
        if (writer.IsPeerGone()) {
          return false;
        }
        produced.push_back(token);
        if (!shared) {
          MaybeYieldExclusive(lease, yield_count);
        }
        return true;
      });
      stats.elapsed_ms = MillisBetween(gen_started,
                                       std::chrono::steady_clock::now());
      if (!streamed) {
        // A gone peer fails here only on a real error (abort returns a
        // count); anything else is reported when someone listens.
        if (!writer.IsPeerGone()) {
          SendGenerationError(writer, streamed.error());
        }
        return;
      }
      stats.completion_tokens = produced.size();
      auto text = tokenizer.Decode(produced);
      if (session != nullptr) {
        session->Append(AssistantTurn(text ? *text : std::string(),
                                      writer.IsPeerGone(), stats));
      }
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
      response.Set("stop_reason",
                   Json::String(streamed->reason == FinishReason::Length
                                    ? "max_tokens"
                                    : "end_turn"));
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
    std::string text;
    std::size_t count = 0;
    const auto gen_started = std::chrono::steady_clock::now();
    auto streamed = drive([&](std::uint32_t token) {
      if (writer.IsPeerGone()) {
        return false;
      }
      auto piece =
          tokenizer.Decode(std::span<const std::uint32_t>(&token, 1));
      if (piece) {
        text += *piece;
      }
      ++count;
      Json delta = Json::Object();
      delta.Set("type", Json::String("content_block_delta"));
      delta.Set("index", Json::Number(0));
      Json text_delta = Json::Object();
      text_delta.Set("type", Json::String("text_delta"));
      text_delta.Set("text", Json::String(piece ? *piece : std::string()));
      delta.Set("delta", std::move(text_delta));
      WriteSse(writer, "content_block_delta", delta, true);
      if (!shared) {
        MaybeYieldExclusive(lease, yield_count);
      }
      return true;
    });
    stats.elapsed_ms = MillisBetween(gen_started,
                                     std::chrono::steady_clock::now());
    stats.completion_tokens = count;
    if (session != nullptr) {
      session->Append(AssistantTurn(text, writer.IsPeerGone(), stats));
    }
    Json stop = Json::Object();
    stop.Set("type", Json::String("content_block_stop"));
    stop.Set("index", Json::Number(0));
    WriteSse(writer, "content_block_stop", stop, true);
    Json message_delta = Json::Object();
    message_delta.Set("type", Json::String("message_delta"));
    Json md = Json::Object();
    md.Set("stop_reason",
           Json::String(streamed && streamed->reason == FinishReason::Length
                            ? "max_tokens"
                            : "end_turn"));
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
    const auto chat_started = std::chrono::steady_clock::now();
    auto streamed = drive([&](std::uint32_t token) {
      if (writer.IsPeerGone()) {
        return false;
      }
      produced.push_back(token);
      if (!shared) {
        MaybeYieldExclusive(lease, yield_count);
      }
      return true;
    });
    stats.elapsed_ms = MillisBetween(chat_started,
                                     std::chrono::steady_clock::now());
    if (!streamed) {
      // A gone peer fails here only on a real error (abort returns a
      // count); anything else is reported when someone listens.
      if (!writer.IsPeerGone()) {
        SendGenerationError(writer, streamed.error());
      }
      return;
    }
    stats.completion_tokens = produced.size();
    auto text = tokenizer.Decode(produced);
    if (!text) {
      SendError(writer, 500, "detokenization failed");
      return;
    }
    // Split the model's thinking out of the reply: the template opened
    // the <think> block, so `SplitThink` (via the streamer) recovers the
    // reasoning before the `</think>` closer. Without this the reasoning
    // is returned as `content` and the real answer never surfaces.
    ThinkStreamer streamer(ThinkExpected(prompt, enable_thinking));
    const ThinkStreamer::Deltas part = streamer.Push(*text);
    const ThinkStreamer::Deltas tail = streamer.Finish();
    const std::string reasoning_text = part.reasoning + tail.reasoning;
    const std::string content_text = part.content + tail.content;
    if (session != nullptr) {
      session->Append(AssistantTurn(*text, writer.IsPeerGone(), stats));
    }
    Json message = Json::Object();
    message.Set("role", Json::String("assistant"));
    if (!reasoning_text.empty()) {
      message.Set("reasoning_content", Json::String(reasoning_text));
    }
    message.Set("content", Json::String(content_text));
    Json choice = Json::Object();
    choice.Set("index", Json::Number(0));
    choice.Set("message", std::move(message));
    choice.Set("finish_reason", Json::String(FinishReasonName(streamed->reason)));
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
  ThinkStreamer streamer(ThinkExpected(prompt, enable_thinking));
  const auto stream_piece = [&](const ThinkStreamer::Deltas& part) {
    if (part.reasoning.empty() && part.content.empty()) {
      return;
    }
    Json delta = Json::Object();
    if (!part.reasoning.empty()) {
      delta.Set("reasoning_content", Json::String(part.reasoning));
    }
    if (!part.content.empty()) {
      delta.Set("content", Json::String(part.content));
    }
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
  };
  std::string text;
  std::size_t count = 0;
  const auto chat_started = std::chrono::steady_clock::now();
  auto streamed = drive([&](std::uint32_t token) {
    if (writer.IsPeerGone()) {
      return false;
    }
    auto piece = tokenizer.Decode(std::span<const std::uint32_t>(&token, 1));
    if (piece) {
      text += *piece;
    }
    ++count;
    stream_piece(streamer.Push(piece ? *piece : std::string()));
    if (!shared) {
      MaybeYieldExclusive(lease, yield_count);
    }
    return true;
  });
  stream_piece(streamer.Finish());
  stats.elapsed_ms = MillisBetween(chat_started,
                                   std::chrono::steady_clock::now());
  stats.completion_tokens = count;
  if (session != nullptr) {
    session->Append(AssistantTurn(text, writer.IsPeerGone(), stats));
  }
  // Final chunk carries the end reason; the deltas above leave it null
  // (OpenAI convention) so a client that reads finish_reason sees "stop"
  // for a completed turn and "length" only on a real truncation.
  Json end_delta = Json::Object();
  Json end_choice = Json::Object();
  end_choice.Set("index", Json::Number(0));
  end_choice.Set("delta", std::move(end_delta));
  end_choice.Set("finish_reason",
                 Json::String(streamed ? FinishReasonName(streamed->reason)
                                       : "length"));
  Json end_choices = Json::Array();
  end_choices.Push(std::move(end_choice));
  Json end_chunk = Json::Object();
  end_chunk.Set("id", Json::String("chatcmpl-0"));
  end_chunk.Set("object", Json::String("chat.completion.chunk"));
  end_chunk.Set("model", Json::String(model_name));
  end_chunk.Set("choices", std::move(end_choices));
  WriteSse(writer, "", end_chunk, false);
  (void)writer.Write("data: [DONE]\n\n");
}

}  // namespace tessera::serve
