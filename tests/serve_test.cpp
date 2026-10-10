#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <thread>

#include "serve/auto_title.hpp"
#include "serve/http.hpp"
#include "serve/readiness.hpp"
#include "serve/respond.hpp"
#include "serve/session.hpp"
#include "serve/session_chat.hpp"
#include "serve/tools/tool_call.hpp"
#include "core/json.hpp"
#include "tessera/log.hpp"
#include "tessera/serve.hpp"

using tessera::serve::HttpRequest;
using tessera::core::Json;
using tessera::serve::ResponseWriter;
using tessera::serve::RunHttpServer;

TEST(ServeTest, JsonParseAndDump) {
  auto value = Json::Parse(R"({"a":1,"b":[true,false,null],"c":"x\n"})");
  ASSERT_NE(value, nullptr);
  ASSERT_TRUE(value->isObject());
  const Json* a = value->Find("a");
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a->AsNumber(), 1.0);
  const Json* b = value->Find("b");
  ASSERT_NE(b, nullptr);
  ASSERT_TRUE(b->isArray());
  ASSERT_EQ(b->AsArray().size(), 3u);
  EXPECT_TRUE(b->AsArray()[0].AsBool());
  const Json* c = value->Find("c");
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c->AsString(), "x\n");
  // A parse/dump round trip preserves the values.
  auto again = Json::Parse(value->Dump());
  ASSERT_NE(again, nullptr);
  EXPECT_EQ(again->Dump(), value->Dump());
}

TEST(ServeTest, JsonRejectsMalformed) {
  EXPECT_EQ(Json::Parse("{"), nullptr);
  EXPECT_EQ(Json::Parse("{\"a\":}"), nullptr);
  EXPECT_EQ(Json::Parse("[1,2"), nullptr);
  EXPECT_EQ(Json::Parse("nul"), nullptr);
  EXPECT_EQ(Json::Parse("1 2"), nullptr);
}

namespace {

int ConnectLocal(std::uint16_t port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

}  // namespace

// The HTTP server answers a POST and reports the request path back.
TEST(ServeTest, HttpLoopback) {
  constexpr std::uint16_t kPort = 18099;
  std::atomic<bool> stop{false};
  tessera::log::Diagnostics log;
  std::thread server([&stop, &log] {
    (void)RunHttpServer(
        "127.0.0.1", kPort,
        [](const HttpRequest& request, ResponseWriter& writer) {
          Json body = Json::Object();
          body.Set("path", Json::String(request.path));
          body.Set("body", Json::String(request.body));
          (void)writer.SendHeaders(200, "application/json");
          (void)writer.Write(body.Dump());
        },
        &stop, log);
  });
  int fd = -1;
  for (int i = 0; i < 200 && fd < 0; ++i) {
    fd = ConnectLocal(kPort);
    if (fd < 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  ASSERT_GE(fd, 0) << "server did not come up";
  const std::string request =
      "POST /v1/completions HTTP/1.1\r\nHost: x\r\n"
      "Content-Length: 7\r\nConnection: close\r\n\r\n{\"a\":1}";
  ASSERT_EQ(::send(fd, request.data(), request.size(), 0),
            static_cast<ssize_t>(request.size()));
  std::string response;
  char chunk[512];
  ssize_t got = 0;
  while ((got = ::recv(fd, chunk, sizeof(chunk), 0)) > 0) {
    response.append(chunk, static_cast<std::size_t>(got));
  }
  ::close(fd);
  EXPECT_NE(response.find("200 OK"), std::string::npos);
  EXPECT_NE(response.find("\"path\":\"/v1/completions\""), std::string::npos);
  EXPECT_NE(response.find("{\\\"a\\\":1}"), std::string::npos);
  stop.store(true);
  const int dummy = ConnectLocal(kPort);  // unblock accept
  if (dummy >= 0) {
    ::close(dummy);
  }
  server.join();
}

// A streaming handler writes chunked SSE frames.
TEST(ServeTest, HttpStreamingChunked) {
  constexpr std::uint16_t kPort = 18100;
  std::atomic<bool> stop{false};
  tessera::log::Diagnostics log;
  std::thread server([&stop, &log] {
    (void)RunHttpServer(
        "127.0.0.1", kPort,
        [](const HttpRequest&, ResponseWriter& writer) {
          (void)writer.SendHeaders(200, "text/event-stream", true);
          (void)writer.Write("data: one\n\n");
          (void)writer.Write("data: two\n\n");
        },
        &stop, log);
  });
  int fd = -1;
  for (int i = 0; i < 200 && fd < 0; ++i) {
    fd = ConnectLocal(kPort);
    if (fd < 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  ASSERT_GE(fd, 0);
  const std::string request =
      "GET /stream HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n";
  ASSERT_EQ(::send(fd, request.data(), request.size(), 0),
            static_cast<ssize_t>(request.size()));
  std::string response;
  char chunk[512];
  ssize_t got = 0;
  while ((got = ::recv(fd, chunk, sizeof(chunk), 0)) > 0) {
    response.append(chunk, static_cast<std::size_t>(got));
  }
  ::close(fd);
  EXPECT_NE(response.find("Transfer-Encoding: chunked"), std::string::npos);
  EXPECT_NE(response.find("data: one\n\n"), std::string::npos);
  EXPECT_NE(response.find("data: two\n\n"), std::string::npos);
  stop.store(true);
  const int dummy = ConnectLocal(kPort);
  if (dummy >= 0) {
    ::close(dummy);
  }
  server.join();
}

// Starting on a port another socket holds fails with an actionable
// line: the call, the endpoint, the errno text and what to do next.
TEST(ServeTest, HttpBindReportsPortInUse) {
  constexpr std::uint16_t kPort = 18101;
  const int blocker = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(blocker, 0);
  sockaddr_in held{};
  held.sin_family = AF_INET;
  held.sin_port = htons(kPort);
  held.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ASSERT_EQ(::bind(blocker, reinterpret_cast<sockaddr*>(&held), sizeof(held)),
            0);
  ASSERT_EQ(::listen(blocker, 1), 0);

  tessera::log::Diagnostics log;
  std::string lines;
  log.SetSink([&lines](tessera::log::Level, std::string_view prefix,
                       std::string_view message) {
    lines += std::string(prefix) + ": " + std::string(message) + "\n";
  });
  std::atomic<bool> stop{false};
  const auto started = RunHttpServer(
      "127.0.0.1", kPort, [](const HttpRequest&, ResponseWriter&) {}, &stop,
      log);
  ASSERT_FALSE(started.has_value());
  EXPECT_EQ(started.error(), tessera::StatusCode::DeviceError);
  EXPECT_NE(lines.find("bind on 127.0.0.1:18101 failed:"), std::string::npos);
  EXPECT_NE(lines.find("Address already in use"), std::string::npos);
  EXPECT_NE(lines.find("--port"), std::string::npos);
  ::close(blocker);
}

// One XML tool call parses to a named call with a JSON object.
TEST(ServeTest, ParseToolCallsSingle) {
  const std::string text =
      "<think>\nprivate\n</think>\n\n<tool_call>\n<function=read>\n"
      "<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>";
  std::string before;
  const std::vector<tessera::serve::ToolCall> calls =
      tessera::serve::ParseToolCalls(text, &before);
  ASSERT_EQ(calls.size(), 1u);
  EXPECT_EQ(calls[0].id, "call_0");
  EXPECT_EQ(calls[0].name, "read");
  EXPECT_EQ(calls[0].arguments, "{\"path\":\"/tmp/x\"}");
  EXPECT_TRUE(before.empty());
}

// Non-string parameter bodies keep their JSON types; surrounding text
// is reported as the text before the call.
TEST(ServeTest, ParseToolCallsTypedParams) {
  const std::string text =
      "looking it up\n<tool_call>\n<function=run>\n"
      "<parameter=count>\n3\n</parameter>\n"
      "<parameter=flags>\n{\"v\":true}\n</parameter>\n"
      "</function>\n</tool_call>\ntrailing";
  std::string before;
  const std::vector<tessera::serve::ToolCall> calls =
      tessera::serve::ParseToolCalls(text, &before);
  ASSERT_EQ(calls.size(), 1u);
  EXPECT_EQ(calls[0].arguments, "{\"count\":3,\"flags\":{\"v\":true}}");
  EXPECT_EQ(before, "looking it up");
}

// Several blocks parse in order; a truncated tail block is skipped.
TEST(ServeTest, ParseToolCallsMultiple) {
  const std::string text =
      "<tool_call>\n<function=a>\n</function>\n</tool_call>\n"
      "<tool_call>\n<function=b>\n<parameter=x>\n1\n</parameter>\n"
      "</function>\n</tool_call>\n<tool_call>\n<function=c>\n";
  std::string before;
  const std::vector<tessera::serve::ToolCall> calls =
      tessera::serve::ParseToolCalls(text, &before);
  ASSERT_EQ(calls.size(), 2u);
  EXPECT_EQ(calls[0].name, "a");
  EXPECT_EQ(calls[0].arguments, "{}");
  EXPECT_EQ(calls[1].id, "call_1");
  EXPECT_EQ(calls[1].name, "b");
}

// Malformed blocks are skipped without losing the valid ones.
TEST(ServeTest, ParseToolCallsSkipsMalformed) {
  const std::string text =
      "<tool_call>\nno function here\n</tool_call>\n"
      "<tool_call>\n<function=ok>\n</function>\n</tool_call>";
  std::string before;
  const std::vector<tessera::serve::ToolCall> calls =
      tessera::serve::ParseToolCalls(text, &before);
  ASSERT_EQ(calls.size(), 1u);
  EXPECT_EQ(calls[0].name, "ok");
}

// Plain text without blocks parses to no calls and the whole text.
TEST(ServeTest, ParseToolCallsPlainText) {
  std::string before;
  const std::vector<tessera::serve::ToolCall> calls =
      tessera::serve::ParseToolCalls("just an answer", &before);
  EXPECT_TRUE(calls.empty());
  EXPECT_EQ(before, "just an answer");
}

// The generation prompt ends with an open <think>, so the reply text
// may hold reasoning with no opener. It is still stripped.
TEST(ServeTest, ParseToolCallsStripsOpenThink) {
  const std::string text =
      "private reasoning\n</think>\n\n<tool_call>\n<function=read>\n"
      "</function>\n</tool_call>";
  std::string before;
  const std::vector<tessera::serve::ToolCall> calls =
      tessera::serve::ParseToolCalls(text, &before);
  ASSERT_EQ(calls.size(), 1u);
  EXPECT_EQ(calls[0].name, "read");
  EXPECT_TRUE(before.empty());
}

// The stripped reasoning is reported, so the client still sees the
// thinking block that never belongs in the reply content.
TEST(ServeTest, ParseToolCallsReportsReasoning) {
  const std::string text =
      "<think>\nprivate\n</think>\n\n<tool_call>\n<function=read>\n"
      "</function>\n</tool_call>";
  std::string before;
  std::string reasoning = "untouched";
  const std::vector<tessera::serve::ToolCall> calls =
      tessera::serve::ParseToolCalls(text, &before, &reasoning);
  ASSERT_EQ(calls.size(), 1u);
  EXPECT_TRUE(before.empty());
  EXPECT_EQ(reasoning, "private");
}

// Reasoning without an opener (the prompt ends with <think>) is
// reported too, and the answer text stays intact beside a call.
TEST(ServeTest, ParseToolCallsReportsOpenReasoning) {
  const std::string text =
      "private reasoning\n</think>\nthe answer\n<tool_call>\n<function=read>\n"
      "</function>\n</tool_call>";
  std::string before;
  std::string reasoning;
  const std::vector<tessera::serve::ToolCall> calls =
      tessera::serve::ParseToolCalls(text, &before, &reasoning);
  ASSERT_EQ(calls.size(), 1u);
  EXPECT_EQ(before, "the answer");
  EXPECT_EQ(reasoning, "private reasoning");
}

// Plain text reports no reasoning and keeps the whole text.
TEST(ServeTest, ParseToolCallsNoReasoningWithoutThink) {
  std::string before;
  std::string reasoning = "untouched";
  const std::vector<tessera::serve::ToolCall> calls =
      tessera::serve::ParseToolCalls("just an answer", &before, &reasoning);
  EXPECT_TRUE(calls.empty());
  EXPECT_EQ(before, "just an answer");
  EXPECT_TRUE(reasoning.empty());
}

// A think span without a closer stays in the text (an unfinished turn
// keeps its words); only closed spans count as reasoning.
TEST(ServeTest, ParseToolCallsKeepsUnclosedThink) {
  const std::string text = "answer <think>unfinished";
  std::string before;
  std::string reasoning = "untouched";
  const std::vector<tessera::serve::ToolCall> calls =
      tessera::serve::ParseToolCalls(text, &before, &reasoning);
  EXPECT_TRUE(calls.empty());
  EXPECT_EQ(before, text);
  EXPECT_TRUE(reasoning.empty());
}

// The effective tool list is absent, empty, or disabled by choice.
TEST(ServeTest, EffectiveToolsSelection) {
  std::string error;
  auto none = Json::Parse(R"({"messages":[]})");
  ASSERT_NE(none, nullptr);
  EXPECT_EQ(tessera::serve::EffectiveTools(*none, &error), nullptr);
  EXPECT_TRUE(error.empty());
  auto empty = Json::Parse(R"({"messages":[],"tools":[]})");
  ASSERT_NE(empty, nullptr);
  EXPECT_EQ(tessera::serve::EffectiveTools(*empty, &error), nullptr);
  auto off = Json::Parse(
      R"({"messages":[],"tools":[{"type":"function"}],"tool_choice":"none"})");
  ASSERT_NE(off, nullptr);
  EXPECT_EQ(tessera::serve::EffectiveTools(*off, &error), nullptr);
  auto bad = Json::Parse(R"({"messages":[],"tools":{}})");
  ASSERT_NE(bad, nullptr);
  EXPECT_EQ(tessera::serve::EffectiveTools(*bad, &error), nullptr);
  EXPECT_EQ(error, "tools must be an array");
  error.clear();
  auto choice = Json::Parse(
      R"({"messages":[],"tools":[{"type":"function"}],"tool_choice":7})");
  ASSERT_NE(choice, nullptr);
  EXPECT_EQ(tessera::serve::EffectiveTools(*choice, &error), nullptr);
  EXPECT_EQ(error, "tool_choice must be a string or an object");
}

// Assistant history calls convert JSON-string arguments to objects.
TEST(ServeTest, NormalizeToolHistoryConvertsArguments) {
  auto body = Json::Parse(
      R"({"messages":[
        {"role":"user","content":"hi"},
        {"role":"assistant","content":null,"tool_calls":[
          {"id":"call_0","type":"function",
           "function":{"name":"read","arguments":"{\"path\":\"/tmp/x\"}"}}]},
        {"role":"tool","content":"data","tool_call_id":"call_0"}]})");
  ASSERT_NE(body, nullptr);
  Json out = Json::Object();
  std::string error;
  ASSERT_TRUE(tessera::serve::NormalizeToolHistory(*body, &out, &error));
  ASSERT_TRUE(out.isArray());
  ASSERT_EQ(out.AsArray().size(), 3u);
  const Json* calls = out.AsArray()[1].Find("tool_calls");
  ASSERT_NE(calls, nullptr);
  ASSERT_EQ(calls->AsArray().size(), 1u);
  const Json* function = calls->AsArray()[0].Find("function");
  ASSERT_NE(function, nullptr);
  const Json* arguments = function->Find("arguments");
  ASSERT_NE(arguments, nullptr);
  ASSERT_TRUE(arguments->isObject());
  EXPECT_EQ(arguments->Find("path")->AsString(), "/tmp/x");
  EXPECT_EQ(out.AsArray()[1].Find("content")->type(), Json::Type::Null);
  EXPECT_EQ(out.AsArray()[2].Find("content")->AsString(), "data");
}

// Malformed history is rejected with a reason.
TEST(ServeTest, NormalizeToolHistoryRejectsMalformed) {
  Json out = Json::Object();
  std::string error;
  auto missing = Json::Parse(R"({})");
  ASSERT_NE(missing, nullptr);
  EXPECT_FALSE(
      tessera::serve::NormalizeToolHistory(*missing, &out, &error));
  EXPECT_EQ(error, "messages must be an array");
  error.clear();
  auto args = Json::Parse(
      R"({"messages":[{"role":"assistant","tool_calls":[
        {"function":{"name":"read","arguments":"oops"}}]}]})");
  ASSERT_NE(args, nullptr);
  EXPECT_FALSE(tessera::serve::NormalizeToolHistory(*args, &out, &error));
  EXPECT_EQ(error, "tool call arguments must parse to an object");
  error.clear();
  auto tool = Json::Parse(
      R"({"messages":[{"role":"tool","content":42}]})");
  ASSERT_NE(tool, nullptr);
  EXPECT_FALSE(tessera::serve::NormalizeToolHistory(*tool, &out, &error));
  EXPECT_EQ(error, "tool messages need string content");
}

// Sessions create in order with default titles, list, fetch and delete.
TEST(ServeTest, SessionStoreCrud) {
  tessera::serve::SessionStore store;
  auto first = store.Create({});
  auto second = store.Create("custom");
  EXPECT_EQ(first->Id(), "s1");
  EXPECT_EQ(second->Id(), "s2");
  auto infos = store.List();
  ASSERT_EQ(infos.size(), 2u);
  EXPECT_EQ(infos[0].id, "s1");
  EXPECT_EQ(infos[0].title, "New chat");
  EXPECT_EQ(infos[0].message_count, 0u);
  EXPECT_EQ(infos[1].title, "custom");
  auto view = store.View("s1");
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->title, "New chat");
  EXPECT_TRUE(view->messages.empty());
  EXPECT_FALSE(store.View("s9").has_value());
  EXPECT_TRUE(store.Remove("s1"));
  EXPECT_FALSE(store.Remove("s1"));
  EXPECT_EQ(store.List().size(), 1u);
}

// History appends, trailing assistant turns pop for retry, and the
// turn guard admits one generation at a time.
TEST(ServeTest, SessionHistoryAndTurnGuard) {
  tessera::serve::SessionStore store;
  auto session = store.Create({});
  session->Append({"user", "hi", {}, {}, {}, false, {}, 0});
  session->Append({"assistant", "hello", "thinking", {}, {}, false, {}, 0});
  session->Append({"assistant", "again", {}, {}, {}, false, {}, 0});
  EXPECT_EQ(session->View().messages.size(), 3u);
  EXPECT_EQ(session->PopTrailingAssistant(), 2u);
  EXPECT_EQ(session->View().messages.size(), 1u);
  EXPECT_EQ(session->PopTrailingAssistant(), 0u);
  EXPECT_TRUE(session->TryBegin());
  EXPECT_FALSE(session->TryBegin());
  EXPECT_TRUE(session->PollControl());
  session->RequestStop();
  EXPECT_FALSE(session->PollControl());
  session->End();
  EXPECT_TRUE(session->TryBegin());
  session->SetPaused(true);
  session->RequestStop();
  EXPECT_FALSE(session->PollControl());
  session->End();
  auto view = store.View(session->Id());
  ASSERT_TRUE(view.has_value());
  EXPECT_FALSE(view->busy);
}

// Titles come from the first user line, capped and trimmed.
TEST(ServeTest, SessionTitleFromText) {
  EXPECT_EQ(tessera::serve::TitleFromText("  hello there  "), "hello there");
  EXPECT_EQ(tessera::serve::TitleFromText("first\nsecond"), "first");
  EXPECT_EQ(tessera::serve::TitleFromText(""), "New chat");
  EXPECT_EQ(tessera::serve::TitleFromText(std::string(100, 'x')).size(), 48u);
}

// Live think splitting: pieces across tag boundaries still divide
// reasoning from content, and deltas concatenate exactly.
TEST(ServeTest, ThinkStreamerSplitsLive) {
  tessera::serve::ThinkStreamer streamer(/*think_expected=*/true);
  std::string reasoning;
  std::string content;
  for (std::string_view piece : {"<th", "ink>id", "ea</th", "ink>ans", "wer"}) {
    const auto deltas = streamer.Push(piece);
    reasoning += deltas.reasoning;
    content += deltas.content;
  }
  const auto tail = streamer.Finish();
  reasoning += tail.reasoning;
  content += tail.content;
  EXPECT_EQ(reasoning, "idea");
  EXPECT_EQ(content, "answer");
}

// Without a think-mode prompt everything streams as content.
TEST(ServeTest, ThinkStreamerPassesPlainText) {
  tessera::serve::ThinkStreamer streamer(/*think_expected=*/false);
  std::string content;
  for (std::string_view piece : {"hel", "lo"}) {
    const auto deltas = streamer.Push(piece);
    EXPECT_TRUE(deltas.reasoning.empty());
    content += deltas.content;
  }
  const auto tail = streamer.Finish();
  content += tail.content;
  EXPECT_TRUE(tail.reasoning.empty());
  EXPECT_EQ(content, "hello");
}

// Expected thinking streams live as reasoning: nothing waits for the
// closer, and Finish adds nothing once everything flowed.
TEST(ServeTest, ThinkStreamerStreamsLiveThinking) {
  tessera::serve::ThinkStreamer streamer(/*think_expected=*/true);
  auto first = streamer.Push("The user said hi. ");
  EXPECT_EQ(first.reasoning, "The user said hi. ");
  EXPECT_TRUE(first.content.empty());
  auto second = streamer.Push("Be brief.");
  EXPECT_EQ(second.reasoning, "Be brief.");
  EXPECT_TRUE(second.content.empty());
  const auto tail = streamer.Finish();
  EXPECT_TRUE(tail.reasoning.empty());
  EXPECT_TRUE(tail.content.empty());
}

// A trailing tag fragment is withheld until it resolves, then the
// closer vanishes as markup while the answer flows as content.
TEST(ServeTest, ThinkStreamerWithholdsTagFragmentLive) {
  tessera::serve::ThinkStreamer streamer(/*think_expected=*/true);
  auto first = streamer.Push("idea</thi");
  EXPECT_EQ(first.reasoning, "idea");
  EXPECT_TRUE(first.content.empty());
  auto second = streamer.Push("nk>answer");
  EXPECT_TRUE(second.reasoning.empty());
  EXPECT_EQ(second.content, "answer");
}

// An unclosed think block never strands text: with no closer it all
// stays thinking, including the Finish remainder.
TEST(ServeTest, ThinkStreamerFlushesUnclosedThink) {
  tessera::serve::ThinkStreamer streamer(/*think_expected=*/true);
  auto pushed = streamer.Push("abc");
  EXPECT_EQ(pushed.reasoning, "abc");
  EXPECT_TRUE(pushed.content.empty());
  const auto tail = streamer.Finish();
  EXPECT_TRUE(tail.reasoning.empty());
  EXPECT_TRUE(tail.content.empty());
}

// Live thinking over realistic word pieces: every pre-close delta
// carries reasoning and no content, the closer itself streams nothing,
// and the answer flows as content.
TEST(ServeTest, ThinkStreamerStreamsLiveWordPieces) {
  tessera::serve::ThinkStreamer streamer(/*think_expected=*/true);
  std::string reasoning;
  std::string content;
  for (std::string_view piece : {"We", " need", " to", " answer", "."}) {
    const auto deltas = streamer.Push(piece);
    EXPECT_FALSE(deltas.reasoning.empty());
    EXPECT_TRUE(deltas.content.empty());
    reasoning += deltas.reasoning;
    content += deltas.content;
  }
  const auto close = streamer.Push("</think>");
  EXPECT_TRUE(close.reasoning.empty());
  EXPECT_TRUE(close.content.empty());
  for (std::string_view piece : {"Hi", "!"}) {
    const auto deltas = streamer.Push(piece);
    EXPECT_TRUE(deltas.reasoning.empty());
    EXPECT_FALSE(deltas.content.empty());
    reasoning += deltas.reasoning;
    content += deltas.content;
  }
  const auto tail = streamer.Finish();
  reasoning += tail.reasoning;
  content += tail.content;
  EXPECT_EQ(reasoning, "We need to answer.");
  EXPECT_EQ(content, "Hi!");
}

// A mid-stream opener falls back to withholding; a cutoff there still
// flushes everything as reasoning, never as answer text.
TEST(ServeTest, ThinkStreamerFlushesOpenerFallbackAsReasoning) {
  tessera::serve::ThinkStreamer streamer(/*think_expected=*/true);
  auto pushed = streamer.Push("foo <think> bar");
  EXPECT_TRUE(pushed.reasoning.empty());
  EXPECT_TRUE(pushed.content.empty());
  const auto tail = streamer.Finish();
  EXPECT_EQ(tail.reasoning, "foo <think> bar");
  EXPECT_TRUE(tail.content.empty());
}

// An unclosed span is withheld, never leaked: nothing streams until
// it closes (or Finish flushes it), then each word lands exactly once.
TEST(ServeTest, ThinkStreamerWithholdsUnclosedSpan) {
  tessera::serve::ThinkStreamer streamer(/*think_expected=*/false);
  auto first = streamer.Push("<think>abc");
  EXPECT_TRUE(first.reasoning.empty());
  EXPECT_TRUE(first.content.empty());
  auto second = streamer.Push("def</think>ghi");
  EXPECT_EQ(second.reasoning, "abcdef");
  EXPECT_EQ(second.content, "ghi");
  const auto tail = streamer.Finish();
  EXPECT_TRUE(tail.reasoning.empty());
  EXPECT_TRUE(tail.content.empty());
}

// Deltas always concatenate to the post-hoc split, whatever the
// piece boundaries (fixed seed). Non-expected mode: leading plain
// text is content here (expected mode treats pre-close text as
// thinking, like the reference runtimes).
TEST(ServeTest, ThinkStreamerMatchesPostHocSplit) {
  const std::string text =
      "lead <think>deep thought</think> middle <think>more</think> tail";
  std::mt19937 rng(11);
  for (int round = 0; round < 20; ++round) {
    tessera::serve::ThinkStreamer streamer(/*think_expected=*/false);
    std::string reasoning;
    std::string content;
    std::size_t pos = 0;
    while (pos < text.size()) {
      const std::size_t len = 1 + rng() % 7;
      const auto deltas =
          streamer.Push(std::string_view(text).substr(pos, len));
      reasoning += deltas.reasoning;
      content += deltas.content;
      pos += len;
    }
    const auto tail = streamer.Finish();
    reasoning += tail.reasoning;
    content += tail.content;
    std::string before;
    std::string ref_reasoning;
    tessera::serve::ParseToolCalls(text, &before, &ref_reasoning);
    EXPECT_EQ(reasoning, ref_reasoning);
    EXPECT_EQ(content, before);
  }
}

// The template ends the think prompt with "<think>\n": the
// streamer must still withhold thinking until its closer.
TEST(ServeTest, ThinkExpectedAllowsTrailingWhitespace) {
  using tessera::serve::ThinkExpected;
  EXPECT_TRUE(ThinkExpected("<|im_start|>assistant\n<think>", true));
  EXPECT_TRUE(ThinkExpected("<|im_start|>assistant\n<think>\n", true));
  EXPECT_TRUE(ThinkExpected("<think> \t\r\n", true));
  EXPECT_FALSE(ThinkExpected("<|im_start|>assistant\n<think>\n", false));
  EXPECT_FALSE(
      ThinkExpected("<|im_start|>assistant\n<think>\n\n</think>\n\n", true));
  EXPECT_FALSE(ThinkExpected("plain prompt", true));
}

// Regression: with a "<think>\n" prompt the greeting thinking never
// leaks into the content and never streams twice.
TEST(ServeTest, ThinkStreamerWithholdsLeadingThinkNewline) {
  using tessera::serve::ThinkExpected;
  using tessera::serve::ThinkStreamer;
  ASSERT_TRUE(ThinkExpected("<|im_start|>assistant\n<think>\n", true));
  ThinkStreamer streamer(
      ThinkExpected("<|im_start|>assistant\n<think>\n", true));
  std::string reasoning;
  std::string content;
  for (std::string_view piece :
       {"The user said hi. ", "Be brief.</think>", "Hi", "!", " How"}) {
    const auto deltas = streamer.Push(piece);
    reasoning += deltas.reasoning;
    content += deltas.content;
  }
  const auto tail = streamer.Finish();
  reasoning += tail.reasoning;
  content += tail.content;
  EXPECT_EQ(reasoning, "The user said hi. Be brief.");
  EXPECT_EQ(content, "Hi! How");
  EXPECT_EQ(content.find("The user said"), std::string::npos);
}

// The default is unlimited (0): an unspecified request fills the
// remaining context instead of stopping after a fixed budget.
TEST(ServeTest, DefaultMaxCompletionTokensIsZero) {
  EXPECT_EQ(tessera::kDefaultMaxCompletionTokens, 0u);
  EXPECT_EQ(tessera::ServeOptions{}.default_max_completion_tokens,
            tessera::kDefaultMaxCompletionTokens);
  auto empty = Json::Parse(R"({})");
  ASSERT_NE(empty, nullptr);
  EXPECT_EQ(tessera::serve::MaxCompletionTokensFrom(*empty, 7u), 7u);
  auto capped = Json::Parse(R"({"max_completion_tokens":0})");
  ASSERT_NE(capped, nullptr);
  EXPECT_EQ(tessera::serve::MaxCompletionTokensFrom(*capped, 7u), 7u);
  auto explicit_count = Json::Parse(R"({"max_completion_tokens":64})");
  ASSERT_NE(explicit_count, nullptr);
  EXPECT_EQ(tessera::serve::MaxCompletionTokensFrom(*explicit_count, 7u),
            64u);
  // The retired name is ignored: an old client falls back to the default.
  auto retired = Json::Parse(R"({"max_tokens":64})");
  ASSERT_NE(retired, nullptr);
  EXPECT_EQ(tessera::serve::MaxCompletionTokensFrom(*retired, 7u), 7u);
}

// The thinking budget defaults to unlimited and needs a positive value.
TEST(ServeTest, MaxThinkingTokensFromReadsBudget) {
  auto empty = Json::Parse(R"({})");
  ASSERT_NE(empty, nullptr);
  EXPECT_EQ(tessera::serve::MaxThinkingTokensFrom(*empty), 0u);
  auto budget = Json::Parse(R"({"max_thinking_tokens":16})");
  ASSERT_NE(budget, nullptr);
  EXPECT_EQ(tessera::serve::MaxThinkingTokensFrom(*budget), 16u);
  auto zero = Json::Parse(R"({"max_thinking_tokens":0})");
  ASSERT_NE(zero, nullptr);
  EXPECT_EQ(tessera::serve::MaxThinkingTokensFrom(*zero), 0u);
}

// The retired field warns with its replacement, and only then.
TEST(ServeTest, WarnRetiredMaxTokensNamesReplacement) {
  tessera::log::Diagnostics log;
  std::string lines;
  log.SetSink([&lines](tessera::log::Level, std::string_view prefix,
                       std::string_view message) {
    lines += std::string(prefix) + ": " + std::string(message) + "\n";
  });
  auto retired = Json::Parse(R"({"max_tokens":64})");
  ASSERT_NE(retired, nullptr);
  tessera::serve::WarnRetiredMaxTokens(log, *retired);
  EXPECT_NE(lines.find("max_completion_tokens"), std::string::npos);
  lines.clear();
  auto current = Json::Parse(R"({"max_completion_tokens":64})");
  ASSERT_NE(current, nullptr);
  tessera::serve::WarnRetiredMaxTokens(log, *current);
  EXPECT_TRUE(lines.empty());
  auto empty = Json::Parse(R"({})");
  ASSERT_NE(empty, nullptr);
  tessera::serve::WarnRetiredMaxTokens(log, *empty);
  EXPECT_TRUE(lines.empty());
}

// The served page carries the same default: one constant, every
// surface. Pages without the placeholder pass through untouched.
TEST(ServeTest, InjectWebDefaultsFillsMaxTokens) {
  EXPECT_EQ(tessera::serve::InjectWebDefaults("no placeholders here"),
            "no placeholders here");
  const std::string want = "<input value=\"" +
                           std::to_string(tessera::kDefaultMaxCompletionTokens) +
                           "\">";
  EXPECT_EQ(tessera::serve::InjectWebDefaults(
                R"(<input value="@TESSERA_DEFAULT_MAX_COMPLETION_TOKENS@">)"),
            want);
}

// A closed peer reads as gone; a live or data-bearing one does not.
// The polls only bound a broken kernel, they never gate a pass.
TEST(ServeTest, PeerGoneDetectsClosedConnection) {
  int pair[2] = {-1, -1};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, pair), 0);
  tessera::serve::ResponseWriter writer(pair[0]);
  EXPECT_FALSE(writer.IsPeerGone());
  const auto poll_gone = [&](bool want, int tries) {
    for (int i = 0; i < tries; ++i) {
      if (writer.IsPeerGone() == want) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return writer.IsPeerGone() == want;
  };
  ASSERT_EQ(::send(pair[1], "x", 1, 0), 1);
  EXPECT_TRUE(poll_gone(false, 200));
  char drop = 0;
  ASSERT_EQ(::recv(pair[0], &drop, sizeof(drop), 0), 1);
  ::close(pair[1]);
  EXPECT_TRUE(poll_gone(true, 200));
  ::close(pair[0]);
}
// streaming generation never blocks control requests. The guard time
// only bounds a regression hang, it never gates a pass.
TEST(ServeTest, HttpServesConcurrentConnections) {
  constexpr std::uint16_t kPort = 18102;
  std::atomic<bool> stop{false};
  std::atomic<bool> release{false};
  tessera::log::Diagnostics log;
  std::thread server([&stop, &release, &log] {
    (void)RunHttpServer(
        "127.0.0.1", kPort,
        [&](const HttpRequest& request, ResponseWriter& writer) {
          if (request.path == "/block") {
            const auto start = std::chrono::steady_clock::now();
            while (!release.load()) {
              if (std::chrono::steady_clock::now() - start >
                  std::chrono::seconds(20)) {
                break;
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
          } else if (request.path == "/release") {
            release.store(true);
          }
          (void)writer.SendHeaders(200, "application/json");
          (void)writer.Write(release.load() ? R"({"ok":true})"
                                            : R"({"ok":false})");
        },
        &stop, log);
  });
  const auto get = [&](const char* target) {
    int fd = -1;
    for (int i = 0; i < 200 && fd < 0; ++i) {
      fd = ConnectLocal(kPort);
      if (fd < 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    if (fd < 0) {
      return std::string();
    }
    const std::string request = std::string("GET ") + target +
                                " HTTP/1.1\r\nHost: x\r\nConnection: close"
                                "\r\n\r\n";
    if (::send(fd, request.data(), request.size(), 0) < 0) {
      ::close(fd);
      return std::string();
    }
    std::string response;
    char chunk[512];
    ssize_t got = 0;
    while ((got = ::recv(fd, chunk, sizeof(chunk), 0)) > 0) {
      response.append(chunk, static_cast<std::size_t>(got));
    }
    ::close(fd);
    return response;
  };
  // The server needs a moment to listen; both requests retry their
  // connects instead of sleeping a fixed span.
  std::string blocked;
  std::thread waiter([&] { blocked = get("/block"); });
  std::string released = get("/release");
  waiter.join();
  EXPECT_NE(released.find("\"ok\":true"), std::string::npos);
  EXPECT_NE(blocked.find("\"ok\":true"), std::string::npos);
  stop.store(true);
  const int dummy = ConnectLocal(kPort);  // unblock accept
  if (dummy >= 0) {
    ::close(dummy);
  }
  server.join();
}

// session_id reads the OpenAI session field; anything else is stateless.
TEST(ServeTest, SessionIdFromReadsSessionField) {
  auto present = Json::Parse(R"({"session_id":"s3","messages":[]})");
  ASSERT_NE(present, nullptr);
  EXPECT_EQ(tessera::serve::SessionIdFrom(*present), "s3");
  auto absent = Json::Parse(R"({"messages":[]})");
  ASSERT_NE(absent, nullptr);
  EXPECT_TRUE(tessera::serve::SessionIdFrom(*absent).empty());
  auto typed = Json::Parse(R"({"session_id":7})");
  ASSERT_NE(typed, nullptr);
  EXPECT_TRUE(tessera::serve::SessionIdFrom(*typed).empty());
  auto empty = Json::Parse(R"({"session_id":""})");
  ASSERT_NE(empty, nullptr);
  EXPECT_TRUE(tessera::serve::SessionIdFrom(*empty).empty());
}

// Prompt-row equality ignores response metadata, so a resent OpenAI
// history (no reasoning) still matches the stored turns.
TEST(ServeTest, SamePromptRowIgnoresResponseMetadata) {
  const tessera::serve::SessionMessage stored{"assistant", "hi", "thinking",
                                              {}, {}, false, {}, 0};
  const tessera::serve::SessionMessage resent{"assistant", "hi", {}, {}, {},
                                              true, {}, 0};
  EXPECT_TRUE(tessera::serve::SamePromptRow(stored, resent));
  const tessera::serve::SessionMessage other{"assistant", "bye", "thinking",
                                             {}, {}, false, {}, 0};
  EXPECT_FALSE(tessera::serve::SamePromptRow(stored, other));
  const tessera::serve::SessionMessage role{"user", "hi", {}, {}, {}, false, {}, 0};
  EXPECT_FALSE(tessera::serve::SamePromptRow(stored, role));
  const tessera::serve::SessionMessage tools{"assistant", "hi", "thinking",
                                             "[{}]", {}, false, {}, 0};
  EXPECT_FALSE(tessera::serve::SamePromptRow(stored, tools));
  const tessera::serve::SessionMessage timed{"assistant", "hi", "thinking",
                                             {}, {}, false, {10, 50, 2000},
                                             12345};
  EXPECT_TRUE(tessera::serve::SamePromptRow(stored, timed));
}

// A full-history resend converges without duplicating the prefix.
TEST(ServeTest, SyncHistoryMergesFullResend) {
  tessera::serve::SessionStore store;
  auto session = store.Create({});
  session->Append({"user", "hi", {}, {}, {}, false, {}, 0});
  session->Append({"assistant", "hello", "thinking", {}, {}, false, {}, 0});
  std::vector<tessera::serve::SessionMessage> incoming{
      {"user", "hi", {}, {}, {}, false, {}, 0},
      {"assistant", "hello", {}, {}, {}, false, {}, 0},
      {"user", "again", {}, {}, {}, false, {}, 0},
  };
  session->SyncHistory(std::move(incoming));
  const auto view = session->View();
  ASSERT_EQ(view.messages.size(), 3u);
  EXPECT_EQ(view.messages[0].content, "hi");
  EXPECT_EQ(view.messages[1].content, "hello");
  EXPECT_EQ(view.messages[1].reasoning, "thinking");
  EXPECT_EQ(view.messages[2].content, "again");
}

// A delta-only client (new turns without the history) appends cleanly.
TEST(ServeTest, SyncHistoryAppendsDeltaOnly) {
  tessera::serve::SessionStore store;
  auto session = store.Create({});
  session->Append({"user", "hi", {}, {}, {}, false, {}, 0});
  std::vector<tessera::serve::SessionMessage> incoming{
      {"user", "next", {}, {}, {}, false, {}, 0},
  };
  session->SyncHistory(std::move(incoming));
  const auto view = session->View();
  ASSERT_EQ(view.messages.size(), 2u);
  EXPECT_EQ(view.messages[1].content, "next");
}
// A divergent history truncates the stored tail, like a branch switch.
TEST(ServeTest, SyncHistoryTruncatesOnDiverge) {

  tessera::serve::SessionStore store;
  auto session = store.Create({});
  session->Append({"user", "hi", {}, {}, {}, false, {}, 0});
  session->Append({"assistant", "hello", {}, {}, {}, false, {}, 0});
  std::vector<tessera::serve::SessionMessage> incoming{
      {"user", "hi", {}, {}, {}, false, {}, 0},
      {"assistant", "changed", {}, {}, {}, false, {}, 0},
  };
  session->SyncHistory(std::move(incoming));
  const auto view = session->View();
  ASSERT_EQ(view.messages.size(), 2u);
  EXPECT_EQ(view.messages[1].content, "changed");
}

// An empty incoming history leaves the stored turns alone.
TEST(ServeTest, SyncHistoryKeepsHistoryOnEmpty) {
  tessera::serve::SessionStore store;
  auto session = store.Create({});
  session->Append({"user", "hi", {}, {}, {}, false, {}, 0});
  std::vector<tessera::serve::SessionMessage> incoming;
  session->SyncHistory(std::move(incoming));
  const auto view = session->View();
  ASSERT_EQ(view.messages.size(), 1u);
  EXPECT_EQ(view.messages[0].content, "hi");
}

// OpenAI histories parse to stored turns; null content reads as empty
// (tool-call-only assistant messages), tool turns round-trip.
TEST(ServeTest, SessionMessagesFromJsonParsesHistory) {
  auto body = Json::Parse(
      R"({"messages":[
        {"role":"user","content":"hi"},
        {"role":"assistant","content":null,"tool_calls":[
          {"id":"call_0","type":"function",
           "function":{"name":"read","arguments":"{}"}}]},
        {"role":"tool","content":"data","tool_call_id":"call_0"}]})");
  ASSERT_NE(body, nullptr);
  const Json* messages = body->Find("messages");
  ASSERT_NE(messages, nullptr);
  std::vector<tessera::serve::SessionMessage> out;
  std::string error;
  ASSERT_TRUE(
      tessera::serve::SessionMessagesFromJson(*messages, &out, &error));
  EXPECT_TRUE(error.empty());
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0].role, "user");
  EXPECT_EQ(out[0].content, "hi");
  EXPECT_TRUE(out[1].content.empty());
  EXPECT_FALSE(out[1].tool_calls_json.empty());
  EXPECT_EQ(out[2].tool_call_id, "call_0");
}

// Non-string content cannot round-trip through the text history.
TEST(ServeTest, SessionMessagesFromJsonRejectsBlocks) {
  auto body = Json::Parse(
      R"({"messages":[{"role":"user","content":[{"type":"text"}]}]})");
  ASSERT_NE(body, nullptr);
  std::vector<tessera::serve::SessionMessage> out;
  std::string error;
  EXPECT_FALSE(
      tessera::serve::SessionMessagesFromJson(*body->Find("messages"), &out,
                                              &error));
  EXPECT_EQ(error, "session history needs string content");
  auto missing = Json::Parse(R"({})");
  ASSERT_NE(missing, nullptr);
  EXPECT_FALSE(tessera::serve::SessionMessagesFromJson(*missing, &out, &error));
}

// History survives a JSON round trip, including tool turns.
TEST(ServeTest, SessionHistoryJsonRoundTripsTools) {
  tessera::serve::SessionStore store;
  auto session = store.Create({});
  session->Append({"user", "hi", {}, {}, {}, false, {}, 0});
  session->Append({"assistant", "", "thinking", R"([{"id":"call_0"}])", {},
                   false});
  session->Append({"tool", "data", {}, {}, "call_0", false});
  const Json history =
      tessera::serve::SessionHistoryJson(session->View());
  ASSERT_TRUE(history.isArray());
  ASSERT_EQ(history.AsArray().size(), 3u);
  std::vector<tessera::serve::SessionMessage> back;
  std::string error;
  ASSERT_TRUE(
      tessera::serve::SessionMessagesFromJson(history, &back, &error));
  ASSERT_EQ(back.size(), 3u);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(tessera::serve::SamePromptRow(
        session->View().messages[i], back[i]));
  }
  EXPECT_EQ(back[1].tool_calls_json, R"([{"id":"call_0"}])");
  EXPECT_EQ(back[2].tool_call_id, "call_0");
}

// Session JSON exposes tool turns for API clients; the info entry
// feeds the sidebar and GET /slots.
TEST(ServeTest, SessionJsonExposesToolTurns) {
  tessera::serve::SessionMessage turn{"assistant", "", "thinking",
                                      R"([{"id":"call_0"}])", {}, false,
                                      {}, 0};
  tessera::serve::SessionView view{"s1", "title", {turn}, true, false};
  const Json json = tessera::serve::SessionJson(view);
  const Json* messages = json.Find("messages");
  ASSERT_NE(messages, nullptr);
  ASSERT_EQ(messages->AsArray().size(), 1u);
  const Json* calls = messages->AsArray()[0].Find("tool_calls");
  ASSERT_NE(calls, nullptr);
  ASSERT_TRUE(calls->isArray());
  const Json info =
      tessera::serve::SessionInfoJson({"s1", "title", 1, true});
  EXPECT_EQ(info.Find("id")->AsString(), "s1");
  EXPECT_EQ(info.Find("title")->AsString(), "title");
  EXPECT_EQ(info.Find("message_count")->AsNumber(), 1.0);
  EXPECT_TRUE(info.Find("busy")->AsBool());
}

// The shared turn guard ends the turn when it drops.
TEST(ServeTest, SessionTurnGuardReleases) {
  tessera::serve::SessionStore store;
  auto session = store.Create({});
  EXPECT_TRUE(session->TryBegin());
  {
    tessera::serve::SessionTurn guard;
    guard.session = session;
    EXPECT_FALSE(session->TryBegin());
  }
  EXPECT_TRUE(session->TryBegin());
  session->End();
}

// A free device grants the queue immediately.
TEST(ServeTest, WaitForGpuAcquiresFreeDevice) {
  int pair[2] = {-1, -1};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, pair), 0);
  tessera::serve::ResponseWriter writer(pair[0]);
  std::mutex generation;
  auto slot = tessera::serve::WaitForGpu(generation, writer);
  ASSERT_TRUE(slot.has_value());
  EXPECT_TRUE(slot->owns_lock());
  slot.reset();
  ::close(pair[0]);
  ::close(pair[1]);
}

// A waiter behind a held device proceeds once it frees.
TEST(ServeTest, WaitForGpuQueuesBehindHeldDevice) {
  int pair[2] = {-1, -1};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, pair), 0);
  tessera::serve::ResponseWriter writer(pair[0]);
  std::mutex generation;
  std::unique_lock<std::mutex> held(generation);
  std::optional<std::unique_lock<std::mutex>> slot;
  std::thread waiter([&] { slot = tessera::serve::WaitForGpu(generation,
                                                             writer); });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  held.unlock();
  waiter.join();
  ASSERT_TRUE(slot.has_value());
  EXPECT_TRUE(slot->owns_lock());
  slot.reset();
  ::close(pair[0]);
  ::close(pair[1]);
}

// A peer that disconnects while queued gives up its place.
TEST(ServeTest, WaitForGpuAbortsWhenPeerGone) {
  int pair[2] = {-1, -1};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, pair), 0);
  std::mutex generation;
  std::unique_lock<std::mutex> held(generation);
  ::close(pair[1]);  // the peer goes away before queueing
  tessera::serve::ResponseWriter writer(pair[0]);
  std::optional<std::unique_lock<std::mutex>> slot;
  std::thread waiter([&] { slot = tessera::serve::WaitForGpu(generation,
                                                             writer); });
  waiter.join();
  EXPECT_FALSE(slot.has_value());
  held.unlock();
  ::close(pair[0]);
}

// Turn speed is completions per generation second; unknown stays zero.
TEST(ServeTest, TurnStatsTokensPerSecond) {
  const tessera::serve::TurnStats known{128, 50, 2000};
  EXPECT_DOUBLE_EQ(known.TokensPerSecond(), 25.0);
  EXPECT_DOUBLE_EQ(tessera::serve::TurnStats{}.TokensPerSecond(), 0.0);
  EXPECT_DOUBLE_EQ((tessera::serve::TurnStats{128, 50, 0}).TokensPerSecond(),
                   0.0);
  EXPECT_DOUBLE_EQ((tessera::serve::TurnStats{128, 0, 2000}).TokensPerSecond(),
                   0.0);
}

// The clock helper reports whole millis between two readings.
TEST(ServeTest, MillisBetweenMeasuresElapsed) {
  const auto point = std::chrono::steady_clock::now();
  EXPECT_EQ(tessera::serve::MillisBetween(point, point), 0);
  EXPECT_GE(tessera::serve::MillisBetween(
                point, point + std::chrono::milliseconds(7)),
            7);
}

// Appends stamp the store time unless the message carries one.
TEST(ServeTest, AppendStampsCreatedMillis) {
  tessera::serve::SessionStore store;
  auto session = store.Create({});
  session->Append({"user", "hi", {}, {}, {}, false, {}, 0});
  const auto stamped = session->View();
  ASSERT_EQ(stamped.messages.size(), 1u);
  EXPECT_NE(stamped.messages[0].created_ms, 0u);
  session->Append({"user", "again", {}, {}, {}, false, {}, 999});
  EXPECT_EQ(session->View().messages[1].created_ms, 999u);
}

// Sync keeps stored stamps and stamps new arrivals.
TEST(ServeTest, SyncHistoryStampsArrivals) {
  tessera::serve::SessionStore store;
  auto session = store.Create({});
  session->Append({"user", "hi", {}, {}, {}, false, {}, 4242});
  std::vector<tessera::serve::SessionMessage> incoming{
      {"user", "hi", {}, {}, {}, false, {}, 0},
      {"assistant", "hello", {}, {}, {}, false, {}, 0},
  };
  session->SyncHistory(std::move(incoming));
  const auto view = session->View();
  ASSERT_EQ(view.messages.size(), 2u);
  EXPECT_EQ(view.messages[0].created_ms, 4242u);
  EXPECT_NE(view.messages[1].created_ms, 0u);
}

// Session JSON carries the per-message timestamp and turn speed.
TEST(ServeTest, SessionJsonExposesMessageStats) {
  tessera::serve::TurnStats stats{128, 50, 2000};
  tessera::serve::SessionMessage turn{"assistant", "hello", {}, {}, {},
                                      false, stats, 1700000000000u};
  tessera::serve::SessionView view{"s1", "title", {turn}, false, false};
  const Json json = tessera::serve::SessionJson(view);
  const Json* messages = json.Find("messages");
  ASSERT_NE(messages, nullptr);
  ASSERT_EQ(messages->AsArray().size(), 1u);
  const Json& first = messages->AsArray()[0];
  EXPECT_DOUBLE_EQ(first.Find("created_ms")->AsNumber(), 1700000000000.0);
  EXPECT_DOUBLE_EQ(first.Find("prompt_tokens")->AsNumber(), 128.0);
  EXPECT_DOUBLE_EQ(first.Find("completion_tokens")->AsNumber(), 50.0);
  EXPECT_DOUBLE_EQ(first.Find("tokens_per_second")->AsNumber(), 25.0);
}

// The title prompt carries the first user text with thinking off.
TEST(ServeTest, TitlePromptBodyDisablesThinking) {
  const Json body = tessera::serve::TitlePromptBody("hello there");
  const Json* messages = body.Find("messages");
  ASSERT_NE(messages, nullptr);
  ASSERT_TRUE(messages->isArray());
  ASSERT_EQ(messages->AsArray().size(), 1u);
  const Json& first = messages->AsArray()[0];
  EXPECT_EQ(first.Find("role")->AsString(), "user");
  EXPECT_NE(first.Find("content")->AsString().find("hello there"),
            std::string::npos);
  const Json* thinking = body.Find("enable_thinking");
  ASSERT_NE(thinking, nullptr);
  EXPECT_FALSE(thinking->AsBool());
}

// Serve options default to model-generated titles.
TEST(ServeTest, ServeOptionsDefaultAutoTitle) {
  EXPECT_TRUE(tessera::ServeOptions{}.auto_title);
}

// A fresh readiness starts loading with no model, so /health answers 503.
TEST(ServeTest, ReadinessStartsLoading) {
  tessera::serve::Readiness readiness;
  const auto status = readiness.Get();
  EXPECT_EQ(status.phase, tessera::serve::Readiness::Phase::kLoading);
  EXPECT_EQ(status.model, nullptr);
  EXPECT_EQ(status.tokenizer, nullptr);
}

// Loading, ready and failed each publish the matching phase and detail,
// and ready is the only phase that names a model.
TEST(ServeTest, ReadinessTracksPhasesAndDetail) {
  tessera::serve::Readiness readiness;
  readiness.Loading("loading weights");
  auto status = readiness.Get();
  EXPECT_EQ(status.phase, tessera::serve::Readiness::Phase::kLoading);
  EXPECT_EQ(status.detail, "loading weights");
  EXPECT_EQ(status.model, nullptr);

  auto* model = reinterpret_cast<tessera::Model*>(std::uintptr_t{0x10});
  auto* tokenizer =
      reinterpret_cast<const tessera::Tokenizer*>(std::uintptr_t{0x20});
  readiness.Ready(model, tokenizer);
  status = readiness.Get();
  EXPECT_EQ(status.phase, tessera::serve::Readiness::Phase::kReady);
  EXPECT_TRUE(status.detail.empty());
  EXPECT_EQ(status.model, model);
  EXPECT_EQ(status.tokenizer, tokenizer);

  readiness.Failed("bad header");
  status = readiness.Get();
  EXPECT_EQ(status.phase, tessera::serve::Readiness::Phase::kFailed);
  EXPECT_EQ(status.detail, "bad header");
  EXPECT_EQ(status.model, nullptr);
  EXPECT_EQ(status.tokenizer, nullptr);
}

// A reader running while the loader flips phases always sees a
// consistent snapshot (never ready without a model).
TEST(ServeTest, ReadinessSafeUnderConcurrentAccess) {
  tessera::serve::Readiness readiness;
  std::atomic<bool> stop{false};
  std::atomic<bool> inconsistent{false};
  std::thread reader([&] {
    while (!stop.load()) {
      const auto status = readiness.Get();
      if (status.phase == tessera::serve::Readiness::Phase::kReady &&
          status.model == nullptr) {
        inconsistent = true;
      }
    }
  });
  auto* model = reinterpret_cast<tessera::Model*>(std::uintptr_t{0x1});
  for (int i = 0; i < 1000; ++i) {
    readiness.Loading("step");
    readiness.Ready(model, nullptr);
  }
  stop = true;
  reader.join();
  EXPECT_FALSE(inconsistent.load());
}
