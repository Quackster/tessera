#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <random>
#include <string>
#include <string_view>
#include <thread>

#include "serve/http.hpp"
#include "serve/respond.hpp"
#include "serve/session.hpp"
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
  session->Append({"user", "hi", {}, false});
  session->Append({"assistant", "hello", "thinking", false});
  session->Append({"assistant", "again", {}, false});
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

// An unclosed think block never strands text: Finish flushes it.
TEST(ServeTest, ThinkStreamerFlushesUnclosedThink) {
  tessera::serve::ThinkStreamer streamer(/*think_expected=*/true);
  EXPECT_TRUE(streamer.Push("abc").content.empty());
  const auto tail = streamer.Finish();
  EXPECT_TRUE(tail.reasoning.empty());
  EXPECT_EQ(tail.content, "abc");
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
// piece boundaries (fixed seed).
TEST(ServeTest, ThinkStreamerMatchesPostHocSplit) {
  const std::string text =
      "lead <think>deep thought</think> middle <think>more</think> tail";
  std::mt19937 rng(11);
  for (int round = 0; round < 20; ++round) {
    tessera::serve::ThinkStreamer streamer(/*think_expected=*/true);
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

// The default cap is 32k tokens: an unspecified request never
// decodes to the end of a large context window.
TEST(ServeTest, DefaultMaxTokensIs32k) {
  EXPECT_EQ(tessera::kDefaultMaxTokens, 32u * 1024u);
  EXPECT_EQ(tessera::ServeOptions{}.default_max_tokens,
            tessera::kDefaultMaxTokens);
  auto empty = Json::Parse(R"({})");
  ASSERT_NE(empty, nullptr);
  EXPECT_EQ(tessera::serve::MaxTokensFrom(*empty, 7u), 7u);
  auto capped = Json::Parse(R"({"max_tokens":0})");
  ASSERT_NE(capped, nullptr);
  EXPECT_EQ(tessera::serve::MaxTokensFrom(*capped, 7u), 7u);
  auto explicit_count = Json::Parse(R"({"max_tokens":64})");
  ASSERT_NE(explicit_count, nullptr);
  EXPECT_EQ(tessera::serve::MaxTokensFrom(*explicit_count, 7u), 64u);
}

// The served page carries the same default: one constant, every
// surface. Pages without the placeholder pass through untouched.
TEST(ServeTest, InjectWebDefaultsFillsMaxTokens) {
  EXPECT_EQ(tessera::serve::InjectWebDefaults("no placeholders here"),
            "no placeholders here");
  const std::string want = "<input value=\"" +
                           std::to_string(tessera::kDefaultMaxTokens) + "\">";
  EXPECT_EQ(tessera::serve::InjectWebDefaults(
                R"(<input value="@TESSERA_DEFAULT_MAX_TOKENS@">)"),
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
