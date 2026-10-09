#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include "serve/http.hpp"
#include "serve/tools/tool_call.hpp"
#include "core/json.hpp"

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
  std::thread server([&stop] {
    (void)RunHttpServer(
        "127.0.0.1", kPort,
        [](const HttpRequest& request, ResponseWriter& writer) {
          Json body = Json::Object();
          body.Set("path", Json::String(request.path));
          body.Set("body", Json::String(request.body));
          (void)writer.SendHeaders(200, "application/json");
          (void)writer.Write(body.Dump());
        },
        &stop);
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
  std::thread server([&stop] {
    (void)RunHttpServer(
        "127.0.0.1", kPort,
        [](const HttpRequest&, ResponseWriter& writer) {
          (void)writer.SendHeaders(200, "text/event-stream", true);
          (void)writer.Write("data: one\n\n");
          (void)writer.Write("data: two\n\n");
        },
        &stop);
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
