#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include "serve/http.hpp"
#include "serve/json.hpp"

using tessera::serve::HttpRequest;
using tessera::serve::Json;
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
