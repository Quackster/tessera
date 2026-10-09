#pragma once

#include <atomic>
#include <cstdint>
#include <expected>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tessera/log.hpp"
#include "tessera/types.hpp"

// Minimal blocking HTTP/1.1 server for the serving layer. One request
// per connection. A handler either buffers a response (Content-Length)
// or streams it (chunked transfer encoding) for SSE.

namespace tessera::serve {

struct HttpRequest {
  std::string method;
  std::string path;
  std::string body;
  std::vector<std::pair<std::string, std::string>> headers;

  // Case-insensitive header lookup; empty when absent.
  [[nodiscard]] std::string Header(std::string_view name) const;
};

// Response sink handed to the handler. Either Write a buffered body and
// let Close send it with Content-Length, or call SendHeaders(status,
// type, true) and Write chunks for a streaming (SSE) response.
class ResponseWriter {
 public:
  explicit ResponseWriter(int fd) : fd_(fd) {}

  void SetHeader(std::string name, std::string value);
  // Send the status line and headers. `streaming` selects chunked
  // transfer (SSE) over Content-Length.
  [[nodiscard]] std::expected<void, StatusCode> SendHeaders(
      int status, std::string content_type, bool streaming = false);
  [[nodiscard]] std::expected<void, StatusCode> Write(std::string_view data);
  void Close();

 private:
  int fd_ = -1;
  bool headers_sent_ = false;
  bool streaming_ = false;
  bool failed_ = false;
  std::vector<std::pair<std::string, std::string>> headers_;
  // Status line + Content-Type + extra headers (buffered until Close).
  std::string header_prefix_;
  std::string body_;
};

using HttpHandler = std::function<void(const HttpRequest&, ResponseWriter&)>;

// Listen on `host:port` and serve until `stop` becomes true or a fatal
// socket error occurs. The handler runs on the accept thread. Logs the
// listening endpoint once the socket is ready. Every socket failure is
// logged through `log` with the failing call, the endpoint and the
// errno text, and reported as DeviceError (InvalidArgument for a host
// that is not an IPv4 literal).
//
// Usage:
//   RunHttpServer("127.0.0.1", 8080, handler, &stop, engine.Diagnostics());
[[nodiscard]] std::expected<void, StatusCode> RunHttpServer(
    const std::string& host, std::uint16_t port, const HttpHandler& handler,
    const std::atomic<bool>* stop, log::Diagnostics& log);

}  // namespace tessera::serve
