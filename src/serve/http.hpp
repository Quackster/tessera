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
  // True when the peer went away (closed connection): a non-blocking
  // peek sees EOF or a reset. Generation hooks poll this so a
  // cancelled stream or a closed window aborts the turn and frees the
  // device instead of decoding into the void.
  [[nodiscard]] bool IsPeerGone() const;
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

// A bound listening socket with its log endpoint. RunHttpServerOn
// takes it by value and closes it on every return.
struct BoundListener {
  int fd = -1;
  std::string endpoint;
};

// Bind `host:port` for listening without accepting yet, so the serve
// layer fails fast on a busy port before loading any weights. Logs
// the listening endpoint on success.
[[nodiscard]] std::expected<BoundListener, StatusCode> BindHttpServer(
    const std::string& host, std::uint16_t port, log::Diagnostics& log);

// Accept on a bound listener until `stop` becomes true or a fatal
// socket error occurs. Each connection runs the handler on its own
// worker thread; the handler must be thread-safe and serialize
// device work itself. Closes the listener on return.
[[nodiscard]] std::expected<void, StatusCode> RunHttpServerOn(
    BoundListener listener, const HttpHandler& handler,
    const std::atomic<bool>* stop, log::Diagnostics& log);

}  // namespace tessera::serve
