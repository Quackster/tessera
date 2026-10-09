#include "serve/http.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <system_error>
#include <thread>

namespace tessera::serve {

namespace {

constexpr std::size_t kMaxBodyBytes = 1u << 22;  // 4 MiB
constexpr std::size_t kMaxHeaderBytes = 1u << 16;

std::string Lower(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

std::string StatusText(int status) {
  switch (status) {
    case 200: return "OK";
    case 201: return "Created";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 422: return "Unprocessable Entity";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default: return "Error";
  }
}

// One actionable line: which call failed, on which endpoint, and why.
std::string SocketFailure(std::string_view call, std::string_view endpoint,
                          int code) {
  return std::string(call) + " on " + std::string(endpoint) + " failed: " +
         std::error_code(code, std::system_category()).message();
}

// What to do next for the bind failures a serve operator runs into.
std::string BindHint(int code) {
  if (code == EADDRINUSE) {
    return "; another process already listens on that address. Stop it or "
           "pass --port with a free port";
  }
  if (code == EADDRNOTAVAIL) {
    return "; that address does not belong to this host";
  }
  return {};
}

bool WriteAll(int fd, std::string_view data) {
  std::size_t sent = 0;
  while (sent < data.size()) {
    const ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, 0);
    if (n <= 0) {
      return false;
    }
    sent += static_cast<std::size_t>(n);
  }
  return true;
}

bool ReadHeaders(int fd, std::string& buffer, std::size_t& header_end) {
  header_end = std::string::npos;
  char chunk[4096];
  while (true) {
    const auto found = buffer.find("\r\n\r\n");
    if (found != std::string::npos) {
      header_end = found + 4;
      return true;
    }
    if (buffer.size() > kMaxHeaderBytes) {
      return false;
    }
    const ssize_t got = ::recv(fd, chunk, sizeof(chunk), 0);
    if (got <= 0) {
      return false;
    }
    buffer.append(chunk, static_cast<std::size_t>(got));
  }
}

// One connection: read the request, run the handler, close. Runs on
// a worker thread per connection so a streaming generation never
// blocks pause/stop/session requests (handlers must be thread-safe;
// generation itself is serialized by the caller).
bool ParseHead(const std::string& head, HttpRequest& request,
               std::size_t* content_length);

void ServeConnection(const HttpHandler& handler, int client) {
  std::string buffer;
  std::size_t header_end = std::string::npos;
  if (!ReadHeaders(client, buffer, header_end)) {
    ::close(client);
    return;
  }
  HttpRequest request;
  std::size_t content_length = 0;
  const std::string head = buffer.substr(0, header_end);
  if (!ParseHead(head, request, &content_length)) {
    ResponseWriter writer(client);
    (void)writer.SendHeaders(400, "application/json");
    (void)writer.Write(R"({"error":"bad request line"})");
    writer.Close();
    ::close(client);
    return;
  }
  if (content_length > kMaxBodyBytes) {
    ResponseWriter writer(client);
    (void)writer.SendHeaders(413, "application/json");
    (void)writer.Write(R"({"error":"body too large"})");
    writer.Close();
    ::close(client);
    return;
  }
  std::string body = buffer.substr(header_end);
  char chunk[4096];
  while (body.size() < content_length) {
    const ssize_t got = ::recv(client, chunk, sizeof(chunk), 0);
    if (got <= 0) {
      break;
    }
    body.append(chunk, static_cast<std::size_t>(got));
  }
  if (body.size() < content_length) {
    ResponseWriter writer(client);
    (void)writer.SendHeaders(400, "application/json");
    (void)writer.Write(R"({"error":"truncated body"})");
    writer.Close();
    ::close(client);
    return;
  }
  request.body = body.substr(0, content_length);
  ResponseWriter writer(client);
  handler(request, writer);
  writer.Close();
  ::close(client);
}

// Parse method/path/headers from the header block.
bool ParseHead(const std::string& head, HttpRequest& request,
               std::size_t* content_length) {
  const std::size_t line_end = head.find("\r\n");
  const std::string request_line =
      line_end == std::string::npos ? head : head.substr(0, line_end);
  const std::size_t sp1 = request_line.find(' ');
  const std::size_t sp2 =
      sp1 == std::string::npos ? std::string::npos
                               : request_line.find(' ', sp1 + 1);
  if (sp1 == std::string::npos || sp2 == std::string::npos) {
    return false;
  }
  request.method = request_line.substr(0, sp1);
  request.path = request_line.substr(sp1 + 1, sp2 - sp1 - 1);
  *content_length = 0;
  std::size_t search = line_end == std::string::npos ? head.size() : line_end + 2;
  while (search < head.size()) {
    const std::size_t nl = head.find("\r\n", search);
    if (nl == std::string::npos) {
      break;
    }
    const std::string line = head.substr(search, nl - search);
    const std::size_t colon = line.find(':');
    if (colon != std::string::npos) {
      std::size_t vstart = colon + 1;
      while (vstart < line.size() &&
             (line[vstart] == ' ' || line[vstart] == '\t')) {
        ++vstart;
      }
      std::string name = Lower(line.substr(0, colon));
      std::string value = line.substr(vstart);
      if (name == "content-length") {
        *content_length =
            static_cast<std::size_t>(std::strtoul(value.c_str(), nullptr, 10));
      }
      request.headers.emplace_back(std::move(name), std::move(value));
    }
    search = nl + 2;
  }
  return true;
}

}  // namespace

std::string HttpRequest::Header(std::string_view name) const {
  const std::string key = Lower(name);
  for (const auto& [k, v] : headers) {
    if (k == key) {
      return v;
    }
  }
  return {};
}

void ResponseWriter::SetHeader(std::string name, std::string value) {
  headers_.emplace_back(std::move(name), std::move(value));
}

std::expected<void, StatusCode> ResponseWriter::SendHeaders(
    int status, std::string content_type, bool streaming) {
  if (headers_sent_) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  headers_sent_ = true;
  streaming_ = streaming;
  header_prefix_ = "HTTP/1.1 " + std::to_string(status) + " " +
                   StatusText(status) + "\r\n";
  header_prefix_ += "Content-Type: " + content_type + "\r\n";
  for (const auto& [name, value] : headers_) {
    header_prefix_ += name + ": " + value + "\r\n";
  }
  if (streaming) {
    // Headers must flush before the first chunk.
    header_prefix_ += "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    if (!WriteAll(fd_, header_prefix_)) {
      failed_ = true;
      return std::unexpected(StatusCode::DeviceError);
    }
  }
  return {};
}

std::expected<void, StatusCode> ResponseWriter::Write(std::string_view data) {
  if (!headers_sent_ || failed_) {
    return std::unexpected(StatusCode::DeviceError);
  }
  if (!streaming_) {
    body_.append(data);
    return {};
  }
  if (data.empty()) {
    return {};
  }
  char len[32];
  std::snprintf(len, sizeof(len), "%zx\r\n", data.size());
  if (!WriteAll(fd_, len) || !WriteAll(fd_, data) || !WriteAll(fd_, "\r\n")) {
    failed_ = true;
    return std::unexpected(StatusCode::DeviceError);
  }
  return {};
}

void ResponseWriter::Close() {
  if (failed_) {
    return;
  }
  if (!headers_sent_) {
    (void)SendHeaders(200, "application/json");
  }
  if (streaming_) {
    (void)WriteAll(fd_, "0\r\n\r\n");
    return;
  }
  std::string out = header_prefix_;
  out += "Content-Length: " + std::to_string(body_.size()) + "\r\n";
  out += "Connection: close\r\n\r\n";
  out += body_;
  (void)WriteAll(fd_, out);
}

std::expected<void, StatusCode> RunHttpServer(
    const std::string& host, std::uint16_t port, const HttpHandler& handler,
    const std::atomic<bool>* stop, log::Diagnostics& log) {
  const std::string endpoint = host + ":" + std::to_string(port);
  const int server = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server < 0) {
    log.Error("serve", SocketFailure("socket", endpoint, errno));
    return std::unexpected(StatusCode::DeviceError);
  }
  int one = 1;
  ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    ::close(server);
    log.Error("serve", "bind on " + endpoint +
                           " failed: the host is not an IPv4 literal");
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (::bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    const int code = errno;
    ::close(server);
    log.Error("serve", SocketFailure("bind", endpoint, code) + BindHint(code));
    return std::unexpected(StatusCode::DeviceError);
  }
  if (::listen(server, 8) < 0) {
    const int code = errno;
    ::close(server);
    log.Error("serve", SocketFailure("listen", endpoint, code));
    return std::unexpected(StatusCode::DeviceError);
  }
  log.Info("serve", "listening on " + endpoint);
  while (stop == nullptr || !stop->load()) {
    sockaddr_in peer{};
    socklen_t peer_len = sizeof(peer);
    const int client =
        ::accept(server, reinterpret_cast<sockaddr*>(&peer), &peer_len);
    if (client < 0) {
      // A peer that vanishes mid-handshake is normal; a signal is not
      // a failure. Anything else kills the listener, so say why.
      if (errno == EINTR || errno == ECONNABORTED) {
        continue;
      }
      const int code = errno;
      ::close(server);
      log.Error("serve", SocketFailure("accept", endpoint, code) +
                             "; the server stops accepting requests");
      return std::unexpected(StatusCode::DeviceError);
    }
    // One worker thread per connection: a streaming generation never
    // blocks control requests on other connections.
    std::thread(ServeConnection, handler, client).detach();
  }
  ::close(server);
  return {};
}

}  // namespace tessera::serve
