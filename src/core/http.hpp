#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "tessera/types.hpp"

// A minimal HTTP(S) GET client for the MTP head fetch. It supports one
// optional byte range so a safetensors shard is read without downloading
// it whole, follows redirects, and returns the status line's status plus
// the Content-Range response header.
//
// The client is built on libcurl when the build finds it; otherwise
// HttpAvailable() is false and HttpGet returns UnsupportedFeature, so the
// engine still builds without a network library.
//
// Thread-safety: HttpGet is safe for concurrent calls; each call owns its
// transfer.
namespace tessera::core {

// One inclusive byte range [first, last], both ends included.
struct ByteRange {
  std::uint64_t first = 0;
  std::uint64_t last = 0;
};

struct HttpRequest {
  std::string url;
  // When set, the request carries `Range: bytes=first-last`. The caller
  // must check that the reply is a 206 with the matching Content-Range.
  std::optional<ByteRange> range;
  // Per-request timeout in seconds; a value <= 0 uses the default.
  long timeout_seconds = 120;
};

struct HttpResponse {
  // HTTP status code, for example 200 or 206. 0 when the transport failed
  // before a status line arrived.
  long status = 0;
  // The `Content-Range` response header, empty when absent.
  std::string content_range;
  // The response body.
  std::vector<std::byte> body;
};

// True when the build linked libcurl and can perform requests.
[[nodiscard]] bool HttpAvailable();

// Perform one GET. Returns NetworkError on a transport failure (DNS, TLS,
// connection, timeout); an HTTP error status is returned in the response,
// not as a failure, so the caller can decide.
//
// Usage:
//   auto reply = HttpGet(HttpRequest{url, ByteRange{0, 7}});
[[nodiscard]] std::expected<HttpResponse, StatusCode> HttpGet(
    const HttpRequest& request);

}  // namespace tessera::core
