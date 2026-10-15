#include "core/http.hpp"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <mutex>

#if defined(TESSERA_HAVE_CURL)
#include <curl/curl.h>
#endif

namespace tessera::core {

namespace {

#if defined(TESSERA_HAVE_CURL)
std::once_flag g_curl_once;

void EnsureCurl() {
  std::call_once(g_curl_once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

struct Sink {
  std::vector<std::byte>* body = nullptr;
  std::string* content_range = nullptr;
};

std::size_t WriteBody(char* ptr, std::size_t size, std::size_t count,
                      void* user) {
  auto* sink = static_cast<Sink*>(user);
  const std::size_t total = size * count;
  const auto* bytes = reinterpret_cast<const std::byte*>(ptr);
  sink->body->insert(sink->body->end(), bytes, bytes + total);
  return total;
}

// Capture `Content-Range` from the response headers. A range read is only
// valid when the reply carries the header the caller asked for, so it must
// survive to the caller.
std::size_t WriteHeader(char* ptr, std::size_t size, std::size_t count,
                        void* user) {
  auto* sink = static_cast<Sink*>(user);
  const std::size_t total = size * count;
  const std::string line(ptr, total);
  constexpr std::string_view kKey = "content-range:";
  if (line.size() >= kKey.size()) {
    bool match = true;
    for (std::size_t i = 0; i < kKey.size(); ++i) {
      if (std::tolower(static_cast<unsigned char>(line[i])) != kKey[i]) {
        match = false;
        break;
      }
    }
    if (match) {
      std::string value = line.substr(kKey.size());
      const std::size_t start = value.find_first_not_of(" \t");
      const std::size_t end = value.find_last_not_of(" \r\n");
      *sink->content_range =
          start == std::string::npos ? std::string{}
                                     : value.substr(start, end - start + 1);
    }
  }
  return total;
}
#endif  // TESSERA_HAVE_CURL

}  // namespace

bool HttpAvailable() {
#if defined(TESSERA_HAVE_CURL)
  return true;
#else
  return false;
#endif
}

std::expected<HttpResponse, StatusCode> HttpGet(const HttpRequest& request) {
#if defined(TESSERA_HAVE_CURL)
  EnsureCurl();
  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return std::unexpected(StatusCode::NetworkError);
  }
  HttpResponse response;
  Sink sink{&response.body, &response.content_range};
  char range[48];
  curl_easy_setopt(curl, CURLOPT_URL, request.url.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "tessera-mtp-fetch");
  if (request.timeout_seconds > 0) {
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, request.timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT,
                     request.timeout_seconds < 30 ? request.timeout_seconds : 30L);
  }
  if (request.range.has_value()) {
    std::snprintf(range, sizeof(range), "%llu-%llu",
                  static_cast<unsigned long long>(request.range->first),
                  static_cast<unsigned long long>(request.range->last));
    curl_easy_setopt(curl, CURLOPT_RANGE, range);
  }
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteBody);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &sink);
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, WriteHeader);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, &sink);
  const CURLcode code = curl_easy_perform(curl);
  if (code == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.status);
  }
  curl_easy_cleanup(curl);
  if (code != CURLE_OK) {
    return std::unexpected(StatusCode::NetworkError);
  }
  return response;
#else
  (void)request;
  return std::unexpected(StatusCode::UnsupportedFeature);
#endif
}

}  // namespace tessera::core
