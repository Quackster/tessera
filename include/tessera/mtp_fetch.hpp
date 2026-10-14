#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tessera/types.hpp"

// Fetch the Qwen3.8-Flash-Next MTP head over HTTP. The served GGUF ships no
// MTP head; the BF16 checkpoint (Qwen/Qwen3.8-Flash-Next) carries the 31
// `mtp.*` tensors, and its safetensors headers let HTTP range requests read
// only those tensors. This is the C++ port of Strata's tools/mtp_fetch.py
// (see CREDITS.md): inventory, resumable fetch, offline verify.
//
// The checkpoint revision is pinned so every install reads the same tensors
// and can be checked against the pinned SHA256 digests. The environment
// variables TESSERA_MTP_REVISION (a revision, for example `main`) and
// HF_ENDPOINT (a mirror) override the defaults.
//
// The functions are confined to the calling thread and touch no device. The
// build must link libcurl for HTTP; without it HttpAvailable() is false and
// the functions return UnsupportedFeature.
namespace tessera {

// One MTP tensor's pinned SHA256 digest (the checkpoint contract).
struct MtpTensorDigest {
  std::string_view name;
  std::string_view sha256;
};

// The 31 pinned MTP tensors with their pinned-revision digests, sorted by
// name.
//
// Usage:
//   for (const auto& d : MtpTensorDigests()) ...
[[nodiscard]] std::span<const MtpTensorDigest> MtpTensorDigests();

struct MtpFetchOptions {
  // Directory the tensors, inventory and manifest are written under.
  std::filesystem::path out;
  // Only tensors whose name contains this substring; empty means all.
  std::string only;
  // Optional progress sink for `fetch`: called with the tensor name and the
  // bytes downloaded and expected after each chunk (and once with `done` at
  // the resume point before the first chunk). A null sink logs nothing.
  std::function<void(std::string_view name, std::uint64_t done,
                     std::uint64_t total)>
      progress;
};

// Read `model.safetensors.index.json` and each MTP shard's header over
// HTTP, then write `<out>/mtp-inventory.json`. Every byte range must be a
// 206 with the asked Content-Range, so a mirror that ignores the Range
// header is refused. NetworkError on a transport failure, MalformedFile on
// a malformed header.
//
// Usage:
//   auto status = MtpInventory(MtpFetchOptions{dir, ""});
[[nodiscard]] std::expected<void, StatusCode> MtpInventory(
    const MtpFetchOptions& options);

// Download every MTP tensor named in the inventory, resumable, and write
// `<out>/tensors/<name>.bin` plus `<out>/mtp-manifest.json`. Each tensor is
// checked against its pinned SHA256; wrong bytes are fetched again once,
// then the call fails with MalformedFile. A tensor from another revision is
// fetched whole, not resumed.
//
// Usage:
//   auto status = MtpFetch(MtpFetchOptions{dir, ""});
[[nodiscard]] std::expected<void, StatusCode> MtpFetch(
    const MtpFetchOptions& options);

// Offline SHA256 check of the fetched tensors against the pinned digests.
// Returns the names that are missing or wrong (empty when all are right, or
// when the configured revision is not the pinned one, so the digests do not
// apply). No network access.
//
// Usage:
//   auto bad = MtpVerify(dir);
[[nodiscard]] std::expected<std::vector<std::string>, StatusCode> MtpVerify(
    const std::filesystem::path& out);

}  // namespace tessera
