#pragma once

#include <cstddef>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tessera/types.hpp"

// Small shared helpers for the tessera-cli front end (AGENTS.md rule 2:
// one canonical implementation of the flag mapping). The heavy logic
// stays in the library.

namespace tessera::cli {

// Match a `--kv-*` flag and set `type`. Returns false when `arg` is not a
// KV-cache flag, so callers fall through to the next case.
inline bool MatchKvType(std::string_view arg, KvCacheType& type) {
  if (arg == "--kv-f16") {
    type = KvCacheType::F16;
    return true;
  }
  if (arg == "--kv-q8") {
    type = KvCacheType::Q8;
    return true;
  }
  if (arg == "--kv-q4") {
    type = KvCacheType::Q4;
    return true;
  }
  if (arg == "--kv-fp8") {
    type = KvCacheType::FP8;
    return true;
  }
  return false;
}

// Match a diagnostic flag that maps to an environment override and set the
// environment, returning true when handled. `i` advances past the flag's
// value for the flags that take one. The library reads these environment
// variables, so the flag is the command-line form of the same diagnostic.
inline bool ApplyDiagnosticFlag(std::string_view arg, int& i, char** argv,
                                int argc) {
  const auto set = [](const char* name, const char* value) {
    setenv(name, value, 1);
  };
  if (arg == "--wmma") {
    set("TESSERA_MXFP4_WMMA", "1");
    return true;
  }
  if (arg == "--no-wmma") {
    set("TESSERA_MXFP4_WMMA", "0");
    return true;
  }
  if (arg == "--w4a8") {
    set("TESSERA_MXFP4_W4A8", "1");
    return true;
  }
  if (arg == "--target-bf16") {
    set("TESSERA_TARGET_BF16", "1");
    return true;
  }
  if ((arg == "--tiled-min-rows" || arg == "--tiled-min-cols" ||
        arg == "--prefill-attn-pairs") &&
       i + 1 < argc) {
    const char* value = argv[++i];
    const char* name = arg == "--tiled-min-rows"   ? "TESSERA_TILED_MIN_ROWS"
                       : arg == "--tiled-min-cols" ? "TESSERA_TILED_MIN_COLS"
                                                   : "TESSERA_PREFILL_ATTN_PAIRS";
    set(name, value);
    return true;
  }
  return false;
}

namespace detail {

// Split a comma-separated value and convert each non-empty item. Shared
// by the device list and the layer-count list.
template <typename T, typename Convert>
inline std::vector<T> ParseCommaList(std::string_view list, Convert convert) {
  std::vector<T> out;
  std::size_t pos = 0;
  while (pos <= list.size()) {
    const std::size_t comma = list.find(',', pos);
    const std::string_view item =
        list.substr(pos, comma == std::string_view::npos ? std::string_view::npos
                                                         : comma - pos);
    if (!item.empty()) {
      out.push_back(convert(std::string(item)));
    }
    if (comma == std::string_view::npos) {
      break;
    }
    pos = comma + 1;
  }
  return out;
}

}  // namespace detail

// Parse a `--gpus` value into indices: one device ("0") or a
// comma-separated list ("0,1"). Empty items are skipped.
inline std::vector<int> ParseGpuList(std::string_view list) {
  return detail::ParseCommaList<int>(
      list, [](const std::string& item) { return std::stoi(item); });
}

// Parse a `--pipeline-split` value into per-stage layer counts: a
// comma-separated list ("27,21"). Empty items are skipped.
inline std::vector<std::size_t> ParseSizeList(std::string_view list) {
  return detail::ParseCommaList<std::size_t>(
      list, [](const std::string& item) {
        return static_cast<std::size_t>(std::stoul(item));
      });
}

// Which command is routing a multi-device `--gpus` list.
enum class DeviceRouteKind {
  kRunSingle,      // run: one device unless a split flag is set
  kServeReplicas,  // serve: one model replica per device
};

// The hint for a multi-device `--gpus` list, or empty for one device or
// fewer. `kRunSingle` names the split flags when none is active, so a run
// that uses only the first device is explained; `kServeReplicas` reports
// the replica layout and names the split flags as the alternative.
inline std::string DeviceRoutingHint(std::span<const int> devices,
                                     DeviceRouteKind kind, bool split_active) {
  if (devices.size() <= 1) {
    return {};
  }
  std::string list;
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (i != 0) {
      list += ", ";
    }
    list += std::to_string(devices[i]);
  }
  constexpr std::string_view kSplits =
      "--pipeline, --tp, --cp, --spec-split or --hybrid";
  if (kind == DeviceRouteKind::kServeReplicas) {
    return "cli: --gpus " + list +
           " serves one model replica per gpu, so the devices answer "
           "requests in parallel. To split one model across the devices, "
           "run with one of " +
           std::string(kSplits) + ".";
  }
  if (split_active) {
    return {};
  }
  return "cli: --gpus " + list + " names " + std::to_string(devices.size()) +
         " devices but no split flag is set, so only gpu " +
         std::to_string(devices.front()) +
         " is used. To spread one model across the devices, add one of " +
         std::string(kSplits) + ".";
}

}  // namespace tessera::cli
